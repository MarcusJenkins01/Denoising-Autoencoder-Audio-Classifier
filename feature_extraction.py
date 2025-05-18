import random

import numpy as np
from scipy.fftpack import dct
import matplotlib.pyplot as plt

sr = 44100
f_max = sr // 2  # Maximum frequency (Nyquist's theorem states the maximum frequency needed is sample rate / 2)

# Fixed parameters for Mel spectrogram and MFFCs
n_fft = int(sr * (20 / 1000))  # FFT window size (20 ms is chosen for speech)
hop_length = n_fft // 2  # The window overlap
f_min = 0  # The minimum frequency (hz)

# Parameters for delta and delta-delta
delta_window_size = 2

dataset_spec_mean = -35.774660811707975
dataset_spec_std = 19.45601474758142

dataset_mean = 0.11858076303909708
dataset_std = 10.648327634379863

random.seed(42)


def hamming_window(M):
    n = np.linspace(0, M - 1, M)
    return 0.54 - 0.46 * np.cos((2 * np.pi * n) / (M - 1))


# Use STFT (short-time Fourier transform) to get the frequency distribution across each time frame
def stft(signal, n_fft, hop_length):
    window = hamming_window(n_fft)

    num_frames = int((len(signal) - n_fft) / hop_length) + 1
    stft_out = np.empty((num_frames, n_fft), dtype=complex)

    for i in range(num_frames):
        cursor = i * hop_length
        frame = signal[cursor:cursor + n_fft] * window
        stft_out[i, :] = np.fft.fft(frame, n=n_fft)

    return np.abs(stft_out.T), np.angle(stft_out.T)


def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


# Create Mel filter bank
def mel_filter_bank(n_fft, n_mels, sr):
    # Mel filter bank calculation
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(f_max), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor(hz_points * (n_fft + 1) / sr).astype(int)

    # Create the filter that converts each frequency range to a discrete Mel value (binning) when dot product is used
    filter_bank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        # print(i - 1, bin_points[i - 1], bin_points[i])  # debug start and end points of Mel band range
        # print(bin_points[i] - bin_points[i - 1])  # debug the length of the first half of the triangle
        # print(bin_points[i + 1] - bin_points[i])  # debug the length of the second half of the triangle

        bin_lower = bin_points[i - 1]
        bin_centre = bin_points[i]
        bin_upper = bin_points[i + 1]

        # Triangular window
        filter_bank[i - 1, bin_lower:bin_centre] = (
            np.linspace(0, 1, bin_centre - bin_lower))
        filter_bank[i - 1, bin_centre:bin_upper] = (
            np.linspace(1, 0, bin_upper - bin_centre))

    return filter_bank


def compute_spectrogram(signal, power=True):
    # Use STFT to get a time-frequency spectrogram
    stft_magnitude, _ = stft(signal, n_fft, hop_length)

    # Convert the spectrogram to an abs or power spectrogram
    spectrogram = np.abs(stft_magnitude)
    if power:
        spectrogram = spectrogram ** 2

    return spectrogram[:n_fft // 2 + 1, :]


def compute_mel_spectrogram(signal, n_mels, power=True):
    spectrogram = compute_spectrogram(signal, power=power)

    # Apply Mel filters to convert to a Mel spectrogram (power of each frequency is scaled by the Mel scale)
    mel_filter = mel_filter_bank(n_fft, n_mels, sr)
    mel_spectrogram = np.dot(mel_filter, spectrogram)  # only use first half of the spectrogram

    return mel_spectrogram


def mel_spectrogram_to_db(mel_spectrogram, power=True):
    # Convert the powers across each frequency range to the powers of 10
    eps = 1e-9  # add a small eps value to prevent log of zero (undefined)
    return (10. if power else 20.) * np.log10(mel_spectrogram + eps)


def normalise(x, mean, std):
    return (x - mean) / std


def compute_deltas(x, n_mfcc, window_size):
    # Pad to keep the number of frames intact (the window will use zeros for the convolutions on boundary frames)
    x_pad = np.pad(x, ((0, 0), (window_size, window_size)), mode="edge")

    _, n_frames = x.shape
    deltas = np.empty((n_mfcc, n_frames))
    denominator = 2 * np.sum(np.power(np.arange(1, window_size + 1), 2))

    # Use a 1d convolutional approach to calculate the gradients
    weights = np.arange(-window_size, window_size + 1)
    for ri in range(n_mfcc):
        mfcc_row = x_pad[ri, :]
        deltas[ri, :] = np.correlate(mfcc_row, weights, mode="valid") / denominator

    return deltas


def get_spectrogram(signal, n_mels):
    # Detect if stereo
    if len(signal.shape) > 1:
        signal = np.mean(signal, axis=1)  # average the stereo channels to mono

    mel_spectrogram = compute_mel_spectrogram(signal, n_mels, power=True)
    mel_db_spectrogram = mel_spectrogram_to_db(mel_spectrogram, power=True)

    mel_db_spectrogram = normalise(mel_db_spectrogram, dataset_spec_mean, dataset_spec_std)

    return mel_db_spectrogram


def get_mfccs(signal, n_mfcc, n_mels, denoiser=None):
    mel_db_spectrogram = get_spectrogram(signal, n_mels)

    if denoiser is not None:
        mel_db_spectrogram = denoiser(mel_db_spectrogram[np.newaxis, :, :, np.newaxis]).numpy().squeeze(axis=(0, -1))

    _, n_frames = mel_db_spectrogram.shape
    mfccs = np.empty((n_mfcc, n_frames))

    for fi, frame in enumerate(mel_db_spectrogram.T):
        mfccs[:, fi] = dct(frame)[:n_mfcc]

    return mfccs


def get_mfccs_deltas(signal, n_mfcc, n_mels, denoiser=None):
    mfccs = get_mfccs(signal, n_mfcc, n_mels, denoiser)
    deltas = compute_deltas(mfccs, n_mfcc=n_mfcc, window_size=delta_window_size)
    delta_deltas = compute_deltas(deltas, n_mfcc=n_mfcc, window_size=delta_window_size)

    features = np.concatenate([mfccs, deltas, delta_deltas], axis=0)
    features = normalise(features, mean=dataset_mean, std=dataset_std)
    return features

