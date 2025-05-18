import pydub as pd
import soundfile as sf
import glob
import time
import random
import numpy as np
from pydub.utils import ratio_to_db


def segment_to_np(segment):
    return np.array(segment.get_array_of_samples()) / 32768.0


def get_audio_segment(audio, sample_rate):
    channels = 1 if len(audio.shape) == 1 else audio.shape[1]
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_segment = pd.AudioSegment(audio_int16.tobytes(), frame_rate=sample_rate,
                                    sample_width=audio_int16.dtype.itemsize, channels=channels)
    return audio_segment, channels


def get_random_noise(audio_segment, channels, args, seed):
    random.seed(seed)
    
    # Select a random background noise from MUSAN
    noise_files = glob.glob("musan/*.wav")

    random.shuffle(noise_files)
    noise_path = noise_files[0]
    noise = pd.AudioSegment.from_file(noise_path)

    # Apply random volume change to the noise
    volume = random.uniform(args.noise_min_volume, args.noise_max_volume)
    noise = noise.apply_gain(ratio_to_db(volume))
    noise = noise.set_channels(channels)
    noise = noise.set_sample_width(audio_segment.sample_width)

    # Randomly select a start point for the noise
    start_point = random.randint(0, len(noise) - args.audio_duration * 1000)
    end_point = start_point + args.audio_duration * 1000
    noise_segment = noise[start_point:end_point]

    return noise_segment


def add_random_noise_active(audio, sample_rate, args, seed):
    # Convert audio to audio segment from numpy array
    audio_segment, channels = get_audio_segment(audio, sample_rate)

    # Get a random noise segment
    noise_segment = get_random_noise(audio_segment, channels, args, seed)

    # Overlay noise onto the clean recording
    noisy_audio = audio_segment.overlay(noise_segment)

    return segment_to_np(noisy_audio)
