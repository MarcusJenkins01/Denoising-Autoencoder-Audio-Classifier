import math
import random
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
import os
import soundfile as sf

from feature_extraction import get_mfccs_deltas
from data_augmentation import add_random_noise_active


def one_hot_encoding(class_name, class_list):
    one_hot = np.zeros((len(class_list),), dtype=np.float32)
    class_idx = class_list.index(class_name)
    one_hot[class_idx] = 1.
    return one_hot


def load_audio(audio_path, args, augment=True, seed=42):
    random.seed(seed)
    signal, sr = sf.read(audio_path)

    # Data augmentation
    if args.augmentation and augment:
        if args.noise_enabled:
            noise_chance = random.uniform(0, 1)
            if noise_chance < args.p_noise:
                signal = add_random_noise_active(signal, sr, args, seed=seed)

    return signal, sr


class AudioLoader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset_dir, split, args, shuffle=False, seed=42, augment=True, denoiser=None):
        super(AudioLoader).__init__()
        self.batch_size = batch_size
        self.augment = augment
        self.args = args
        self.audio_paths = [x for x in glob.glob(os.path.join(dataset_dir, split, "*/*.wav"))]
        self.denoiser = denoiser
        self.seed = seed

        if shuffle:
            random.seed(seed)
            random.shuffle(self.audio_paths)

        self.labels = [os.path.basename(os.path.dirname(f))
                       for f in self.audio_paths]  # get class from the folder name

    def load_audio_features(self, f):
        signal, sr = load_audio(f, args=self.args, augment=self.augment, seed=self.seed)
        return get_mfccs_deltas(signal, n_mfcc=self.args.n_mfcc, n_mels=self.args.n_mels,
                                denoiser=self.denoiser)[:, :, np.newaxis]

    def __getitem__(self, idx):
        lower = idx * self.batch_size
        upper = min(lower + self.batch_size, len(self.audio_paths))
        audio_paths_batch = self.audio_paths[lower:upper]
        labels_batch = self.labels[lower:upper]

        # Batch of MFCCs and temporal derivatives features
        audio_batch = np.array([self.load_audio_features(f) for f in audio_paths_batch])

        # One hot encoded
        labels_batch = np.array([one_hot_encoding(cls, self.args.classes) for cls in labels_batch])

        return audio_batch, labels_batch

    def __len__(self):
        return math.ceil(len(self.audio_paths) / self.batch_size)

