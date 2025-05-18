import datetime
import glob
import math
import os
import random
from pathlib import Path

import tensorflow as tf
import keras
from dotmap import DotMap
from matplotlib import pyplot as plt
from tensorflow.keras import layers, Model, optimizers, losses
import numpy as np

from model_cnn import ResidualBlock
from dataloader import load_audio
from feature_extraction import get_spectrogram

from tqdm import tqdm


class ResidualBlockDown(layers.Layer):
    def __init__(self, channels_out):
        super().__init__()
        self.conv1 = layers.Conv2D(channels_out, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.batch_norm1 = layers.BatchNormalization(axis=3)
        self.relu = layers.Activation("relu")
        self.conv2 = layers.Conv2D(channels_out, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.batch_norm2 = layers.BatchNormalization(axis=3)
        self.res_conv = layers.Conv2D(channels_out, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.add = layers.Add()
        self.downsample = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, x, training=None, mask=None):
        res = x
        
        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)

        # Residual connection
        res = self.res_conv(res)  # pass residual through conv layer so that #filters matches x for adding
        x = self.add((x, res))  # add the residual to x

        x = self.relu(x)
        x_downsampled = self.downsample(x)
        return x_downsampled, x


class ResidualBlockUp(layers.Layer):
    def __init__(self, channels_out):
        super().__init__()
        self.conv1 = layers.Conv2D(channels_out, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.batch_norm1 = layers.BatchNormalization(axis=3)
        self.relu = layers.Activation("relu")
        self.conv2 = layers.Conv2D(channels_out, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.batch_norm2 = layers.BatchNormalization(axis=3)
        self.res_conv = layers.Conv2D(channels_out, kernel_size=(1, 1), padding="same")
        self.add = layers.Add()
        # self.upsample = layers.UpSampling2D(size=(2, 2))
        self.upsample = layers.Conv2DTranspose(channels_out, kernel_size=(3, 3), strides=(2, 2), padding="same")

    def call(self, x, training=None, mask=None):
        res = x

        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)

        # Residual connection
        res = self.res_conv(res)
        x = self.add((x, res))

        x = self.relu(x)
        x = self.upsample(x)
        return x


class ResidualBlockDownW(ResidualBlockDown):
    def __init__(self, channels_out):
        super(ResidualBlockDownW, self).__init__(channels_out)
        self.downsample = layers.AveragePooling2D(pool_size=(1, 2), strides=(1, 2))


class ResidualBlockUpW(ResidualBlockUp):
    def __init__(self, channels_out):
        super(ResidualBlockUpW, self).__init__(channels_out)
        self.upsample = layers.Conv2DTranspose(channels_out, kernel_size=(3, 3), strides=(1, 2), padding="same")


class Denoiser(Model):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.padding = layers.ZeroPadding2D(padding=((0, 0), (2, 3)))
        self.res_down1 = ResidualBlockDown(64)
        self.res_down2 = ResidualBlockDown(128)
        self.res_down3 = ResidualBlockDown(256)
        self.res_down_h = ResidualBlockDownW(256)
        self.res_up_h = ResidualBlockUpW(256)
        self.res_up1 = ResidualBlockUp(256)
        self.res_up2 = ResidualBlockUp(128)
        self.res_up3 = ResidualBlockUp(64)
        self.res_final = ResidualBlock(32)
        self.conv_out = layers.Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.add = layers.Add()
        self.crop = layers.Cropping2D(cropping=((0, 0), (2, 3)))

    def call(self, x, training=None, mask=None):
        x = self.padding(x, training=training)

        x_down1, x_down1_res = self.res_down1(x, training=training)
        x_down2, x_down2_res = self.res_down2(x_down1, training=training)
        x_down3, x_down3_res = self.res_down3(x_down2, training=training)
        x_down_h, _ = self.res_down_h(x_down3, training=training)

        x_up_h = self.res_up_h(x_down_h, training=training)

        x_up1 = self.res_up1(x_up_h, training=training)
        # x_up1 = self.add((x_up1, x_down3_res))

        x_up2 = self.res_up2(x_up1, training=training)
        # x_up2 = self.add((x_up2, x_down2_res))

        x_up3 = self.res_up3(x_up2, training=training)
        # x_up3 = self.add((x_up3, x_down1_res))

        x_out = self.res_final(x_up3, training=training)
        x_out = self.conv_out(x_out, training=training)
        x_out = self.crop(x_out, training=training)

        return x_out


class DenoiserDataLoader(tf.keras.utils.Sequence):
    def __init__(self, batch_size, dataset_dir, split, args, seed):
        super(DenoiserDataLoader).__init__()
        self.batch_size = batch_size
        self.args = args
        self.audio_paths = glob.glob(os.path.join(dataset_dir, split, "*/*.wav"))
        self.seed = seed

        random.seed(seed)
        random.shuffle(self.audio_paths)

    def __getitem__(self, idx):
        lower = idx * self.batch_size
        upper = min(lower + self.batch_size, len(self.audio_paths))
        audio_paths_batch = self.audio_paths[lower:upper]

        audio_batch = np.array([get_spectrogram(load_audio(f, args=self.args, augment=True, seed=self.seed)[0],
                                                 n_mels=self.args.n_mels)[:, :, np.newaxis]
                                for f in audio_paths_batch])
        targets_batch = np.array([get_spectrogram(load_audio(f, args=self.args, augment=False, seed=self.seed)[0],
                                                 n_mels=self.args.n_mels)[:, :, np.newaxis]
                                for f in audio_paths_batch])

        return audio_batch, targets_batch

    def __len__(self):
        return math.ceil(len(self.audio_paths) / self.batch_size)


def validate(model, val_loader, loss_fn):
    total_loss = np.array([0.])
    n_batches = 0

    for audio_batch, labels_batch in tqdm(val_loader, desc="Validating", unit="batch"):
        if len(audio_batch) == 0:
            break

        n_batches += 1
        pred = model(audio_batch, training=False)
        loss = tf.reduce_mean(loss_fn(labels_batch, pred))
        total_loss[0] += loss.numpy()

    return total_loss / n_batches  # return the mean validation loss


def train(args):
    model = Denoiser()

    # Dataloaders for train and val sets
    train_loader = DenoiserDataLoader(args.batch_size, args.dataset_dir, "train", args, seed=45)
    val_loader = DenoiserDataLoader(args.batch_size, args.dataset_dir, "val", args, seed=49)

    # optimiser = optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum, decay=args.weight_decay)
    optimiser = optimizers.Adam(learning_rate=0.0001)
    loss_fn = losses.mean_squared_error

    # Set up logs
    os.makedirs(args.logs_dir, exist_ok=True)
    log_path = os.path.join(args.logs_dir, f"denoiser_train_log_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt")

    best_val_loss = float("inf")

    with open(log_path, "a") as f_log:
        for epoch in range(args.epochs):
            total_loss = np.array([0.])
            n_batches = 0

            for noisy_audio, clean_audio in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
                if len(noisy_audio) == 0:
                    break

                n_batches += 1

                with tf.GradientTape() as tape:
                    pred = model(noisy_audio, training=True)
                    loss = tf.reduce_mean(loss_fn(clean_audio, pred))
                    total_loss[0] += loss.numpy()

                # Get the gradient of loss with respect to each trainable parameter and perform grad descent
                grads = tape.gradient(loss, model.trainable_variables)
                optimiser.apply_gradients(list(zip(grads, model.trainable_variables)))

            mean_loss = total_loss / n_batches

            # Validate after every epoch
            val_loss = validate(model, val_loader, loss_fn)

            if val_loss < best_val_loss:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                model.save_weights(os.path.join(args.checkpoint_path, "best.ckpt".format(epoch=epoch)))
                best_val_loss = val_loss

            tqdm.write(f"Train Loss: {mean_loss.item()}, Val Loss: {val_loss.item()}")
            f_log.write(f"Epoch: {epoch}/{args.epochs}, Train Loss: {mean_loss.item()}, Val Loss: {val_loss.item()}\n")

    return best_val_loss


def test(args):
    model = Denoiser()
    test_loader = DenoiserDataLoader(1, args.dataset_dir, "test", args, seed=52)
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(args.checkpoint_path)

    for noisy_audio, clean_audio in tqdm(test_loader, desc="Testing", unit="samples"):
        pred = model(noisy_audio, training=False)

        fig, axs = plt.subplots(1, 3, figsize=(20, 10))

        axs[0].imshow(noisy_audio[0])
        axs[0].axis("off")  # hide axes for a cleaner look
        axs[0].set_title("Noisy Input")

        # Display each image in its respective subplot
        axs[1].imshow(pred[0])
        axs[1].axis("off")
        axs[1].set_title('Denoised (Prediction)')

        axs[2].imshow(clean_audio[0])
        axs[2].axis("off")
        axs[2].set_title("Clean Audio")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mode = "train"

    if mode == "train":
        args = DotMap()
        args.batch_size = 8
        args.dataset_dir = "splits"
        args.logs_dir = "logs"
        args.checkpoint_path = "denoiser_checkpoints"
        args.epochs = 1000
        args.num_classes = 20
        args.learning_rate = 0.001
        args.momentum = 0.9
        args.weight_decay = 0.00001
        args.audio_duration = 3
        args.augmentation = True
        args.noise_enabled = True
        args.p_noise = 0.75
        args.noise_min_volume = 0.5
        args.noise_max_volume = 1.0
        args.n_mels = 40

        train(args)
    elif mode == "test":
        args = DotMap()
        args.batch_size = 8
        args.dataset_dir = "splits"
        args.logs_dir = "logs"
        args.checkpoint_path = "denoiser_checkpoints/best.ckpt"
        args.epochs = 300
        args.num_classes = 20
        args.learning_rate = 0.001
        args.momentum = 0.9
        args.weight_decay = 0.001
        args.audio_duration = 3
        args.augmentation = True
        args.noise_enabled = True
        args.p_noise = 0.75
        args.noise_min_volume = 0.5
        args.noise_max_volume = 1.0
        args.n_mels = 40

        test(args)
