import os
import random

import numpy as np

from denoiser import Denoiser
from denoiser import train as train_denoiser
from dataloader import AudioLoader
from model_cnn import CNNClassifier
from dotmap import DotMap
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tqdm import tqdm
import datetime


def validate(model, val_loader, loss_fn):
    total_loss = np.array([0.])
    n_samples = 0

    for audio_batch, labels_batch in tqdm(val_loader, desc="Validating", unit="batch"):
        if len(audio_batch) == 0:
            break

        n_samples += len(audio_batch)
        pred = model(audio_batch, training=False)
        loss = loss_fn(labels_batch, pred)
        total_loss[0] += loss.numpy()

    return total_loss / n_samples  # return the mean validation loss


def train(args):
    model = CNNClassifier(num_classes=args.num_classes)

    denoiser = Denoiser()

    # Dataloaders for train and val sets
    train_loader = AudioLoader(args.batch_size, args.dataset_dir, "train", args=args, shuffle=True, seed=43, denoiser=denoiser)
    val_loader = AudioLoader(args.batch_size, args.dataset_dir, "val", args=args, augment=True, seed=62, denoiser=denoiser)

    optimiser = optimizers.SGD(learning_rate=args.learning_rate, momentum=args.momentum, decay=args.weight_decay)
    loss_fn = losses.CategoricalCrossentropy()

    # Set up logs directory
    os.makedirs(args.logs_dir, exist_ok=True)
    log_path = os.path.join(args.logs_dir, f"train_log_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt")

    best_val_loss = float("inf")

    with open(log_path, "a") as f_log:
        for epoch in range(args.epochs):
            total_loss = np.array([0.])
            n_samples = 0

            for audio_batch, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch"):
                if len(audio_batch) == 0:
                    break

                n_samples += len(audio_batch)

                with tf.GradientTape() as tape:
                    pred = model(audio_batch, training=True)
                    loss = loss_fn(labels_batch, pred)
                    total_loss[0] += loss.numpy()

                # Get the gradient of loss with respect to each trainable parameter and perform grad descent
                grads = tape.gradient(loss, model.trainable_variables)
                optimiser.apply_gradients(list(zip(grads, model.trainable_variables)))

            mean_loss = total_loss / n_samples

            # Validate after every epoch
            val_loss = validate(model, val_loader, loss_fn)

            if val_loss < best_val_loss:
                model.save_weights(os.path.join(args.checkpoint_path, "den_temp/best.ckpt".format(epoch=epoch)))
                best_val_loss = val_loss

            tqdm.write(f"Train Loss: {mean_loss.item()}, Val Loss: {val_loss.item()}")
            f_log.write(f"Epoch: {epoch}/{args.epochs}, Train Loss: {mean_loss.item()}, Val Loss: {val_loss.item()}\n")

    return best_val_loss


if __name__ == "__main__":
    # Train the audio classifier
    random.seed(43)
    args = DotMap()
    args.batch_size = 8
    args.dataset_dir = "splits"
    args.logs_dir = "logs"
    args.checkpoint_path = "checkpoints"
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
    args.n_mfcc = 19

    train(args)

