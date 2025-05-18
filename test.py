import os.path

import numpy as np
from tqdm import tqdm
from dotmap import DotMap
import tensorflow as tf

from denoiser import Denoiser
from dataloader import AudioLoader
from model_cnn import CNNClassifier


def test(args):
    model = CNNClassifier(num_classes=args.num_classes)
    checkpoint = tf.train.Checkpoint(model)
    checkpoint.restore(args.checkpoint_path)

    denoiser = Denoiser()
    checkpoint = tf.train.Checkpoint(denoiser)
    checkpoint.restore("denoiser_checkpoints/best.ckpt")

    test_loader = AudioLoader(1, args.dataset_dir, "test", args=args, augment=False, denoiser=denoiser)

    n_correct = 0
    n_samples = 0

    y_pred = []
    y_true = []

    for audio_batch, labels_batch in tqdm(test_loader, desc="Testing", unit="samples"):
        if len(audio_batch) == 0:
            break

        n_samples += len(audio_batch)

        pred = model(audio_batch, training=False)
        pred_idx = int(np.argmax(pred[0]))
        class_idx = int(np.argmax(labels_batch[0]))

        y_pred.append(pred_idx)
        y_true.append(class_idx)

        if class_idx == pred_idx:
            n_correct += 1

    accuracy = n_correct / n_samples
    return accuracy, y_true, y_pred


if __name__ == "__main__":
    args = DotMap()
    args.dataset_dir = "splits"
    args.logs_dir = "logs"
    args.num_classes = 20
    args.checkpoint_path = os.path.abspath("./checkpoints/best.ckpt")
    args.augmentation = False
    args.n_mels = 40
    args.n_mfcc = 19

    accuracy, y_true, y_pred = test(args)
    print(f"Test accuracy: {accuracy}")
