import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models, Model
import numpy as np


class ResidualBlock(layers.Layer):
    def __init__(self, channels_out):
        super().__init__()
        self.conv1 = layers.Conv2D(channels_out, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.batch_norm1 = layers.BatchNormalization(axis=3)
        self.relu = layers.Activation("relu")
        self.conv2 = layers.Conv2D(channels_out, kernel_size=(3, 3), strides=(1, 1), padding="same")
        self.batch_norm2 = layers.BatchNormalization(axis=3)
        self.res_conv = layers.Conv2D(channels_out, kernel_size=(1, 1), padding="same")
        self.add = layers.Add()

    def call(self, x, training=None, mask=None):
        res = x

        # Core of the ResNet block
        x = self.conv1(x)
        x = self.batch_norm1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x, training=training)

        # Residual connection
        res = self.res_conv(res)  # pass residual through conv layer so that #filters matches x for adding
        x = self.add((x, res))  # add the residual to x

        x = self.relu(x)
        return x


def residual_block_layer(channels_out, n_layers=1):
    return models.Sequential([ResidualBlock(channels_out)
                              for _ in range(n_layers)])


class CNNClassifier(Model):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.conv1 = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), activation="relu")
        self.max_pool = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))
        self.global_pool = layers.GlobalAveragePooling2D()

        self.res_blocks = models.Sequential([
            residual_block_layer(64, n_layers=2),
            residual_block_layer(128, n_layers=2),
            # layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)),
            residual_block_layer(256, n_layers=2),
            residual_block_layer(512, n_layers=2),
        ])

        self.linear_out = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=None, mask=None):
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.res_blocks(x, training=training)
        x = self.global_pool(x)
        x = self.linear_out(x)
        return x


# model = CNNClassifier(num_classes=20)
#
# out = model(mel_db_spectrogram, training=True)
# print(out.shape)
#
# model.summary()
