# Denoising Autoencoder Audio Classifier

This uses a ResNet-based CNN for audio classification of Mel-frequency cepstral coefficients and delta and delta-delta features. It also includes a denoising autoencoder (DAE) to remove background noise from audio samples.
The DAE requires the musan dataset (https://www.openslr.org/17/). All musan audio files should be placed in a folder called "musan" in the root directory, and all audio files should be in WAV format.

The size of the Mel features depends on the length of the audio clip, and so all audio lengths should fit into the convolutional strides to prevent non-homogenous tensors. In future we may fix this by adding automatic padding.
