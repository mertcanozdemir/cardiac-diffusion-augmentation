This project implements a Denoising Diffusion Probabilistic Model (DDPM) for generating high-resolution cardiac MRI images. The framework leverages attention-enhanced UNet architectures and progressive noise scheduling to reconstruct realistic grayscale images from noise.

Overview
Architecture: Modified UNet with attention blocks.

Dataset: Cardiac cine MRI slices in grayscale.

Diffusion Framework: diffusers library with custom scheduler and training logic.

Training: Mixed precision with accelerate, cosine learning rate decay, and checkpoint resumption.

Output: Synthesized images and models saved per defined intervals.

Components
main.py: Launches training with configuration, dataloaders, and scheduler.

model.py: Defines the attention-augmented UNet structure.

train.py: Training loop with logging, evaluation, and checkpointing.

generate.py: Generates synthetic MRI samples using a trained model.

dataset.py: Custom PyTorch Dataset with augmentations.

config.py: Training hyperparameters and device setup.

utils.py: Utility functions for evaluation, checkpoint loading, and image grid creation.

Usage
bash
Kopyala
Düzenle
python main.py
After training, run:

bash
Kopyala
Düzenle
python generate.py
Requirements
Python ≥ 3.8

PyTorch

diffusers

accelerate

torchvision

PIL, tqdm
