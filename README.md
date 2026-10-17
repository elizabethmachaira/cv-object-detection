# Evaluation of Pre-Trained CV Architectures for Object Detection

Benchmarks pre-trained computer vision architectures for object detection on an animal image dataset, with a focus on the impact of data augmentation techniques on model performance.

## Overview

Transfer learning from ImageNet-pretrained models is evaluated across multiple architectures. The study examines how augmentation strategies (flipping, rotation, colour jitter, etc.) affect classification accuracy, generalisation, and robustness.

## Notebooks

| Notebook | Description |
|---|---|
| `FinalNotebook.ipynb` | Main results: architecture comparison and final evaluation |
| `transfer_learning.ipynb` | Transfer learning baseline experiments |
| `Augmentation.ipynb` | Augmentation strategies and their effect on performance |
| `transfer_learning_augment.ipynb` | Combined transfer learning + augmentation experiments |

## Architectures Evaluated

- ResNet
- VGG
- EfficientNet

## Methods

- Fine-tuning of pre-trained ImageNet models
- Systematic data augmentation pipeline
- Performance comparison: accuracy, loss, confusion matrices

## Stack

Python · PyTorch · torchvision · matplotlib
