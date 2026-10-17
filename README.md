# Evaluating Pre-Trained CV Models for Animal Image Classification

Benchmarks pre-trained computer vision architectures for **image classification** across 90 animal species, with a systematic comparison of model performance with and without custom data augmentation.

## Overview

The project evaluates how transfer learning from ImageNet-pretrained models performs on fine-grained animal classification, and whether hand-crafted augmentation strategies (random cropping, Gaussian blur, colour jitter) improve generalisation.

All models are evaluated with **5-fold cross-validation**.

## Notebooks

| Notebook | Description |
|---|---|
| `FinalNotebook.ipynb` | Full pipeline: augmentation, ResNet18, ResNet50, ViT — comparison and results |
| `transfer_learning.ipynb` | Transfer learning baseline experiments |
| `Augmentation.ipynb` | Custom augmentation techniques |
| `transfer_learning_augment.ipynb` | Combined transfer learning + augmentation experiments |

## Architectures

| Model | Notes |
|---|---|
| ResNet18 | Included for completeness; ResNet50 was the primary CNN |
| ResNet50 | Main CNN — deeper, stronger generalisation |
| ViT (Vision Transformer) | `google/vit-base-patch16-224-in21k` — significant improvement over CNN baseline |

## Methods

- Custom augmentation pipeline: random cropping, Gaussian blur, colour jitter, distortions
- Fine-tuning of ImageNet-pretrained weights
- 5-fold cross-validation for robust evaluation
- Confusion matrices, accuracy comparisons across augmentation conditions

## Stack

Python · PyTorch · torchvision · HuggingFace (`transformers`) · scikit-learn · OpenCV
