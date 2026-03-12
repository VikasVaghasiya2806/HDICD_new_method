# HDICD New Method

This repository contains a unified framework that combines the strengths of **HypCD** (Hyperbolic Category Discovery) and **HIDISC** (Hyperbolic Domain Generalized Category Discovery) to improve performance on Generalized Category Discovery (GCD) and Domain Generalized Category Discovery (DG-GCD).

## Project Structure
```
HDICD_new_method/
├── augmentation/      # Tangent CutMix and Domain Augmentations
├── backbone/          # DINO/DINOv2 backbone
├── configs/           # YAML configurations
├── datasets/          # CIFAR-100, CUB-200, PACS, OfficeHome loaders
├── hyperbolic/        # Poincaré mathematical ops and Möbius layers
├── losses/            # InfoNCE, Busemann, Outlier Repulsion, and Classifier losses
├── models/            # Main HDICD model integrating all parts
├── scripts/           # Training and testing scripts
└── training/          # Trainer and optimizer setups
```

## Setup & Running Experiments

### Requirements
- PyTorch >= 1.13.0
- torchvision >= 0.14.0
- pyyaml
- tqdm
- scipy
- numpy

### Training
To train the model on CIFAR-100, you can run:
```bash
python scripts/train.py --config configs/config.yaml
```

### Evaluation
To evaluate a trained model:
```bash
python scripts/evaluate.py --config configs/config.yaml
```

## Key Features
1. **Hyperbolic Mapping**: Maps Euclidean feature vectors to the Poincaré ball using learnable curvature.
2. **Tangent CutMix**: Generates synthetic pseudo-novel class embeddings in the tangent Euclidean space before mapping back.
3. **Hyperbolic Prototype Classifier**: Predicts class labels directly in hyperbolic space.
4. **Unified Loss**: A hybrid composition of Busemann Alignment Loss, Hyperbolic Contrastive InfoNCE Loss, Adaptive Outlier Repulsion, and Prototype Classification Loss.
