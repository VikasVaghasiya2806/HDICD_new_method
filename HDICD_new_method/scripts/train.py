import os
import sys
import yaml
import torch
import argparse
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hdicd_model import HDICDModel
from training.optimizer import get_optimizer, get_scheduler
from training.trainer import HDICDTrainer
from datasets.cifar_loader import get_cifar100_dataloaders
from augmentation.domain_augment import get_domain_augmentations

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    device = torch.device(f"cuda:{config['system']['device_id']}" if torch.cuda.is_available() else "cpu")

    # Load Data
    transform = get_domain_augmentations(224)
    # Using CIFAR100 logic for testing script
    train_loader, test_loader = get_cifar100_dataloaders(
        root=config['dataset']['data_path'],
        batch_size=config['dataset']['batch_size'],
        train_transform=transform, test_transform=transform
    )

    # Initialize Model
    model = HDICDModel(
        arch=config['model']['arch'],
        patch_size=config['model']['patch_size'],
        emb_dim=config['model']['emb_dim'],
        num_classes=config['dataset']['num_classes'],
        c=config['model']['c'],
        do_hyperbolic=config['model']['do_hyperbolic']
    ).to(device)

    # Optimizers
    optimizer = get_optimizer(model, lr=config['training']['lr'], weight_decay=config['training']['weight_decay'], momentum=config['training']['momentum'])
    scheduler = get_scheduler(optimizer, epochs=config['training']['epochs'], lr=config['training']['lr'])

    # Trainer
    trainer = HDICDTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=config['dataset']['num_classes'],
        emb_dim=config['model']['emb_dim'],
        penalty_value=config['training']['penalty_value']
    )

    print("Initializing prototypes...")
    trainer.initialize_prototypes(proto_epochs=config['training']['proto_epochs'], proto_lr=config['training']['proto_lr'])

    print("Starting training...")
    for epoch in range(config['training']['epochs']):
        loss = trainer.train_epoch(train_loader, epoch, config['training']['epochs'], n_views=config['training']['n_views'], alpha_d=config['training']['alpha_d'])
        print(f"Epoch {epoch} Loss: {loss:.4f}")

if __name__ == '__main__':
    main()
