import os
import sys
import yaml
import torch
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hdicd_model import HDICDModel
from datasets.cifar_loader import get_cifar100_dataloaders
from augmentation.domain_augment import get_test_augmentations

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    print(f"Accuracy: {100.0 * correct / total:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{config['system']['device_id']}" if torch.cuda.is_available() else "cpu")

    # Assuming we have a saved model
    model = HDICDModel(
        arch=config['model']['arch'],
        patch_size=config['model']['patch_size'],
        emb_dim=config['model']['emb_dim'],
        num_classes=config['dataset']['num_classes'],
        c=config['model']['c'],
        do_hyperbolic=config['model']['do_hyperbolic']
    ).to(device)
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        print("Warning: No checkpoint provided or found. Evaluating with initialized weights.")

    # Dataloader
    test_transform = get_test_augmentations()
    _, test_loader = get_cifar100_dataloaders(
        root=config['dataset']['data_path'],
        batch_size=config['dataset']['batch_size'],
        test_transform=test_transform
    )

    print("Evaluating...")
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
