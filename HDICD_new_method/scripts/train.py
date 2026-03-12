import os
import sys
import yaml
import json
import torch
import argparse
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hdicd_model import HDICDModel
from training.optimizer import get_optimizer, get_scheduler
from training.trainer import HDICDTrainer
from datasets.cifar_loader import get_cifar100_dataloaders
from augmentation.domain_augment import get_train_augmentations, get_test_augmentations

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def hungarian_match(preds, labels):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if len(preds) == 0: return np.array([])
    num_classes = max(preds.max(), labels.max()) + 1
    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(len(preds)):
        cost_matrix[preds[i], labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(p, p) for p in preds])

def evaluate(model, dataloader, device):
    """Evaluates the model and returns accuracy."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            _, _, logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    if len(all_labels) == 0: return 0.0
    matched_preds = hungarian_match(all_preds, all_labels)
    acc = (matched_preds == all_labels).mean() * 100.0
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode (2 epochs, 10 batches)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['system']['seed'])
    device = torch.device(f"cuda:{config['system']['device_id']}" if torch.cuda.is_available() else "cpu")

    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load Data
    train_transform = get_train_augmentations()
    test_transform = get_test_augmentations()
    
    if config['dataset']['name'].lower() == 'domainnet':
        from datasets.domainnet_loader import get_domainnet_dataloaders
        train_loader, test_loader, num_classes, num_old_classes = get_domainnet_dataloaders(
            root=config['dataset']['data_path'],
            source_domain=config['dataset'].get('source_domain', 'real'),
            target_domain=config['dataset'].get('target_domain', 'sketch'),
            batch_size=config['dataset']['batch_size'],
            train_transform=train_transform,
            test_transform=test_transform,
            old_class_ratio=config['dataset'].get('old_class_ratio', 0.5)
        )
        config['dataset']['num_classes'] = num_classes
        config['dataset']['num_old_classes'] = num_old_classes
    else:
        # Using CIFAR100 logic for testing script
        train_loader, test_loader = get_cifar100_dataloaders(
            root=config['dataset']['data_path'],
            batch_size=config['dataset']['batch_size'],
            train_transform=train_transform, test_transform=test_transform
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

    start_epoch = 0
    best_acc = 0.0
    
    # Resume functionality
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("Initializing prototypes...")
        # Only initialize prototypes if not resuming
        trainer.initialize_prototypes(
            proto_epochs=2 if args.debug else config['training']['proto_epochs'], 
            proto_lr=config['training']['proto_lr']
        )

    print("Starting training...")
    total_epochs = 2 if args.debug else config['training']['epochs']
    
    # Setup json log file
    log_file = os.path.join('logs', 'train_log.json')
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            json.dump([], f)

    for epoch in range(start_epoch, total_epochs):
        metrics = trainer.train_epoch(
            train_loader, 
            epoch, 
            total_epochs, 
            n_views=config['training']['n_views'], 
            alpha_d=config['training']['alpha_d'],
            debug=args.debug
        )
        
        # Log to console
        print(f"\nEpoch {epoch}")
        print(f"Total Loss: {metrics['total_loss']:.2f}")
        print(f"Busemann Loss: {metrics['busemann_loss']:.2f}")
        print(f"Contrastive Loss: {metrics['contrastive_loss']:.2f}")
        print(f"Outlier Loss: {metrics['outlier_loss']:.2f}")
        print(f"Classifier Loss: {metrics['classifier_loss']:.2f}")
        
        # Save metrics to JSON
        metrics['epoch'] = epoch
        with open(log_file, 'r') as f:
            logs = json.load(f)
        logs.append(metrics)
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=4)
            
        # Evaluation for Best Model
        val_acc = evaluate(model, test_loader, device)
        print(f"Validation Accuracy: {val_acc:.2f}%\n")
        
        # Checkpointing
        checkpoint_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "loss": metrics['total_loss'],
            "val_acc": val_acc
        }
        
        # Save current epoch checkpoint
        epoch_ckpt_path = os.path.join('checkpoints', f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint_state, epoch_ckpt_path)
        
        # Update latest checkpoint
        latest_ckpt_path = os.path.join('checkpoints', 'latest_checkpoint.pth')
        torch.save(checkpoint_state, latest_ckpt_path)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_ckpt_path = os.path.join('checkpoints', 'best_model.pth')
            torch.save(checkpoint_state, best_ckpt_path)
            print(f"*** New best model saved with accuracy {best_acc:.2f}% ***\n")

if __name__ == '__main__':
    main()
