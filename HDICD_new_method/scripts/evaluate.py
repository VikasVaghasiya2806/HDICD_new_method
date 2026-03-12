import os
import sys
import yaml
import torch
import argparse
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hdicd_model import HDICDModel
from datasets.cifar_loader import get_cifar100_dataloaders
from augmentation.domain_augment import get_test_augmentations

def hungarian_match(preds, labels):
    if len(preds) == 0:
        return np.array([])
    num_classes = max(preds.max(), labels.max()) + 1
    cost_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for i in range(len(preds)):
        cost_matrix[preds[i], labels[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    return np.array([mapping.get(p, p) for p in preds])

def evaluate(model, dataloader, device, num_old_classes):
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
    
    if len(all_labels) == 0:
        print("No samples to evaluate.")
        return 0.0

    # Hungarian Matching
    matched_preds = hungarian_match(all_preds, all_labels)
    
    # Metrics
    nmi = nmi_score(all_labels, all_preds)
    ari = ari_score(all_labels, all_preds)
    
    old_mask = all_labels < num_old_classes
    new_mask = all_labels >= num_old_classes
    
    acc_all = (matched_preds == all_labels).mean() * 100
    acc_old = (matched_preds[old_mask] == all_labels[old_mask]).mean() * 100 if old_mask.sum() > 0 else 0
    acc_new = (matched_preds[new_mask] == all_labels[new_mask]).mean() * 100 if new_mask.sum() > 0 else 0
    
    print("\nEvaluation Results")
    print("------------------")
    print(f"All Accuracy : {acc_all:.2f}%")
    print(f"Old Accuracy : {acc_old:.2f}%")
    print(f"New Accuracy : {acc_new:.2f}%")
    print(f"NMI          : {nmi:.4f}")
    print(f"ARI          : {ari:.4f}\n")
    
    return acc_all

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

    num_classes = config['dataset'].get('num_classes', 100)
    old_class_ratio = config['dataset'].get('old_class_ratio', 0.5)

    if config['dataset']['name'].lower() == 'domainnet':
        from datasets.domainnet_loader import get_domainnet_dataloaders
        _, test_loader, num_classes, num_old_classes = get_domainnet_dataloaders(
            root=config['dataset']['data_path'],
            source_domain=config['dataset'].get('source_domain', 'real'),
            target_domain=config['dataset'].get('target_domain', 'sketch'),
            batch_size=config['dataset']['batch_size'],
            train_transform=get_test_augmentations(),
            test_transform=get_test_augmentations(),
            old_class_ratio=old_class_ratio
        )
    else:
        test_transform = get_test_augmentations()
        _, test_loader = get_cifar100_dataloaders(
            root=config['dataset']['data_path'],
            batch_size=config['dataset']['batch_size'],
            test_transform=test_transform
        )
        num_old_classes = int(num_classes * old_class_ratio)

    print("Evaluating...")
    evaluate(model, test_loader, device, num_old_classes)

if __name__ == '__main__':
    main()
