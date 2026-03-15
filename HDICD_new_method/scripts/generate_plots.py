import os
import sys
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.hdicd_model import HDICDModel
from datasets.cifar_loader import get_cifar100_dataloaders
from augmentation.domain_augment import get_test_augmentations
from scripts.evaluate import evaluate
from visualization.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(f"cuda:{config['system']['device_id']}" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = HDICDModel(
        arch=config['model']['arch'],
        patch_size=config['model']['patch_size'],
        emb_dim=config['model']['emb_dim'],
        num_classes=config['dataset']['num_classes'],
        c=config['model']['c'],
        do_hyperbolic=config['model']['do_hyperbolic']
    ).to(device)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"Error: Checkpoint {args.checkpoint} not found!")
        return

    # Load Data
    num_classes = config['dataset'].get('num_classes', 100)
    old_class_ratio = config['dataset'].get('old_class_ratio', 0.5)

    dataset_name = config['dataset']['name'].lower()
    if dataset_name == 'domainnet':
        from datasets.domainnet_loader import get_domainnet_dataloaders
        _, test_loader, num_classes, num_old_classes = get_domainnet_dataloaders(
            root=config['dataset']['data_path'],
            source_domain=config['dataset'].get('source_domain', 'real'),
            target_domain=config['dataset'].get('target_domain', 'sketch'),
            batch_size=config['dataset']['batch_size'],
            train_transform=get_test_augmentations(),
            test_transform=get_test_augmentations(),
            old_class_ratio=old_class_ratio,
            download=False
        )
    elif dataset_name == 'officehome':
        from datasets.office_home_loader import get_office_home_dataloaders
        _, test_loader, num_classes, num_old_classes = get_office_home_dataloaders(
            root=config['dataset']['data_path'],
            source_domains=config['dataset'].get('source_domains', ['Art', 'Clipart', 'Product']),
            target_domain=config['dataset'].get('target_domain', 'Real_World'),
            batch_size=config['dataset']['batch_size'],
            train_transform=get_test_augmentations(),
            test_transform=get_test_augmentations(),
            old_class_ratio=old_class_ratio,
            download=False
        )
    elif dataset_name == 'pacs':
        from datasets.pacs_loader import get_pacs_dataloaders
        _, test_loader, num_classes, num_old_classes = get_pacs_dataloaders(
            root=config['dataset']['data_path'],
            source_domain=config['dataset'].get('source_domain', 'photo'),
            target_domain=config['dataset'].get('target_domain', 'art_painting'),
            batch_size=config['dataset']['batch_size'],
            train_transform=get_test_augmentations(),
            test_transform=get_test_augmentations(),
            old_class_ratio=old_class_ratio,
            download=False
        )
    elif dataset_name in ('cub200', 'cub'):
        from datasets.cub_loader import get_cub_dataloaders
        _, test_loader, num_classes, num_old_classes = get_cub_dataloaders(
            root=config['dataset']['data_path'],
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

    print("Extracting features from the dataset...")
    model.eval()
    all_emb = []
    all_hyp_emb = []
    all_images = []
    
    # We will use evaluate() to get the metrics and matched labels
    metrics_dict = evaluate(model, test_loader, device, num_old_classes)
    
    # Run another pass just to save embeddings for TSNE, limiting to a max subset to avoid memory crashes
    max_samples = 5000 
    collected = 0
    with torch.no_grad():
        for images, _ in tqdm(test_loader, desc="Extracting embeddings"):
            images = images.to(device)
            emb, hyp_emb, _ = model(images)
            
            all_emb.extend(emb.cpu().numpy())
            all_hyp_emb.extend(hyp_emb.cpu().numpy())
            all_images.extend(images.cpu())
            
            collected += images.size(0)
            if collected > max_samples:
                break
                
    all_emb = np.array(all_emb)[:max_samples]
    all_hyp_emb = np.array(all_hyp_emb)[:max_samples]
    # convert list of tensors to a single tensor
    all_images = torch.stack(all_images[:max_samples])
    
    # Truncate labels and preds to max_samples for alignment in visualization
    mapped_preds = metrics_dict['preds'][:max_samples]
    true_labels = metrics_dict['labels'][:max_samples]

    os.makedirs('results', exist_ok=True)
    visualizer = Visualizer(save_dir='plots', dpi=300)

    # 1. t-SNE Euclidean & Hyperbolic
    visualizer.plot_tsne(all_emb, true_labels, "tsne_euclidean.png", title="t-SNE (Euclidean Embeddings)")
    visualizer.plot_tsne(all_hyp_emb, true_labels, "tsne_hyperbolic.png", title="t-SNE (Hyperbolic Embeddings)")
    
    # 2. Poincaré Ball
    visualizer.plot_poincare(all_hyp_emb, true_labels, "poincare_embedding.png")

    # 3. Confusion Matrix
    visualizer.plot_confusion_matrix(metrics_dict['preds'], metrics_dict['labels'], "confusion_matrix.png")

    # 4. Accuracy Comparison
    visualizer.plot_accuracy_comparison(
        acc_old=metrics_dict['acc_old'],
        acc_new=metrics_dict['acc_new'],
        acc_all=metrics_dict['acc_all'],
        filename="accuracy_comparison.png"
    )

    # 5. Training Curves
    log_file = os.path.join('logs', 'train_log.json')
    visualizer.plot_training_curves(log_file)
    
    # 6. Cluster Samples
    visualizer.plot_cluster_samples(all_images, mapped_preds, num_clusters=5, samples_per_cluster=5, filename="cluster_samples.png")

    print("\nAll visualizations have been successfully generated in the /plots directory!")

if __name__ == "__main__":
    main()
