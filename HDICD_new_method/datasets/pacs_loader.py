"""
PACS Dataset Loader
Domains: photo, art_painting, cartoon, sketch
Classes: 7

Automatically downloads from official GitHub mirror via git clone.
"""
import os
import subprocess
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch


class PACSGCDDataset(Dataset):
    REPO_URL = 'https://github.com/MachineLearning2020/Homework3-PACS.git'

    def __init__(self, root, domain, transform=None, old_class_ratio=0.5, is_train=True, download=True):
        if download:
            try:
                self._download(root)
            except Exception as e:
                print(f"Warning: Automatic download failed: {e}")

        # Check multiple possible folder layouts
        possible_paths = [
            os.path.join(root, 'Homework3-PACS-master', 'PACS', domain),
            os.path.join(root, 'PACS', domain),
            os.path.join(root, domain)
        ]

        domain_path = None
        for path in possible_paths:
            if os.path.isdir(path):
                domain_path = path
                break

        if domain_path is None:
            raise RuntimeError(
                f"PACS domain folder not found for '{domain}' in {root}.\n"
                f"Checked: {possible_paths}\n"
                f"Please run: git clone --depth 1 {self.REPO_URL} {root}/Homework3-PACS-master"
            )

        self.dataset = datasets.ImageFolder(domain_path, transform=transform)
        self.num_classes = len(self.dataset.classes)
        self.num_old_classes = int(self.num_classes * old_class_ratio)
        self.is_train = is_train

    def _download(self, root):
        os.makedirs(root, exist_ok=True)
        target_dir = os.path.join(root, 'Homework3-PACS-master')
        if os.path.isdir(os.path.join(target_dir, 'PACS', 'photo')):
            return
        print(f"Cloning PACS dataset from GitHub into {target_dir}...")
        subprocess.run(['git', 'clone', '--depth', '1', self.REPO_URL, target_dir], check=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


class ContrastiveCollate:
    """Custom collate to handle list-of-views from ContrastiveLearningViewGenerator."""
    def __call__(self, batch):
        images_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        if isinstance(images_list[0], list):
            n_views = len(images_list[0])
            views = [torch.stack([images_list[i][v] for i in range(len(images_list))]) for v in range(n_views)]
            return views, labels
        return torch.stack(images_list), labels


def get_pacs_dataloaders(
    root,
    source_domains,
    target_domain,
    batch_size,
    train_transform,
    test_transform,
    old_class_ratio=0.5,
    download=True,
    num_workers=4
):
    """
    Returns train/test dataloaders for PACS in Domain Generalization setup.
    """
    train_datasets = []
    num_classes = None
    num_old_classes = None

    for domain in source_domains:
        ds = PACSGCDDataset(root, domain, transform=train_transform, old_class_ratio=old_class_ratio, is_train=True, download=download)
        train_datasets.append(ds)
        if num_classes is None:
            num_classes = ds.num_classes
            num_old_classes = ds.num_old_classes

    combined_train = ConcatDataset(train_datasets)
    test_ds = PACSGCDDataset(root, target_domain, transform=test_transform, old_class_ratio=old_class_ratio, is_train=False, download=False)

    collate_fn = ContrastiveCollate()

    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, drop_last=True, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, drop_last=False, pin_memory=True)

    return train_loader, test_loader, num_classes, num_old_classes
