"""
CUB-200-2011 Dataset Loader for Generalized Category Discovery (GCD)
Dataset: CUB_200_2011 (11,788 images, 200 bird species)

Unlike Office-Home (which has domains), CUB-200 is a standard single-domain GCD benchmark.
The split is class-based only:
  - Old Classes (Seen):   First N classes, labeled during training
  - New Classes (Novel):  Remaining classes, unlabeled during training
  - Test Set:             All classes jointly (evaluate old + new accuracy)

Manual Dataset Setup (run in terminal):
    mkdir -p ./data/cub200
    wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz -O ./data/cub200/CUB_200_2011.tgz
    tar -xzf ./data/cub200/CUB_200_2011.tgz -C ./data/cub200/
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


class ContrastiveCollate:
    """Handle batches that may contain multi-view lists from ContrastiveLearningViewGenerator."""
    def __call__(self, batch):
        images_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        if isinstance(images_list[0], list):
            n_views = len(images_list[0])
            views = [torch.stack([images_list[i][v] for i in range(len(images_list))]) for v in range(n_views)]
            return views, labels
        return torch.stack(images_list), labels


class CUBGCDDataset(Dataset):
    """
    CUB-200-2011 dataset with Generalized Category Discovery (GCD) splits.

    Args:
        root: Path to the CUB_200_2011 folder (containing images/, image_class_labels.txt, etc.)
        transform: Torchvision transforms
        old_class_ratio: Fraction of the 200 bird species treated as 'known' (default 0.5 → 100 old classes)
        split: 'train', 'test', or 'all'
    """
    def __init__(self, root, transform=None, old_class_ratio=0.5, split='train'):
        self.root = root
        self.transform = transform
        self.old_class_ratio = old_class_ratio
        self.split = split

        # Validate folder structure
        cub_path = self._find_cub_root(root)
        if cub_path is None:
            raise RuntimeError(
                f"\n[!] CUB-200 dataset not found in '{root}'.\n"
                f"Please download it manually:\n"
                f"  mkdir -p {root}\n"
                f"  wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz -O {root}/CUB_200_2011.tgz\n"
                f"  tar -xzf {root}/CUB_200_2011.tgz -C {root}/\n"
            )

        # Load metadata files
        images_file = os.path.join(cub_path, 'images.txt')
        labels_file = os.path.join(cub_path, 'image_class_labels.txt')
        split_file  = os.path.join(cub_path, 'train_test_split.txt')
        images_dir  = os.path.join(cub_path, 'images')

        img_paths, img_labels, is_train_flags = {}, {}, {}
        with open(images_file) as f:
            for line in f:
                idx, path = line.strip().split(' ', 1)
                img_paths[int(idx)] = os.path.join(images_dir, path)
        with open(labels_file) as f:
            for line in f:
                idx, label = line.strip().split()
                img_labels[int(idx)] = int(label) - 1  # 0-indexed
        with open(split_file) as f:
            for line in f:
                idx, flag = line.strip().split()
                is_train_flags[int(idx)] = int(flag)

        # Class split: old=first N, new=rest
        all_classes = sorted(set(img_labels.values()))
        num_old = int(len(all_classes) * old_class_ratio)
        self.old_class_ids = set(all_classes[:num_old])
        self.num_classes = len(all_classes)
        self.num_old_classes = num_old

        # Filter by official split flag, then by class set
        self.samples = []
        for idx in sorted(img_paths.keys()):
            is_train = is_train_flags[idx] == 1
            label = img_labels[idx]
            path = img_paths[idx]

            if split == 'train':
                # Training: labeled old classes + unlabeled new classes
                if is_train:
                    self.samples.append((path, label))
            elif split == 'test':
                # Test: all classes, only test images
                if not is_train:
                    self.samples.append((path, label))
            else:  # 'all'
                self.samples.append((path, label))

    def _find_cub_root(self, root):
        """Handle multiple possible extraction structures."""
        candidates = [
            root,
            os.path.join(root, 'CUB_200_2011'),
        ]
        for path in candidates:
            if os.path.isfile(os.path.join(path, 'images.txt')):
                return path
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cub_dataloaders(root, batch_size, train_transform, test_transform, old_class_ratio=0.5, num_workers=4):
    """
    Build train and test DataLoaders for CUB-200 with GCD class splits.

    Returns:
        train_loader, test_loader, num_classes (200), num_old_classes
    """
    train_ds = CUBGCDDataset(root, transform=train_transform, old_class_ratio=old_class_ratio, split='train')
    test_ds  = CUBGCDDataset(root, transform=test_transform,  old_class_ratio=old_class_ratio, split='test')

    num_classes = train_ds.num_classes
    num_old_classes = train_ds.num_old_classes

    collate_fn = ContrastiveCollate()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )

    return train_loader, test_loader, num_classes, num_old_classes
