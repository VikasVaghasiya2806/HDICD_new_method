"""
Office-Home Dataset Loader (Manual Edition)
Domains: Art, Clipart, Product, Real World
Classes: 65

This version assumes you have manually downloaded and extracted the dataset.
Expected structure:
    data/office_home/
        Art/
        Clipart/
        Product/
        Real World/
"""
import os
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch

class ContrastiveCollate:
    """Custom collate function to handle list-of-views from ContrastiveLearningViewGenerator."""
    def __call__(self, batch):
        images_list = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        if isinstance(images_list[0], list):
            n_views = len(images_list[0])
            views = [torch.stack([images_list[i][v] for i in range(len(images_list))]) for v in range(n_views)]
            return views, labels
        else:
            return torch.stack(images_list), labels

class OfficeHomeGCDDataset(Dataset):
    def __init__(self, root, domain, transform=None, old_class_ratio=0.5, is_train=True, download=False):
        """
        Args:
            root: Root directory where data/office_home should exist
            domain: One of 'Art', 'Clipart', 'Product', 'Real World'
        """
        # Check multiple possible folder layouts
        possible_paths = [
            os.path.join(root, domain),                            # root/Art
            os.path.join(root, 'OfficeHomeDataset_10072016', domain), # root/OfficeHomeDataset_10072016/Art
            os.path.join(root, 'OfficeHome', domain),             # root/OfficeHome/Art
        ]

        domain_path = None
        for path in possible_paths:
            if os.path.isdir(path):
                domain_path = path
                break

        if domain_path is None:
            raise RuntimeError(
                f"\n[!] Office-Home domain '{domain}' folder not found in {root}.\n"
                f"Please manually setup the dataset using these steps:\n"
                f"1. Create the directory: mkdir -p {root}\n"
                f"2. Download the zip: wget -O OfficeHome.zip \"https://huggingface.co/datasets/Kellter/OfficeHomeDataset/resolve/main/Office-Home.zip?download=true\"\n"
                f"3. Unzip it: unzip OfficeHome.zip -d {root}\n"
                f"4. Move folders if needed: mv {root}/OfficeHomeDataset_10072016/* {root}/\n"
            )

        self.dataset = datasets.ImageFolder(domain_path, transform=transform)
        self.num_classes = len(self.dataset.classes)
        self.num_old_classes = int(self.num_classes * old_class_ratio)
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

def get_office_home_dataloaders(root, source_domains, target_domain, batch_size, train_transform, test_transform, old_class_ratio=0.5, download=False, num_workers=4):
    train_datasets = []
    num_classes = None
    num_old_classes = None

    for domain in source_domains:
        ds = OfficeHomeGCDDataset(root, domain, transform=train_transform, old_class_ratio=old_class_ratio, is_train=True)
        train_datasets.append(ds)
        if num_classes is None:
            num_classes = ds.num_classes
            num_old_classes = ds.num_old_classes

    combined_train = ConcatDataset(train_datasets)
    test_ds = OfficeHomeGCDDataset(root, target_domain, transform=test_transform, old_class_ratio=old_class_ratio, is_train=False)

    collate_fn = ContrastiveCollate()

    train_loader = DataLoader(
        combined_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False, pin_memory=True
    )

    return train_loader, test_loader, num_classes, num_old_classes
