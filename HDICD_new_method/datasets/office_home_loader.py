"""
Office-Home Dataset Loader
Domains: Art, Clipart, Product, Real World
Classes: 65

Automatically downloads the dataset using the official Dassl toolkit mirror.
Folder structure after extraction:
    data/office_home/
        Art/
            Alarm_Clock/
            ...
        Clipart/
        Product/
        Real World/
"""
import os
import subprocess
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# Official domain names for Office-Home
OFFICEHOME_DOMAINS = ['Art', 'Clipart', 'Product', 'Real World']

class ContrastiveCollate:
    """Custom collate function to handle list-of-views from ContrastiveLearningViewGenerator."""
    def __call__(self, batch):
        import torch
        images_list = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        labels = torch.tensor(labels)
        # If images are a list of views (from ContrastiveLearningViewGenerator)
        if isinstance(images_list[0], list):
            n_views = len(images_list[0])
            views = [torch.stack([images_list[i][v] for i in range(len(images_list))]) for v in range(n_views)]
            return views, labels
        else:
            images = torch.stack(images_list)
            return images, labels


class OfficeHomeGCDDataset(Dataset):
    """
    Dataset wrapper for Office-Home that supports:
    - Automated download via git clone
    - Multiple folder structure conventions
    - Old/new class split for GCD
    """
    
    def __init__(self, root, domain, transform=None, old_class_ratio=0.5, is_train=True, download=True):
        """
        Args:
            root: Root directory where data/office_home should exist
            domain: One of 'Art', 'Clipart', 'Product', 'Real World'
            transform: Torchvision transforms to apply
            old_class_ratio: Fraction of classes treated as 'known'
            is_train: Whether this is training set
            download: Whether to attempt auto download
        """
        if download:
            try:
                self._download(root)
            except Exception as e:
                print(f"Warning: Automatic download failed: {e}")
                print("Checking if dataset already exists locally...")

        # Check multiple possible folder layouts after extraction
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
                f"Office-Home domain '{domain}' not found in {root}.\n"
                f"Expected one of: {possible_paths}\n"
                f"Please run the download command manually:\n"
                f"  cd {root} && wget https://mega.nz/file/YF5CQILK -O OfficeHomeDataset.zip\n"
                f"Or use the manual steps in README.md"
            )

        self.dataset = datasets.ImageFolder(domain_path, transform=transform)
        self.num_classes = len(self.dataset.classes)
        self.num_old_classes = int(self.num_classes * old_class_ratio)
        self.is_train = is_train

    def _download(self, root):
        """Try to retrieve the Office-Home dataset using gdown from the official Google Drive."""
        os.makedirs(root, exist_ok=True)

        # Check if already present in any known layout
        for domain in ['Art', 'Clipart', 'Product']:
            if os.path.isdir(os.path.join(root, domain)):
                print("Office-Home dataset already found locally.")
                return
            if os.path.isdir(os.path.join(root, 'OfficeHomeDataset_10072016', domain)):
                print("Office-Home dataset already found locally.")
                return

        print(f"Downloading Office-Home dataset into {root}...")
        zip_path = os.path.join(root, 'OfficeHomeDataset.zip')
        
        if not os.path.exists(zip_path):
            # Use gdown to download from Google Drive (1.5GB)
            # Official Google Drive ID for OfficeHomeDataset_10072016.zip
            file_id = '0B81rNlvomiwed0V1YUxQdC1uOTg'
            try:
                import gdown
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, zip_path, quiet=False)
            except Exception as e:
                print(f"gdown failed: {e}")
                raise RuntimeError(
                    "Could not automatically download Office-Home dataset.\n"
                    "Please download it manually:\n"
                    "  pip install gdown\n"
                    f"  gdown 'https://drive.google.com/uc?id={file_id}' -O {zip_path}\n"
                    f"  unzip {zip_path} -d {root}"
                )

        # Extract the archive
        if os.path.exists(zip_path):
            print("Extracting Office-Home dataset...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(root)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label


def get_office_home_dataloaders(
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
    Build train (multi-source) and test (single-target) DataLoaders for Office-Home.

    Args:
        root: directory containing domain folders (e.g. ./data/office_home)
        source_domains: list of domain names for training
        target_domain: domain name for evaluation
        batch_size: batch size for both loaders
        train_transform: transform applied to training images
        test_transform: transform applied to test images
        old_class_ratio: fraction of classes treated as old/known
        download: attempt automated download if data is missing
        num_workers: dataloader worker count

    Returns:
        train_loader, test_loader, num_classes, num_old_classes
    """
    train_datasets = []
    num_classes = None
    num_old_classes = None

    for domain in source_domains:
        ds = OfficeHomeGCDDataset(
            root, domain,
            transform=train_transform,
            old_class_ratio=old_class_ratio,
            is_train=True,
            download=download
        )
        train_datasets.append(ds)
        if num_classes is None:
            num_classes = ds.num_classes
            num_old_classes = ds.num_old_classes

    combined_train = ConcatDataset(train_datasets)
    test_ds = OfficeHomeGCDDataset(
        root, target_domain,
        transform=test_transform,
        old_class_ratio=old_class_ratio,
        is_train=False,
        download=False   # Don't re-download for test split
    )

    collate_fn = ContrastiveCollate()

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,         # Important for contrastive losses
        pin_memory=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True
    )

    return train_loader, test_loader, num_classes, num_old_classes
