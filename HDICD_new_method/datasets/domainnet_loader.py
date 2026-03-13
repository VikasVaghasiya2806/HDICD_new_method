import os
import numpy as np
import urllib.error
from torchvision import datasets
from torchvision.datasets.utils import download_and_extract_archive
from torch.utils.data import DataLoader, Dataset

class DomainNetGCDDataset(Dataset):
    DOMAIN_URLS = {
        'clipart': 'http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip',
        'painting': 'http://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip',
        'real': 'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
        'sketch': 'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
        'infograph': 'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
        'quickdraw': 'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
    }

    def __init__(self, root, domain, transform=None, old_class_ratio=0.5, is_train=True, download=True):
        if download:
            self._download(root, domain)
            
        domain_path = os.path.join(root, domain)
        if not os.path.exists(domain_path):
            raise RuntimeError(f"Dataset not found at {domain_path}. You can use download=True to download it.")
            
        self.dataset = datasets.ImageFolder(domain_path, transform=transform)
        self.num_classes = len(self.dataset.classes)
        self.num_old_classes = int(self.num_classes * old_class_ratio)
        self.is_train = is_train

    def _download(self, root, domain):
        if domain not in self.DOMAIN_URLS:
            raise ValueError(f"Domain {domain} not a valid DomainNet domain.")
            
        domain_path = os.path.join(root, domain)
        if os.path.exists(domain_path) and len(os.listdir(domain_path)) > 0:
            return  # Already downloaded
            
        url = self.DOMAIN_URLS[domain]
        os.makedirs(root, exist_ok=True)
        print(f"Downloading DomainNet domain: {domain}...")
        try:
            download_and_extract_archive(url, download_root=root, extract_root=root)
        except Exception as e:
            print(f"Error downloading from primary URL: {e}")
            raise RuntimeError(f"Could not download {domain}. Manual download required from http://ai.bu.edu/M3SDA/")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We return the ground truth label.
        # In a strict GCD trainer, we would mask the label if it's >= num_old_classes.
        img, label = self.dataset[idx]
        return img, label

def get_domainnet_dataloaders(root, source_domain, target_domain, batch_size, train_transform, test_transform, old_class_ratio=0.5, download=True):
    """
    Returns dataloaders for the DomainNet GCD protocol.
    source_domain: Domain used for training (e.g., 'real')
    target_domain: Domain used for evaluation (e.g., 'sketch')
    """
    train_dataset = DomainNetGCDDataset(root, source_domain, transform=train_transform, old_class_ratio=old_class_ratio, is_train=True, download=download)
    test_dataset = DomainNetGCDDataset(root, target_domain, transform=test_transform, old_class_ratio=old_class_ratio, is_train=False, download=download)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    return train_loader, test_loader, train_dataset.num_classes, train_dataset.num_old_classes
