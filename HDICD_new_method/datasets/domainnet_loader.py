import os
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset

class DomainNetGCDDataset(Dataset):
    def __init__(self, root, domain, transform=None, old_class_ratio=0.5, is_train=True):
        self.dataset = datasets.ImageFolder(os.path.join(root, domain), transform=transform)
        self.num_classes = len(self.dataset.classes)
        self.num_old_classes = int(self.num_classes * old_class_ratio)
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # We return the ground truth label.
        # In a strict GCD trainer, we would mask the label if it's >= num_old_classes.
        img, label = self.dataset[idx]
        return img, label

def get_domainnet_dataloaders(root, source_domain, target_domain, batch_size, train_transform, test_transform, old_class_ratio=0.5):
    """
    Returns dataloaders for the DomainNet GCD protocol.
    source_domain: Domain used for training (e.g., 'real')
    target_domain: Domain used for evaluation (e.g., 'sketch')
    """
    train_dataset = DomainNetGCDDataset(root, source_domain, transform=train_transform, old_class_ratio=old_class_ratio, is_train=True)
    test_dataset = DomainNetGCDDataset(root, target_domain, transform=test_transform, old_class_ratio=old_class_ratio, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    return train_loader, test_loader, train_dataset.num_classes, train_dataset.num_old_classes
