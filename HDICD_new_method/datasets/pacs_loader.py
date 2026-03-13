import os
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset, ConcatDataset

class PACSGCDDataset(Dataset):
    def __init__(self, root, domain, transform=None, old_class_ratio=0.5, is_train=True):
        domain_path = os.path.join(root, domain)
        if not os.path.exists(domain_path):
            raise RuntimeError(f"PACS domain folder not found at {domain_path}. Please download the dataset.")
            
        self.dataset = datasets.ImageFolder(domain_path, transform=transform)
        self.num_classes = len(self.dataset.classes)
        self.num_old_classes = int(self.num_classes * old_class_ratio)
        self.is_train = is_train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label

def get_pacs_dataloaders(root, source_domains, target_domain, batch_size, train_transform, test_transform, old_class_ratio=0.5):
    """
    Returns dataloaders for the PACS Dataset in a Domain Generalization setup.
    source_domains: List of domains used for training (e.g., ['photo', 'cartoon', 'art_painting'])
    target_domain: Domain used for evaluation (e.g., 'sketch')
    """
    train_datasets = []
    num_classes = None
    num_old_classes = None
    
    for domain in source_domains:
        ds = PACSGCDDataset(root, domain, transform=train_transform, old_class_ratio=old_class_ratio, is_train=True)
        train_datasets.append(ds)
        if num_classes is None:
            num_classes = ds.num_classes
            num_old_classes = ds.num_old_classes
            
    # Combine all source domain datasets into a single training dataset
    combined_train_dataset = ConcatDataset(train_datasets)
    
    test_dataset = PACSGCDDataset(root, target_domain, transform=test_transform, old_class_ratio=old_class_ratio, is_train=False)
    
    train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    return train_loader, test_loader, num_classes, num_old_classes
