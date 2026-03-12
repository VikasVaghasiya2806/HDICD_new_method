from torchvision import datasets
from torch.utils.data import DataLoader

def get_officehome_dataloaders(root_train, root_test, batch_size=128, train_transform=None, test_transform=None):
    """
    Generic dataloader for ImageFolder datasets like OfficeHome.
    """
    train_set = datasets.ImageFolder(root=root_train, transform=train_transform)
    test_set = datasets.ImageFolder(root=root_test, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, test_loader
