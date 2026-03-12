import torch
import torch.nn as nn

class ClassifierLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, logits, targets):
        """
        logits: (B, num_classes)
        targets: (B,)
        """
        return self.criterion(logits, targets)
