import torch
import torch.nn as nn
import math

class DINOBackbone(nn.Module):
    def __init__(self, arch='vit_base', patch_size=16):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        
        if arch == 'vit_base' and patch_size == 16:
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.out_dim = 768
        elif arch == 'vit_small' and patch_size == 16:
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            self.out_dim = 384
        else:
            raise ValueError(f"Unsupported architecture {arch} with patch size {patch_size}")
            
        # Freeze backbone if needed, usually we fine-tune block by block or keep frozen
        # For GCD, we often fine-tune the last few blocks.
        
    def forward(self, x):
        # Return CLS token
        return self.model(x)

    def get_intermediate_layers(self, x, n=1):
        return self.model.get_intermediate_layers(x, n)
