import torch
import numpy as np

def tangent_cutmix(features, alpha=1.0):
    """
    Synthesize pseudo-novel samples by mixing features in the tangent space
    (Euclidean space at the origin) before mapping to the hyperbolic manifold.
    
    Args:
        features (torch.Tensor): Euclidean features (tangent vectors at origin) of shape (B, D).
        alpha (float): Beta distribution parameter.
        
    Returns:
        torch.Tensor: Mixed Euclidean features.
    """
    B = features.size(0)
    device = features.device
    idx = torch.randperm(B, device=device)
    f1, f2 = features, features[idx]
    
    # Sample lambda from Beta distribution
    lam = torch.from_numpy(
        np.random.beta(alpha, alpha, size=(B, 1))
    ).float().to(device)
    
    # Mix in tangent space
    f_mix = lam * f1 + (1 - lam) * f2
    return f_mix
