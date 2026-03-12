import torch
import torch.nn as nn
from hyperbolic.poincare_ops import mobius_add

class PenalizedBusemannLoss(nn.Module):
    """
    Implements the penalized Busemann loss for hyperbolic learning:
    
        ℓ(z, p) = log(||p - z||²) - (phi + 1) * log(1 - ||z||²)
    
    where:
      - z are the hyperbolic embeddings (assumed to lie in the Poincaré ball, ||z|| < 1)
      - p are the corresponding class prototypes on the ideal boundary (||p|| = 1)
      - phi is the penalty scalar.
    
    An epsilon is added for numerical stability.
    """
    def __init__(self, phi=0.75, eps=1e-6):
        super(PenalizedBusemannLoss, self).__init__()
        self.phi = phi
        self.eps = eps

    def forward(self, z, p):
        """
        Args:
            z: Tensor of shape (batch_size, dims) representing hyperbolic embeddings.
            p: Tensor of shape (batch_size, dims) corresponding to the ideal prototypes for each example.
        
        Returns:
            The scalar penalized Busemann loss averaged over the batch.
        """
        z_norm_sq = torch.clamp(torch.sum(z ** 2, dim=1), 0, 1 - self.eps)
        diff_norm_sq = torch.sum((p - z) ** 2, dim=1) + self.eps
        loss = torch.log(diff_norm_sq) - (self.phi + 1) * torch.log(1 - z_norm_sq)
        return torch.mean(loss)
