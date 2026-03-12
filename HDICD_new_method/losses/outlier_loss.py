import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperbolic.poincare_ops import dist_matrix

class AdaptiveOutlierLoss(nn.Module):
    """
    Adaptive Outlier Repulsion: Ensures novel samples are separated from known prototypes.
    """
    def __init__(self, c=1.0):
        super().__init__()
        self.c = c

    def forward(self, z_mix, prototypes, repel_margin):
        """
        z_mix: (B, D) mixed features in hyperbolic space
        prototypes: (num_classes, D) prototypes
        repel_margin: scalar adaptive margin
        """
        # Distance between z_mix and each prototype
        # dist_matrix returns distance between points in hyperbolic space
        dist2proto = dist_matrix(z_mix, prototypes, c=self.c)
        min_dist, _ = dist2proto.min(dim=1)
        outlier_loss = F.relu(repel_margin - min_dist).mean()
        return outlier_loss
