import torch
import torch.nn as nn
from hyperbolic.mobius_layers import HyperbolicMLR

class HyperbolicPrototypeClassifier(nn.Module):
    """
    Hyperbolic Prototype Classifier using Hyperbolic Multinomial Logistic Regression.
    """
    def __init__(self, emb_dim, num_classes, c=1.0):
        super().__init__()
        self.mlr = HyperbolicMLR(ball_dim=emb_dim, n_classes=num_classes, c=c)
        
    def forward(self, x, c=None):
        """
        x: features mapped to the Poincare ball.
        """
        return self.mlr(x, c)
