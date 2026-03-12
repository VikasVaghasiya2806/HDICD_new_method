import torch
import torch.nn as nn
from backbone.dino_backbone import DINOBackbone
from hyperbolic.mobius_layers import ToPoincare
from hyperbolic.hyp_classifier import HyperbolicPrototypeClassifier

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class HDICDModel(nn.Module):
    def __init__(self, arch='vit_base', patch_size=16, emb_dim=32, num_classes=100, c=0.05, do_hyperbolic=True):
        super().__init__()
        self.do_hyperbolic = do_hyperbolic
        self.backbone = DINOBackbone(arch, patch_size)
        self.projection_head = DINOHead(in_dim=self.backbone.out_dim, out_dim=emb_dim, nlayers=3)
        self.curvature = nn.Parameter(torch.tensor(c, dtype=torch.float32), requires_grad=True)
        self.to_poincare = ToPoincare(c=self.curvature, train_c=False, riemannian=True)
        self.classifier = HyperbolicPrototypeClassifier(emb_dim=emb_dim, num_classes=num_classes, c=self.curvature)
        
    def forward(self, x):
        # 1. Feature extraction
        feats = self.backbone(x)
        
        # 2. Projection to Euclidean embedding space
        emb = self.projection_head(feats)
        
        # 3. Map to hyperbolic
        if self.do_hyperbolic:
            # We map the embeddings to hyperbolic utilizing expmap0
            # ToPoincare wraps expmap0
            hyp_emb = self.to_poincare(emb)
        else:
            hyp_emb = emb
            
        # 4. Classification
        logits = self.classifier(hyp_emb, c=self.curvature)
        
        return emb, hyp_emb, logits
