import torch
import torch.nn.functional as F
from tqdm import tqdm
from augmentation.tangent_cutmix import tangent_cutmix
from losses.busemann_loss import PenalizedBusemannLoss
from losses.contrastive_loss import hyperbolic_info_nce_loss
from losses.outlier_loss import AdaptiveOutlierLoss
from losses.classifier_loss import ClassifierLoss

class HDICDTrainer:
    def __init__(self, model, optimizer, scheduler, device, num_classes, emb_dim, penalty_value=0.75):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Losses
        self.busemann_criterion = PenalizedBusemannLoss(phi=penalty_value).to(device)
        self.outlier_criterion = AdaptiveOutlierLoss(c=model.curvature).to(device)
        self.classifier_criterion = ClassifierLoss().to(device)
        
        # Prototypes for Busemann and Outlier loss
        self.prototypes = torch.randn(num_classes, emb_dim, device=device)
        self.prototypes = nn.Parameter(F.normalize(self.prototypes, p=2, dim=1))
        # Initial adaptive margin
        self.repel_margin = None

    def initialize_prototypes(self, proto_epochs=1000, proto_lr=0.1):
        from training.optimizer import get_proto_optimizer
        proto_opt = get_proto_optimizer(self.prototypes, lr=proto_lr)
        
        for _ in range(proto_epochs):
            proto_opt.zero_grad()
            sim_matrix = torch.matmul(self.prototypes, self.prototypes.t()) + 1.0
            sim_matrix.fill_diagonal_(0)
            loss_per_proto, _ = sim_matrix.max(dim=1)
            loss_proto = loss_per_proto.mean()
            loss_proto.backward()
            proto_opt.step()
            with torch.no_grad():
                self.prototypes.div_(self.prototypes.norm(dim=1, keepdim=True))

    def train_epoch(self, dataloader, epoch, total_epochs, n_views=2, alpha_d=0.5):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
            
            # Since n_views=2 for Contrastive Learning View Generator, images might be a list of 2 tensors
            if isinstance(images, list):
                images = torch.cat(images, dim=0).to(self.device)
                labels = torch.cat([labels, labels], dim=0).to(self.device)
            else:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
            emb, hyp_emb, logits = self.model(images)
            
            # 1. Tangent CutMix
            # Generate pseudo novel samples in tangent space
            f_mix = tangent_cutmix(emb, alpha=1.0)
            
            # Map pseudo samples to hyperbolic
            if self.model.do_hyperbolic:
                z_mix = self.model.to_poincare(f_mix)
            else:
                z_mix = f_mix
                
            z_mix = torch.tanh(z_mix)
            z = torch.tanh(hyp_emb)
            
            batch_prototypes = self.prototypes[labels]
            
            # 2. Compute Losses
            # Busemann loss (prototypes vs embeddings)
            loss_busemann = self.busemann_criterion(z, batch_prototypes)
            
            # InfoNCE contrastive loss
            loss_contrastive = hyperbolic_info_nce_loss(hyp_emb, n_views=n_views, c=self.model.curvature, alpha_d=alpha_d)
            
            # Adaptive Outlier Repulsion
            if epoch == 0 and batch_idx == 0:
                from hyperbolic.poincare_ops import dist_matrix
                dists = dist_matrix(z_mix, self.prototypes, c=self.model.curvature)
                mins = dists.min(dim=1).values
                self.repel_margin = torch.quantile(mins, 0.8).detach()
                
            loss_outlier = self.outlier_criterion(z_mix, self.prototypes, self.repel_margin)
            
            # Prototype Classifier CrossEntropy
            loss_classifier = self.classifier_criterion(logits, labels)
            
            # Total Loss
            loss = 1.0 * loss_busemann + 1.0 * loss_contrastive + 1.0 * loss_outlier + 1.0 * loss_classifier
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        self.scheduler.step()

