import torch
import torch.nn.functional as F
import torch.nn as nn
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
        self.num_classes = num_classes
        
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

    def train_epoch(self, dataloader, epoch, total_epochs, n_views=2, alpha_d=0.5, debug=False, log_interval=10):
        self.model.train()
        
        # Track multiple losses
        metrics = {
            'total_loss': 0.0,
            'busemann_loss': 0.0,
            'contrastive_loss': 0.0,
            'outlier_loss': 0.0,
            'classifier_loss': 0.0
        }
        
        num_batches = len(dataloader)
        if debug:
            num_batches = min(num_batches, 10)
            
        pbar = tqdm(total=num_batches, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if debug and batch_idx >= 10:
                break
                
            # Since n_views=2 for Contrastive Learning View Generator, images might be a list of 2 tensors
            if isinstance(images, list):
                views = images.copy()
                view1, view2 = views[0], views[1]
                
                if epoch == 0 and batch_idx == 0:
                    print(view1.shape)
                    print(view2.shape)
                    
                # Feed both views to the model
                images = torch.cat([view1, view2], dim=0).to(self.device)
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
            
            # For Busemann Loss: only use non-doubled labels (first half)
            # because we only need one prototype per sample, not per view
            base_labels = labels[:labels.size(0) // n_views] if labels.size(0) > images.size(0) // 2 else labels
            batch_prototypes = self.prototypes[base_labels % self.prototypes.size(0)]
            
            # Scale points to unit ball for Busemann norm calculations
            c_val = self.model.curvature.abs().clamp(min=1e-4, max=10.0)
            c_sqrt = torch.sqrt(c_val)
            z_mix_scaled = z_mix * c_sqrt
            z_scaled = hyp_emb * c_sqrt
            
            # Busemann loss (prototypes vs embeddings)
            loss_busemann = self.busemann_criterion(z_scaled[:base_labels.size(0)], batch_prototypes)
            
            # InfoNCE contrastive loss (uses all views)
            loss_contrastive = hyperbolic_info_nce_loss(hyp_emb, n_views=n_views, c=c_val, alpha_d=alpha_d)
            
            # Adaptive Outlier Repulsion
            if epoch == 0 and batch_idx == 0:
                from hyperbolic.poincare_ops import dist_matrix
                dists = dist_matrix(z_mix_scaled, self.prototypes, c=c_val)
                mins = dists.min(dim=1).values
                self.repel_margin = torch.quantile(mins, 0.8).detach()
                
            loss_outlier = self.outlier_criterion(z_mix_scaled, self.prototypes, self.repel_margin)
            
            # Prototype Classifier CrossEntropy (use base labels)
            loss_classifier = self.classifier_criterion(logits[:batch_prototypes.size(0)], base_labels % self.num_classes)
            
            # Total Loss
            loss = 1.0 * loss_busemann + 1.0 * loss_contrastive + 1.0 * loss_outlier + 1.0 * loss_classifier
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            metrics['total_loss'] += loss.item()
            metrics['busemann_loss'] += loss_busemann.item()
            metrics['contrastive_loss'] += loss_contrastive.item()
            metrics['outlier_loss'] += loss_outlier.item()
            metrics['classifier_loss'] += loss_classifier.item()
            
            with torch.no_grad():
                self.model.curvature.clamp_(min=1e-4, max=10.0)
            
            # Update progress bar
            pbar.update(1)
            if (batch_idx + 1) % log_interval == 0:
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'Buse': f"{loss_busemann.item():.4f}",
                    'Cont': f"{loss_contrastive.item():.4f}"
                })
                
        pbar.close()
            
        self.scheduler.step()
        
        # Average metrics
        for k in metrics.keys():
            metrics[k] /= num_batches
            
        return metrics
