import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import torchvision

class Visualizer:
    def __init__(self, save_dir='plots', dpi=300):
        self.save_dir = save_dir
        self.dpi = dpi
        os.makedirs(save_dir, exist_ok=True)
        # Set styling to be paper-quality
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
        })
        
    def plot_tsne(self, embeddings, labels, filename, title="t-SNE Visualization"):
        print(f"Generating {filename}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20', s=15, alpha=0.8)
        plt.colorbar(scatter, label="Classes")
        plt.title(title)
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_poincare(self, embeddings, labels, filename="poincare_embedding.png"):
        print(f"Generating {filename}...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        emb_2d = tsne.fit_transform(embeddings)

        # Map to poincare disk for visualization purposes if not already
        norms = np.linalg.norm(emb_2d, axis=1, keepdims=True)
        # Scale to unit disk
        emb_2d = emb_2d / (norms.max() + 1e-5) * 0.95

        fig, ax = plt.subplots(figsize=(8, 8))
        circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', linewidth=1.5)
        ax.add_artist(circle)
        
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap='tab20', s=15, alpha=0.8)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        plt.title("Poincaré Ball Visualization")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_confusion_matrix(self, preds, labels, filename="confusion_matrix.png"):
        print(f"Generating {filename}...")
        # Since number of classes can be huge (345), we might want to plot a dense matrix without annotations
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap='Blues', cbar=True, xticklabels=False, yticklabels=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_training_curves(self, log_file, filename_loss="training_loss_curve.png", filename_acc="validation_accuracy_curve.png"):
        print(f"Generating training curves...")
        if not os.path.exists(log_file):
            print(f"Log file {log_file} not found. Skipping training curves.")
            return

        with open(log_file, 'r') as f:
            logs = json.load(f)

        epochs = [log['epoch'] for log in logs]
        total_loss = [log['total_loss'] for log in logs]
        # Val acc might not be in the log if we didn't save it there, checking:
        # Actually in trainer we just appended metrics. Let's look for val_acc in the checkpoints instead, 
        # but to keep it simple, we plot the loss components.
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, total_loss, marker='o', linestyle='-', color='b', label="Total Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename_loss), dpi=self.dpi)
        plt.close()

        # If val_acc exists, plot it, otherwise plot busmann/contrastive vs epoch
        busemann = [log.get('busemann_loss', 0) for log in logs]
        contrastive = [log.get('contrastive_loss', 0) for log in logs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, busemann, marker='s', linestyle='-', color='r', label="Busemann Loss")
        plt.plot(epochs, contrastive, marker='^', linestyle='-', color='g', label="Contrastive Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss Components")
        plt.title("Loss Components Curve")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename_acc), dpi=self.dpi)
        plt.close()

    def plot_accuracy_comparison(self, acc_old, acc_new, acc_all, filename="accuracy_comparison.png"):
        print(f"Generating {filename}...")
        labels = ['Old Classes', 'New Classes', 'All Classes']
        accs = [acc_old, acc_new, acc_all]

        plt.figure(figsize=(8, 6))
        bars = plt.bar(labels, accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.8)
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Comparison")
        plt.ylim(0, 100)
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
            
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi)
        plt.close()

    def plot_cluster_samples(self, images_tensor, preds, num_clusters=5, samples_per_cluster=5, filename="cluster_samples.png"):
        print(f"Generating {filename}...")
        # images_tensor: shape (N, 3, H, W)
        unique_clusters = np.unique(preds)
        selected_clusters = np.random.choice(unique_clusters, min(num_clusters, len(unique_clusters)), replace=False)
        
        grid_images = []
        for c in selected_clusters:
            idx = np.where(preds == c)[0]
            sampled_idx = np.random.choice(idx, min(samples_per_cluster, len(idx)), replace=False)
            
            for i in sampled_idx:
                img = images_tensor[i]
                # Denormalize ImageNet
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img.cpu() * std + mean
                img = torch.clamp(img, 0, 1)
                grid_images.append(img)
                
        # Make a grid where each row is a cluster
        if len(grid_images) > 0:
            grid = torchvision.utils.make_grid(grid_images, nrow=samples_per_cluster, padding=2, normalize=False)
            plt.figure(figsize=(samples_per_cluster * 2, num_clusters * 2))
            plt.imshow(grid.permute(1, 2, 0).numpy())
            plt.axis('off')
            plt.title("Discovered Cluster Samples")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, filename), dpi=self.dpi)
            plt.close()
