import torch
import torch.nn.functional as F
from hyperbolic.poincare_ops import mobius_add

def hyperbolic_similarity_matrix(features, c, alpha_d):
    """
    Compute combined similarity matrix using both hyperbolic distance and cosine similarity.
    """
    N = features.size(0)

    # Compute hyperbolic pairwise distance matrix
    x_i = features.unsqueeze(1)  # (N, 1, D)
    x_j = features.unsqueeze(0)  # (1, N, D)
    mobius_diff = mobius_add(-x_i, x_j, c)  # (N, N, D)
    dist_matrix = torch.norm(mobius_diff, dim=-1)  # (N, N)
    dist_sim = -2 / (c**0.5) * torch.atanh(torch.clamp(c**0.5 * dist_matrix, max=1 - 1e-5))

    # Cosine similarity matrix (angle-based)
    features_norm = F.normalize(features, dim=1)
    cos_sim = torch.matmul(features_norm, features_norm.T)  # (N, N)

    # Combine both similarities
    sim_matrix = alpha_d * dist_sim + (1 - alpha_d) * cos_sim
    return sim_matrix

def hyperbolic_info_nce_loss(features, n_views=2, temperature=0.7, c=1.0, alpha_d=0.5):
    """
    Compute InfoNCE loss in hyperbolic space.
    """
    device = features.device
    b_ = int(features.size(0) // n_views)
    labels = torch.cat([torch.arange(b_) for _ in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

    # Similarity matrix in hyperbolic space
    sim_matrix = hyperbolic_similarity_matrix(features, c, alpha_d)

    # Remove self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

    # Positive and negative splits
    positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

    return F.cross_entropy(logits, labels)
