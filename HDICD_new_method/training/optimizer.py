import torch

def get_optimizer(model, lr=0.01, weight_decay=5e-5, momentum=0.9):
    """
    Setup the optimizer and scheduler.
    """
    classifier_params = [p for p in model.classifier.parameters() if p is not model.curvature]
    param_groups = [
        {'params': filter(lambda p: p.requires_grad, model.backbone.model.parameters()), 'lr': lr},
        {'params': model.projection_head.parameters(), 'lr': lr},
        {'params': classifier_params, 'lr': lr},
        {'params': [model.curvature], 'lr': lr * 1e-3}
    ]
    optimizer = torch.optim.SGD(param_groups, momentum=momentum, weight_decay=float(weight_decay))
    return optimizer

def get_scheduler(optimizer, epochs, lr):
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-3
    )
    return scheduler

def get_proto_optimizer(prototypes, lr=0.1, momentum=0.9):
    return torch.optim.SGD([prototypes], lr=lr, momentum=momentum)
