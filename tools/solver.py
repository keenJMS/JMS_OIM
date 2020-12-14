
import torch

def get_optimizer(cfg, model):
    lr = cfg.SOLVER.BASE_LR
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr':lr * (cfg.SOLVER.DOUBLE_BIAS + 1),
                            'weight_decay': cfg.SOLVER.WEIGHT_DECAY and cfg.SOLVER.WEIGHT_DECAY_BIAS or 0}]
            else:
                params += [{'params': [value], 'lr':lr,
                            'weight_decay': cfg.SOLVER.WEIGHT_DECAY}]

    optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    return optimizer


def get_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.LR_DECAY_MILESTONES is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES,
            gamma=cfg.SOLVER.GAMMA)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.SOLVER.STEP_SIZE,
            gamma=cfg.SOLVER.GAMMA)

    return scheduler


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)