
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
    if cfg.SOLVER.OPTIMIZER_NAME != 'SGD':
        optimizer = torch.optim.__dict__[cfg.SOLVER.OPTIMIZER_NAME](params)
    else:
        optimizer = torch.optim.SGD(params,momentum=cfg.SOLVER.MOMENTUM)

    return optimizer


def get_lr_scheduler(cfg, optimizer):
    if cfg.SOLVER.WARMUP_ITERS>0:
        scheduler= warmup_lr_scheduler(optimizer,cfg)
        print("using warmup")
        return scheduler
    elif cfg.SOLVER.LR_DECAY_MILESTONES is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES,
            gamma=cfg.SOLVER.GAMMA)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.SOLVER.STEP_SIZE,
            gamma=cfg.SOLVER.GAMMA)

    return scheduler


def warmup_lr_scheduler(optimizer, cfg):

    def f(x):
        if x >= cfg.SOLVER.WARMUP_ITERS:
            if cfg.SOLVER.WARMUP_METHOD=='constant':
                return 1
            elif cfg.SOLVER.WARMUP_METHOD=='step':
                return cfg.SOLVER.WARMUP_GAMMA**(int((x-cfg.SOLVER.WARMUP_ITERS)/cfg.SOLVER.WARMUP_DECAY_STEP))
            else:
                raise RuntimeError("wrong WARMUP_METHOD")
        alpha = float(x) / cfg.SOLVER.WARMUP_ITERS

        return cfg.SOLVER.WARMUP_FACTOR * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)