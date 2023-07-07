from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, eta_min=1e-6, last_epoch=-1):
        self.base_lr = base_lr
        self.eta_min = eta_min
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        self.power = 2
        super(PolyScheduler, self).__init__(optimizer, -1, False)
        self.last_epoch = last_epoch

    def get_lr(self):
        if self.last_epoch == -1:
            return [self.eta_min for _ in self.optimizer.param_groups]
        
        if self.last_epoch < self.warmup_steps:
            alpha = pow(float(self.last_epoch) / float(self.warmup_steps), self.power)
        else:
            alpha = pow(
                1.0
                - float(self.last_epoch - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power
            )

        return [(self.base_lr - self.eta_min) * alpha + self.eta_min for _ in self.optimizer.param_groups]
