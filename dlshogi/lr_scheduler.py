import math

from torch.optim.lr_scheduler import LRScheduler, StepLR


class CosineLRScheduler(LRScheduler):
    """ This code is based on the Cosine Learning Rate Scheduler implementation found at:
    https://github.com/huggingface/pytorch-image-models/blob/main/timm/scheduler/cosine_lr.py
    """
    def __init__(
        self,
        optimizer,
        t_initial,
        lr_min=0.0,
        cycle_mul=1.0,
        cycle_decay=1.0,
        cycle_limit=1,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        k_decay=1.0,
        last_epoch=-1,
    ):
        self.t_initial = t_initial
        self.lr_min = lr_min
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.k_decay = k_decay

        if last_epoch == -1:
            base_lrs = [group["lr"] for group in optimizer.param_groups]
        else:
            base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in base_lrs]
        else:
            self.warmup_steps = [1 for _ in base_lrs]

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        t = self.last_epoch
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(
                    math.log(
                        1 - t / self.t_initial * (1 - self.cycle_mul), self.cycle_mul
                    )
                )
                t_i = self.cycle_mul**i * self.t_initial
                t_curr = (
                    t - (1 - self.cycle_mul**i) / (1 - self.cycle_mul) * self.t_initial
                )
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.cycle_decay**i
            lr_max_values = [v * gamma for v in self.base_lrs]
            k = self.k_decay

            if i < self.cycle_limit:
                lrs = [
                    self.lr_min
                    + 0.5
                    * (lr_max - self.lr_min)
                    * (1 + math.cos(math.pi * t_curr**k / t_i**k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_lrs]

        return lrs

class WarmupStepLR(StepLR):
    def __init__(
        self,
        optimizer,
        step_size,
        gamma=0.1,
        warmup_t=0,
        warmup_lr_init=0,
        last_epoch=-1
    ):
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init

        if last_epoch == -1:
            base_lrs = [group["lr"] for group in optimizer.param_groups]
        else:
            base_lrs = [group["initial_lr"] for group in optimizer.param_groups]
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in base_lrs]
        else:
            self.warmup_steps = [1 for _ in base_lrs]

        super().__init__(optimizer, step_size, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_t:
            return [self.warmup_lr_init + self.last_epoch * s for s in self.warmup_steps]
        elif (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma
                for group in self.optimizer.param_groups]
