from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ 
    ref: https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(
            self,
            optimizer,
            multiplier,
            warmup_iterations,
            after_scheduler):
        self.multiplier = multiplier
        self.warmup_iterations = warmup_iterations
        self.after_scheduler = after_scheduler
        super().__init__(optimizer)
        self.after_scheduler.base_lrs = [
            base_lr * self.multiplier for base_lr in self.base_lrs]

    def get_lr(self):
        if self.last_epoch <= self.warmup_iterations:
            return [base_lr *
                    ((self.multiplier -
                      1.) *
                     self.last_epoch /
                     self.warmup_iterations +
                     1.) for base_lr in self.base_lrs]

        self.after_scheduler.step(
            max(self.last_epoch - self.warmup_iterations, 0))

        return self.after_scheduler.get_last_lr()

    def set_step(self, step):
        self.last_epoch = step

    def lr(self):
        return self.get_lr()