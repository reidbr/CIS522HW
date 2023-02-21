from typing import List
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Set the learning rate of each parameter group using a cosine annealing
    schedule

    """

    def __init__(self, optimizer, last_epoch=-1, max_epochs=100, warmup_epochs=10):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns the list of LR from the custom scheduler
        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        if self.last_epoch < self.warmup_epochs:
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]

        return [
            base_lr
            * (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / 2
            for base_lr in self.base_lrs
        ]
