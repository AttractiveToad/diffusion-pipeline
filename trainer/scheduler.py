import math
import logging
from torch.optim.lr_scheduler import CosineAnnealingLR


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.current_step = 0

        logging.info(f"WarmupCosineScheduler: warmup_steps={warmup_steps}, "
                     f"total_steps={total_steps}, max_lr={max_lr}, min_lr={min_lr}")

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            lr = self.max_lr * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class SchedulerFactory:
    @staticmethod
    def create_scheduler(optimizer, config, dataset_size):
        training_config = config.get("training", {})
        lr_config = training_config.get("lr_scheduler", {})
        scheduler_type = lr_config.get("type", "constant")

        logging.info(f"Creating scheduler of type: {scheduler_type}")

        if scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=training_config.get("epochs", 50),
                eta_min=float(lr_config.get("min_lr", 1e-6))
            )
            return scheduler, "epoch"

        elif scheduler_type == "warmup_cosine":
            # Calculate steps
            dataset_config = config.get("dataset", {})
            batch_size = dataset_config.get("batch_size", 10)
            steps_per_epoch = dataset_size // batch_size
            total_steps = steps_per_epoch * training_config.get("epochs", 50)
            warmup_steps = lr_config.get("warmup_steps", 500)

            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                max_lr=float(training_config.get("learning_rate", 0.0002)),
                min_lr=float(lr_config.get("min_lr", 1e-6))
            )
            return scheduler, "step"

        else:  # constant
            logging.info("Using constant learning rate (no scheduler)")
            return None, None