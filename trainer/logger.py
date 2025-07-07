import os
import logging
from torch.utils.tensorboard import SummaryWriter


class TrainingLogger:
    def __init__(self, save_dir, config):
        self.config = config

        # Get logging config
        logging_config = self.config.get("logging", {})
        tensorboard_config = logging_config.get("tensorboard", {})

        # Setup tensorboard (nur wenn aktiviert)
        self.use_tensorboard = tensorboard_config.get("enabled", True)

        if self.use_tensorboard:
            paths_config = self.config.get("paths", {})
            self.tensorboard_dir = paths_config.get("tensorboard_dir", os.path.join(save_dir, "tensorboard"))
            os.makedirs(self.tensorboard_dir, exist_ok=True)

            self.writer = SummaryWriter(self.tensorboard_dir)
            logging.info(f"TensorBoard enabled: {self.tensorboard_dir}")
        else:
            self.writer = None
            logging.info("TensorBoard disabled")

        # Minimal settings
        self.log_frequency = tensorboard_config.get("log_frequency", 50)  # Weniger häufig

    def log_step(self, loss, lr, step):
        """Log nur alle X steps"""
        if not self.writer or step % self.log_frequency != 0:
            return

        self.writer.add_scalar("Loss/Train", loss, step)
        self.writer.add_scalar("LR", lr, step)

    def log_epoch(self, epoch_loss, lr, epoch):
        """Log epoch - nur Loss"""
        if not self.writer:
            return

        self.writer.add_scalar("Loss/Epoch", epoch_loss, epoch)

    def log_validation(self, val_images, test_prompts, epoch):
        """Log nur das erste Validation Image"""
        if not self.writer or len(val_images) == 0:
            return

        # Nur das erste Bild
        first_image = val_images[0:1]
        self.writer.add_image("Validation/Sample", first_image[0], epoch)

    def log_best_loss(self, best_loss, epoch):
        """Log best loss"""
        if not self.writer:
            return

        self.writer.add_scalar("Loss/Best", best_loss, epoch)

    def close(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()