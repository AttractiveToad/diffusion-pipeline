import os
import logging
import json
import torch
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

            # Config in TensorBoard speichern
            self._log_config()
        else:
            self.writer = None
            logging.info("TensorBoard disabled")

        # Minimal settings
        self.log_frequency = tensorboard_config.get("log_frequency", 10)

    def _log_config(self):
        """Log die gesamte Config in TensorBoard"""
        if not self.writer:
            return

        # Als Text speichern
        config_text = json.dumps(self.config, indent=2)
        self.writer.add_text("config", config_text, 0)

        # Wichtige Hyperparameter als Hyperparameter-Tab
        hparams = self._extract_hparams()
        if hparams:
            # Dummy metrics für hparams (TensorBoard braucht das)
            metrics = {"training_loss": 0.0}
            self.writer.add_hparams(hparams, metrics)

    def _extract_hparams(self):
        """Extrahiere wichtige Hyperparameter aus der Config"""
        hparams = {}

        # Training params
        training_config = self.config.get("training", {})
        hparams["learning_rate"] = training_config.get("learning_rate", 1e-4)
        hparams["epochs"] = training_config.get("epochs", 50)

        # Dataset params
        dataset_config = self.config.get("dataset", {})
        hparams["batch_size"] = dataset_config.get("batch_size", 4)

        # Optimizer params
        optimizer_config = training_config.get("optimizer", {})
        hparams["weight_decay"] = optimizer_config.get("weight_decay", 0.01)

        # Model params falls vorhanden
        model_config = self.config.get("models", {})
        if "unet" in model_config:
            unet_config = model_config["unet"]
            hparams["unet_channels"] = str(unet_config.get("channels", [128, 256, 512]))

        return hparams

    def log_step(self, loss, lr, step):
        """Log training loss und learning rate"""
        if not self.writer or step % self.log_frequency != 0:
            return

        self.writer.add_scalar("training_loss", loss, step)
        self.writer.add_scalar("learning_rate", lr, step)

    def log_epoch(self, epoch_loss, lr, epoch):
        """Log epoch loss"""
        if not self.writer:
            return

        self.writer.add_scalar("epoch_loss", epoch_loss, epoch)
        self.writer.add_scalar("learning_rate_epoch", lr, epoch)

    def log_gradients(self, model, step):
        """Log gradient norms für debugging"""
        if not self.writer or step % (self.log_frequency * 10) != 0:
            return

        total_norm = 0
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                # Log einzelne Layer-Gradienten
                self.writer.add_scalar(f"gradients/{name}", param_norm.item(), step)

        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar("gradient_norm_total", total_norm, step)

    def log_validation(self, val_images, test_prompts, epoch):
        """Log validation images"""
        if not self.writer or len(val_images) == 0:
            return

        # Mehrere Bilder als Grid
        if len(val_images) > 1:
            import torchvision.utils as vutils
            grid = vutils.make_grid(val_images, nrow=2, normalize=True, value_range=(0, 1))
            self.writer.add_image("validation_images", grid, epoch)
        else:
            self.writer.add_image("validation_images", val_images[0], epoch)

    def log_best_loss(self, best_loss, epoch):
        """Log best loss"""
        if not self.writer:
            return

        self.writer.add_scalar("best_loss", best_loss, epoch)

    def close(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()
