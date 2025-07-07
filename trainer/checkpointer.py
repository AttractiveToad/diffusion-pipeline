import os
import yaml
import logging


class ModelCheckpointer:
    def __init__(self, config):
        self.config = config

        # Get paths config
        paths_config = self.config.get("paths", {})
        self.save_dir = paths_config.get("save_dir", "./models")
        self.checkpoints_dir = paths_config.get("checkpoints_dir", os.path.join(self.save_dir, "checkpoints"))

        # Get checkpointing config
        checkpointing_config = self.config.get("checkpointing", {})
        self.save_best = checkpointing_config.get("save_best", True)
        self.save_frequency = checkpointing_config.get("save_frequency", None)
        self.keep_top_k = checkpointing_config.get("keep_top_k", 3)

        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        logging.info(f"Checkpointer: save_dir={self.save_dir}, save_best={self.save_best}")

    def save_models(self, unet, vae, clip_params):
        """Save all models"""
        if not self.save_best:
            return

        # UNet speichern
        unet_path = os.path.join(self.save_dir, "unet")
        os.makedirs(unet_path, exist_ok=True)
        unet.save_pretrained(unet_path)

        # VAE speichern
        if vae:
            vae_path = os.path.join(self.save_dir, "vae")
            os.makedirs(vae_path, exist_ok=True)
            vae.save_pretrained(vae_path)

        logging.info("Beste Modelle gespeichert")

    def save_config(self, unet_params, vae_params, clip_params):
        """Save training configuration"""
        config_path = os.path.join(self.save_dir, "training_config.yaml")

        # Extract relevant training config
        training_config = self.config.get("training", {})
        image_config = self.config.get("image", {})
        dataset_config = self.config.get("dataset", {})

        save_config = {
            'model_params': {
                'unet_params': unet_params,
                'vae_params': vae_params,
                'clip_params': clip_params
            },
            'training_config': {
                'max_size': image_config.get("max_size"),
                'batch_size': dataset_config.get("batch_size"),
                'learning_rate': training_config.get("learning_rate"),
                'epochs': training_config.get("epochs")
            },
            'full_config': self.config
        }

        with open(config_path, 'w') as f:
            yaml.dump(save_config, f, default_flow_style=False)