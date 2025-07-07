import torch
import logging
import os
from tqdm import tqdm
from .model_builder import ModelBuilder
from .scheduler import SchedulerFactory
from .validator import Validator
from .logger import TrainingLogger
from .checkpointer import ModelCheckpointer
from clip_module import encode_text_prompt, get_unconditional_embeddings
from utils import get_latent_from_vae


class Trainer:
    def __init__(self, config, datamodule):
        self.config = config
        self.device = torch.device(config.get("device", "cuda"))
        self.datamodule = datamodule
        self.global_step = 0

        # Create directories
        self._setup_directories()

        # Setup components
        self._setup_models()
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_logging()
        self._setup_validation()
        self._setup_checkpointing()

    def _setup_directories(self):
        """Create necessary directories"""
        paths_config = self.config.get("paths", {})

        for path_name, path_value in paths_config.items():
            if path_value and not os.path.exists(path_value):
                os.makedirs(path_value, exist_ok=True)
                logging.info(f"Created directory: {path_value}")

    def _setup_models(self):
        """Setup all models"""
        logging.info("Setting up models...")
        builder = ModelBuilder(self.config, self.device)

        self.unet, self.unet_params = builder.build_unet()
        self.vae, self.vae_params = builder.build_vae()
        self.text_encoder, self.tokenizer, self.clip_config, self.clip_params = builder.build_text_encoder()

        self.use_vae = self.vae is not None

    def _setup_optimizer(self):
        """Setup optimizer"""
        training_config = self.config.get("training", {})
        optimizer_config = training_config.get("optimizer", {})

        # Collect parameters
        params = list(self.unet.parameters())
        if self.vae:
            params.extend(list(self.vae.parameters()))

        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(training_config.get("learning_rate", 0.0002)),
            betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
            weight_decay=optimizer_config.get("weight_decay", 0.01)
        )

        logging.info(f"Optimizer: lr={training_config.get('learning_rate')}")

    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        dataset_size = len(self.datamodule.dataset["train"])
        self.scheduler, self.scheduler_step = SchedulerFactory.create_scheduler(
            self.optimizer, self.config, dataset_size
        )

    def _setup_logging(self):
        """Setup logging"""
        paths_config = self.config.get("paths", {})
        save_dir = paths_config.get("save_dir", "./models")
        self.logger = TrainingLogger(save_dir, self.config)

    def _setup_validation(self):
        """Setup validation"""
        self.validator = Validator(
            self.unet, self.vae, self.text_encoder, self.tokenizer,
            self.clip_config, self.device, self.use_vae, self.config
        )

    def _setup_checkpointing(self):
        """Setup model checkpointing"""
        self.checkpointer = ModelCheckpointer(self.config)
        self.checkpointer.save_config(
            self.unet_params, self.vae_params, self.clip_params
        )

    def train_step(self, batch):
        """Single training step"""
        images = batch["pixel_values"].to(self.device)
        text_prompts = batch.get("text", [""] * images.shape[0])

        # VAE Encoding
        if self.use_vae:
            latents = get_latent_from_vae(self.vae, images)
        else:
            latents = images

        # Text Embeddings
        with torch.no_grad():
            text_embeddings = encode_text_prompt(
                self.tokenizer, self.text_encoder, text_prompts,
                device=self.device, max_length=self.clip_config["max_length"]
            )

            uncond_embeddings = get_unconditional_embeddings(
                self.tokenizer, self.text_encoder, batch_size=len(text_prompts),
                device=self.device, max_length=self.clip_config["max_length"]
            )

        # Noise Addition
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (latents.shape[0],), device=self.device)
        noisy_latents = latents + noise

        # Classifier-free guidance training
        training_config = self.config.get("training", {})
        cfg_prob = training_config.get("classifier_free_guidance_prob", 0.1)

        if torch.rand(1) < cfg_prob:
            encoder_hidden_states = uncond_embeddings
        else:
            encoder_hidden_states = text_embeddings

        # Forward pass
        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None and self.scheduler_step == "step":
            self.scheduler.step()

        return loss.item()

    def train_epoch(self, epoch):
        """Train single epoch"""
        self.unet.train()
        if self.vae:
            self.vae.train()
        self.text_encoder.eval()

        epoch_loss = 0.0
        running_loss = 0.0
        batch_count = 0

        loader = self.datamodule.train_dataloader()
        training_config = self.config.get("training", {})
        epochs = training_config.get("epochs", 50)

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")

        for i, batch in enumerate(pbar):
            loss = self.train_step(batch)

            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_step(loss, current_lr, self.global_step)

            # Loss tracking
            running_loss += loss
            epoch_loss += loss
            batch_count += 1
            self.global_step += 1

            # Progress update
            logging_config = self.config.get("logging", {})
            console_config = logging_config.get("console", {})
            progress_freq = console_config.get("progress_frequency", 10)

            if (i + 1) % progress_freq == 0:
                avg_loss = running_loss / progress_freq
                pbar.set_postfix({
                    "loss": f"{avg_loss:.6f}",
                    "lr": f"{current_lr:.2e}"
                })
                running_loss = 0.0

        return epoch_loss / batch_count

    def train(self):
        """Main training loop"""
        best_loss = float('inf')
        training_config = self.config.get("training", {})
        validation_config = self.config.get("validation", {})

        epochs = training_config.get("epochs", 50)
        val_frequency = validation_config.get("frequency", 5)

        for epoch in range(epochs):
            # Train epoch
            avg_epoch_loss = self.train_epoch(epoch)

            # Scheduler step
            if self.scheduler is not None and self.scheduler_step == "epoch":
                self.scheduler.step()

            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.log_epoch(avg_epoch_loss, current_lr, epoch)
            logging.info(f"Epoch {epoch + 1}/{epochs} Loss: {avg_epoch_loss:.6f}")

            # Validation
            if (epoch + 1) % val_frequency == 0:
                self.validate(epoch)

            # Checkpointing
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                self.checkpointer.save_models(self.unet, self.vae, self.clip_params)
                self.logger.log_best_loss(best_loss, epoch)

        self.logger.close()
        logging.info(f"Training completed. Best loss: {best_loss:.6f}")

    def validate(self, epoch):
        """Run validation"""
        logging.info("Generating validation images...")
        val_images = self.validator.generate_validation_images(epoch)
        self.logger.log_validation(val_images, self.validator.test_prompts, epoch)