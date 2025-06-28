import yaml
import torch
import os
from tqdm import tqdm
from dataset import DataModule
from unet_module import build_unet
from vae_module import build_vae
from utils import batch_to_device, get_latent_from_vae

import logconf
import logging

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.max_size = config["max_image_size"]
        self.use_vae = self.max_size > 32
        self.in_ch = self.out_ch = 4 if self.use_vae else 3

        # UNet Konfiguration
        unet_params = {
            "sample_size": self.max_size,
            "in_c": self.in_ch,
            "out_c": self.out_ch,
        }
        if "channels" in config:
            unet_params["channels"] = config["channels"]

        logging.info(f"Baue UNet auf, sample_size={self.max_size}, in_c={self.in_ch}, out_c={self.out_ch}")
        self.unet = build_unet(**unet_params).to(self.device)

        if self.use_vae:
            # VAE Konfiguration
            vae_params = {
                "image_size": self.max_size,
            }
            if "block_out_channels" in config:
                vae_params["block_out_channels"] = config["block_out_channels"]

            logging.info("Baue VAE auf")
            self.vae = build_vae(**vae_params).to(self.device)
        else:
            self.vae = None

        params = list(self.unet.parameters()) + (list(self.vae.parameters()) if self.vae else [])
        self.optimizer = torch.optim.AdamW(params, lr=float(config["learning_rate"]))

        logging.info('Initialisiere DataModule...')
        self.datamodule = DataModule(config)
        self.datamodule.setup()

    def train(self):
        self.unet.train()
        if self.vae:
            self.vae.train()
        for epoch in range(self.config["epochs"]):
            logging.info(f"Starte Training: Epoch {epoch+1}/{self.config['epochs']}")
            running_loss = 0.0
            loader = self.datamodule.train_dataloader()
            pbar = tqdm(loader, desc=f"Ep {epoch+1}/{self.config['epochs']}")
            for i, batch in enumerate(pbar):
                images = batch["pixel_values"].to(self.device)
                if self.use_vae:
                    latents = get_latent_from_vae(self.vae, images)
                else:
                    latents = images

                noise = torch.randn_like(latents)
                bsz = latents.size(0)
                timesteps = torch.randint(0, 1000, (bsz,), device=self.device).long()
                noisy_latents = latents + noise
                noise_pred = self.unet(noisy_latents, timesteps).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if (i + 1) % 10 == 0:
                    avg = running_loss / 10
                    pbar.set_postfix({"loss": avg})
                    logging.debug(f'Epoch {epoch+1}, Step {i+1}: Loss={avg:.6f}')
                    running_loss = 0.0
            logging.info(f"Epoch {epoch+1} abgeschlossen.")

        # Speichern
        logging.info("Speichere Modelle...")
        os.makedirs(f"{self.config['save_dir']}/unet", exist_ok=True)
        self.unet.save_pretrained(f"{self.config['save_dir']}/unet")
        if self.use_vae:
            os.makedirs(f"{self.config['save_dir']}/vae", exist_ok=True)
            self.vae.save_pretrained(f"{self.config['save_dir']}/vae")
        logging.info("Modelle gespeichert.")

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    trainer = Trainer(config)
    logging.info("Starte Training")
    trainer.train()
    logging.info("Training abgeschlossen")

if __name__ == "__main__":
    main()