import yaml
import torch
from PIL import Image
from torchvision import transforms
from unet_module import build_unet
from vae_module import build_vae
from safetensors.torch import load_file
import os

import logconf
import logging

class Inferencer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.device = torch.device(self.config["device"])
        self.out_dir = self.config["out_dir"]
        os.makedirs(self.out_dir, exist_ok=True)

        logging.info(f"Lade Modelle für infer ({self.config['save_dir']}) ...")
        self.max_size = self.config["max_image_size"]

        # UNet Konfiguration
        unet_params = {
            "sample_size": self.max_size,
        }
        if "channels" in self.config:
            unet_params["channels"] = self.config["channels"]

        self.unet = build_unet(**unet_params).to(self.device)
        self.unet.load_state_dict(
            load_file(f"{self.config['save_dir']}/unet/diffusion_pytorch_model.safetensors")
        )

        # VAE Konfiguration
        vae_params = {
            "image_size": self.max_size,
        }
        if "vae_channels" in self.config:
            vae_params["vae_channels"] = self.config["vae_channels"]
        if "norm_groups" in self.config:
            vae_params["norm_groups"] = self.config["norm_groups"]

        self.vae = build_vae(**vae_params).to(self.device)
        state_dict = load_file(f"{self.config['save_dir']}/vae/diffusion_pytorch_model.safetensors")
        self.vae.load_state_dict(state_dict)
        
        self.unet.eval()
        self.vae.eval()
        logging.info("Modelle geladen und eval gesetzt")

    def postprocess_pixelart(self, image, grid=32, out_size=256, palette_colors=16):
        logging.debug(f'Pixelart-Postprocessing: grid={grid}, out_size={out_size}, palette_colors={palette_colors}')
        image = image.resize((grid, grid), Image.NEAREST)
        image = image.quantize(colors=palette_colors, method=2)
        return image.resize((out_size, out_size), Image.NEAREST)

    def infer(self, grid, out_px, palette_colors=16):
        with torch.no_grad():
            logging.info(f"Starte Inferenz für grid={grid}, out_px={out_px}, palette_colors={palette_colors}")
            latent = torch.randn(1, 4, grid, grid, device=self.device)
            if grid < self.max_size:
                pad = (0, self.max_size - grid, 0, self.max_size - grid)
                latent = torch.nn.functional.pad(latent, pad)
                logging.debug(f'Latent padded to max_size={self.max_size}')

            for t in reversed(range(1000)):
                latent = self.unet(latent, torch.tensor([t], device=self.device)).sample
                if t % 200 == 0:
                    logging.debug(f"Inferenz: t={t}")

            if grid < self.max_size:
                out_latent = latent[:, :, :grid, :grid]
            else:
                out_latent = latent
            image = self.vae.decode(out_latent / 0.18215).sample
            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = transforms.ToPILImage()(image.squeeze().cpu())
        return self.postprocess_pixelart(image, grid, out_px, palette_colors)

    def create_asset(self, grid, out_px, palette_colors=16):
        outname = f"{self.out_dir}/asset_{grid}_{out_px}_pal{palette_colors}.png"
        result = self.infer(grid, out_px, palette_colors)
        result.save(outname)
        logging.info(f"Asset gespeichert: {outname}")

def main():
    inferencer = Inferencer("config.yaml")
    grid = int(input("Welche Gridgröße? (16, 32 ... 512): ").strip())
    out_px = int(input("Endgröße (Pixel, z.B. 256): ").strip())
    palette_colors = int(input("Anzahl der Farben (z.B. 16): ").strip())
    inferencer.create_asset(grid, out_px, palette_colors)

if __name__ == "__main__":
    main()