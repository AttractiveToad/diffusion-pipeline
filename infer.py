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
        self.palette_colors = self.config["palette_colors"]
        self.out_dir = self.config["out_dir"]
        os.makedirs(self.out_dir, exist_ok=True)

        logging.info(f"Lade Modelle für infer ({self.config['save_dir']}) ...")
        self.max_size = self.config["max_image_size"]
        self.channels = self.config["channels"]
        self.vae_channels = self.config["vae_channels"]

        self.unet = build_unet(
            sample_size=self.max_size, channels=self.channels
        ).to(self.device)
        self.unet.load_state_dict(
            load_file(f"{self.config['save_dir']}/unet/diffusion_pytorch_model.safetensors")
        )
        self.vae = build_vae(
            image_size=self.max_size, vae_channels=self.vae_channels
        ).to(self.device)
        state_dict = load_file(f"{self.config['save_dir']}/vae/diffusion_pytorch_model.safetensors")
        self.vae.load_state_dict(state_dict)
        self.unet.eval()
        self.vae.eval()
        logging.info("Modelle geladen und eval gesetzt")

    def postprocess_pixelart(self, image, grid=32, out_size=256):
        logging.debug(f'Pixelart-Postprocessing: grid={grid}, out_size={out_size}')
        image = image.resize((grid, grid), Image.NEAREST)
        image = image.quantize(colors=self.palette_colors, method=2)
        return image.resize((out_size, out_size), Image.NEAREST)

    def infer(self, grid, out_px):
        with torch.no_grad():
            logging.info(f"Starte Inferenz für grid={grid}, out_px={out_px}")
            latent = torch.randn(1, 4, grid, grid, device=self.device)
            if grid < self.max_size:
                pad = (0, self.max_size - grid, 0, self.max_size - grid)
                latent = torch.nn.functional.pad(latent, pad)
                logging.debug(f'Latent padded to max_size={self.max_size}')

            for t in reversed(range(1000)):
                latent = self.unet(latent, torch.tensor([t], device=self.device)).sample
                if t % 200 == 0:
                    logging.debug(f"Inferenz: t={t}")

            # Crop auf grid
            if grid < self.max_size:
                out_latent = latent[:, :, :grid, :grid]
            else:
                out_latent = latent
            image = self.vae.decode(out_latent / 0.18215).sample
            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = transforms.ToPILImage()(image.squeeze().cpu())
        return self.postprocess_pixelart(image, grid, out_px)

    def create_asset(self, grid, out_px):
        outname = f"{self.out_dir}/asset_{grid}_{out_px}.png"
        result = self.infer(grid, out_px)
        result.save(outname)
        logging.info(f"Asset gespeichert: {outname}")

def main():
    inferencer = Inferencer("config.yaml")
    grid = int(input("Welche Gridgröße? (16, 32 ... 512): ").strip())
    out_px = int(input("Endgröße (Pixel, z.B. 256): ").strip())
    inferencer.create_asset(grid, out_px)

if __name__ == "__main__":
    main()