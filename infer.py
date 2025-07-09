import yaml
import torch
from PIL import Image
from torchvision import transforms
import os
from diffusers import UNet2DConditionModel, AutoencoderKL
from clip_module import build_clip_text_encoder, encode_text_prompt, get_unconditional_embeddings

import logconf
import logging


class Inferencer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config["device"])
        self.max_size = self.config["image"]["max_size"]
        self.use_vae = self.max_size > 32

        # check and create directories
        self.out_dir = self.config["paths"]["output_dir"]
        os.makedirs(self.config["paths"]["output_dir"], exist_ok=True)

        logging.info(f"Loading models for inference ({self.config['paths']['save_dir']}) ...")

        # load UNet (now UNet2DConditionModel)
        unet_path = f"{self.config['paths']['save_dir']}/unet"
        self.unet = UNet2DConditionModel.from_pretrained(unet_path).to(self.device)
        self.unet.eval()

        # load VAE
        if self.use_vae:
            vae_path = f"{self.config['paths']['save_dir']}/vae"
            self.vae = AutoencoderKL.from_pretrained(vae_path).to(self.device)
            self.vae.eval()
        else:
            self.vae = None

        # load CLIP text encoder
        clip_config_path = f"{self.config['paths']['save_dir']}/clip_config.yaml"
        if os.path.exists(clip_config_path):
            with open(clip_config_path, 'r') as f:
                clip_params = yaml.safe_load(f)
        else:
            # Fallback to default config
            clip_config = self.config.get("models", {}).get("clip", {})
            clip_params = {
                "model_name": clip_config.get("model_name", "runwayml/stable-diffusion-v1-5"),
                "max_length": clip_config.get("max_length", 77)
            }

        logging.info(f"Loading CLIP with parameters: {clip_params}")
        self.text_encoder, self.tokenizer, self.clip_config = build_clip_text_encoder(**clip_params)
        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder.eval()

        # Classifier-free guidance scale
        clip_config = self.config.get("models", {}).get("clip", {})
        self.guidance_scale = clip_config.get("guidance_scale", 7.5)

        logging.info("Models loaded and set to eval mode")

    def postprocess_pixelart(self, image, grid=32, out_size=256, palette_colors=16):
        logging.debug(f'Pixelart-Postprocessing: grid={grid}, out_size={out_size}, palette_colors={palette_colors}')
        image = image.resize((grid, grid), Image.NEAREST)
        image = image.quantize(colors=palette_colors, method=2)
        return image.resize((out_size, out_size), Image.NEAREST)

    def infer(self, grid, out_px, palette_colors=16, prompt="", negative_prompt=""):
        with torch.no_grad():
            logging.info(f"Starting inference with grid={grid}, out_px={out_px}, palette_colors={palette_colors}")
            logging.info(f"Prompt: '{prompt}'")
            if negative_prompt:
                logging.info(f"Negative prompt: '{negative_prompt}'")

            # Generate text embeddings
            text_embeddings = encode_text_prompt(
                self.tokenizer,
                self.text_encoder,
                [prompt],
                device=self.device,
                max_length=self.clip_config["max_length"]
            )

            # Generate unconditional embeddings for classifier-free guidance
            uncond_embeddings = get_unconditional_embeddings(
                self.tokenizer,
                self.text_encoder,
                batch_size=1,
                device=self.device,
                max_length=self.clip_config["max_length"]
            )

            # If negative prompt is provided, use it instead of empty prompt
            if negative_prompt:
                negative_embeddings = encode_text_prompt(
                    self.tokenizer,
                    self.text_encoder,
                    [negative_prompt],
                    device=self.device,
                    max_length=self.clip_config["max_length"]
                )
                uncond_embeddings = negative_embeddings

            # Initialize latent
            latent = torch.randn(1, 4, grid, grid, device=self.device)
            if grid < self.max_size:
                pad = (0, self.max_size - grid, 0, self.max_size - grid)
                latent = torch.nn.functional.pad(latent, pad)
                logging.debug(f'Latent padded to max_size={self.max_size}')

            # Denoising loop with classifier-free guidance
            for t in reversed(range(1000)):
                # Unconditional prediction
                noise_pred_uncond = self.unet(
                    latent,
                    torch.tensor([t], device=self.device),
                    encoder_hidden_states=uncond_embeddings
                ).sample

                # Conditional prediction
                noise_pred_text = self.unet(
                    latent,
                    torch.tensor([t], device=self.device),
                    encoder_hidden_states=text_embeddings
                ).sample

                # Classifier-free guidance
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Update latent (simplified scheduler - in real implementation use proper scheduler)
                alpha = 0.99  # simplified - should use proper scheduling
                latent = alpha * latent + (1 - alpha) * noise_pred

                if t % 200 == 0:
                    logging.debug(f"Inferenz: t={t}")

            # Crop back to original size if padded
            if grid < self.max_size:
                out_latent = latent[:, :, :grid, :grid]
            else:
                out_latent = latent

            # Decode through VAE
            if self.vae:
                image = self.vae.decode(out_latent).sample
            else:
                image = out_latent

            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = transforms.ToPILImage()(image.squeeze().cpu())

        return self.postprocess_pixelart(image, grid, out_px, palette_colors)

    def create_asset(self, grid, out_px, palette_colors=16, prompt="", negative_prompt=""):
        filename_prompt = prompt.replace(" ", "_").replace(",", "")[:50] if prompt else "no_prompt"
        outname = f"{self.out_dir}/asset_{grid}_{out_px}_pal{palette_colors}_{filename_prompt}.png"
        result = self.infer(grid, out_px, palette_colors, prompt, negative_prompt)
        result.save(outname)
        logging.info(f"Asset saved: {outname}")


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    inferencer = Inferencer(config)

    # User input
    grid = int(input("Welche Gridgröße? (16, 32 ... 512): ").strip())
    out_px = int(input("Endgröße (Pixel, z.B. 256): ").strip())
    palette_colors = int(input("Anzahl der Farben (z.B. 16): ").strip())

    # Text prompts
    prompt = input("Text prompt (optional): ").strip()
    negative_prompt = input("Negative prompt (optional): ").strip()

    inferencer.create_asset(grid, out_px, palette_colors, prompt, negative_prompt)


if __name__ == "__main__":
    main()