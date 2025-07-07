import torch
import logging
from unet_module import build_unet
from vae_module import build_vae
from clip_module import build_clip_text_encoder


class ModelBuilder:
    def __init__(self, config, device):
        self.config = config
        self.device = device

        # Image configuration
        image_config = self.config.get("image", {})
        self.max_size = image_config.get("max_size", 64)

        # Model settings
        self.use_vae = self.max_size > 32
        self.in_ch = self.out_ch = 4 if self.use_vae else 3

        logging.info(f"ModelBuilder: max_size={self.max_size}, use_vae={self.use_vae}")

    def build_unet(self):
        """Build UNet model"""
        unet_config = self.config.get("models", {}).get("unet", {})

        unet_params = {
            "sample_size": self.max_size,
            "in_c": self.in_ch,
            "out_c": self.out_ch,
            "channels": unet_config.get("channels", [128, 256, 512]),
            "norm_num_groups": unet_config.get("norm_num_groups", 32)
        }

        logging.info(f"UNet Parameter: {unet_params}")
        unet = build_unet(**unet_params).to(self.device)
        return unet, unet_params

    def build_vae(self):
        """Build VAE model if needed"""
        if not self.use_vae:
            logging.info("VAE not needed for this image size")
            return None, None

        vae_config = self.config.get("models", {}).get("vae", {})
        vae_params = {
            "sample_size": self.max_size,
            "block_out_channels": vae_config.get("block_out_channels", [64, 128]),
        }

        logging.info(f"VAE Parameter: {vae_params}")
        vae = build_vae(**vae_params).to(self.device)
        return vae, vae_params

    def build_text_encoder(self):
        """Build CLIP text encoder"""
        clip_config = self.config.get("models", {}).get("clip", {})

        clip_params = {
            "model_name": clip_config.get("model_name", "runwayml/stable-diffusion-v1-5"),
            "max_length": clip_config.get("max_length", 77)
        }

        logging.info(f"CLIP Parameter: {clip_params}")
        text_encoder, tokenizer, loaded_clip_config = build_clip_text_encoder(**clip_params)
        text_encoder = text_encoder.to(self.device)

        # Freeze CLIP text encoder
        for param in text_encoder.parameters():
            param.requires_grad = False

        logging.info("CLIP text encoder geladen und eingefroren")
        return text_encoder, tokenizer, loaded_clip_config, clip_params