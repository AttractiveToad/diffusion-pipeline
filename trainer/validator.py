import torch
import logging
from clip_module import encode_text_prompt, get_unconditional_embeddings


class Validator:
    def __init__(self, unet, vae, text_encoder, tokenizer, clip_config, device, use_vae, config):
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.clip_config = clip_config
        self.device = device
        self.use_vae = use_vae
        self.config = config

        # Get validation config
        validation_config = self.config.get("validation", {})
        self.test_prompts = validation_config.get("prompts", [
            "pixel art house",
            "pixel art car",
            "pixel art character",
            "pixel art tree",
            "pixel art castle"
        ])

        # Get inference settings
        inference_config = validation_config.get("inference", {})
        self.timesteps = inference_config.get("timesteps", [800, 600, 400, 200, 50])
        self.guidance_scale = inference_config.get("guidance_scale", 7.5)
        self.grid_size = inference_config.get("grid_size", 32)

        logging.info(f"Validator initialized with {len(self.test_prompts)} prompts")

    def generate_validation_images(self, epoch):
        """Generate validation images for TensorBoard"""
        self.unet.eval()
        if self.vae:
            self.vae.eval()

        validation_images = []

        with torch.no_grad():
            for prompt in self.test_prompts:
                image = self._generate_single_image(prompt)
                validation_images.append(image)

        # Zurück in Training-Modus
        self.unet.train()
        if self.vae:
            self.vae.train()

        return torch.cat(validation_images, dim=0)

    def _generate_single_image(self, prompt):
        """Generate single image from prompt"""
        # Text embeddings
        text_embeddings = encode_text_prompt(
            self.tokenizer,
            self.text_encoder,
            [prompt],
            device=self.device,
            max_length=self.clip_config["max_length"]
        )

        uncond_embeddings = get_unconditional_embeddings(
            self.tokenizer,
            self.text_encoder,
            batch_size=1,
            device=self.device,
            max_length=self.clip_config["max_length"]
        )

        # Initialize latent
        if self.use_vae:
            latent = torch.randn(1, 4, self.grid_size, self.grid_size, device=self.device)
        else:
            latent = torch.randn(1, 3, self.grid_size, self.grid_size, device=self.device)

        # Simplified denoising
        for t in self.timesteps:
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

            # Simple update
            beta = 0.1
            latent = latent - beta * noise_pred

        # Decode
        if self.vae:
            try:
                image = self.vae.decode(latent).sample
            except:
                image = latent[:, :3, :, :]
        else:
            image = latent

        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image