from diffusers import AutoencoderKL

def build_vae(image_size=512, vae_channels=[128], norm_groups=8):
    return AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=4,
        block_out_channels=tuple(vae_channels),
        norm_num_groups=norm_groups,
        sample_size=image_size,
    )