from diffusers import AutoencoderKL

DEFAULT_VAE_CONFIG = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,

    "latent_channels": 4,
    "block_out_channels": (64, 128),
    "scaling_factor": 0.18215,
    # advanced config
    "down_block_types": ("DownEncoderBlock2D", "DownEncoderBlock2D"),
    "up_block_types": ("UpDecoderBlock2D", "UpDecoderBlock2D"),
    "act_fn": "silu",
    "mid_block_add_attention": True
}

def build_vae(sample_size=None, block_out_channels=None):
    """
    Build VAE with defaults that can be overridden by config
    """
    config = DEFAULT_VAE_CONFIG.copy()
    
    if sample_size is not None:
        config["sample_size"] = sample_size
    if block_out_channels is not None:
        config["block_out_channels"] = block_out_channels

    return AutoencoderKL(
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        latent_channels=config["latent_channels"],
        block_out_channels=config["block_out_channels"],
        down_block_types=config["down_block_types"],
        up_block_types=config["up_block_types"],
        act_fn=config["act_fn"],
        scaling_factor=config["scaling_factor"],
        sample_size=config["sample_size"],
        mid_block_add_attention=config["mid_block_add_attention"]
    )