from diffusers import UNet2DModel

DEFAULT_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,
    "layers_per_block": 2,
    "channels": [256, 512]
}


def build_unet(sample_size=None, in_c=None, out_c=None, channels=None):
    """
    Build UNet with defaults that can be overridden by config
    """
    config = DEFAULT_UNET_CONFIG.copy()

    if sample_size is not None:
        config["sample_size"] = sample_size
    if in_c is not None:
        config["in_channels"] = in_c
    if out_c is not None:
        config["out_channels"] = out_c
    if channels is not None:
        config["channels"] = channels

    return UNet2DModel(
        sample_size=config["sample_size"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        layers_per_block=config["layers_per_block"],
        block_out_channels=tuple(config["channels"]),
        down_block_types=("DownBlock2D",) * len(config["channels"]),
        up_block_types=("UpBlock2D",) * len(config["channels"]),
    )