from diffusers import UNet2DModel

def build_unet(sample_size=512, in_c=4, out_c=4, channels=[128, 256, 512]):
    return UNet2DModel(
        sample_size=sample_size,
        in_channels=in_c,
        out_channels=out_c,
        layers_per_block=2,
        block_out_channels=tuple(channels),
        down_block_types=("DownBlock2D",) * len(channels),
        up_block_types=("UpBlock2D",) * len(channels),
    )