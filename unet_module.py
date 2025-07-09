from diffusers import UNet2DConditionModel

DEFAULT_UNET_CONFIG = {
    "sample_size": 64,
    "in_channels": 4,
    "out_channels": 4,
    "center_input_sample": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,

    # Block configuration
    "down_block_types": (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D"
    ),
    "mid_block_type": "UNetMidBlock2DCrossAttn",
    "up_block_types": (
        "UpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D"
    ),

    "only_cross_attention": False,
    "block_out_channels": (256, 512, 1024),
    "layers_per_block": 2,
    "downsample_padding": 1,
    "mid_block_scale_factor": 1.0,
    "dropout": 0.0,
    "act_fn": "silu",
    "norm_num_groups": 32,
    "norm_eps": 1e-5,

    # Cross attention configuration
    "cross_attention_dim": 768,  # CLIP text embedding dimension
    "transformer_layers_per_block": 1,
    "reverse_transformer_layers_per_block": None,

    # Encoder configuration
    "encoder_hid_dim": None,
    "encoder_hid_dim_type": None,

    # Attention configuration
    "attention_head_dim": 8,
    "num_attention_heads": None,  # If None, defaults to attention_head_dim
    "resnet_time_scale_shift": "default",

    # Class and additional embeddings
    "class_embed_type": None,
    "addition_embed_type": None,
    "addition_time_embed_dim": None,
    "num_class_embeds": None,

    # Time embedding configuration
    "time_embedding_type": "positional",
    "time_embedding_dim": None,
    "time_embedding_act_fn": None,
    "timestep_post_act": None,
    "time_cond_proj_dim": None,

    # Convolution configuration
    "conv_in_kernel": 3,
    "conv_out_kernel": 3,

    # Additional class embedding configuration
    "projection_class_embeddings_input_dim": None,
    "class_embeddings_concat": False,
    "mid_block_only_cross_attention": None
}


def build_unet(sample_size=None, in_c=None, out_c=None, channels=None, norm_num_groups=None):
    """
    Build UNet with defaults that can be overridden by config
    Uses UNet2DConditionModel for text conditioning support
    """
    config = DEFAULT_UNET_CONFIG.copy()

    if sample_size is not None:
        config["sample_size"] = sample_size
    if in_c is not None:
        config["in_channels"] = in_c
    if out_c is not None:
        config["out_channels"] = out_c
    if channels is not None:
        config["block_out_channels"] = tuple(channels)
    if norm_num_groups is not None:
        config["norm_num_groups"] = norm_num_groups

    return UNet2DConditionModel(
        sample_size=config["sample_size"],
        in_channels=config["in_channels"],
        out_channels=config["out_channels"],
        center_input_sample=config["center_input_sample"],
        flip_sin_to_cos=config["flip_sin_to_cos"],
        freq_shift=config["freq_shift"],
        down_block_types=config["down_block_types"],
        mid_block_type=config["mid_block_type"],
        up_block_types=config["up_block_types"],
        only_cross_attention=config["only_cross_attention"],
        block_out_channels=config["block_out_channels"],
        layers_per_block=config["layers_per_block"],
        downsample_padding=config["downsample_padding"],
        mid_block_scale_factor=config["mid_block_scale_factor"],
        dropout=config["dropout"],
        act_fn=config["act_fn"],
        norm_num_groups=config["norm_num_groups"],
        norm_eps=config["norm_eps"],
        cross_attention_dim=config["cross_attention_dim"],
        transformer_layers_per_block=config["transformer_layers_per_block"],
        reverse_transformer_layers_per_block=config["reverse_transformer_layers_per_block"],
        encoder_hid_dim=config["encoder_hid_dim"],
        encoder_hid_dim_type=config["encoder_hid_dim_type"],
        attention_head_dim=config["attention_head_dim"],
        num_attention_heads=config["num_attention_heads"],
        resnet_time_scale_shift=config["resnet_time_scale_shift"],
        class_embed_type=config["class_embed_type"],
        addition_embed_type=config["addition_embed_type"],
        addition_time_embed_dim=config["addition_time_embed_dim"],
        num_class_embeds=config["num_class_embeds"],
        time_embedding_type=config["time_embedding_type"],
        time_embedding_dim=config["time_embedding_dim"],
        time_embedding_act_fn=config["time_embedding_act_fn"],
        timestep_post_act=config["timestep_post_act"],
        time_cond_proj_dim=config["time_cond_proj_dim"],
        conv_in_kernel=config["conv_in_kernel"],
        conv_out_kernel=config["conv_out_kernel"],
        projection_class_embeddings_input_dim=config["projection_class_embeddings_input_dim"],
        class_embeddings_concat=config["class_embeddings_concat"],
        mid_block_only_cross_attention=config["mid_block_only_cross_attention"]
    )