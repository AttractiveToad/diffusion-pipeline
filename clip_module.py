import torch
from transformers import CLIPTextModel, CLIPTokenizer

DEFAULT_CLIP_CONFIG = {
    "model_name": "runwayml/stable-diffusion-v1-5",  # Hat CLIP text encoder
    "max_length": 77,
    "output_hidden_states": True,
    "return_attention_mask": True,
    "pad_token_id": 49407,
    "truncation": True,
    "padding": "max_length"
}


def build_clip_text_encoder(model_name=None, max_length=None):
    """
    Build CLIP text encoder with defaults that can be overridden by config
    Uses pretrained model to avoid breaking pixel diffusion pipeline
    """
    config = DEFAULT_CLIP_CONFIG.copy()

    if model_name is not None:
        config["model_name"] = model_name
    if max_length is not None:
        config["max_length"] = max_length

    try:
        # Versuche zuerst das direkte Modell zu laden
        if config["model_name"] == "openai/clip-vit-base-patch32":
            # Für OpenAI CLIP verwenden wir from_tf=True
            text_encoder = CLIPTextModel.from_pretrained(
                config["model_name"],
                from_tf=True
            )
            tokenizer = CLIPTokenizer.from_pretrained(config["model_name"])
        else:
            # Für Stable Diffusion Modelle mit CLIP text encoder
            text_encoder = CLIPTextModel.from_pretrained(
                config["model_name"],
                subfolder="text_encoder"
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                config["model_name"],
                subfolder="tokenizer"
            )
    except Exception as e:
        print(f"Fehler beim Laden des CLIP-Modells: {e}")
        print("Verwende Fallback-Modell: runwayml/stable-diffusion-v1-5")
        # Fallback auf Stable Diffusion CLIP
        text_encoder = CLIPTextModel.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="text_encoder"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="tokenizer"
        )
        config["model_name"] = "runwayml/stable-diffusion-v1-5"

    return text_encoder, tokenizer, config


def encode_text_prompt(tokenizer, text_encoder, prompt, device="cuda", max_length=77):
    """
    Encode text prompt to embeddings using CLIP
    """
    # Tokenize text
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    # Move to device
    text_input_ids = text_inputs.input_ids.to(device)

    # Get text embeddings
    with torch.no_grad():
        text_embeddings = text_encoder(text_input_ids)[0]

    return text_embeddings


def get_unconditional_embeddings(tokenizer, text_encoder, batch_size=1, device="cuda", max_length=77):
    """
    Get unconditional (empty) text embeddings for classifier-free guidance
    """
    uncond_tokens = [""] * batch_size
    return encode_text_prompt(tokenizer, text_encoder, uncond_tokens, device, max_length)