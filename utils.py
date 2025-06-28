import torch

def batch_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

def get_latent_from_vae(vae, images):
    with torch.no_grad():
        posterior = vae.encode(images).latent_dist
        latents = posterior.sample()
        return 0.18215 * latents