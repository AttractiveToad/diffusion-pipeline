from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import random
from PIL import Image

import logconf
import logging


class DynamicSizeTransform:
    """
    Randomly resize images between min_size and max_size and pad to pad_size for batching.
    """
    def __init__(self, min_size, max_size, pad_size):
        self.min_size = min_size
        self.max_size = max_size
        self.pad_size = pad_size
        logging.debug(f'DynamicSizeTransform initialized with min_size={min_size}, max_size={max_size}, pad_size={pad_size}')

    def __call__(self, batch):
        target_size = random.choice(range(self.min_size, self.max_size + 1, 8))
        images = []
        logging.debug(f"Processing batch of {len(batch['image'])} images to target_size={target_size}")
        for idx, img in enumerate(batch["image"]):
            img = img.convert("RGB")
            orig_w, orig_h = img.size
            scale = target_size / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_resized = img.resize((new_w, new_h), resample=Image.NEAREST)
            logging.debug(
                f"[Image {idx}] Original size: {orig_w}x{orig_h}, Resized to: {new_w}x{new_h} (scale: {scale:.3f})"
            )

            # Pad auf zentrales Quadrat
            new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            paste_x = (target_size - new_w) // 2
            paste_y = (target_size - new_h) // 2
            new_img.paste(img_resized, (paste_x, paste_y))
            logging.debug(
                f"[Image {idx}] Image padded to {target_size}x{target_size} at ({paste_x},{paste_y})"
            )

            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            tensor = t(new_img)

            # Optional: nochmal auf pad_size paddieren, falls batch-size ein größeres Pad braucht
            if self.pad_size > target_size:
                pad = (0, self.pad_size - target_size, 0, self.pad_size - target_size)
                tensor = torch.nn.functional.pad(tensor, pad)
                logging.debug(
                    f"[Image {idx}] Further padded to {self.pad_size}x{self.pad_size}"
                )
            images.append(tensor)
        batch_tensor = torch.stack(images)
        logging.debug(f'Batch processing done, shape: {batch_tensor.shape}')
        return {"pixel_values": batch_tensor}

class DataModule:
    def __init__(self, config):
        self.config = config
        self.transform = DynamicSizeTransform(
            config.get("min_image_size", 32),
            config.get("max_image_size", 512),
            config.get("max_image_size", 512)
        )
        logging.debug('DataModule initialized')

    def setup(self):
        logging.info('Lade Dataset...')
        self.dataset = load_dataset(self.config["dataset_name"], self.config["dataset_config"])
        logging.info(f'Dataset geladen: {self.config["dataset_name"]} ({self.config["dataset_config"]})')
        self.dataset = self.dataset.with_transform(self.transform)
        logging.info(f'Dataset-Transformation gesetzt.')

    def train_dataloader(self):
        logging.info('Erzeuge DataLoader...')
        return DataLoader(
            self.dataset["train"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4
        )