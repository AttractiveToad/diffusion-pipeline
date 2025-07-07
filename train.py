import yaml
import logging
import logconf
from dataset import DataModule
from trainer.trainer import Trainer


def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup data
    dataset_config = config.get("dataset", {})
    image_config = config.get("image", {})

    # Create DataModule with flattened config for backward compatibility
    datamodule_config = {
        "dataset_name": dataset_config.get("name"),
        "dataset_config": dataset_config.get("config"),
        "batch_size": dataset_config.get("batch_size"),
        "min_image_size": image_config.get("min_size"),
        "max_image_size": image_config.get("max_size")
    }

    datamodule = DataModule(datamodule_config)
    datamodule.setup()

    # Create trainer
    trainer = Trainer(config, datamodule)

    # Train
    logging.info("Starting training with structured config")
    trainer.train()
    logging.info("Training completed")


if __name__ == "__main__":
    main()