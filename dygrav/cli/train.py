import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl

from dygrav.data.datamodule import VLNDataModule
from dygrav.lightning.training_module import TrainingLightningModule


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("Starting training script...")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.train.seed)

    # Instantiate DataModule
    print("Instantiating DataModule...")
    datamodule = VLNDataModule(**cfg.data)

    # Instantiate LightningModule
    print("Instantiating LightningModule...")
    model = TrainingLightningModule(cfg)

    # Instantiate Trainer
    print("Instantiating Trainer...")
    trainer = pl.Trainer(**cfg.train.trainer)

    # Start training
    print("Starting training...")
    trainer.fit(model, datamodule)
    print("Training finished.")


if __name__ == "__main__":
    main()
