import pytorch_lightning as pl
from ..lightning.module import NavLightningModule
from ..data.datamodule import SyntheticDataModule

def main():
    dm = SyntheticDataModule(batch_size=32)
    model = NavLightningModule(lr=1e-3)
    trainer = pl.Trainer(max_epochs=2, enable_checkpointing=False, logger=False)
    trainer.fit(model, train_dataloaders=dm.train_dataloader())

if __name__ == "__main__":
    main()
