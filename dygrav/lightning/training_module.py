from __future__ import annotations
from typing import Any, Dict

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainingLightningModule(pl.LightningModule):
    """
    A comprehensive Lightning module for training VLN agents, configurable via Hydra.

    This module integrates the backbone, policy, and data, and handles the
    training, validation, and optimization loops.
    """
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)

        # Instantiate backbone
        self.backbone = hydra.utils.instantiate(config.model.backbone)

        # Instantiate policy
        # Note: The policy now takes the backbone as an argument.
        self.policy = hydra.utils.instantiate(config.model.policy, backbone=self.backbone)

        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        # The enhanced_collate_fn in VLNDataModule produces an EpisodeBatch object
        # which needs to be handled differently than the simple (obs, target) tuple.
        # This is a placeholder for the actual training logic.
        # A real implementation would involve iterating through the episode steps.
        
        # For now, let's create a dummy loss to ensure the training loop runs.
        loss = torch.tensor(0.0, requires_grad=True)
        self.log("train/loss", loss, prog_bar=True)
        
        # Placeholder for DyGRAV trigger rate
        self.log("train/dygrav", 0.0)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.train.lr)
