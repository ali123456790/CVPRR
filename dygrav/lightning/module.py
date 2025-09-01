from __future__ import annotations
from typing import Any, Dict
import torch
from torch import nn
import pytorch_lightning as pl

from ..core.policy import PolicyWithDygrav
from ..detectors.yolo import SimpleDetector

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 2)

    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        x = obs["feat"]  # [B, 8]
        logits = self.lin(x)
        probs = torch.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values.mean().item()
        ent = (- (probs * probs.clamp_min(1e-9).log()).sum(dim=-1)).mean().item()
        return {"logits": logits, "confidence": conf, "attn_entropy": ent, "current_phrase": "ceramic bowl"}

    def bias(self, policy_out, dygrav_signal):
        if dygrav_signal and dygrav_signal.chosen_region:
            logits = policy_out["logits"]
            policy_out["logits"] = logits + torch.tensor([0.25, -0.25], device=logits.device)
            policy_out["biased"] = True
        return policy_out

class DummyTokenizer:
    def contains_attribute(self, phrase: str) -> bool:
        return True

class NavLightningModule(pl.LightningModule):
    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = TinyBackbone()
        self.policy = PolicyWithDygrav(
            backbone=self.backbone,
            detector=SimpleDetector(),
            tokenizer=DummyTokenizer(),
            ambiguity_cfg={"tau_conf": 0.55, "tau_entropy": 1.25, "max_candidates": 3},
            vlm_cfg={"model_name": "ViT-L-14", "pretrained": "openai", "device": "cpu"},
            sg_cfg={"next_to_thresh": 0.5},
        )
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        obs, target = batch
        # Handle batched input - process each sample individually
        batch_size = target.shape[0]
        all_logits = []
        dygrav_triggers = 0
        
        for i in range(batch_size):
            # Extract single sample
            sample_obs = {
                "feat": obs["feat"][i],  # [8]
                "rgb": obs["rgb"][i]     # PIL Image
            }
            out = self.policy.step(sample_obs)
            all_logits.append(out["logits"])
            dygrav_triggers += 1 if out["dygrav"] else 0
        
        # Stack all logits
        logits = torch.stack(all_logits)  # [B, 2]
        loss = self.criterion(logits, target)
        
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/dygrav", float(dygrav_triggers / batch_size))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
