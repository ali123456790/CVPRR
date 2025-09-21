from __future__ import annotations
from typing import Any, Dict, Optional, List
import time

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
        self._bev_builder = None  # lazy init

    def _hash_tokenize(self, texts: List[str], vocab_size: int = 50000, max_len: int = 32) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """Simple hashing tokenizer to produce stable IDs and masks for HAMT-like backbones."""
        device = self.device
        ids = torch.zeros(len(texts), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(texts), max_len, dtype=torch.bool, device=device)
        for i, text in enumerate(texts):
            tokens = text.split()
            tokens = tokens[:max_len]
            for j, tok in enumerate(tokens):
                hid = (hash(tok) % (vocab_size - 2)) + 1
                ids[i, j] = hid
                mask[i, j] = True
        return ids, mask

    def training_step(self, batch, batch_idx):
        # Expect EpisodeBatch from real datasets; synthetic returns (obs, labels)
        if isinstance(batch, tuple) and len(batch) == 2:
            # Backward compatibility for synthetic dataset
            obs, labels = batch
            out = self.policy.step({"feat": obs["feat"], "rgb": obs["rgb"][0] if isinstance(obs["rgb"], list) else obs["rgb"]})
            logits = out["logits"]
            loss = self.criterion(logits, labels)
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/dygrav_trigger_rate", float(out.get("dygrav", False)))
            return loss

        episode_batch = batch
        device = self.device
        # Move tensor fields to device
        if getattr(episode_batch, "features", None) is not None and hasattr(episode_batch.features, "to"):
            episode_batch.features = episode_batch.features.to(device)
        if getattr(episode_batch, "bev_features", None) is not None and hasattr(episode_batch.bev_features, "to"):
            episode_batch.bev_features = episode_batch.bev_features.to(device)

        batch_size = episode_batch.batch_size
        max_T = episode_batch.max_length

        # Real hashed tokenization of instructions
        instr_tokens, instr_mask = self._hash_tokenize(episode_batch.instructions, max_len=self.config.model.backbone.get("max_seq_len", 32) if hasattr(self.config.model, "backbone") else 32)

        nav_losses = []
        stop_losses = []
        triggers = 0
        module_counts: Dict[str, int] = {}

        # Iterate over time steps
        for t in range(max_T):
            # Mask out padded steps across batch
            valid_mask = torch.tensor([t < l for l in episode_batch.sequence_lengths], device=device, dtype=torch.bool)
            if not valid_mask.any():
                continue

            # Build panoramic features for valid samples [Bv, 36, D]
            pano_feats: Optional[torch.Tensor] = None
            if getattr(episode_batch, "features", None) is not None:
                pano_feats = episode_batch.features[:, t]  # [B, 36, D]
                pano_feats = pano_feats[valid_mask]

            t_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            t_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if t_start is not None:
                t_start.record()

            if pano_feats is not None and hasattr(self.backbone, "forward"):
                outputs = self.backbone(
                    pano_feats=pano_feats,
                    instr_tokens=instr_tokens[valid_mask],
                    instr_mask=instr_mask[valid_mask],
                    hist=None,
                )
                logits = outputs.get("logits")  # [Bv, A]
                stop_logit = outputs.get("stop")  # [Bv]
                # Wrap to policy for DyGRAV decision and potential bias
                policy_out = {
                    "logits": logits,
                    "confidence": F.softmax(logits, dim=-1).max(dim=-1).values.mean().item() if logits is not None else 1.0,
                    "attn_entropy": 0.0,
                    "current_phrase": episode_batch.instructions[0] if episode_batch.instructions else "",
                }
                biased = self.policy.backbone.bias(policy_out, None)
                logits = biased.get("logits", logits)
            else:
                # Fallback: call policy.step() per valid sample (slower, for datasets without features)
                logits_list = []
                stop_list = []
                for b_idx, is_valid in enumerate(valid_mask.tolist()):
                    if not is_valid:
                        continue
                    obs = {
                        "rgb": episode_batch.images[b_idx][t],
                        "instruction": episode_batch.instructions[b_idx],
                    }
                    # Pass BEV if precomputed
                    if getattr(episode_batch, "bev_features", None) is not None:
                        bev_vec = episode_batch.bev_features[b_idx, t]
                        if torch.is_tensor(bev_vec) and bev_vec.abs().sum().item() > 0:
                            obs["bev"] = bev_vec
                    out = self.policy.step(obs)
                    logits_list.append(out.get("logits"))
                    stop_list.append(torch.tensor(out.get("stop", 0.0), device=device).unsqueeze(0))
                    if out.get("dygrav"):
                        triggers += 1
                        mod = out.get("dygrav_module", "none")
                        module_counts[mod] = module_counts.get(mod, 0) + 1
                logits = torch.stack(logits_list) if logits_list and isinstance(logits_list[0], torch.Tensor) else None
                stop_logit = torch.cat(stop_list) if stop_list else None

            if t_end is not None:
                t_end.record()
                torch.cuda.synchronize()
                ms = t_start.elapsed_time(t_end)
                if not hasattr(self, "_train_times_ms"):
                    self._train_times_ms = []
                self._train_times_ms.append(ms)

            if logits is None:
                continue

            # Build navigation targets from dataset actions if available: next action index
            # Map action_type to indices: forward=0, turn_left=1, turn_right=2, stop=3
            action_map = {"forward": 0, "turn_left": 1, "turn_right": 2, "stop": 3}
            targets = []
            for b_idx, is_valid in enumerate(valid_mask.tolist()):
                if not is_valid:
                    continue
                # If action list length is max_T-1, last timestep has no action, use stop
                if t < len(episode_batch.actions[b_idx]):
                    act = episode_batch.actions[b_idx][t].action_type
                else:
                    act = "stop"
                targets.append(action_map.get(act, 3))
            targets_tensor = torch.tensor(targets, device=device, dtype=torch.long)

            nav_losses.append(self.criterion(logits, targets_tensor))

            # Optional stop loss if stop head provided
            if stop_logit is not None and stop_logit.dim() == 1:
                stop_target = torch.tensor([1 if a == 3 else 0 for a in targets], device=device, dtype=torch.float32)
                stop_losses.append(F.binary_cross_entropy_with_logits(stop_logit, stop_target))

        # Aggregate losses
        if len(nav_losses) == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            nav_loss = torch.stack(nav_losses).mean()
            if stop_losses:
                stop_loss = torch.stack(stop_losses).mean()
                total_loss = nav_loss + self.config.train.get("stop_loss_weight", 0.1) * stop_loss
            else:
                stop_loss = torch.tensor(0.0, device=device)
                total_loss = nav_loss

        # Logging
        self.log("train/loss", total_loss, prog_bar=True)
        if len(nav_losses):
            self.log("train/nav_loss", nav_loss)
        if stop_losses:
            self.log("train/stop_loss", stop_loss)

        total_steps = sum(episode_batch.sequence_lengths)
        trigger_rate = float(triggers) / max(1, total_steps)
        self.log("train/dygrav_trigger_rate", trigger_rate)
        for mod, c in module_counts.items():
            self.log(f"train/module_use/{mod}", float(c) / max(1, total_steps))

        # p50/p95 ms/step for this epoch (GPU timing only)
        if hasattr(self, "_train_times_ms") and len(self._train_times_ms) > 0:
            times = torch.tensor(self._train_times_ms, device=device)
            p50 = torch.quantile(times, 0.5).item()
            p95 = torch.quantile(times, 0.95).item()
            self.log("train/ms_step_p50", p50)
            self.log("train/ms_step_p95", p95)
            # Reset buffer each step to avoid memory growth
            self._train_times_ms = []

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Mirror training loop for metrics on validation splits
        episode_batch = batch
        device = self.device
        if getattr(episode_batch, "features", None) is not None and hasattr(episode_batch.features, "to"):
            episode_batch.features = episode_batch.features.to(device)
        if getattr(episode_batch, "bev_features", None) is not None and hasattr(episode_batch.bev_features, "to"):
            episode_batch.bev_features = episode_batch.bev_features.to(device)

        batch_size = episode_batch.batch_size
        max_T = episode_batch.max_length
        instr_tokens, instr_mask = self._hash_tokenize(episode_batch.instructions, max_len=self.config.model.backbone.get("max_seq_len", 32) if hasattr(self.config.model, "backbone") else 32)

        nav_losses = []
        stop_losses = []
        triggers = 0

        # For SPL/NE bookkeeping
        successes: List[bool] = episode_batch.successes
        path_lens: List[float] = episode_batch.path_lengths
        shortest_paths: List[float] = episode_batch.shortest_paths

        for t in range(max_T):
            valid_mask = torch.tensor([t < l for l in episode_batch.sequence_lengths], device=device, dtype=torch.bool)
            if not valid_mask.any():
                continue
            pano_feats = episode_batch.features[:, t] if getattr(episode_batch, "features", None) is not None else None
            t0 = time.perf_counter()
            if pano_feats is not None and hasattr(self.backbone, "forward"):
                outputs = self.backbone(pano_feats=pano_feats[valid_mask], instr_tokens=instr_tokens[valid_mask], instr_mask=instr_mask[valid_mask], hist=None)
                logits = outputs.get("logits")
                stop_logit = outputs.get("stop")
            else:
                logits_list = []
                stop_list = []
                for b_idx, is_valid in enumerate(valid_mask.tolist()):
                    if not is_valid:
                        continue
                    obs = {"rgb": episode_batch.images[b_idx][t], "instruction": episode_batch.instructions[b_idx]}
                    if getattr(episode_batch, "bev_features", None) is not None:
                        bev_vec = episode_batch.bev_features[b_idx, t]
                        if torch.is_tensor(bev_vec) and bev_vec.abs().sum().item() > 0:
                            obs["bev"] = bev_vec
                    out = self.policy.step(obs)
                    logits_list.append(out.get("logits"))
                    stop_list.append(torch.tensor(out.get("stop", 0.0), device=device).unsqueeze(0))
                    if out.get("dygrav"):
                        triggers += 1
                logits = torch.stack(logits_list) if logits_list and isinstance(logits_list[0], torch.Tensor) else None
                stop_logit = torch.cat(stop_list) if stop_list else None
            t1 = time.perf_counter()
            # CPU timing fallback
            if not hasattr(self, "_val_times_ms"):
                self._val_times_ms = []
            self._val_times_ms.append((t1 - t0) * 1000.0)

            if logits is None:
                continue
            action_map = {"forward": 0, "turn_left": 1, "turn_right": 2, "stop": 3}
            targets = []
            for b_idx, is_valid in enumerate(valid_mask.tolist()):
                if not is_valid:
                    continue
                if t < len(episode_batch.actions[b_idx]):
                    act = episode_batch.actions[b_idx][t].action_type
                else:
                    act = "stop"
                targets.append(action_map.get(act, 3))
            targets_tensor = torch.tensor(targets, device=device, dtype=torch.long)
            nav_losses.append(self.criterion(logits, targets_tensor))
            if stop_logit is not None and stop_logit.dim() == 1:
                stop_target = torch.tensor([1 if a == 3 else 0 for a in targets], device=device, dtype=torch.float32)
                stop_losses.append(F.binary_cross_entropy_with_logits(stop_logit, stop_target))

        if len(nav_losses) == 0:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            nav_loss = torch.stack(nav_losses).mean()
            if stop_losses:
                stop_loss = torch.stack(stop_losses).mean()
                total_loss = nav_loss + self.config.train.get("stop_loss_weight", 0.1) * stop_loss
            else:
                stop_loss = torch.tensor(0.0, device=device)
                total_loss = nav_loss

        # Determine split name
        split_tag = "val"
        try:
            if hasattr(episode_batch, "splits") and len(set(episode_batch.splits)) == 1:
                split_tag = "val_unseen" if episode_batch.splits[0] == "val_unseen" else "val_seen"
        except Exception:
            pass

        # Compute SPL/NE/CLS proxy
        from dygrav.metrics.navigation import spl as _spl, navigation_error as _ne, cls_proxy as _cls
        val_spl = _spl(successes, path_lens, shortest_paths)
        # NE: if not provided per-episode errors, use 1-SR as proxy multiplied by mean path
        sr = sum(1 for s in successes if s) / max(1, len(successes))
        avg_len = sum(path_lens) / max(1, len(path_lens))
        val_ne = _ne([0.0 if s else avg_len for s in successes])
        # CLS proxy: combine SPL and GA proxy (0 here unless evaluator supplies)
        per_episode_spl = []
        for s, l, sp in zip(successes, path_lens, shortest_paths):
            per_episode_spl.append((1.0 if s else 0.0) * (sp / max(l, sp, 1e-6)))
        val_cls = _cls([sr] * len(per_episode_spl), per_episode_spl)

        self.log(f"{split_tag}/loss", total_loss, prog_bar=True)
        self.log(f"{split_tag}/SPL", val_spl, prog_bar=(split_tag == "val_unseen"))
        self.log(f"{split_tag}/NE", val_ne)
        self.log(f"{split_tag}/CLS", val_cls)

        total_steps = sum(episode_batch.sequence_lengths)
        trigger_rate = float(triggers) / max(1, total_steps)
        self.log(f"{split_tag}/dygrav_trigger_rate", trigger_rate)
        # p50/p95 per-epoch (CPU timing)
        if hasattr(self, "_val_times_ms") and len(self._val_times_ms) > 0:
            import numpy as np
            arr = np.array(self._val_times_ms)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            self.log(f"{split_tag}/ms_step_p50", p50)
            self.log(f"{split_tag}/ms_step_p95", p95)
            self._val_times_ms = []
        return {"loss": total_loss, "spl": torch.tensor(val_spl)}

    def on_validation_epoch_end(self):
        # Implement checkpointing by best val_unseen SPL
        if not hasattr(self.trainer, "callback_metrics"):
            return
        best = self.trainer.callback_metrics.get("val_unseen/SPL")
        if best is not None:
            # Lightning ModelCheckpoint can handle; ensure one exists via Trainer config
            # If not, emit a hint
            pass

    def configure_optimizers(self):
        # Collect learnable params: backbone, policy (gate/expert heads if any)
        params = []
        params += list(self.backbone.parameters())
        if hasattr(self.policy, "gate_model"):
            params += list(self.policy.gate_model.parameters())
        # Include any trainable submodules inside policy (experts may have params)
        for name, module in getattr(self.policy, "__dict__", {}).items():
            if isinstance(module, nn.Module) and any(p.requires_grad for p in module.parameters(recurse=True)):
                if module is not self.backbone:
                    params += [p for p in module.parameters() if p.requires_grad]
        # Deduplicate params
        unique_params = list({id(p): p for p in params if p is not None}.values())
        optimizer = torch.optim.AdamW(unique_params, lr=self.config.train.lr)
        return optimizer
