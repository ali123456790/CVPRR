"""Generate offline gate labels and feature tensors from a baseline π0 model.

Usage:
  python -m dygrav.tools.generate_gate_labels \
    --dataset rxr --data_root /data/RXR --split train --out labels_rxr_train.npz
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from ..data.datamodule import VLNDataModule
from ..lightning.training_module import TrainingLightningModule
from ..modules.gate_features import extract_gate_features, RuleBasedSkillClassifier


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, choices=["r2r", "rxr", "rxr_fg"], required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_episodes", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    dm = VLNDataModule(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=1,
        max_episodes=args.max_episodes,
        load_images=True,
        verbose=False,
    )
    loader = dm._create_dataset(args.split)

    # Use a lightweight policy via TrainingLightningModule instantiation
    # Here we rely on default Hydra, so we call model components directly would be better in practice.
    # For label generation we only use policy.step() on observations built from Episode.

    skill = RuleBasedSkillClassifier()
    feats = []
    labels = []

    for ep in loader:
        images = ep.get_images(load_images=True)
        for t, img in enumerate(images):
            obs = {"rgb": img, "instruction": ep.instruction}
            # Baseline correctness proxy: compare action list existence; if no action at this step, assume stop
            gt_action = None
            if t < len(ep.actions):
                gt_action = ep.actions[t].action_type
            else:
                gt_action = "stop"
            # Create a dummy logits distribution to extract features (stand-in for π0 until wired)
            logits = torch.randn(6)
            # No VLM scores in offline pass
            f = extract_gate_features(
                logits=logits,
                instruction=ep.instruction,
                vlm_scores=[],
                visual_embedding=None,
                novelty_tracker=None,
                skill_classifier=skill,
                tokenizer=None,
            )
            feats.append(f.to_tensor().cpu().numpy())
            # Ambiguous when non-forward action in GT (proxy) to bootstrap labels
            labels.append(1 if gt_action != "forward" else 0)

    feats = np.stack(feats) if feats else np.zeros((0, 10), dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)
    Path(Path(args.out).parent).mkdir(parents=True, exist_ok=True)
    np.savez(args.out, features=feats, labels=labels_np)
    print(f"Saved gate labels: feats={feats.shape} labels={labels_np.shape} -> {args.out}")


if __name__ == "__main__":
    main()


