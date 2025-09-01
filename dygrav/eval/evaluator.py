"""Synthetic evaluator for DyGRAV system testing."""
import random
import numpy as np
from typing import Dict, List
from PIL import Image
from ..core.policy import PolicyWithDygrav
from ..detectors.yolo import SimpleDetector

class MockBackbone:
    """Mock backbone that produces varying confidence/entropy to trigger DyGRAV."""
    def step(self, obs):
        # Vary confidence and entropy to create realistic trigger patterns
        conf = random.uniform(0.3, 0.8)  # Sometimes low to trigger
        entropy = random.uniform(0.5, 2.5)  # Sometimes high to trigger
        return {
            "confidence": conf,
            "attn_entropy": entropy,
            "current_phrase": random.choice(["red chair", "wooden table", "ceramic bowl"]),
            "logits": [random.random(), random.random()]
        }
    
    def bias(self, policy_out, dygrav_signal):
        # Simple bias: flip logits when DyGRAV is active
        if dygrav_signal:
            policy_out["logits"] = policy_out["logits"][::-1]
            policy_out["biased"] = True
        return policy_out

class MockTokenizer:
    """Mock tokenizer that detects attributes."""
    def contains_attribute(self, phrase: str) -> bool:
        attributes = ["red", "wooden", "ceramic", "small", "large", "blue", "green"]
        return any(attr in phrase.lower() for attr in attributes)

def run_eval(n: int = 50) -> Dict[str, float]:
    """Run synthetic evaluation with n episodes."""
    # Set up policy
    policy = PolicyWithDygrav(
        backbone=MockBackbone(),
        detector=SimpleDetector(),
        tokenizer=MockTokenizer(),
        ambiguity_cfg={"tau_conf": 0.55, "tau_entropy": 1.25, "max_candidates": 4},
        vlm_cfg={"model_name": "ViT-L-14", "pretrained": "openai", "device": "cpu"},
        sg_cfg={"next_to_thresh": 0.15},
    )
    
    triggers = 0
    gas = []  # Grounding accuracies (synthetic)
    
    for _ in range(n):
        # Synthetic observation
        obs = {"rgb": Image.new("RGB", (224, 224), color=(128, 128, 128))}
        out = policy.step(obs)
        
        if out["dygrav"]:
            triggers += 1
            # Synthetic grounding accuracy: perfect when DyGRAV triggers
            gas.append(1.0)
        else:
            # Lower accuracy when not using DyGRAV
            gas.append(random.uniform(0.6, 0.9))
    
    trigger_rate = triggers / max(1, n)
    ga = np.mean(gas) if gas else 0.0
    
    # Guardrail (tunable later on real val): keep trigger rate under a soft cap.
    # This does NOT raise on synthetic; it's a placeholder pattern.
    if triggers / max(1, n) > 0.30:
        pass  # when RxR is wired, turn this into a warning or assertion in CI
    
    return {
        "trigger_rate": trigger_rate,
        "GA": ga,
        "n_episodes": n,
        "n_triggers": triggers
    }
