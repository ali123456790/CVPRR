import numpy as np
from PIL import Image
from dygrav.core.policy import PolicyWithDygrav
from dygrav.detectors.yolo import SimpleDetector

class MockBackbone:
    def step(self, obs):
        return {
            "confidence": 0.4,            # low to trigger
            "attn_entropy": 2.0,          # high to trigger
            "current_phrase": "ceramic bowl",
            "logits": [0.1, 0.9]
        }
    def bias(self, policy_out, dygrav_signal):
        # flip logits as a visible effect
        policy_out["logits"] = [0.9, 0.1]
        policy_out["biased"] = True
        return policy_out

class MockTokenizer:
    def contains_attribute(self, phrase: str) -> bool:
        return True

def test_policy_triggers_dygrav_and_biases_output():
    policy = PolicyWithDygrav(
        backbone=MockBackbone(),
        detector=SimpleDetector(),
        tokenizer=MockTokenizer(),
        ambiguity_cfg={"tau_conf":0.55, "tau_entropy":1.25, "max_candidates":3},
        vlm_cfg={"model_name":"ViT-L-14", "pretrained":"openai", "device":"cpu"},
        sg_cfg={"next_to_thresh":0.5},
    )
    obs = {"rgb": np.array(Image.new("RGB", (200, 100), color=(255,255,255)))}
    out = policy.step(obs)
    assert out["dygrav"] is True
    assert out.get("biased", False) is True
    assert "vlm_top" in out["debug_dygrav"]
