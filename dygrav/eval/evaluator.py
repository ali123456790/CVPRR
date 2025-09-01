from __future__ import annotations
from typing import List, Tuple
from PIL import Image
from ..core.policy import PolicyWithDygrav
from ..detectors.yolo import SimpleDetector
from ..modules.ambiguity import AmbiguityDetector
from ..modules.scene_graph import SceneGraphBuilder
from ..modules.vlm_query import VLMQuery
from ..metrics.grounding import grounding_accuracy

class DummyBackbone:
    def step(self, obs):
        return {"logits": None, "confidence": 0.45, "attn_entropy": 1.8, "current_phrase": "ceramic bowl"}
    def bias(self, out, sig): return out

class DummyTokenizer:
    def contains_attribute(self, phrase: str) -> bool: return True

def run_eval(n: int = 20) -> dict:
    policy = PolicyWithDygrav(
        backbone=DummyBackbone(),
        detector=SimpleDetector(),
        tokenizer=DummyTokenizer(),
        ambiguity_cfg={"tau_conf":0.55,"tau_entropy":1.25,"max_candidates":3},
        vlm_cfg={"model_name":"ViT-L-14","pretrained":"openai","device":"cpu"},
        sg_cfg={"next_to_thresh":0.5},
    )
    img = Image.new("RGB", (200, 100), color=(255,255,255))
    chosen: List[Tuple[int,int,int,int]] = []
    refer:  List[Tuple[int,int,int,int]] = []
    triggers = 0
    for _ in range(n):
        out = policy.step({"rgb": img, "feat": None})
        triggers += 1 if out["dygrav"] else 0
        if out["dygrav"] and out["debug_dygrav"]:
            chosen.append((80,15,130,65))
            refer.append((80,15,130,65))
        else:
            chosen.append((10,10,60,60))
            refer.append((80,15,130,65))
    ga = grounding_accuracy(chosen, refer, tau=0.5)
    return {"trigger_rate": triggers / max(1,n), "GA": ga}