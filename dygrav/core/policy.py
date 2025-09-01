from typing import Dict, List
from .types import DygravSignal, Region
from ..modules.ambiguity import AmbiguityDetector
from ..modules.vlm_query import VLMQuery
from ..modules.scene_graph import SceneGraphBuilder
from ..modules.fusion import DygravFusion

class PolicyWithDygrav:
    def __init__(self, backbone, detector, tokenizer, ambiguity_cfg, vlm_cfg, sg_cfg):
        self.backbone = backbone
        self.detector = detector
        self.tokenizer = tokenizer
        self.amb = AmbiguityDetector(**ambiguity_cfg)
        self.vlm = VLMQuery(**vlm_cfg)
        self.sg  = SceneGraphBuilder(**sg_cfg)
        self.fuse = DygravFusion()

    def step(self, obs: Dict) -> Dict:
        """One env step. If ambiguous, invoke DyGRAV modules, fuse, and bias the policy."""
        policy_out = self.backbone.step(obs)
        conf = policy_out.get("confidence", 1.0)
        attn_entropy = policy_out.get("attn_entropy", 0.0)
        phrase = policy_out.get("current_phrase", "")
        attr_present = self.tokenizer.contains_attribute(phrase)

        image = obs.get("rgb")
        regions: List[Region] = self.detector.propose(image, phrase)

        dec = self.amb(conf, attn_entropy, attr_present, regions)
        dygrav_signal: DygravSignal | None = None

        if dec.trigger and dec.candidates:
            cand = [regions[i] for i in dec.candidates]
            vlm_scores = self.vlm.score(image, cand, phrase)
            rels = self.sg.build(cand)
            dygrav_signal = self.fuse.build_signal(cand, vlm_scores, rels)
            policy_out = self.backbone.bias(policy_out, dygrav_signal)

        policy_out["dygrav"] = bool(dygrav_signal)
        policy_out["debug_dygrav"] = dygrav_signal.debug if dygrav_signal else {}
        return policy_out
