from dataclasses import dataclass
from typing import Optional, List
from ..core.types import Region

@dataclass
class AmbiguityDecision:
    trigger: bool
    reason: str
    candidates: List[int]  # indices of regions to disambiguate

class AmbiguityDetector:
    """
    Decide when to activate DyGRAV.
    Heuristics: low policy confidence; multiple region matches to attribute tokens;
    attention entropy high; near-tie top-k object candidates.
    """
    def __init__(self, tau_conf: float = 0.55, tau_entropy: float = 1.25, max_candidates: int = 4):
        self.tau_conf = tau_conf
        self.tau_entropy = tau_entropy
        self.max_candidates = max_candidates

    def __call__(self,
                 policy_confidence: float,
                 attention_entropy: float,
                 textual_attribute_present: bool,
                 candidate_regions: List[Region]) -> AmbiguityDecision:
        reasons = []
        if policy_confidence < self.tau_conf:
            reasons.append("low_policy_conf")
        if attention_entropy > self.tau_entropy:
            reasons.append("high_attn_entropy")
        if textual_attribute_present and len(candidate_regions) >= 2:
            reasons.append("multi_attr_candidates")

        trigger = len(reasons) > 0
        cand_idx = list(range(min(len(candidate_regions), self.max_candidates))) if trigger else []
        return AmbiguityDecision(trigger=trigger, reason="+".join(reasons), candidates=cand_idx)
