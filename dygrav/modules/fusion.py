from typing import List, Optional
from ..core.types import Region, VLMResult, RelationEdge, DygravSignal

class DygravFusion:
    """Fuse VLM ranking + relations to pick a referent and augment policy state."""
    def select_region(self, regions: List[Region], vlm: List[VLMResult], rels: List[RelationEdge],
                      prefer_rel: Optional[str] = None) -> Optional[Region]:
        if not regions or not vlm: 
            return None
        top = max(vlm, key=lambda r: r.score)
        chosen = regions[top.region_idx]
        return chosen

    def build_signal(self, regions, vlm, rels) -> DygravSignal:
        chosen = self.select_region(regions, vlm, rels)
        dbg = {"vlm_top": max([v.score for v in vlm], default=0.0), "n_rel": float(len(rels))}
        return DygravSignal(chosen_region=chosen, vlm_scores=vlm, relations=rels, debug=dbg)
