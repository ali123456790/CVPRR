from typing import Dict, List, Optional
from .types import DygravSignal, Region
from ..modules.ambiguity import AmbiguityDetector
from ..modules.vlm_query import VLMQuery
from ..modules.scene_graph import SceneGraphBuilder
from ..modules.fusion import DygravFusion
from ..utils.caching import ProposalCache, ScoreCache

class PolicyWithDygrav:
    def __init__(
        self, 
        backbone, 
        detector, 
        tokenizer, 
        ambiguity_cfg, 
        vlm_cfg, 
        sg_cfg,
        cache_cfg: Optional[Dict] = None
    ):
        self.backbone = backbone
        self.detector = detector
        self.tokenizer = tokenizer
        self.amb = AmbiguityDetector(**ambiguity_cfg)
        self.vlm = VLMQuery(**vlm_cfg)
        self.sg  = SceneGraphBuilder(**sg_cfg)
        self.fuse = DygravFusion()
        
        # Initialize caches
        cache_cfg = cache_cfg or {}
        self.proposal_cache = ProposalCache(
            max_size=cache_cfg.get("proposal_cache_size", 1000),
            verbose=cache_cfg.get("verbose", False)
        )
        self.score_cache = ScoreCache(
            max_size=cache_cfg.get("score_cache_size", 5000),
            verbose=cache_cfg.get("verbose", False)
        )

    def step(self, obs: Dict) -> Dict:
        """One env step. If ambiguous, invoke DyGRAV modules, fuse, and bias the policy."""
        # Forward fast path / backbone; pass through BEV if supported
        policy_out = self.backbone.step(obs)
        conf = policy_out.get("confidence", 1.0)
        attn_entropy = policy_out.get("attn_entropy", 0.0)
        phrase = policy_out.get("current_phrase", "")
        attr_present = self.tokenizer.contains_attribute(phrase)

        image = obs.get("rgb")
        
        # Check proposal cache first
        regions: List[Region] = self.proposal_cache.get(image)
        if regions is None:
            # Cache miss - compute proposals and cache result
            regions = self.detector.propose(image, phrase)
            self.proposal_cache.put(image, regions)

        dec = self.amb(conf, attn_entropy, attr_present, regions)
        dygrav_signal: DygravSignal | None = None

        if dec.trigger and dec.candidates:
            cand = [regions[i] for i in dec.candidates]
            
            # Check score cache for VLM queries
            vlm_scores = []
            uncached_regions = []
            
            for region in cand:
                # Extract crop for caching
                x1, y1, x2, y2 = region.xyxy
                crop = image[y1:y2, x1:x2] if image is not None else None
                
                if crop is not None:
                    cached_score = self.score_cache.get(crop, phrase)
                    if cached_score is not None:
                        # Cache hit - use cached score
                        from .types import VLMResult
                        vlm_result = VLMResult(
                            region=region,
                            score=cached_score,
                            phrase=phrase,
                            metadata={"cached": True}
                        )
                        vlm_scores.append(vlm_result)
                    else:
                        # Cache miss - need to compute
                        uncached_regions.append(region)
                else:
                    # No image available - add to uncached
                    uncached_regions.append(region)
            
            # Compute scores for uncached regions
            if uncached_regions:
                new_vlm_scores = self.vlm.score(image, uncached_regions, phrase)
                vlm_scores.extend(new_vlm_scores)
                
                # Cache the new scores
                for vlm_result in new_vlm_scores:
                    region = vlm_result.region
                    x1, y1, x2, y2 = region.xyxy
                    crop = image[y1:y2, x1:x2] if image is not None else None
                    if crop is not None:
                        self.score_cache.put(crop, phrase, vlm_result.score)
            
            # Build scene graph and fuse signal
            rels = self.sg.build(cand)
            dygrav_signal = self.fuse.build_signal(cand, vlm_scores, rels)
            policy_out = self.backbone.bias(policy_out, dygrav_signal)

        policy_out["dygrav"] = bool(dygrav_signal)
        policy_out["debug_dygrav"] = dygrav_signal.debug if dygrav_signal else {}

        # BEV telemetry passthrough
        if "bev" in obs:
            policy_out["bev_used"] = True
        else:
            policy_out["bev_used"] = False
        
        # Add cache statistics to debug info
        if dygrav_signal and hasattr(dygrav_signal, 'debug'):
            dygrav_signal.debug.update({
                "proposal_cache_hit_rate": self.proposal_cache.hit_rate(),
                "score_cache_hit_rate": self.score_cache.hit_rate(),
            })
        
        return policy_out
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics for monitoring."""
        return {
            "proposal_cache": self.proposal_cache.stats(),
            "score_cache": self.score_cache.stats(),
        }
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.proposal_cache.clear()
        self.score_cache.clear()
