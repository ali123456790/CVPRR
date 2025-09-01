from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

try:
    import torch  # type: ignore
    Tensor = torch.Tensor
except Exception:  # pragma: no cover
    Tensor = Any  # lightweight fallback for environments without torch

@dataclass(frozen=True)
class Region:
    """Axis-aligned bounding box in image coords."""
    x1: int; y1: int; x2: int; y2: int
    score: float
    cls: Optional[str] = None  # detector label if any

@dataclass(frozen=True)
class VLMResult:
    region_idx: int
    text: str
    score: float

@dataclass(frozen=True)
class RelationEdge:
    src_idx: int
    dst_idx: int
    rel_type: str       # "left_of", "right_of", "on", "next_to", "between"
    confidence: float

@dataclass
class DygravSignal:
    """Signals injected back into policy."""
    chosen_region: Optional[Region]
    vlm_scores: List[VLMResult]
    relations: List[RelationEdge]
    debug: Dict[str, float]
