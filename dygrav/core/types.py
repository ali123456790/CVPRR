from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

try:
    import torch  # type: ignore
    Tensor = torch.Tensor
except Exception:  # pragma: no cover
    Tensor = Any  # lightweight fallback for environments without torch

@dataclass(frozen=True)
class Region:
    """Axis-aligned bounding box in image coords."""
    xyxy: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    score: float
    label: str  # detector label
    
    # Backward compatibility properties
    @property
    def x1(self) -> int:
        return self.xyxy[0]
    
    @property
    def y1(self) -> int:
        return self.xyxy[1]
    
    @property
    def x2(self) -> int:
        return self.xyxy[2]
    
    @property
    def y2(self) -> int:
        return self.xyxy[3]
    
    @property
    def cls(self) -> str:
        return self.label

@dataclass(frozen=True)
class VLMResult:
    region: Region
    score: float
    phrase: str
    metadata: Dict[str, Any] = field(default_factory=dict)

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
