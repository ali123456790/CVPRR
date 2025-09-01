from typing import List
import math
from ..core.types import Region, RelationEdge

class SceneGraphBuilder:
    """Build micro scene graph among visible regions using simple geometric predicates."""
    def __init__(self, next_to_thresh: float = 0.15):
        self.next_to_thresh = next_to_thresh

    def _center(self, r: Region):
        return ((r.x1 + r.x2) / 2.0, (r.y1 + r.y2) / 2.0)

    def build(self, regions: List[Region]) -> List[RelationEdge]:
        edges: List[RelationEdge] = []
        for i, a in enumerate(regions):
            ax, ay = self._center(a)
            for j, b in enumerate(regions):
                if i == j: 
                    continue
                bx, by = self._center(b)
                dx, dy = bx - ax, by - ay
                rel = "left_of" if bx < ax else "right_of"
                conf = min(1.0, abs(dx) / max(1.0, abs(dy) + 1e-6))
                edges.append(RelationEdge(i, j, rel, conf))
                diag = math.hypot(a.x2 - a.x1, a.y2 - a.y1)
                dist = math.hypot(dx, dy) / max(1.0, diag)
                if dist < self.next_to_thresh:
                    edges.append(RelationEdge(i, j, "next_to", 1.0 - dist / self.next_to_thresh))
        return edges
