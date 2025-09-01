from typing import List, Tuple

def iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = max(1e-6, area_a + area_b - inter)
    return inter / union

def grounding_accuracy(chosen_boxes: List[Tuple[int,int,int,int]],
                       referent_boxes: List[Tuple[int,int,int,int]],
                       tau: float = 0.5) -> float:
    hits = 0
    for c, r in zip(chosen_boxes, referent_boxes):
        hits += 1 if iou_xyxy(c, r) >= tau else 0
    return hits / max(1, len(chosen_boxes))
