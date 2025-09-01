from PIL import Image
from dygrav.modules.vlm_query import VLMQuery
from dygrav.core.types import Region

def test_vlm_dummy_prefers_rightmost_region():
    # Create a blank image; VLM dummy ignores content and scores by x-center.
    img = Image.new("RGB", (200, 100), color=(255,255,255))
    regions = [
        Region(10,10,30,30,0.9),   # center x ~ 20
        Region(120,10,160,30,0.9), # center x ~ 140 -> should be top
    ]
    q = VLMQuery()
    scores = q.score(img, regions, "anything")
    best = max(scores, key=lambda r: r.score)
    assert best.region_idx == 1
