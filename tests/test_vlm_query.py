from PIL import Image
from dygrav.modules.vlm_query import VLMQuery
from dygrav.core.types import Region

def test_vlm_dummy_prefers_rightmost_region():
    # Create a blank image; VLM dummy ignores content and scores by x-center.
    img = Image.new("RGB", (200, 100), color=(255,255,255))
    regions = [
        Region(xyxy=(10,10,30,30), score=0.9, label="obj1"),   # center x ~ 20
        Region(xyxy=(120,10,160,30), score=0.9, label="obj2"), # center x ~ 140 -> should be top
    ]
    q = VLMQuery()
    scores = q.score(img, regions, "anything")
    best = max(scores, key=lambda r: r.score)
    assert best.region == regions[1]
