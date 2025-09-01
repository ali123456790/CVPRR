from typing import List
from ..core.types import Region

class SimpleDetector:
    """Stub detector that proposes a couple of boxes regardless of image."""
    def propose(self, image, phrase: str) -> List[Region]:
        return [
            Region(10, 10, 60, 60, score=0.8, cls="objA"),
            Region(80, 15, 130, 65, score=0.75, cls="objB"),
            Region(140, 20, 180, 60, score=0.72, cls="objC"),
        ]
