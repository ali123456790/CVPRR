"""Enhanced caching utilities with thread-safe LRU caches."""

from typing import Any, Dict, Optional, List
import hashlib
import threading
import time
from collections import OrderedDict
import numpy as np

from ..core.types import Region


def image_hash(image: np.ndarray) -> str:
    """Compute hash of image array for caching."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(image)}")
    
    # Use a subset of pixels for faster hashing on large images
    h, w = image.shape[:2]
    if h * w > 100000:  # Large image
        sample = image[::10, ::10]
        return hashlib.md5(sample.tobytes()).hexdigest()[:16]
    else:
        return hashlib.md5(image.tobytes()).hexdigest()[:16]


def crop_hash(crop: np.ndarray, text: str) -> str:
    """Compute hash of crop and text for VLM caching."""
    crop_h = image_hash(crop)
    text_h = hashlib.md5(text.encode()).hexdigest()[:16]
    return f"{crop_h}_{text_h}"


class ProposalCache:
    """Thread-safe LRU cache for object detection proposals."""
    
    def __init__(self, max_size: int = 1000, verbose: bool = False):
        self.max_size = max_size
        self.verbose = verbose
        self._cache: OrderedDict[str, List[Region]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        if self.verbose:
            print(f"✓ ProposalCache initialized (max_size={max_size})")
    
    def get(self, image: np.ndarray) -> Optional[List[Region]]:
        """Get cached proposals for an image."""
        img_hash = image_hash(image)
        
        with self._lock:
            if img_hash in self._cache:
                proposals = self._cache.pop(img_hash)
                self._cache[img_hash] = proposals  # Move to end
                self._hits += 1
                return proposals
            else:
                self._misses += 1
                return None
    
    def put(self, image: np.ndarray, proposals: List[Region]) -> None:
        """Cache proposals for an image."""
        img_hash = image_hash(image)
        
        with self._lock:
            if img_hash in self._cache:
                del self._cache[img_hash]
            
            self._cache[img_hash] = proposals
            
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self.hit_rate(),
            }


class ScoreCache:
    """Thread-safe LRU cache for VLM similarity scores."""
    
    def __init__(self, max_size: int = 5000, verbose: bool = False):
        self.max_size = max_size
        self.verbose = verbose
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
        if self.verbose:
            print(f"✓ ScoreCache initialized (max_size={max_size})")
    
    def get(self, crop: np.ndarray, text: str) -> Optional[float]:
        """Get cached VLM score for a crop-text pair."""
        cache_key = crop_hash(crop, text)
        
        with self._lock:
            if cache_key in self._cache:
                score = self._cache.pop(cache_key)
                self._cache[cache_key] = score  # Move to end
                self._hits += 1
                return score
            else:
                self._misses += 1
                return None
    
    def put(self, crop: np.ndarray, text: str, score: float) -> None:
        """Cache VLM score for a crop-text pair."""
        cache_key = crop_hash(crop, text)
        
        with self._lock:
            if cache_key in self._cache:
                del self._cache[cache_key]
            
            self._cache[cache_key] = score
            
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
    
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": self.hit_rate(),
            }


# Legacy compatibility
class Cache:
    """Legacy cache class for backward compatibility."""
    def __init__(self):
        self._d = {}
    def get(self, k): 
        return self._d.get(k)
    def set(self, k, v): 
        self._d[k] = v
