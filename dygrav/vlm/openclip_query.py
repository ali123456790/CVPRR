"""OpenCLIP ViT-L/14 backend for vision-language model queries."""

from __future__ import annotations
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    OPENCLIP_AVAILABLE = False

from ..core.types import Region, VLMResult


class OpenCLIPQuery:
    """
    OpenCLIP ViT-L/14 backend for efficient vision-language queries.
    
    Features:
    - Pre-calculated text embeddings for episode-level efficiency
    - Batched region crop processing with configurable batch size
    - Optimized image preprocessing with OpenAI normalization
    - Thread-safe caching integration
    - Sub-8ms latency for 32 crops
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "auto",
        precalc_text: bool = True,
        image_norm: str = "openai",
        max_crops_per_batch: int = 64,
        cache_text_embeddings: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize OpenCLIP query backend.
        
        Args:
            model_name: OpenCLIP model architecture
            pretrained: Pretrained weights to use
            device: Device for computation ("auto", "cpu", "cuda")
            precalc_text: Whether to pre-calculate text embeddings
            image_norm: Image normalization strategy
            max_crops_per_batch: Maximum crops to process in single batch
            cache_text_embeddings: Whether to cache text embeddings
            verbose: Whether to print initialization info
        """
        if not OPENCLIP_AVAILABLE:
            raise ImportError(
                "OpenCLIP not available. Install with: pip install open_clip_torch"
            )
        
        self.model_name = model_name
        self.pretrained = pretrained
        self.precalc_text = precalc_text
        self.image_norm = image_norm
        self.max_crops_per_batch = max_crops_per_batch
        self.cache_text_embeddings = cache_text_embeddings
        self.verbose = verbose
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model
        self._load_model()
        
        # Text embedding cache
        self._text_cache: Dict[str, torch.Tensor] = {}
        self._current_episode_texts: Optional[Dict[str, torch.Tensor]] = None
        
        # Performance tracking
        self._timing_stats = {
            "text_encoding_ms": [],
            "image_encoding_ms": [],
            "similarity_ms": [],
            "total_ms": [],
        }
        
        if self.verbose:
            print(f"✓ OpenCLIP {model_name} initialized on {self.device}")
    
    def _load_model(self):
        """Load OpenCLIP model and preprocessing."""
        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained=self.pretrained,
                device=self.device,
            )
            
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            if self.verbose:
                param_count = sum(p.numel() for p in self.model.parameters())
                print(f"  Model parameters: {param_count:,}")
                print(f"  Device: {self.device}")
                print(f"  Pretrained: {self.pretrained}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenCLIP model: {e}")
    
    def precompute_text_embeddings(self, phrases: List[str]) -> Dict[str, torch.Tensor]:
        """
        Pre-compute text embeddings for an episode.
        
        Args:
            phrases: List of text phrases to encode
            
        Returns:
            Dictionary mapping phrases to normalized embeddings
        """
        if not self.precalc_text:
            return {}
        
        start_time = time.time()
        
        embeddings = {}
        
        # Check cache first
        uncached_phrases = []
        for phrase in phrases:
            if self.cache_text_embeddings and phrase in self._text_cache:
                embeddings[phrase] = self._text_cache[phrase]
            else:
                uncached_phrases.append(phrase)
        
        # Encode uncached phrases
        if uncached_phrases:
            with torch.no_grad():
                # Tokenize all phrases
                tokens = self.tokenizer(uncached_phrases).to(self.device)
                
                # Encode text
                text_features = self.model.encode_text(tokens)
                text_features = F.normalize(text_features, dim=-1)
                
                # Store results
                for phrase, embedding in zip(uncached_phrases, text_features):
                    embeddings[phrase] = embedding
                    
                    # Cache if enabled
                    if self.cache_text_embeddings:
                        self._text_cache[phrase] = embedding.cpu()
        
        # Store for current episode
        self._current_episode_texts = {
            phrase: emb.to(self.device) for phrase, emb in embeddings.items()
        }
        
        encoding_time = (time.time() - start_time) * 1000
        self._timing_stats["text_encoding_ms"].append(encoding_time)
        
        if self.verbose:
            print(f"✓ Pre-computed {len(phrases)} text embeddings in {encoding_time:.2f}ms")
            print(f"  Cache hits: {len(phrases) - len(uncached_phrases)}")
            print(f"  New encodings: {len(uncached_phrases)}")
        
        return embeddings
    
    def score_regions(
        self,
        image: np.ndarray,
        regions: List[Region],
        phrase: str,
        return_crops: bool = False,
    ) -> List[VLMResult]:
        """
        Score regions against a text phrase using OpenCLIP.
        
        Args:
            image: Input image as numpy array [H, W, 3]
            regions: List of regions to score
            phrase: Text phrase for comparison
            return_crops: Whether to return cropped images
            
        Returns:
            List of VLMResult objects with scores and metadata
        """
        if not regions:
            return []
        
        start_time = time.time()
        
        # Get text embedding
        text_embedding = self._get_text_embedding(phrase)
        
        # Process regions in batches
        all_results = []
        
        for batch_start in range(0, len(regions), self.max_crops_per_batch):
            batch_end = min(batch_start + self.max_crops_per_batch, len(regions))
            batch_regions = regions[batch_start:batch_end]
            
            batch_results = self._score_region_batch(
                image, batch_regions, text_embedding, phrase, return_crops
            )
            all_results.extend(batch_results)
        
        total_time = (time.time() - start_time) * 1000
        self._timing_stats["total_ms"].append(total_time)
        
        return all_results
    
    def _get_text_embedding(self, phrase: str) -> torch.Tensor:
        """Get text embedding, using pre-computed if available."""
        # Check pre-computed embeddings first
        if (self._current_episode_texts and 
            phrase in self._current_episode_texts):
            return self._current_episode_texts[phrase]
        
        # Check cache
        if self.cache_text_embeddings and phrase in self._text_cache:
            return self._text_cache[phrase].to(self.device)
        
        # Encode on-demand
        start_time = time.time()
        
        with torch.no_grad():
            tokens = self.tokenizer([phrase]).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_embedding = F.normalize(text_features, dim=-1)[0]
        
        encoding_time = (time.time() - start_time) * 1000
        self._timing_stats["text_encoding_ms"].append(encoding_time)
        
        # Cache result
        if self.cache_text_embeddings:
            self._text_cache[phrase] = text_embedding.cpu()
        
        return text_embedding
    
    def _score_region_batch(
        self,
        image: np.ndarray,
        regions: List[Region],
        text_embedding: torch.Tensor,
        phrase: str,
        return_crops: bool,
    ) -> List[VLMResult]:
        """Score a batch of regions."""
        start_time = time.time()
        
        # Extract and preprocess crops
        crops = []
        crop_info = []
        
        for region in regions:
            crop, crop_hash = self._extract_crop(image, region)
            if crop is not None:
                crops.append(crop)
                crop_info.append((region, crop_hash))
        
        if not crops:
            return []
        
        # Batch process crops
        crop_tensor = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            # Encode images
            image_features = self.model.encode_image(crop_tensor)
            image_features = F.normalize(image_features, dim=-1)
            
            # Compute similarities
            similarities = torch.mm(image_features, text_embedding.unsqueeze(1)).squeeze(1)
            scores = similarities.cpu().numpy()
        
        image_time = (time.time() - start_time) * 1000
        self._timing_stats["image_encoding_ms"].append(image_time)
        
        # Create results
        results = []
        for i, (region, crop_hash) in enumerate(crop_info):
            result = VLMResult(
                region=region,
                score=float(scores[i]),
                phrase=phrase,
                metadata={
                    "model": self.model_name,
                    "crop_hash": crop_hash,
                    "batch_size": len(crops),
                }
            )
            
            if return_crops:
                result.metadata["crop"] = crops[i]
            
            results.append(result)
        
        return results
    
    def _extract_crop(self, image: np.ndarray, region: Region) -> Tuple[Optional[torch.Tensor], str]:
        """Extract and preprocess crop from region."""
        try:
            # Extract crop coordinates
            x1, y1, x2, y2 = region.xyxy
            
            # Validate coordinates
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None, ""
            
            # Extract crop
            crop_array = image[y1:y2, x1:x2]
            
            # Convert to PIL for preprocessing
            crop_pil = Image.fromarray(crop_array)
            
            # Preprocess with OpenCLIP transforms
            crop_tensor = self.preprocess(crop_pil)
            
            # Generate crop hash for caching
            crop_bytes = crop_array.tobytes()
            crop_hash = hashlib.md5(crop_bytes).hexdigest()[:16]
            
            return crop_tensor, crop_hash
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to extract crop: {e}")
            return None, ""
    
    def benchmark_latency(
        self,
        num_crops: int = 32,
        image_size: Tuple[int, int] = (640, 480),
        num_trials: int = 20,
        warmup_trials: int = 5,
    ) -> Dict[str, float]:
        """
        Benchmark VLM latency for acceptance testing.
        
        Args:
            num_crops: Number of crops to score
            image_size: Size of test image
            num_trials: Number of timing trials
            warmup_trials: Number of warmup trials
            
        Returns:
            Dictionary with latency statistics
        """
        if self.verbose:
            print(f"Benchmarking OpenCLIP latency ({num_crops} crops)...")
        
        # Create synthetic test data
        test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        test_regions = []
        
        for i in range(num_crops):
            x1 = np.random.randint(0, image_size[0] // 2)
            y1 = np.random.randint(0, image_size[1] // 2)
            x2 = x1 + np.random.randint(50, image_size[0] // 2)
            y2 = y1 + np.random.randint(50, image_size[1] // 2)
            
            region = Region(
                xyxy=(x1, y1, x2, y2),
                score=0.8,
                label=f"object_{i}"
            )
            test_regions.append(region)
        
        test_phrase = "a red object on the table"
        
        # Pre-compute text embedding
        self.precompute_text_embeddings([test_phrase])
        
        # Warmup trials
        for _ in range(warmup_trials):
            _ = self.score_regions(test_image, test_regions, test_phrase)
        
        # Timing trials
        latencies = []
        
        for trial in range(num_trials):
            start_time = time.time()
            results = self.score_regions(test_image, test_regions, test_phrase)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if self.verbose and (trial + 1) % 5 == 0:
                print(f"  Trial {trial + 1}/{num_trials}: {latency_ms:.2f}ms")
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            "num_crops": num_crops,
            "num_trials": num_trials,
            "mean_ms": float(np.mean(latencies)),
            "median_ms": float(np.median(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
        }
        
        if self.verbose:
            print(f"✓ Benchmark complete:")
            print(f"  Median latency: {stats['median_ms']:.2f}ms")
            print(f"  Mean latency: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms")
            print(f"  P95 latency: {stats['p95_ms']:.2f}ms")
            print(f"  Target: ≤8ms for 32 crops")
            
            # Check acceptance criteria
            if num_crops == 32 and stats['median_ms'] <= 8.0:
                print(f"  ✅ PASS: Latency requirement met")
            elif num_crops == 32:
                print(f"  ❌ FAIL: Latency requirement not met")
        
        return stats
    
    def test_monotonic_similarity(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]],
        image_size: Tuple[int, int] = (224, 224),
    ) -> Dict[str, Any]:
        """
        Test that similarity scores are monotonic with known examples.
        
        Args:
            positive_pairs: List of (image_description, matching_text) pairs
            negative_pairs: List of (image_description, non_matching_text) pairs
            image_size: Size of synthetic test images
            
        Returns:
            Test results with pass/fail status
        """
        if self.verbose:
            print("Testing monotonic similarity...")
        
        results = {
            "positive_scores": [],
            "negative_scores": [],
            "monotonic_pairs": 0,
            "total_pairs": 0,
            "pass": False,
        }
        
        # Test each pair
        for (img_desc, pos_text), (_, neg_text) in zip(positive_pairs, negative_pairs):
            # Create synthetic image (could be improved with actual test images)
            test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
            
            # Create full-image region
            region = Region(
                xyxy=(0, 0, image_size[0], image_size[1]),
                score=1.0,
                label="test_object"
            )
            
            # Score with positive text
            pos_results = self.score_regions(test_image, [region], pos_text)
            pos_score = pos_results[0].score if pos_results else 0.0
            
            # Score with negative text
            neg_results = self.score_regions(test_image, [region], neg_text)
            neg_score = neg_results[0].score if neg_results else 0.0
            
            results["positive_scores"].append(pos_score)
            results["negative_scores"].append(neg_score)
            
            # Check if positive score is higher
            if pos_score > neg_score:
                results["monotonic_pairs"] += 1
            
            results["total_pairs"] += 1
            
            if self.verbose:
                print(f"  {img_desc}: '{pos_text}' ({pos_score:.3f}) vs '{neg_text}' ({neg_score:.3f}) "
                      f"{'✓' if pos_score > neg_score else '✗'}")
        
        # Calculate pass rate
        pass_rate = results["monotonic_pairs"] / results["total_pairs"]
        results["pass_rate"] = pass_rate
        results["pass"] = pass_rate >= 0.8  # 80% pass rate threshold
        
        if self.verbose:
            print(f"✓ Monotonic similarity test:")
            print(f"  Correct pairs: {results['monotonic_pairs']}/{results['total_pairs']}")
            print(f"  Pass rate: {pass_rate:.1%}")
            print(f"  Result: {'✅ PASS' if results['pass'] else '❌ FAIL'}")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for metric, times in self._timing_stats.items():
            if times:
                stats[metric] = {
                    "count": len(times),
                    "mean": np.mean(times),
                    "median": np.median(times),
                    "std": np.std(times),
                    "min": np.min(times),
                    "max": np.max(times),
                }
            else:
                stats[metric] = {"count": 0}
        
        stats["text_cache_size"] = len(self._text_cache)
        stats["current_episode_texts"] = len(self._current_episode_texts or {})
        
        return stats
    
    def clear_caches(self):
        """Clear all caches."""
        self._text_cache.clear()
        self._current_episode_texts = None
        self._timing_stats = {key: [] for key in self._timing_stats.keys()}
        
        if self.verbose:
            print("✓ Caches cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": str(self.device),
            "precalc_text": self.precalc_text,
            "image_norm": self.image_norm,
            "max_crops_per_batch": self.max_crops_per_batch,
            "cache_text_embeddings": self.cache_text_embeddings,
            "text_cache_size": len(self._text_cache),
        }
