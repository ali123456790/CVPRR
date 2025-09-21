#!/usr/bin/env python
"""
VLM Benchmarking Tool

This script benchmarks the OpenCLIP ViT-L/14 backend for latency and accuracy,
ensuring it meets the ≤8ms requirement for scoring 32 crops and monotonic similarity.

Usage:
    python tools/bench_vlm.py --crops 32 --trials 20
    python tools/bench_vlm.py --test-similarity --assert-latency
"""

from __future__ import annotations
import argparse
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dygrav.vlm.openclip_query import OpenCLIPQuery
    from dygrav.core.types import Region
    OPENCLIP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenCLIP not available: {e}")
    OPENCLIP_AVAILABLE = False


class VLMBenchmark:
    """Comprehensive VLM benchmarking with latency and accuracy testing."""
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai", 
        device: str = "auto",
        max_crops_per_batch: int = 64,
        verbose: bool = True,
    ):
        """
        Initialize VLM benchmark.
        
        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained weights
            device: Computation device
            max_crops_per_batch: Batch size for crop processing
            verbose: Whether to print verbose output
        """
        if not OPENCLIP_AVAILABLE:
            raise ImportError("OpenCLIP required for VLM benchmarking")
        
        self.verbose = verbose
        
        # Initialize OpenCLIP query
        self.vlm = OpenCLIPQuery(
            model_name=model_name,
            pretrained=pretrained,
            device=device,
            max_crops_per_batch=max_crops_per_batch,
            precalc_text=True,
            cache_text_embeddings=True,
            verbose=verbose,
        )
        
        if self.verbose:
            print(f"✓ VLM benchmark initialized")
            print(f"  Model: {model_name}")
            print(f"  Device: {self.vlm.device}")
            print(f"  Batch size: {max_crops_per_batch}")
    
    def benchmark_latency(
        self,
        num_crops: int = 32,
        image_size: Tuple[int, int] = (640, 480),
        num_trials: int = 20,
        warmup_trials: int = 5,
        assert_target: bool = False,
        target_latency_ms: float = 8.0,
    ) -> Dict[str, Any]:
        """
        Benchmark VLM latency with acceptance gate.
        
        Args:
            num_crops: Number of crops to score
            image_size: Test image dimensions
            num_trials: Number of timing trials
            warmup_trials: Number of warmup trials
            assert_target: Whether to assert latency target
            target_latency_ms: Target latency threshold
            
        Returns:
            Benchmark results with pass/fail status
        """
        if self.verbose:
            print("\n" + "="*60)
            print("VLM LATENCY BENCHMARK")
            print("="*60)
            print(f"Testing {num_crops} crops, target ≤{target_latency_ms}ms")
        
        # Run benchmark
        stats = self.vlm.benchmark_latency(
            num_crops=num_crops,
            image_size=image_size,
            num_trials=num_trials,
            warmup_trials=warmup_trials,
        )
        
        # Add acceptance testing
        median_latency = stats["median_ms"]
        latency_passed = median_latency <= target_latency_ms
        
        results = {
            "benchmark_type": "latency",
            "num_crops": num_crops,
            "target_latency_ms": target_latency_ms,
            "measured_latency_ms": median_latency,
            "latency_passed": latency_passed,
            "stats": stats,
        }
        
        if self.verbose:
            print(f"\nLatency Test Results:")
            print(f"  Target: ≤{target_latency_ms}ms")
            print(f"  Measured: {median_latency:.2f}ms")
            print(f"  Result: {'✅ PASS' if latency_passed else '❌ FAIL'}")
            
            if not latency_passed:
                print(f"  Suggestions:")
                print(f"    - Reduce max_crops_per_batch to {max(16, num_crops//2)}")
                print(f"    - Use GPU if available")
                print(f"    - Consider model quantization")
        
        # Assert if requested
        if assert_target and not latency_passed:
            raise AssertionError(
                f"Latency assertion failed: {median_latency:.2f}ms > {target_latency_ms}ms"
            )
        
        return results
    
    def test_monotonic_similarity(
        self,
        custom_pairs: Optional[List[Tuple[Tuple[str, str], Tuple[str, str]]]] = None,
        pass_threshold: float = 0.8,
        assert_monotonic: bool = False,
    ) -> Dict[str, Any]:
        """
        Test monotonic similarity with known positive/negative examples.
        
        Args:
            custom_pairs: Custom test pairs, or None for defaults
            pass_threshold: Minimum pass rate for success
            assert_monotonic: Whether to assert monotonic behavior
            
        Returns:
            Similarity test results
        """
        if self.verbose:
            print("\n" + "="*60)
            print("MONOTONIC SIMILARITY TEST")
            print("="*60)
        
        # Default test pairs
        if custom_pairs is None:
            positive_pairs = [
                ("cat sitting on mat", "cat"),
                ("red sports car", "car"),
                ("wooden dining chair", "chair"),
                ("laptop computer", "computer"),
                ("green apple", "apple"),
                ("golden retriever dog", "dog"),
                ("coffee mug", "mug"),
                ("white tennis shoes", "shoes"),
            ]
            
            negative_pairs = [
                ("cat sitting on mat", "dog"),
                ("red sports car", "bicycle"),
                ("wooden dining chair", "table"),
                ("laptop computer", "phone"),
                ("green apple", "orange"),
                ("golden retriever dog", "cat"),
                ("coffee mug", "plate"),
                ("white tennis shoes", "hat"),
            ]
        else:
            positive_pairs = [pair[0] for pair in custom_pairs]
            negative_pairs = [pair[1] for pair in custom_pairs]
        
        # Run similarity test
        results = self.vlm.test_monotonic_similarity(
            positive_pairs=positive_pairs,
            negative_pairs=negative_pairs,
        )
        
        # Add benchmark metadata
        results.update({
            "benchmark_type": "similarity",
            "pass_threshold": pass_threshold,
            "monotonic_passed": results["pass_rate"] >= pass_threshold,
        })
        
        if self.verbose:
            print(f"\nSimilarity Test Results:")
            print(f"  Correct pairs: {results['monotonic_pairs']}/{results['total_pairs']}")
            print(f"  Pass rate: {results['pass_rate']:.1%}")
            print(f"  Threshold: {pass_threshold:.1%}")
            print(f"  Result: {'✅ PASS' if results['monotonic_passed'] else '❌ FAIL'}")
            
            if not results['monotonic_passed']:
                print(f"  Suggestions:")
                print(f"    - Check model weights and preprocessing")
                print(f"    - Verify text tokenization")
                print(f"    - Consider different pretrained checkpoint")
        
        # Assert if requested
        if assert_monotonic and not results['monotonic_passed']:
            raise AssertionError(
                f"Monotonic similarity assertion failed: "
                f"{results['pass_rate']:.1%} < {pass_threshold:.1%}"
            )
        
        return results
    
    def benchmark_text_precomputation(
        self,
        num_phrases: int = 100,
        phrase_length: int = 10,
        num_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark text precomputation efficiency.
        
        Args:
            num_phrases: Number of phrases to precompute
            phrase_length: Average phrase length in words
            num_trials: Number of trials for timing
            
        Returns:
            Precomputation benchmark results
        """
        if self.verbose:
            print("\n" + "="*60)
            print("TEXT PRECOMPUTATION BENCHMARK")
            print("="*60)
        
        # Generate test phrases
        words = ["red", "blue", "large", "small", "wooden", "metal", "round", "square",
                "chair", "table", "car", "dog", "cat", "house", "tree", "book",
                "on", "under", "near", "behind", "in", "next", "to", "the"]
        
        test_phrases = []
        for i in range(num_phrases):
            phrase_words = np.random.choice(words, size=phrase_length, replace=True)
            phrase = " ".join(phrase_words)
            test_phrases.append(phrase)
        
        # Benchmark precomputation
        precomp_times = []
        lookup_times = []
        
        for trial in range(num_trials):
            # Clear cache
            self.vlm.clear_caches()
            
            # Time precomputation
            start_time = time.time()
            self.vlm.precompute_text_embeddings(test_phrases)
            precomp_time = (time.time() - start_time) * 1000
            precomp_times.append(precomp_time)
            
            # Time lookups
            lookup_start = time.time()
            for phrase in test_phrases[:10]:  # Sample lookups
                _ = self.vlm._get_text_embedding(phrase)
            lookup_time = (time.time() - lookup_start) * 1000 / 10  # Per lookup
            lookup_times.append(lookup_time)
        
        results = {
            "benchmark_type": "text_precomputation",
            "num_phrases": num_phrases,
            "num_trials": num_trials,
            "precomputation": {
                "mean_ms": np.mean(precomp_times),
                "std_ms": np.std(precomp_times),
                "median_ms": np.median(precomp_times),
            },
            "lookup": {
                "mean_ms": np.mean(lookup_times),
                "std_ms": np.std(lookup_times),
                "median_ms": np.median(lookup_times),
            },
            "speedup_factor": np.mean(precomp_times) / (np.mean(lookup_times) * num_phrases),
        }
        
        if self.verbose:
            print(f"Precomputation Results ({num_phrases} phrases):")
            print(f"  Precomputation: {results['precomputation']['median_ms']:.2f}ms")
            print(f"  Per-lookup: {results['lookup']['median_ms']:.4f}ms")
            print(f"  Speedup factor: {results['speedup_factor']:.1f}x")
        
        return results
    
    def benchmark_batch_processing(
        self,
        batch_sizes: List[int] = [1, 8, 16, 32, 64, 128],
        image_size: Tuple[int, int] = (640, 480),
        num_trials: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark batch processing efficiency.
        
        Args:
            batch_sizes: List of batch sizes to test
            image_size: Test image dimensions
            num_trials: Number of trials per batch size
            
        Returns:
            Batch processing benchmark results
        """
        if self.verbose:
            print("\n" + "="*60)
            print("BATCH PROCESSING BENCHMARK")
            print("="*60)
        
        # Create test data
        test_image = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        test_phrase = "red object on table"
        
        # Pre-compute text embedding
        self.vlm.precompute_text_embeddings([test_phrase])
        
        results = {
            "benchmark_type": "batch_processing",
            "batch_results": {},
        }
        
        for batch_size in batch_sizes:
            if self.verbose:
                print(f"\nTesting batch size: {batch_size}")
            
            # Create regions
            regions = []
            for i in range(batch_size):
                x1 = np.random.randint(0, image_size[0] // 2)
                y1 = np.random.randint(0, image_size[1] // 2)
                x2 = x1 + np.random.randint(50, image_size[0] // 2)
                y2 = y1 + np.random.randint(50, image_size[1] // 2)
                
                region = Region(xyxy=(x1, y1, x2, y2), score=0.8, label=f"obj_{i}")
                regions.append(region)
            
            # Benchmark batch processing
            batch_times = []
            
            for trial in range(num_trials):
                start_time = time.time()
                _ = self.vlm.score_regions(test_image, regions, test_phrase)
                batch_time = (time.time() - start_time) * 1000
                batch_times.append(batch_time)
            
            # Calculate per-crop latency
            per_crop_times = [t / batch_size for t in batch_times]
            
            batch_stats = {
                "batch_size": batch_size,
                "total_time": {
                    "mean_ms": np.mean(batch_times),
                    "median_ms": np.median(batch_times),
                    "std_ms": np.std(batch_times),
                },
                "per_crop_time": {
                    "mean_ms": np.mean(per_crop_times),
                    "median_ms": np.median(per_crop_times),
                    "std_ms": np.std(per_crop_times),
                },
            }
            
            results["batch_results"][batch_size] = batch_stats
            
            if self.verbose:
                print(f"  Total: {batch_stats['total_time']['median_ms']:.2f}ms")
                print(f"  Per crop: {batch_stats['per_crop_time']['median_ms']:.3f}ms")
        
        # Find optimal batch size
        per_crop_medians = [
            results["batch_results"][bs]["per_crop_time"]["median_ms"]
            for bs in batch_sizes
        ]
        optimal_idx = np.argmin(per_crop_medians)
        results["optimal_batch_size"] = batch_sizes[optimal_idx]
        results["optimal_per_crop_ms"] = per_crop_medians[optimal_idx]
        
        if self.verbose:
            print(f"\nOptimal batch size: {results['optimal_batch_size']}")
            print(f"Optimal per-crop latency: {results['optimal_per_crop_ms']:.3f}ms")
        
        return results
    
    def run_full_benchmark(
        self,
        target_crops: int = 32,
        target_latency_ms: float = 8.0,
        assert_gates: bool = False,
    ) -> Dict[str, Any]:
        """
        Run complete VLM benchmark suite.
        
        Args:
            target_crops: Number of crops for latency testing
            target_latency_ms: Target latency threshold
            assert_gates: Whether to assert acceptance gates
            
        Returns:
            Complete benchmark results
        """
        if self.verbose:
            print("VLM BENCHMARK SUITE")
            print("=" * 80)
            print(f"Model: {self.vlm.model_name}")
            print(f"Device: {self.vlm.device}")
            print(f"Target: {target_crops} crops in ≤{target_latency_ms}ms")
            print("=" * 80)
        
        results = {
            "model_info": self.vlm.get_model_info(),
            "benchmarks": {},
        }
        
        # 1. Latency benchmark
        latency_results = self.benchmark_latency(
            num_crops=target_crops,
            target_latency_ms=target_latency_ms,
            assert_target=assert_gates,
        )
        results["benchmarks"]["latency"] = latency_results
        
        # 2. Monotonic similarity test
        similarity_results = self.test_monotonic_similarity(
            assert_monotonic=assert_gates,
        )
        results["benchmarks"]["similarity"] = similarity_results
        
        # 3. Text precomputation benchmark
        precomp_results = self.benchmark_text_precomputation()
        results["benchmarks"]["text_precomputation"] = precomp_results
        
        # 4. Batch processing benchmark
        batch_results = self.benchmark_batch_processing()
        results["benchmarks"]["batch_processing"] = batch_results
        
        # Overall pass/fail
        overall_pass = (
            latency_results["latency_passed"] and
            similarity_results["monotonic_passed"]
        )
        results["overall_passed"] = overall_pass
        
        if self.verbose:
            print("\n" + "="*80)
            print("BENCHMARK SUMMARY")
            print("="*80)
            print(f"Latency Test: {'✅ PASS' if latency_results['latency_passed'] else '❌ FAIL'}")
            print(f"  {target_crops} crops: {latency_results['measured_latency_ms']:.2f}ms "
                  f"(target: ≤{target_latency_ms}ms)")
            print(f"Similarity Test: {'✅ PASS' if similarity_results['monotonic_passed'] else '❌ FAIL'}")
            print(f"  Pass rate: {similarity_results['pass_rate']:.1%}")
            print(f"Text Precomputation: {precomp_results['speedup_factor']:.1f}x speedup")
            print(f"Optimal Batch Size: {batch_results['optimal_batch_size']} "
                  f"({batch_results['optimal_per_crop_ms']:.3f}ms/crop)")
            print(f"Overall Result: {'✅ PASS' if overall_pass else '❌ FAIL'}")
            print("="*80)
        
        return results


def main():
    """Main entry point for VLM benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark OpenCLIP VLM performance")
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-L-14",
        help="OpenCLIP model name"
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="openai",
        help="Pretrained weights"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device"
    )
    parser.add_argument(
        "--crops",
        type=int,
        default=32,
        help="Number of crops for latency testing"
    )
    parser.add_argument(
        "--target-latency",
        type=float,
        default=8.0,
        help="Target latency in milliseconds"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of benchmark trials"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Maximum crops per batch"
    )
    parser.add_argument(
        "--test-similarity",
        action="store_true",
        help="Run monotonic similarity test"
    )
    parser.add_argument(
        "--test-precomputation",
        action="store_true",
        help="Run text precomputation benchmark"
    )
    parser.add_argument(
        "--test-batching",
        action="store_true",
        help="Run batch processing benchmark"
    )
    parser.add_argument(
        "--assert-latency",
        action="store_true",
        help="Assert latency requirement"
    )
    parser.add_argument(
        "--assert-similarity",
        action="store_true",
        help="Assert monotonic similarity"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Create benchmark
    try:
        benchmark = VLMBenchmark(
            model_name=args.model,
            pretrained=args.pretrained,
            device=args.device,
            max_crops_per_batch=args.batch_size,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"❌ Failed to initialize VLM benchmark: {e}")
        sys.exit(1)
    
    # Run benchmarks
    try:
        if (not args.test_similarity and not args.test_precomputation and 
            not args.test_batching):
            # Run full benchmark suite
            results = benchmark.run_full_benchmark(
                target_crops=args.crops,
                target_latency_ms=args.target_latency,
                assert_gates=(args.assert_latency or args.assert_similarity),
            )
        else:
            # Run individual benchmarks
            results = {"benchmarks": {}}
            
            if args.test_similarity:
                similarity_results = benchmark.test_monotonic_similarity(
                    assert_monotonic=args.assert_similarity,
                )
                results["benchmarks"]["similarity"] = similarity_results
            
            if args.test_precomputation:
                precomp_results = benchmark.benchmark_text_precomputation()
                results["benchmarks"]["text_precomputation"] = precomp_results
            
            if args.test_batching:
                batch_results = benchmark.benchmark_batch_processing()
                results["benchmarks"]["batch_processing"] = batch_results
            
            # Always run latency test
            latency_results = benchmark.benchmark_latency(
                num_crops=args.crops,
                target_latency_ms=args.target_latency,
                assert_target=args.assert_latency,
            )
            results["benchmarks"]["latency"] = latency_results
            
            results["overall_passed"] = all(
                bench.get("latency_passed", bench.get("monotonic_passed", True))
                for bench in results["benchmarks"].values()
            )
    
    except AssertionError as e:
        print(f"❌ Benchmark assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / f"vlm_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Exit with appropriate code
    if results.get("overall_passed", False):
        print("🎉 VLM benchmark PASSED!")
        sys.exit(0)
    else:
        print("❌ VLM benchmark FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
