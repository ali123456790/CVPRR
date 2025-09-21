#!/usr/bin/env python
"""
Detector Benchmarking Tool

This script benchmarks the YOLOv10-n detector to ensure it meets the 
<= 20ms latency requirement specified in the Week-1 Sprint Plan.

Usage:
    python tools/bench_detector.py --config configs/detector.yaml
    python tools/bench_detector.py --model yolov10n.pt --input-size 640
"""

from __future__ import annotations
import argparse
import time
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dygrav.detectors.yolo import YoloV10Detector, SimpleDetector
from dygrav.utils.seed import seed_everything

try:
    import torch
    import cv2
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


class DetectorBenchmark:
    """Comprehensive detector benchmarking with latency gates."""
    
    def __init__(self, config: Dict[str, Any], verbose: bool = True):
        """
        Initialize benchmark with configuration.
        
        Args:
            config: Detector configuration dictionary
            verbose: Whether to print verbose output
        """
        self.config = config
        self.verbose = verbose
        self.detector = None
        
        # Latency requirements
        self.max_median_ms = config.get("latency", {}).get("max_median_ms", 20)
        self.target_gpus = config.get("latency", {}).get("target_gpu", [])
        
    def setup_detector(self) -> bool:
        """Setup detector based on configuration."""
        try:
            detector_config = {
                "model_path": self.config.get("model_path", "yolov10n.pt"),
                "precision": self.config.get("precision", "fp16"),
                "input_size": self.config.get("input_size", 640),
                "conf_thresh": self.config.get("conf_thresh", 0.25),
                "iou_thresh": self.config.get("iou_thresh", 0.6),
                "device": self.config.get("device", "auto"),
                "use_tensorrt": self.config.get("use_tensorrt", True),
                "verbose": self.verbose,
            }
            
            self.detector = YoloV10Detector(**detector_config)
            
            if self.verbose:
                print("✓ Detector setup successful")
                config = self.detector.get_config()
                print(f"  Model: {config['model_path']}")
                print(f"  Device: {config['device']}")
                print(f"  Precision: {config['precision']}")
                print(f"  Input size: {config['input_size']}")
                print(f"  TensorRT: {config['use_tensorrt']}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Detector setup failed: {e}")
            return False
    
    def create_test_images(self, batch_size: int = 1) -> list[np.ndarray]:
        """Create test images for benchmarking."""
        input_size = self.config.get("input_size", 640)
        
        images = []
        
        # Create different types of test images
        for i in range(batch_size):
            if i % 4 == 0:
                # Random noise image
                image = np.random.randint(0, 255, (input_size, input_size, 3), dtype=np.uint8)
            elif i % 4 == 1:
                # Gradient image
                x = np.linspace(0, 255, input_size, dtype=np.uint8)
                y = np.linspace(0, 255, input_size, dtype=np.uint8)
                xx, yy = np.meshgrid(x, y)
                image = np.stack([xx, yy, (xx + yy) // 2], axis=-1)
            elif i % 4 == 2:
                # Checkerboard pattern
                check = np.zeros((input_size, input_size, 3), dtype=np.uint8)
                check[::32, ::32] = 255
                image = check
            else:
                # Solid color
                color = [128, 64, 192]
                image = np.full((input_size, input_size, 3), color, dtype=np.uint8)
            
            images.append(image)
        
        return images
    
    def benchmark_latency(
        self, 
        num_warmup: int = 10, 
        num_runs: int = 100,
        batch_size: int = 1
    ) -> Dict[str, Any]:
        """
        Benchmark detector latency.
        
        Args:
            num_warmup: Number of warmup runs
            num_runs: Number of benchmark runs
            batch_size: Batch size (currently only supports 1)
        
        Returns:
            Dictionary with timing statistics
        """
        if self.detector is None:
            return {"error": "Detector not initialized"}
        
        # Create test images
        test_images = self.create_test_images(batch_size)
        
        if self.verbose:
            print(f"\n🔥 Starting latency benchmark:")
            print(f"  Warmup runs: {num_warmup}")
            print(f"  Benchmark runs: {num_runs}")
            print(f"  Batch size: {batch_size}")
            print(f"  Image size: {test_images[0].shape}")
        
        # Warmup runs
        if self.verbose:
            print("  Running warmup...")
        
        for i in range(num_warmup):
            _ = self.detector.propose(test_images[0])
        
        # Benchmark runs
        if self.verbose:
            print("  Running benchmark...")
        
        times = []
        
        for i in range(num_runs):
            # Use different images to avoid caching effects
            image = test_images[i % len(test_images)]
            
            start_time = time.time()
            regions = self.detector.propose(image)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            times.append(latency_ms)
            
            if self.verbose and (i + 1) % 20 == 0:
                current_median = np.median(times)
                print(f"    Run {i+1}/{num_runs}, current median: {current_median:.2f}ms")
        
        # Calculate statistics
        times = np.array(times)
        
        stats = {
            "num_runs": num_runs,
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
            "times": times.tolist(),
        }
        
        return stats
    
    def check_latency_gate(self, stats: Dict[str, Any]) -> bool:
        """
        Check if detector meets latency requirements (hard gate).
        
        Args:
            stats: Timing statistics from benchmark
        
        Returns:
            True if latency requirements are met
        """
        if "error" in stats:
            return False
        
        median_ms = stats["median_ms"]
        
        if self.verbose:
            print(f"\n🎯 Latency Gate Check:")
            print(f"  Median latency: {median_ms:.2f}ms")
            print(f"  Required: ≤ {self.max_median_ms}ms")
        
        passed = median_ms <= self.max_median_ms
        
        if passed:
            if self.verbose:
                print(f"  ✅ PASSED - Latency requirement met!")
        else:
            if self.verbose:
                print(f"  ❌ FAILED - Latency exceeds requirement by {median_ms - self.max_median_ms:.2f}ms")
        
        return passed
    
    def suggest_fallback_config(self) -> Dict[str, Any]:
        """Suggest fallback configuration for better latency."""
        fallback = self.config.get("fallback", {})
        
        suggestions = {
            "input_size": fallback.get("input_size", 512),
            "model_path": fallback.get("model_path", "yolov10s.pt"),
            "precision": "fp16",
            "use_tensorrt": True,
        }
        
        if self.verbose:
            print(f"\n💡 Fallback Configuration Suggestions:")
            print(f"  Reduce input size: {self.config.get('input_size', 640)} → {suggestions['input_size']}")
            print(f"  Switch model: {self.config.get('model_path', 'yolov10n.pt')} → {suggestions['model_path']}")
            print(f"  Ensure TensorRT: {suggestions['use_tensorrt']}")
            print(f"  Ensure FP16: {suggestions['precision']}")
        
        return suggestions
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmarking context."""
        info = {
            "python_version": sys.version,
            "dependencies_available": DEPS_AVAILABLE,
        }
        
        if DEPS_AVAILABLE:
            info.update({
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            })
            
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory // (1024**3)
        
        return info
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark suite."""
        results = {
            "config": self.config,
            "system_info": self.get_system_info(),
            "timestamp": time.time(),
        }
        
        if self.verbose:
            print("="*70)
            print("DETECTOR BENCHMARK")
            print("="*70)
            
            print(f"\n💻 System Information:")
            sys_info = results["system_info"]
            print(f"  Dependencies available: {sys_info['dependencies_available']}")
            if sys_info.get("cuda_available"):
                print(f"  GPU: {sys_info.get('gpu_name', 'Unknown')}")
                print(f"  CUDA: {sys_info.get('cuda_version', 'Unknown')}")
                print(f"  Memory: {sys_info.get('gpu_memory', 0)}GB")
        
        # Setup detector
        if not self.setup_detector():
            results["error"] = "Failed to setup detector"
            return results
        
        # Run latency benchmark
        latency_stats = self.benchmark_latency(
            num_warmup=10,
            num_runs=100,
            batch_size=1
        )
        
        results["latency_stats"] = latency_stats
        
        # Check latency gate
        latency_passed = self.check_latency_gate(latency_stats)
        results["latency_gate_passed"] = latency_passed
        
        # Suggest fallback if needed
        if not latency_passed:
            results["fallback_suggestions"] = self.suggest_fallback_config()
        
        if self.verbose:
            print(f"\n📊 Final Results:")
            if "error" not in latency_stats:
                print(f"  Mean latency: {latency_stats['mean_ms']:.2f}ms ± {latency_stats['std_ms']:.2f}ms")
                print(f"  Median latency: {latency_stats['median_ms']:.2f}ms")
                print(f"  95th percentile: {latency_stats['p95_ms']:.2f}ms")
                print(f"  99th percentile: {latency_stats['p99_ms']:.2f}ms")
            
            print(f"  Latency gate: {'✅ PASSED' if latency_passed else '❌ FAILED'}")
            print("="*70)
        
        return results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load detector configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def main():
    """Main entry point for detector benchmarking."""
    parser = argparse.ArgumentParser(description="Benchmark YOLOv10 detector latency")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detector.yaml",
        help="Path to detector configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Override model path"
    )
    parser.add_argument(
        "--input-size",
        type=int,
        help="Override input size"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp16", "fp32"],
        help="Override precision"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        help="Override device"
    )
    parser.add_argument(
        "--no-tensorrt",
        action="store_true",
        help="Disable TensorRT optimization"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of benchmark runs"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--assert-latency",
        action="store_true",
        help="Exit with error code if latency gate fails"
    )
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        print(f"Using default configuration")
        config = {
            "model_path": "yolov10n.pt",
            "precision": "fp16",
            "input_size": 640,
            "conf_thresh": 0.25,
            "iou_thresh": 0.6,
            "device": "auto",
            "use_tensorrt": True,
            "latency": {"max_median_ms": 20}
        }
    
    # Apply command line overrides
    if args.model:
        config["model_path"] = args.model
    if args.input_size:
        config["input_size"] = args.input_size
    if args.precision:
        config["precision"] = args.precision
    if args.device:
        config["device"] = args.device
    if args.no_tensorrt:
        config["use_tensorrt"] = False
    
    # Run benchmark
    benchmark = DetectorBenchmark(config, verbose=not args.quiet)
    results = benchmark.run_full_benchmark()
    
    # Check assertion
    if args.assert_latency:
        latency_passed = results.get("latency_gate_passed", False)
        if not latency_passed:
            if not args.quiet:
                print("\n❌ Latency assertion failed!")
            sys.exit(1)
        else:
            if not args.quiet:
                print("\n✅ Latency assertion passed!")
            sys.exit(0)


if __name__ == "__main__":
    main()
