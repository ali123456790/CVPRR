#!/usr/bin/env python
"""
YOLOv10-n Detector Demonstration

This script demonstrates the YOLOv10-n detector implementation for DyGRAV,
including the new Region format, latency benchmarking, and integration examples.
"""

import sys
from pathlib import Path
import numpy as np
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dygrav.detectors import YoloV10Detector, SimpleDetector
from dygrav.core.types import Region


def demo_region_format():
    """Demonstrate the new Region dataclass format."""
    print("="*60)
    print("REGION FORMAT DEMONSTRATION")
    print("="*60)
    
    # Create regions with new format
    regions = [
        Region(xyxy=(10, 10, 100, 80), score=0.95, label="chair"),
        Region(xyxy=(120, 30, 200, 150), score=0.87, label="table"),
        Region(xyxy=(50, 100, 90, 140), score=0.72, label="cup"),
    ]
    
    print("New Region Format:")
    for i, region in enumerate(regions):
        print(f"  Region {i}: {region}")
        print(f"    XYXY: {region.xyxy}")
        print(f"    Score: {region.score:.3f}")
        print(f"    Label: {region.label}")
        print()
    
    # Demonstrate backward compatibility
    print("Backward Compatibility:")
    for i, region in enumerate(regions):
        print(f"  Region {i}: x1={region.x1}, y1={region.y1}, x2={region.x2}, y2={region.y2}, cls={region.cls}")
    
    print("✓ Region format demonstration complete\n")


def demo_simple_detector():
    """Demonstrate SimpleDetector with new format."""
    print("="*60)
    print("SIMPLE DETECTOR DEMONSTRATION")
    print("="*60)
    
    detector = SimpleDetector()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print(f"Test image shape: {test_image.shape}")
    
    # Run detection
    start_time = time.time()
    regions = detector.propose(test_image, phrase="find the chair")
    end_time = time.time()
    
    print(f"Detection time: {(end_time - start_time)*1000:.2f}ms")
    print(f"Number of regions: {len(regions)}")
    
    for i, region in enumerate(regions):
        print(f"  Region {i}: {region.label} @ {region.xyxy} (score: {region.score:.3f})")
    
    print("✓ SimpleDetector demonstration complete\n")


def demo_yolo_detector():
    """Demonstrate YOLOv10Detector (fallback mode)."""
    print("="*60)
    print("YOLOV10 DETECTOR DEMONSTRATION")
    print("="*60)
    
    # Initialize detector (will use fallback if YOLO not available)
    detector = YoloV10Detector(
        model_path="yolov10n.pt",
        precision="fp16",
        input_size=640,
        conf_thresh=0.25,
        iou_thresh=0.6,
        use_tensorrt=True,
        verbose=True
    )
    
    # Get configuration
    config = detector.get_config()
    print(f"Detector Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print(f"Test image shape: {test_image.shape}")
    
    # Run detection with timing
    start_time = time.time()
    regions = detector.propose(test_image, phrase="detect objects")
    end_time = time.time()
    
    print(f"Detection time: {(end_time - start_time)*1000:.2f}ms")
    print(f"Number of regions: {len(regions)}")
    
    for i, region in enumerate(regions):
        print(f"  Region {i}: {region.label} @ {region.xyxy} (score: {region.score:.3f})")
    
    print("✓ YOLOv10Detector demonstration complete\n")


def demo_benchmarking():
    """Demonstrate detector benchmarking."""
    print("="*60)
    print("DETECTOR BENCHMARKING DEMONSTRATION")
    print("="*60)
    
    detector = YoloV10Detector(verbose=False)
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    print("Running benchmark...")
    
    # Run benchmark
    benchmark_results = detector.benchmark(test_image, num_runs=20)
    
    if "error" not in benchmark_results:
        print(f"Benchmark Results:")
        print(f"  Number of runs: {benchmark_results['num_runs']}")
        print(f"  Mean latency: {benchmark_results['mean_ms']:.2f}ms ± {benchmark_results['std_ms']:.2f}ms")
        print(f"  Median latency: {benchmark_results['median_ms']:.2f}ms")
        print(f"  Min latency: {benchmark_results['min_ms']:.2f}ms")
        print(f"  Max latency: {benchmark_results['max_ms']:.2f}ms")
        print(f"  95th percentile: {benchmark_results['p95_ms']:.2f}ms")
        print(f"  99th percentile: {benchmark_results['p99_ms']:.2f}ms")
        
        # Check latency gate
        latency_gate_passed = benchmark_results['median_ms'] <= 20.0
        print(f"  Latency gate (≤20ms): {'✅ PASSED' if latency_gate_passed else '❌ FAILED'}")
    else:
        print(f"Benchmark failed: {benchmark_results['error']}")
    
    print("✓ Benchmarking demonstration complete\n")


def demo_integration():
    """Demonstrate integration with DyGRAV pipeline."""
    print("="*60)
    print("DYGRAV INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # This simulates how the detector would be used in the DyGRAV pipeline
    try:
        from dygrav.core.policy import PolicyWithDygrav
        from dygrav.lightning.module import DummyTokenizer
    except ImportError as e:
        print(f"⚠️  Skipping integration demo due to missing dependencies: {e}")
        print("   Install full dependencies with: pip install -e '.[full]'")
        print("✓ Integration demonstration skipped\n")
        return
    
    print("Creating DyGRAV policy with YOLOv10 detector...")
    
    # Mock backbone
    class MockBackbone:
        def step(self, obs):
            return {
                "confidence": 0.4,
                "attn_entropy": 2.0,
                "current_phrase": "find the wooden chair",
                "logits": [0.1, 0.9]
            }
        def bias(self, policy_out, dygrav_signal):
            policy_out["logits"] = [0.9, 0.1]
            policy_out["biased"] = True
            return policy_out
    
    # Create policy with YOLOv10 detector
    policy = PolicyWithDygrav(
        backbone=MockBackbone(),
        detector=YoloV10Detector(verbose=False),
        tokenizer=DummyTokenizer(),
        ambiguity_cfg={"tau_conf": 0.55, "tau_entropy": 1.25, "max_candidates": 3},
        vlm_cfg={"model_name": "ViT-L-14", "pretrained": "openai", "device": "cpu"},
        sg_cfg={"next_to_thresh": 0.5},
    )
    
    # Create test observation
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    obs = {
        "rgb": test_image,
        "instruction": "Go to the wooden chair in the living room",
    }
    
    print(f"Test observation: {obs.keys()}")
    print(f"Image shape: {obs['rgb'].shape}")
    print(f"Instruction: {obs['instruction']}")
    
    # Run policy step
    start_time = time.time()
    output = policy.step(obs)
    end_time = time.time()
    
    print(f"\nPolicy Step Results:")
    print(f"  Processing time: {(end_time - start_time)*1000:.2f}ms")
    print(f"  DyGRAV triggered: {output.get('dygrav', False)}")
    print(f"  Policy biased: {output.get('biased', False)}")
    print(f"  Debug info keys: {list(output.get('debug_dygrav', {}).keys())}")
    
    if output.get('debug_dygrav'):
        debug = output['debug_dygrav']
        print(f"  VLM top score: {debug.get('vlm_top', 0):.3f}")
        print(f"  Number of relations: {debug.get('n_rel', 0)}")
    
    print("✓ Integration demonstration complete\n")


def demo_configuration():
    """Demonstrate detector configuration options."""
    print("="*60)
    print("DETECTOR CONFIGURATION DEMONSTRATION")
    print("="*60)
    
    configs = [
        {
            "name": "High Accuracy",
            "config": {
                "model_path": "yolov10n.pt",
                "precision": "fp32",
                "input_size": 640,
                "conf_thresh": 0.1,
                "iou_thresh": 0.5,
                "use_tensorrt": False,
            }
        },
        {
            "name": "Balanced",
            "config": {
                "model_path": "yolov10n.pt",
                "precision": "fp16",
                "input_size": 640,
                "conf_thresh": 0.25,
                "iou_thresh": 0.6,
                "use_tensorrt": True,
            }
        },
        {
            "name": "High Speed",
            "config": {
                "model_path": "yolov10n.pt",
                "precision": "fp16",
                "input_size": 512,
                "conf_thresh": 0.4,
                "iou_thresh": 0.7,
                "use_tensorrt": True,
            }
        }
    ]
    
    for config_info in configs:
        print(f"{config_info['name']} Configuration:")
        for key, value in config_info['config'].items():
            print(f"  {key}: {value}")
        print()
    
    print("Configuration Guidelines:")
    print("  • Higher input_size → Better accuracy, slower inference")
    print("  • Lower conf_thresh → More detections, more false positives")
    print("  • FP16 precision → ~2x faster inference on modern GPUs")
    print("  • TensorRT → Additional 20-30% speedup after initial compilation")
    
    print("✓ Configuration demonstration complete\n")


def main():
    """Run all detector demonstrations."""
    print("YOLOV10-N DETECTOR DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        demo_region_format()
        demo_simple_detector()
        demo_yolo_detector()
        demo_benchmarking()
        demo_integration()
        demo_configuration()
        
        print("=" * 80)
        print("🎉 ALL DETECTOR DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Next Steps:")
        print("1. Install YOLOv10 dependencies: pip install ultralytics")
        print("2. Download YOLOv10 weights: yolo download yolov10n.pt")
        print("3. Run latency benchmark: python tools/bench_detector.py")
        print("4. Configure detector: edit configs/detector.yaml")
        print("5. Implement Stage-2 fine-tuning: see docs/DETECTOR_FINETUNING_PLAN.md")
        print()
        print("For more information:")
        print("- YOLOv10 detector: dygrav/detectors/yolo.py")
        print("- Benchmarking tool: tools/bench_detector.py")
        print("- Configuration: configs/detector.yaml")
        print("- Fine-tuning plan: docs/DETECTOR_FINETUNING_PLAN.md")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
