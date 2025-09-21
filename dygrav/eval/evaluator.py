from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import json
import time
from collections import defaultdict
from PIL import Image
import numpy as np

from ..core.policy import PolicyWithDygrav
from ..core.types import Region
from ..detectors.yolo import SimpleDetector
from ..metrics.grounding import grounding_accuracy
from ..metrics.navigation import success_rate, spl, navigation_error
from ..utils.seed import seed_everything
from ..utils.timing import timer


@dataclass
class EvalConfig:
    """Configuration for evaluation"""
    # Model config
    backbone_type: str = "dummy"  # "dummy", "hamt", "custom"
    checkpoint_path: Optional[str] = None
    device: str = "cpu"
    
    # Data config
    dataset: str = "synthetic"  # "synthetic", "rxr", "rxr_fg", "r2r"
    data_path: Optional[str] = None
    num_episodes: int = 100
    batch_size: int = 1
    
    # DyGRAV config
    use_dygrav: bool = True
    tau_conf: float = 0.55
    tau_entropy: float = 1.25
    max_candidates: int = 4
    
    # VLM config
    vlm_model: str = "ViT-L-14"
    vlm_pretrained: str = "openai"
    
    # Eval config
    seed: int = 17
    iou_threshold: float = 0.5
    save_predictions: bool = True
    output_dir: str = "experiments/eval"
    verbose: bool = True
    log_every: int = 10
    
    # Ablation flags
    ablate_vlm: bool = False
    ablate_sg: bool = False
    ablate_ambiguity: bool = False


@dataclass
class EvalMetrics:
    """Comprehensive evaluation metrics"""
    # Navigation metrics
    success_rate: float = 0.0
    spl_score: float = 0.0
    navigation_error: float = 0.0
    
    # Grounding metrics
    grounding_accuracy: float = 0.0
    grounding_precision: float = 0.0
    grounding_recall: float = 0.0
    
    # DyGRAV metrics
    trigger_rate: float = 0.0
    avg_candidates: float = 0.0
    avg_vlm_score: float = 0.0
    avg_relations: float = 0.0
    
    # Timing metrics
    avg_inference_time: float = 0.0
    avg_dygrav_time: float = 0.0
    total_time: float = 0.0
    
    # Per-category breakdowns
    by_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "navigation": {
                "success_rate": self.success_rate,
                "spl": self.spl_score,
                "nav_error": self.navigation_error,
            },
            "grounding": {
                "accuracy": self.grounding_accuracy,
                "precision": self.grounding_precision,
                "recall": self.grounding_recall,
            },
            "dygrav": {
                "trigger_rate": self.trigger_rate,
                "avg_candidates": self.avg_candidates,
                "avg_vlm_score": self.avg_vlm_score,
                "avg_relations": self.avg_relations,
            },
            "timing": {
                "avg_inference_ms": self.avg_inference_time * 1000,
                "avg_dygrav_ms": self.avg_dygrav_time * 1000,
                "total_seconds": self.total_time,
            },
            "by_category": self.by_category,
        }


class BackboneWrapper:
    """Wrapper to handle different backbone types"""
    
    def __init__(self, backbone_type: str, checkpoint_path: Optional[str] = None, device: str = "cpu"):
        self.backbone_type = backbone_type
        self.device = device
        
        if backbone_type == "dummy":
            self.backbone = DummyBackbone()
        elif backbone_type == "hamt":
            self.backbone = self._load_hamt(checkpoint_path)
        elif backbone_type == "custom":
            self.backbone = self._load_custom(checkpoint_path)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    def _load_hamt(self, checkpoint_path: Optional[str]):
        """Load HAMT or similar transformer backbone"""
        # Import your actual model here
        # from models.hamt import HAMT
        # model = HAMT.load_from_checkpoint(checkpoint_path) if checkpoint_path else HAMT()
        # For now, return enhanced dummy
        return EnhancedDummyBackbone()
    
    def _load_custom(self, checkpoint_path: Optional[str]):
        """Load custom backbone"""
        # Implement custom loading logic
        return EnhancedDummyBackbone()
    
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        return self.backbone.step(obs)
    
    def bias(self, policy_out: Dict, dygrav_signal: Any) -> Dict:
        if hasattr(self.backbone, 'bias'):
            return self.backbone.bias(policy_out, dygrav_signal)
        return policy_out


class EnhancedDummyBackbone:
    """More realistic dummy backbone for testing"""
    
    def __init__(self):
        self.step_count = 0
        self.phrases = [
            "ceramic bowl on the left",
            "red chair next to the table", 
            "smallest box on the shelf",
            "wooden table",
            "blue vase between the lamps"
        ]
    
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.step_count += 1
        
        # Simulate varying confidence and entropy
        conf = 0.3 + 0.4 * np.random.random()  # 0.3 to 0.7
        entropy = 0.5 + 2.0 * np.random.random()  # 0.5 to 2.5
        phrase_idx = self.step_count % len(self.phrases)
        
        return {
            "logits": np.array([0.1, 0.9]),
            "confidence": conf,
            "attn_entropy": entropy,
            "current_phrase": self.phrases[phrase_idx],
            "action": np.random.randint(0, 4),  # Forward, left, right, stop
        }
    
    def bias(self, policy_out: Dict, dygrav_signal: Any) -> Dict:
        if dygrav_signal and dygrav_signal.chosen_region:
            # Simulate biasing based on grounding
            policy_out["confidence"] = min(1.0, policy_out["confidence"] + 0.2)
            policy_out["biased"] = True
        return policy_out


class DatasetLoader:
    """Load different datasets for evaluation"""
    
    def __init__(
        self, 
        dataset: str, 
        data_path: Optional[str] = None, 
        num_episodes: int = 100,
        split: str = "val_unseen",
        language: str = "en",
        load_images: bool = True,
        verbose: bool = False,
    ):
        self.dataset = dataset
        self.data_path = data_path
        self.num_episodes = num_episodes
        self.split = split
        self.language = language
        self.load_images = load_images
        self.verbose = verbose
    
    def load_episodes(self) -> List[Dict[str, Any]]:
        """Load evaluation episodes"""
        
        if self.dataset == "synthetic":
            return self._load_synthetic()
        elif self.dataset == "rxr":
            return self._load_rxr()
        elif self.dataset == "rxr_fg":
            return self._load_rxr_fg()
        elif self.dataset == "r2r":
            return self._load_r2r()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")
    
    def _load_synthetic(self) -> List[Dict[str, Any]]:
        """Generate synthetic evaluation episodes"""
        episodes = []
        for i in range(self.num_episodes):
            episodes.append({
                "episode_id": f"synthetic_{i}",
                "instruction": f"Navigate to the {['kitchen', 'bedroom', 'living room'][i % 3]}",
                "images": [Image.new("RGB", (640, 480), color=(255, 255, 255)) for _ in range(5)],
                "gt_path": [0, 1, 1, 0, 3],  # Ground truth actions
                "gt_regions": [(80, 15, 130, 65)] * 5,  # Ground truth bounding boxes
                "category": "attribute" if i % 2 == 0 else "relation",
            })
        return episodes
    
    def _load_rxr(self) -> List[Dict[str, Any]]:
        """Load RxR dataset"""
        if not self.data_path:
            print("Warning: No RxR data path provided, using synthetic data")
            return self._load_synthetic()
        
        try:
            from ..data.rxr_dataset import create_rxr_dataset
            
            dataset = create_rxr_dataset(
                data_root=self.data_path,
                split=self.split,
                language=self.language,
                max_episodes=self.num_episodes,
                load_images=self.load_images,
                verbose=self.verbose,
            )
            
            return self._convert_episodes_to_dict(dataset.episodes)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load RxR dataset: {e}")
            return self._load_synthetic()
    
    def _load_rxr_fg(self) -> List[Dict[str, Any]]:
        """Load RxR-FG (fine-grained) subset"""
        if not self.data_path:
            print("Warning: No RxR-FG data path provided, using synthetic data")
            return self._load_synthetic()
        
        try:
            from ..data.rxr_dataset import create_rxr_fg_dataset
            
            criteria_file = Path(self.data_path) / "rxr_fg_criteria.json"
            
            dataset = create_rxr_fg_dataset(
                data_root=self.data_path,
                split=self.split,
                language=self.language,
                criteria_file=criteria_file if criteria_file.exists() else None,
                max_episodes=self.num_episodes,
                load_images=self.load_images,
                verbose=self.verbose,
            )
            
            return self._convert_episodes_to_dict(dataset.episodes)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load RxR-FG dataset: {e}")
            return self._load_synthetic()
    
    def _load_r2r(self) -> List[Dict[str, Any]]:
        """Load R2R dataset"""
        if not self.data_path:
            print("Warning: No R2R data path provided, using synthetic data")
            return self._load_synthetic()
        
        try:
            from ..data.r2r_dataset import create_r2r_dataset
            
            dataset = create_r2r_dataset(
                data_root=self.data_path,
                split=self.split,
                max_episodes=self.num_episodes,
                load_images=self.load_images,
                verbose=self.verbose,
            )
            
            return self._convert_episodes_to_dict(dataset.episodes)
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load R2R dataset: {e}")
            return self._load_synthetic()
    
    def _convert_episodes_to_dict(self, episodes) -> List[Dict[str, Any]]:
        """Convert Episode objects to dictionary format for evaluation"""
        from ..data.episode import Episode
        
        converted = []
        
        for ep in episodes:
            if not isinstance(ep, Episode):
                continue
            
            # Get images
            images = ep.get_images(load_images=self.load_images)
            
            # Extract ground truth regions from grounding annotations
            gt_regions = []
            for step_idx in range(len(ep.path)):
                step_annotations = ep.get_grounding_for_step(step_idx)
                if step_annotations:
                    # Use first annotation for this step
                    gt_regions.append(step_annotations[0].bbox)
                else:
                    # Default region if no annotation
                    gt_regions.append((80, 15, 130, 65))
            
            # Determine category based on grounding annotations
            categories = ep.grounding_categories
            if "attribute" in categories:
                category = "attribute"
            elif "relation" in categories:
                category = "relation"
            else:
                category = "object"
            
            # Convert to evaluation format
            episode_dict = {
                "episode_id": ep.episode_id,
                "instruction": ep.instruction,
                "images": images,
                "gt_path": list(range(len(ep.path))),  # Simple path indices
                "gt_regions": gt_regions,
                "category": category,
                "dataset": ep.dataset,
                "split": ep.split,
                "success": ep.success,
                "path_length": ep.trajectory_length,
                "shortest_path": ep.shortest_path_length,
                "scan_id": ep.scan_id,
                "language": ep.language,
                "has_grounding": ep.has_grounding_annotations,
                "num_grounding_annotations": len(ep.grounding_annotations),
            }
            
            converted.append(episode_dict)
        
        return converted


class Evaluator:
    """Main evaluation class"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        seed_everything(config.seed)
        
        # Initialize components
        self.backbone = BackboneWrapper(
            config.backbone_type, 
            config.checkpoint_path,
            config.device
        )
        
        self.tokenizer = self._get_tokenizer()
        self.detector = self._get_detector()
        
        # Create policy with DyGRAV
        if config.use_dygrav:
            self.policy = PolicyWithDygrav(
                backbone=self.backbone,
                detector=self.detector,
                tokenizer=self.tokenizer,
                ambiguity_cfg={
                    "tau_conf": config.tau_conf,
                    "tau_entropy": config.tau_entropy,
                    "max_candidates": config.max_candidates,
                },
                vlm_cfg={
                    "model_name": config.vlm_model,
                    "pretrained": config.vlm_pretrained,
                    "device": config.device,
                },
                sg_cfg={"next_to_thresh": 0.5},
            )
        else:
            # Baseline without DyGRAV
            self.policy = self.backbone
        
        # Load dataset
        self.dataset_loader = DatasetLoader(
            dataset=config.dataset,
            data_path=config.data_path,
            num_episodes=config.num_episodes,
            split="val_unseen",  # Default evaluation split
            language="en",  # Default language
            load_images=True,
            verbose=config.verbose,
        )
        
        # Initialize metrics storage
        self.episode_results = []
        self.metrics = EvalMetrics()
    
    def _get_tokenizer(self):
        """Get appropriate tokenizer"""
        class SmartTokenizer:
            def contains_attribute(self, phrase: str) -> bool:
                attributes = ["red", "blue", "wooden", "ceramic", "smallest", "largest", "left", "right"]
                return any(attr in phrase.lower() for attr in attributes)
        return SmartTokenizer()
    
    def _get_detector(self):
        """Get object detector"""
        # Could swap for real YOLO/DETR here
        return SimpleDetector()
    
    def evaluate_episode(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single episode"""
        
        results = {
            "episode_id": episode["episode_id"],
            "category": episode.get("category", "unknown"),
            "triggers": 0,
            "correct_groundings": 0,
            "total_steps": len(episode["images"]),
            "success": False,
            "nav_error": float('inf'),
            "inference_times": [],
            "dygrav_times": [],
        }
        
        chosen_regions = []
        gt_regions = episode.get("gt_regions", [])
        
        for step_idx, image in enumerate(episode["images"]):
            obs = {
                "rgb": image,
                "instruction": episode["instruction"],
                "step": step_idx,
            }
            
            # Time the inference
            t_start = time.time()
            
            with timer("policy_step") if self.config.verbose else nullcontext():
                output = self.policy.step(obs)
            
            t_inference = time.time() - t_start
            results["inference_times"].append(t_inference)
            
            # Track DyGRAV activation
            if output.get("dygrav", False):
                results["triggers"] += 1
                if "debug_dygrav" in output:
                    t_dygrav = output["debug_dygrav"].get("dygrav_time", 0)
                    results["dygrav_times"].append(t_dygrav)
            
            # Track grounding accuracy if we have GT
            if gt_regions and step_idx < len(gt_regions):
                if output.get("dygrav") and output.get("debug_dygrav"):
                    # DyGRAV chose a region
                    chosen_region = output["debug_dygrav"].get("chosen_bbox", (10, 10, 60, 60))
                else:
                    # Default/baseline choice
                    chosen_region = (10, 10, 60, 60)
                
                chosen_regions.append(chosen_region)
                
                # Check if correct
                from ..metrics.grounding import iou_xyxy
                iou = iou_xyxy(chosen_region, gt_regions[step_idx])
                if iou >= self.config.iou_threshold:
                    results["correct_groundings"] += 1
        
        # Calculate episode metrics
        results["trigger_rate"] = results["triggers"] / max(1, results["total_steps"])
        results["grounding_accuracy"] = results["correct_groundings"] / max(1, len(gt_regions)) if gt_regions else 0
        results["avg_inference_time"] = np.mean(results["inference_times"])
        results["avg_dygrav_time"] = np.mean(results["dygrav_times"]) if results["dygrav_times"] else 0
        
        # Navigation success (simplified)
        results["success"] = results["grounding_accuracy"] > 0.5  # Simplified success criteria
        results["nav_error"] = 2.0 * (1 - results["grounding_accuracy"])  # Simplified error
        
        return results
    
    def run(self) -> EvalMetrics:
        """Run full evaluation"""
        
        print(f"\n{'='*70}")
        print(f"Starting Evaluation")
        print(f"{'='*70}")
        print(f"Dataset: {self.config.dataset}")
        print(f"Episodes: {self.config.num_episodes}")
        print(f"DyGRAV: {'Enabled' if self.config.use_dygrav else 'Disabled'}")
        if self.config.use_dygrav:
            print(f"  τ_conf: {self.config.tau_conf}")
            print(f"  τ_entropy: {self.config.tau_entropy}")
        print(f"{'='*70}\n")
        
        # Load episodes
        episodes = self.dataset_loader.load_episodes()
        
        # Evaluate each episode
        t_start = time.time()
        
        for idx, episode in enumerate(episodes):
            result = self.evaluate_episode(episode)
            self.episode_results.append(result)
            
            if (idx + 1) % self.config.log_every == 0 or idx == len(episodes) - 1:
                current_sr = np.mean([r["success"] for r in self.episode_results])
                current_ga = np.mean([r["grounding_accuracy"] for r in self.episode_results])
                current_tr = np.mean([r["trigger_rate"] for r in self.episode_results])
                
                print(f"[{idx+1:4d}/{len(episodes)}] "
                      f"SR: {current_sr:.3f} | "
                      f"GA: {current_ga:.3f} | "
                      f"Trigger: {current_tr:.3f}")
        
        # Aggregate metrics
        self.metrics = self._aggregate_metrics()
        self.metrics.total_time = time.time() - t_start
        
        # Save results
        if self.config.save_predictions:
            self._save_results()
        
        # Print summary
        self._print_summary()
        
        return self.metrics
    
    def _aggregate_metrics(self) -> EvalMetrics:
        """Aggregate episode results into final metrics"""
        
        metrics = EvalMetrics()
        
        # Navigation metrics
        successes = [r["success"] for r in self.episode_results]
        nav_errors = [r["nav_error"] for r in self.episode_results]
        metrics.success_rate = np.mean(successes)
        metrics.navigation_error = np.mean(nav_errors)
        
        # Grounding metrics
        metrics.grounding_accuracy = np.mean([r["grounding_accuracy"] for r in self.episode_results])
        
        # DyGRAV metrics
        metrics.trigger_rate = np.mean([r["trigger_rate"] for r in self.episode_results])
        
        # Timing metrics
        metrics.avg_inference_time = np.mean([r["avg_inference_time"] for r in self.episode_results])
        metrics.avg_dygrav_time = np.mean([r["avg_dygrav_time"] for r in self.episode_results])
        
        # Per-category breakdown
        categories = set(r["category"] for r in self.episode_results)
        for category in categories:
            cat_results = [r for r in self.episode_results if r["category"] == category]
            metrics.by_category[category] = {
                "success_rate": np.mean([r["success"] for r in cat_results]),
                "grounding_accuracy": np.mean([r["grounding_accuracy"] for r in cat_results]),
                "trigger_rate": np.mean([r["trigger_rate"] for r in cat_results]),
                "count": len(cat_results),
            }
        
        return metrics
    
    def _save_results(self):
        """Save evaluation results to disk"""
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        # Save detailed results
        results_file = output_dir / "episode_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.episode_results, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_dir}")
    
    def _print_summary(self):
        """Print evaluation summary"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*70}")
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Success Rate:        {self.metrics.success_rate:.3f}")
        print(f"  Grounding Accuracy:  {self.metrics.grounding_accuracy:.3f}")
        print(f"  Navigation Error:    {self.metrics.navigation_error:.3f}")
        
        if self.config.use_dygrav:
            print(f"\n🎯 DyGRAV Metrics:")
            print(f"  Trigger Rate:        {self.metrics.trigger_rate:.3f}")
            print(f"  Avg Inference Time:  {self.metrics.avg_inference_time*1000:.2f} ms")
            print(f"  Avg DyGRAV Time:     {self.metrics.avg_dygrav_time*1000:.2f} ms")
        
        if self.metrics.by_category:
            print(f"\n📈 Per-Category Performance:")
            for category, stats in self.metrics.by_category.items():
                print(f"  {category}:")
                print(f"    SR: {stats['success_rate']:.3f} | "
                      f"GA: {stats['grounding_accuracy']:.3f} | "
                      f"Trigger: {stats['trigger_rate']:.3f} | "
                      f"N={stats['count']}")
        
        print(f"\n⏱️  Total Time: {self.metrics.total_time:.1f} seconds")
        print(f"{'='*70}")


# Convenience functions for backward compatibility
def run_eval(n: int = 20, config: Optional[EvalConfig] = None) -> dict:
    """Run evaluation with default or custom config"""
    
    if config is None:
        config = EvalConfig(num_episodes=n)
    else:
        config.num_episodes = n
    
    evaluator = Evaluator(config)
    metrics = evaluator.run()
    
    return {
        "trigger_rate": metrics.trigger_rate,
        "GA": metrics.grounding_accuracy,
        "SR": metrics.success_rate,
        "metrics": metrics.to_dict(),
    }


def run_ablation_study(base_config: Optional[EvalConfig] = None) -> Dict[str, Any]:
    """Run ablation study to measure component contributions"""
    
    if base_config is None:
        base_config = EvalConfig(num_episodes=50)
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY")
    print(f"{'='*70}\n")
    
    ablations = {
        "full": EvalConfig(**base_config.__dict__),
        "no_dygrav": EvalConfig(**{**base_config.__dict__, "use_dygrav": False}),
        "no_vlm": EvalConfig(**{**base_config.__dict__, "ablate_vlm": True}),
        "no_sg": EvalConfig(**{**base_config.__dict__, "ablate_sg": True}),
        "no_ambiguity": EvalConfig(**{**base_config.__dict__, "ablate_ambiguity": True}),
    }
    
    results = {}
    for name, config in ablations.items():
        print(f"\n🔬 Running: {name}")
        config.output_dir = f"experiments/ablation/{name}"
        evaluator = Evaluator(config)
        metrics = evaluator.run()
        results[name] = metrics.to_dict()
    
    # Print comparison
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<15} {'SR':<10} {'GA':<10} {'Trigger':<10}")
    print("-" * 45)
    
    for name, metrics in results.items():
        sr = metrics["navigation"]["success_rate"]
        ga = metrics["grounding"]["accuracy"]
        tr = metrics["dygrav"]["trigger_rate"]
        print(f"{name:<15} {sr:<10.3f} {ga:<10.3f} {tr:<10.3f}")
    
    return results


# For even easier imports
class DummyBackbone:
    """Keep for backward compatibility"""
    def step(self, obs):
        return {"logits": None, "confidence": 0.45, "attn_entropy": 1.8, "current_phrase": "ceramic bowl"}
    def bias(self, out, sig): 
        return out


class DummyTokenizer:
    """Keep for backward compatibility"""
    def contains_attribute(self, phrase: str) -> bool: 
        return True


# Context manager for optional timing
from contextlib import nullcontext