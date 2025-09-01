import numpy as np
from typing import Dict, List, Tuple
import json
from pathlib import Path
from PIL import Image

# Import the necessary components directly
from dygrav.core.policy import PolicyWithDygrav
from dygrav.detectors.yolo import SimpleDetector
from dygrav.metrics.grounding import grounding_accuracy

class DummyBackbone:
    """Backbone with controllable confidence for testing thresholds"""
    def __init__(self, base_conf: float = 0.45, base_entropy: float = 1.8):
        self.base_conf = base_conf
        self.base_entropy = base_entropy
        self.step_count = 0
        
    def step(self, obs):
        # Vary confidence and entropy slightly to simulate realistic conditions
        self.step_count += 1
        noise_conf = np.random.uniform(-0.1, 0.1)
        noise_ent = np.random.uniform(-0.3, 0.3)
        
        return {
            "logits": None, 
            "confidence": max(0.1, min(1.0, self.base_conf + noise_conf)),
            "attn_entropy": max(0.0, self.base_entropy + noise_ent),
            "current_phrase": "ceramic bowl"
        }
    
    def bias(self, out, sig): 
        return out

class DummyTokenizer:
    def contains_attribute(self, phrase: str) -> bool: 
        return True

def run_sweep_eval(tau_conf: float, tau_entropy: float, n_episodes: int = 30, 
                   conf_range: Tuple[float, float] = (0.3, 0.7),
                   entropy_range: Tuple[float, float] = (1.0, 2.5)) -> Dict:
    """
    Evaluate with specific threshold values across various confidence/entropy scenarios
    """
    results = {
        'tau_conf': tau_conf,
        'tau_entropy': tau_entropy,
        'trigger_rates': [],
        'grounding_accs': [],
        'details': []
    }
    
    # Test across different confidence and entropy scenarios
    test_scenarios = [
        (0.3, 2.5, "low_conf_high_ent"),   # Should trigger
        (0.7, 0.5, "high_conf_low_ent"),   # Should not trigger
        (0.45, 1.8, "medium_both"),        # Borderline
        (0.35, 1.2, "low_conf_medium_ent"), # Maybe trigger
        (0.6, 2.0, "medium_conf_high_ent"), # Maybe trigger
    ]
    
    for base_conf, base_ent, scenario_name in test_scenarios:
        # Create policy with current thresholds
        backbone = DummyBackbone(base_conf=base_conf, base_entropy=base_ent)
        policy = PolicyWithDygrav(
            backbone=backbone,
            detector=SimpleDetector(),
            tokenizer=DummyTokenizer(),
            ambiguity_cfg={
                "tau_conf": tau_conf,
                "tau_entropy": tau_entropy,
                "max_candidates": 3
            },
            vlm_cfg={"model_name": "ViT-L-14", "pretrained": "openai", "device": "cpu"},
            sg_cfg={"next_to_thresh": 0.5},
        )
        
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        chosen = []
        refer = []
        triggers = 0
        
        for _ in range(n_episodes):
            out = policy.step({"rgb": img, "feat": None})
            triggers += 1 if out["dygrav"] else 0
            
            # Simulate grounding accuracy based on whether DyGRAV triggered
            if out["dygrav"] and out["debug_dygrav"]:
                # When DyGRAV triggers, it usually helps find the right region
                chosen.append((80, 15, 130, 65))  # Correct region
                refer.append((80, 15, 130, 65))
            else:
                # Without DyGRAV, sometimes wrong
                if np.random.random() > 0.6:  # 40% chance of being right without DyGRAV
                    chosen.append((80, 15, 130, 65))
                else:
                    chosen.append((10, 10, 60, 60))  # Wrong region
                refer.append((80, 15, 130, 65))
        
        trigger_rate = triggers / max(1, n_episodes)
        ga = grounding_accuracy(chosen, refer, tau=0.5)
        
        results['trigger_rates'].append(trigger_rate)
        results['grounding_accs'].append(ga)
        results['details'].append({
            'scenario': scenario_name,
            'base_conf': base_conf,
            'base_entropy': base_ent,
            'trigger_rate': trigger_rate,
            'GA': ga
        })
    
    # Calculate aggregate metrics
    results['mean_trigger_rate'] = np.mean(results['trigger_rates'])
    results['mean_GA'] = np.mean(results['grounding_accs'])
    results['std_trigger_rate'] = np.std(results['trigger_rates'])
    results['std_GA'] = np.std(results['grounding_accs'])
    
    return results

def find_optimal_thresholds():
    """
    Grid search to find optimal tau_conf and tau_entropy thresholds
    """
    print("=" * 70)
    print("DyGRAV Threshold Sweep - Finding Optimal Parameters")
    print("=" * 70)
    
    # Define search ranges
    tau_conf_values = np.arange(0.4, 0.7, 0.05)
    tau_entropy_values = np.arange(0.8, 2.0, 0.2)
    
    best_score = -float('inf')
    best_params = None
    all_results = []
    
    print(f"\nTesting {len(tau_conf_values)} x {len(tau_entropy_values)} = "
          f"{len(tau_conf_values) * len(tau_entropy_values)} configurations...\n")
    
    for tau_conf in tau_conf_values:
        for tau_entropy in tau_entropy_values:
            result = run_sweep_eval(tau_conf, tau_entropy, n_episodes=20)
            
            # Score: balance between good GA and reasonable trigger rate (aim for 20-40%)
            # Penalize both too low and too high trigger rates
            target_trigger_rate = 0.3
            trigger_penalty = abs(result['mean_trigger_rate'] - target_trigger_rate)
            score = result['mean_GA'] - 0.5 * trigger_penalty
            
            all_results.append(result)
            
            print(f"τ_conf={tau_conf:.2f}, τ_ent={tau_entropy:.1f}: "
                  f"trigger={result['mean_trigger_rate']:.2f}±{result['std_trigger_rate']:.2f}, "
                  f"GA={result['mean_GA']:.2f}±{result['std_GA']:.2f}, "
                  f"score={score:.3f}")
            
            if score > best_score:
                best_score = score
                best_params = (tau_conf, tau_entropy)
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    # Find configurations by trigger rate buckets
    low_trigger = [r for r in all_results if r['mean_trigger_rate'] < 0.2]
    medium_trigger = [r for r in all_results if 0.2 <= r['mean_trigger_rate'] <= 0.4]
    high_trigger = [r for r in all_results if r['mean_trigger_rate'] > 0.4]
    
    print(f"\n📊 Trigger Rate Distribution:")
    print(f"  Low (<20%):    {len(low_trigger)} configs")
    print(f"  Medium (20-40%): {len(medium_trigger)} configs")
    print(f"  High (>40%):   {len(high_trigger)} configs")
    
    if medium_trigger:
        best_medium = max(medium_trigger, key=lambda x: x['mean_GA'])
        print(f"\n🎯 Best in target range (20-40% trigger):")
        print(f"  τ_conf={best_medium['tau_conf']:.2f}, τ_entropy={best_medium['tau_entropy']:.1f}")
        print(f"  Trigger Rate: {best_medium['mean_trigger_rate']:.2%}")
        print(f"  Grounding Acc: {best_medium['mean_GA']:.2%}")
    
    print(f"\n🏆 Overall Best (balanced score):")
    print(f"  τ_conf={best_params[0]:.2f}, τ_entropy={best_params[1]:.1f}")
    print(f"  Score: {best_score:.3f}")
    
    # Save results
    output_file = Path("experiments/threshold_sweep_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'best_params': {'tau_conf': best_params[0], 'tau_entropy': best_params[1]},
            'best_score': best_score,
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return best_params, all_results

def run_detailed_analysis(tau_conf: float = 0.55, tau_entropy: float = 1.25):
    """
    Run detailed analysis with specific thresholds
    """
    print("\n" + "=" * 70)
    print(f"DETAILED ANALYSIS: τ_conf={tau_conf:.2f}, τ_entropy={tau_entropy:.2f}")
    print("=" * 70)
    
    result = run_sweep_eval(tau_conf, tau_entropy, n_episodes=50)
    
    print(f"\n📈 Overall Performance:")
    print(f"  Mean Trigger Rate: {result['mean_trigger_rate']:.2%} ± {result['std_trigger_rate']:.2%}")
    print(f"  Mean Grounding Accuracy: {result['mean_GA']:.2%} ± {result['std_GA']:.2%}")
    
    print(f"\n📊 Per-Scenario Breakdown:")
    print(f"  {'Scenario':<25} {'Conf':<8} {'Entropy':<10} {'Trigger':<12} {'GA':<8}")
    print("  " + "-" * 65)
    
    for detail in result['details']:
        print(f"  {detail['scenario']:<25} "
              f"{detail['base_conf']:<8.2f} "
              f"{detail['base_entropy']:<10.2f} "
              f"{detail['trigger_rate']:<12.2%} "
              f"{detail['GA']:<8.2%}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if result['mean_trigger_rate'] < 0.15:
        print("  ⚠️  Trigger rate too LOW - DyGRAV rarely activates")
        print("     → Consider decreasing τ_conf or τ_entropy")
    elif result['mean_trigger_rate'] > 0.40:
        print("  ⚠️  Trigger rate too HIGH - excessive computation")
        print("     → Consider increasing τ_conf or τ_entropy")
    else:
        print("  ✅ Trigger rate in optimal range (15-40%)")
    
    if result['mean_GA'] < 0.7:
        print("  ⚠️  Grounding accuracy could be improved")
        print("     → Check VLM scoring and scene graph quality")
    else:
        print("  ✅ Good grounding accuracy")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DyGRAV Threshold Sweep")
    parser.add_argument('--mode', choices=['sweep', 'analyze', 'quick'], 
                       default='quick',
                       help='Mode: sweep (full grid search), analyze (specific values), quick (fast test)')
    parser.add_argument('--tau-conf', type=float, default=0.55,
                       help='Confidence threshold (for analyze mode)')
    parser.add_argument('--tau-entropy', type=float, default=1.25,
                       help='Entropy threshold (for analyze mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'sweep':
        # Full grid search
        best_params, all_results = find_optimal_thresholds()
        
        # Run detailed analysis on best params
        run_detailed_analysis(best_params[0], best_params[1])
        
    elif args.mode == 'analyze':
        # Analyze specific thresholds
        run_detailed_analysis(args.tau_conf, args.tau_entropy)
        
    else:  # quick mode
        # Quick test with default values
        print("Quick stability test (5 trials)...")
        trials = 5
        rates = []
        gas = []
        
        for i in range(trials):
            result = run_sweep_eval(0.55, 1.25, n_episodes=10)
            rates.append(result['mean_trigger_rate'])
            gas.append(result['mean_GA'])
            print(f"  Trial {i+1}: trigger_rate={result['mean_trigger_rate']:.2f}, GA={result['mean_GA']:.2f}")
        
        print(f"\n📊 Summary:")
        print(f"  Trigger Rate: {np.mean(rates):.2f} ± {np.std(rates):.3f}")
        print(f"  Grounding Acc: {np.mean(gas):.2f} ± {np.std(gas):.3f}")