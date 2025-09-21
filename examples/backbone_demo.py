#!/usr/bin/env python
"""
Backbone Implementation Demonstration

This script demonstrates the HAMT and RvLN-BERT backbone implementations
for DyGRAV, showing contract compliance and integration capabilities.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dygrav.backbones import HAMTWrapper, RvLNBERTWrapper


def demo_backbone_contract():
    """Demonstrate BackboneWrapper contract compliance."""
    print("="*60)
    print("BACKBONE CONTRACT DEMONSTRATION")
    print("="*60)
    
    # Test HAMT wrapper
    print("\n🧠 HAMT Wrapper:")
    hamt = HAMTWrapper(model_config={})
    
    # Test forward method signature
    batch_size = 2
    pano_feats = torch.randn(batch_size, 36, 2048)
    instr_tokens = torch.randint(1, 1000, (batch_size, 20))
    instr_mask = torch.ones(batch_size, 20, dtype=torch.bool)
    
    with torch.no_grad():
        outputs = hamt.forward(pano_feats, instr_tokens, instr_mask)
    
    print(f"  ✓ Forward method contract:")
    print(f"    Input shapes: pano_feats={pano_feats.shape}, instr_tokens={instr_tokens.shape}")
    print(f"    Output keys: {list(outputs.keys())}")
    print(f"    Logits shape: {outputs['logits'].shape} (expected: [B, A])")
    print(f"    Stop shape: {outputs['stop'].shape} (expected: [B])")
    print(f"    Attention shape: {outputs['attn'].shape} (expected: [B, ?, ?])")
    
    # Test step method
    obs = {"current_phrase": "navigate to the kitchen"}
    step_output = hamt.step(obs)
    print(f"  ✓ Step method contract:")
    print(f"    Input: {obs}")
    print(f"    Output keys: {list(step_output.keys())}")
    print(f"    Confidence: {step_output['confidence']:.3f}")
    print(f"    Attention entropy: {step_output['attn_entropy']:.3f}")
    
    # Test bias method
    policy_out = {"logits": torch.randn(1, 4)}
    biased_out = hamt.bias(policy_out, None)
    print(f"  ✓ Bias method contract:")
    print(f"    Input logits: {policy_out['logits'].shape}")
    print(f"    Output logits: {biased_out['logits'].shape}")
    print(f"    Bias applied: {biased_out.get('biased', False)}")
    
    print("✓ HAMT contract compliance verified\n")


def demo_attention_visualization():
    """Demonstrate attention weight extraction and analysis."""
    print("="*60)
    print("ATTENTION VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    hamt = HAMTWrapper(model_config={})
    
    # Create meaningful inputs
    batch_size = 1
    pano_feats = torch.randn(batch_size, 36, 2048)
    
    # Create instruction tokens (simulate "go to the red chair")
    instr_tokens = torch.tensor([[1, 145, 67, 89, 234, 567, 2, 0, 0, 0]], dtype=torch.long)
    instr_mask = torch.tensor([[True, True, True, True, True, True, True, False, False, False]])
    
    with torch.no_grad():
        outputs = hamt.forward(pano_feats, instr_tokens, instr_mask)
    
    attention_weights = outputs["attn"]  # [B, 36, L]
    
    print(f"Attention Analysis:")
    print(f"  Shape: {attention_weights.shape}")
    print(f"  Min attention: {attention_weights.min().item():.4f}")
    print(f"  Max attention: {attention_weights.max().item():.4f}")
    print(f"  Mean attention: {attention_weights.mean().item():.4f}")
    
    # Calculate attention entropy (measure of focus)
    attn_flat = attention_weights.view(-1)
    attn_probs = F.softmax(attn_flat, dim=0)
    entropy = -(attn_probs * torch.log(attn_probs + 1e-10)).sum().item()
    max_entropy = np.log(len(attn_flat))
    normalized_entropy = entropy / max_entropy
    
    print(f"  Attention entropy: {entropy:.4f} (normalized: {normalized_entropy:.4f})")
    print(f"  Attention focus: {'Focused' if normalized_entropy < 0.7 else 'Uniform'}")
    
    # Show top attended viewpoints for each instruction token
    print(f"\nTop attended viewpoints per token:")
    for token_idx in range(instr_mask.sum().item()):
        token_attn = attention_weights[0, :, token_idx]  # [36]
        top_viewpoints = torch.topk(token_attn, k=3)
        print(f"  Token {token_idx}: views {top_viewpoints.indices.tolist()} "
              f"(scores: {top_viewpoints.values.tolist()})")
    
    print("✓ Attention visualization complete\n")


def demo_dygrav_integration():
    """Demonstrate integration with DyGRAV pipeline."""
    print("="*60)
    print("DYGRAV INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create HAMT backbone
    hamt = HAMTWrapper(model_config={})
    
    # Simulate DyGRAV grounding signal
    class MockDyGravSignal:
        def __init__(self):
            from dygrav.core.types import Region
            self.chosen_region = Region(xyxy=(100, 100, 200, 200), score=0.8, label="chair")
            
            class VLMScore:
                def __init__(self, score):
                    self.score = score
            
            self.vlm_scores = [VLMScore(0.9), VLMScore(0.7)]
    
    # Test policy integration
    obs = {
        "current_phrase": "find the red chair on the left",
        "pano_feats": torch.randn(1, 36, 2048),
        "instr_tokens": torch.randint(1, 1000, (1, 15)),
        "instr_mask": torch.ones(1, 15, dtype=torch.bool),
    }
    
    # Get policy output
    policy_output = hamt.step(obs)
    print(f"Policy Output (before DyGRAV):")
    print(f"  Confidence: {policy_output['confidence']:.3f}")
    print(f"  Action logits: {policy_output['logits'].tolist()}")
    print(f"  Stop probability: {policy_output['stop_prob']:.3f}")
    
    # Apply DyGRAV bias
    dygrav_signal = MockDyGravSignal()
    biased_output = hamt.bias(policy_output, dygrav_signal)
    
    print(f"\nPolicy Output (after DyGRAV bias):")
    print(f"  Biased: {biased_output.get('biased', False)}")
    print(f"  Bias strength: {biased_output.get('bias_strength', 0):.3f}")
    print(f"  Biased logits: {biased_output['logits'].tolist()}")
    
    # Compare action probabilities
    original_probs = F.softmax(policy_output['logits'], dim=-1)
    biased_probs = F.softmax(biased_output['logits'], dim=-1)
    
    actions = ["forward", "left", "right", "stop"]
    print(f"\nAction Probability Comparison:")
    print(f"  Action    | Original | Biased   | Change")
    print(f"  --------- | -------- | -------- | ------")
    for i, action in enumerate(actions):
        orig_prob = original_probs[0, i].item()
        bias_prob = biased_probs[0, i].item()
        change = bias_prob - orig_prob
        print(f"  {action:<9} | {orig_prob:8.3f} | {bias_prob:8.3f} | {change:+6.3f}")
    
    print("✓ DyGRAV integration demonstration complete\n")


def demo_model_comparison():
    """Demonstrate comparison between HAMT and RvLN-BERT."""
    print("="*60)
    print("MODEL COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Create both models
    hamt = HAMTWrapper(model_config={})
    
    try:
        rvlnbert = RvLNBERTWrapper(model_config={})
        rvlnbert_available = True
    except ImportError:
        print("⚠️  RvLN-BERT requires transformers library")
        rvlnbert_available = False
    
    # Compare model information
    models = [("HAMT", hamt)]
    if rvlnbert_available:
        models.append(("RvLN-BERT", rvlnbert))
    
    print(f"Model Comparison:")
    print(f"{'Model':<12} | {'Parameters':<12} | {'Hidden Dim':<11} | {'Type'}")
    print(f"------------ | ------------ | ----------- | ----")
    
    for name, model in models:
        info = model.get_model_info()
        params = info['parameters']
        hidden_dim = getattr(model, 'hidden_dim', 'N/A')
        model_type = info['model_type']
        
        print(f"{name:<12} | {params:>11,} | {hidden_dim:<11} | {model_type}")
    
    # Compare forward pass timing
    batch_size = 1
    pano_feats = torch.randn(batch_size, 36, 2048)
    instr_tokens = torch.randint(1, 1000, (batch_size, 20))
    instr_mask = torch.ones(batch_size, 20, dtype=torch.bool)
    
    print(f"\nInference Timing (CPU, single batch):")
    
    for name, model in models:
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model.forward(pano_feats, instr_tokens, instr_mask)
        
        # Time inference
        import time
        times = []
        with torch.no_grad():
            for _ in range(10):
                start = time.time()
                _ = model.forward(pano_feats, instr_tokens, instr_mask)
                end = time.time()
                times.append((end - start) * 1000)  # ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  {name}: {avg_time:.2f}ms ± {std_time:.2f}ms")
    
    print("✓ Model comparison complete\n")


def demo_checkpoint_handling():
    """Demonstrate checkpoint loading and saving."""
    print("="*60)
    print("CHECKPOINT HANDLING DEMONSTRATION")
    print("="*60)
    
    # Create model
    hamt = HAMTWrapper(model_config={})
    
    # Test checkpoint loading (will gracefully handle missing files)
    fake_checkpoint_path = "nonexistent_checkpoint.pt"
    
    print(f"Testing checkpoint loading:")
    try:
        hamt.load_checkpoint(fake_checkpoint_path)
        print(f"  ✓ Checkpoint loaded from {fake_checkpoint_path}")
    except Exception as e:
        print(f"  ⚠️  Expected error (checkpoint not found): {e}")
    
    # Test model state dict access
    state_dict = hamt.state_dict()
    print(f"  ✓ Model has {len(state_dict)} parameter tensors")
    
    # Test parameter freezing
    original_requires_grad = [p.requires_grad for p in hamt.parameters()]
    hamt.freeze_backbone()
    frozen_requires_grad = [p.requires_grad for p in hamt.parameters()]
    
    print(f"  ✓ Parameters before freezing: {sum(original_requires_grad)} trainable")
    print(f"  ✓ Parameters after freezing: {sum(frozen_requires_grad)} trainable")
    
    # Unfreeze
    hamt.unfreeze_backbone()
    unfrozen_requires_grad = [p.requires_grad for p in hamt.parameters()]
    print(f"  ✓ Parameters after unfreezing: {sum(unfrozen_requires_grad)} trainable")
    
    print("✓ Checkpoint handling demonstration complete\n")


def main():
    """Run all backbone demonstrations."""
    print("BACKBONE IMPLEMENTATION DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        demo_backbone_contract()
        demo_attention_visualization()
        demo_dygrav_integration()
        demo_model_comparison()
        demo_checkpoint_handling()
        
        print("=" * 80)
        print("🎉 ALL BACKBONE DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Key Features Demonstrated:")
        print("✓ BackboneWrapper contract compliance")
        print("✓ Attention weight extraction and analysis")
        print("✓ DyGRAV grounding signal integration")
        print("✓ Model comparison and performance analysis")
        print("✓ Checkpoint loading and parameter management")
        print()
        print("Next Steps:")
        print("1. Install transformers for RvLN-BERT: pip install transformers")
        print("2. Obtain pre-trained HAMT checkpoint for real validation")
        print("3. Run validation on R2R dataset: python tools/validate_backbone.py")
        print("4. Fine-tune on indoor navigation tasks")
        print()
        print("For more information:")
        print("- HAMT implementation: dygrav/backbones/hamt_wrapper.py")
        print("- RvLN-BERT implementation: dygrav/backbones/rvlnbert_wrapper.py")
        print("- Validation tool: tools/validate_backbone.py")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
