"""
Integration tests for the complete DyGRAV system.
Tests all components working together as specified.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json

from dygrav.core.enhanced_policy import EnhancedPolicyWithGate
from dygrav.core.types import Region
from dygrav.modules.gate_model import GateModel
from dygrav.modules.gate_features import NoveltyTracker, RuleBasedSkillClassifier
from dygrav.training.cost_aware_trainer import CostAwareNavModule, TrainingConfig
from dygrav.training.calibration import calibrate_model, compute_ece
from dygrav.detectors.yolo import SimpleDetector


class MockBackbone:
    """Mock backbone for testing."""
    
    def step(self, obs):
        batch_size = 1
        num_actions = 6
        
        # Generate random logits with some structure
        logits = torch.randn(num_actions)
        logits[0] += 1.0  # Bias towards first action
        
        # Compute derived features
        probs = torch.softmax(logits, dim=0)
        confidence = probs.max().item()
        entropy = -(probs * probs.clamp(min=1e-9).log()).sum().item()
        
        # Generate visual features
        visual_features = torch.randn(512)
        
        return {
            'logits': logits,
            'probs': probs,
            'confidence': confidence,
            'attn_entropy': entropy,
            'visual_features': visual_features,
            'current_phrase': 'go to the ceramic bowl'
        }
    
    def bias(self, policy_out, signal):
        if signal and hasattr(signal, 'bias'):
            policy_out['logits'] = policy_out['logits'] + signal.bias
            policy_out['biased'] = True
        return policy_out


def create_test_observation():
    """Create a test observation."""
    return {
        'rgb': np.array(Image.new('RGB', (640, 480), color=(128, 128, 128))),
        'depth': torch.randn(480, 640) * 5.0,  # Random depth map
        'instruction': 'Go to the ceramic bowl on the left table',
        'regions': [
            Region(100, 100, 200, 200, 'bowl', 0.9),
            Region(300, 100, 400, 200, 'bowl', 0.85),
            Region(200, 300, 350, 450, 'table', 0.95)
        ]
    }


def test_gate_model_initialization():
    """Test gate model can be initialized."""
    gate = GateModel(
        input_dim=10,
        hidden_dim=64,
        modules=['none', 'micro_graph', 'ver_slice', 'llm_tools'],
        depth_bins=3
    )
    
    assert gate is not None
    assert gate.num_modules == 4
    assert gate.temperature.item() == 1.0


def test_gate_forward_pass():
    """Test gate model forward pass."""
    gate = GateModel(input_dim=10, hidden_dim=64)
    
    # Create random features
    features = torch.randn(10)
    
    # Forward pass
    output = gate(features, training=False)
    
    assert output.module in ['none', 'micro_graph', 'ver_slice', 'llm_tools']
    assert 0 <= output.depth <= 100  # Some reasonable range
    assert output.cost >= 0


def test_novelty_tracker():
    """Test novelty tracking."""
    tracker = NoveltyTracker(memory_size=5)
    
    # First observation should be maximally novel
    embedding1 = torch.randn(512)
    novelty1 = tracker.compute_novelty(embedding1)
    assert novelty1 == 1.0
    
    # Add to memory
    tracker.update_memory(embedding1)
    
    # Same embedding should have low novelty
    novelty2 = tracker.compute_novelty(embedding1)
    assert novelty2 < 0.5
    
    # Very different embedding should have high novelty
    embedding3 = torch.randn(512) * 10
    novelty3 = tracker.compute_novelty(embedding3)
    assert novelty3 > 0.8


def test_skill_classifier():
    """Test skill classification."""
    classifier = RuleBasedSkillClassifier()
    
    # Test direction skill
    probs1 = classifier.predict_skills("Turn left at the door")
    assert probs1[0] > 0.5  # Direction skill
    
    # Test vertical skill
    probs2 = classifier.predict_skills("Go up the stairs")
    assert probs2[3] > 0.5  # Vertical skill
    
    # Test numeric skill
    probs3 = classifier.predict_skills("Go to the second door")
    assert probs3[4] > 0.5  # Numeric skill


def test_enhanced_policy_initialization():
    """Test enhanced policy can be initialized."""
    backbone = MockBackbone()
    detector = SimpleDetector()
    gate_model = GateModel(input_dim=10, hidden_dim=64)
    
    policy = EnhancedPolicyWithGate(
        backbone=backbone,
        detector=detector,
        gate_model=gate_model,
        use_learned_gate=True
    )
    
    assert policy is not None
    assert policy.use_learned_gate


def test_enhanced_policy_step():
    """Test full inference step."""
    backbone = MockBackbone()
    detector = SimpleDetector()
    gate_model = GateModel(input_dim=10, hidden_dim=64)
    
    policy = EnhancedPolicyWithGate(
        backbone=backbone,
        detector=detector,
        gate_model=gate_model,
        use_learned_gate=True,
        log_metrics=False  # Disable logging for test
    )
    
    # Create observation
    obs = create_test_observation()
    
    # Run inference
    output = policy.step(obs)
    
    # Check output structure
    assert 'action' in output
    assert 'dygrav_triggered' in output
    assert 'dygrav_module' in output
    assert 'dygrav_cost' in output
    assert 'inference_metrics' in output
    
    # Check metrics
    metrics = output['inference_metrics']
    assert metrics.step == 1
    assert metrics.module_selected in ['none', 'micro_graph', 'ver_slice', 'llm_tools']
    assert metrics.entropy >= 0
    assert metrics.logit_gap >= 0
    assert len(metrics.skill_probs) == 5


def test_cool_down_mechanism():
    """Test cool-down prevents rapid escalation."""
    backbone = MockBackbone()
    detector = SimpleDetector()
    gate_model = GateModel(input_dim=10, hidden_dim=64)
    
    policy = EnhancedPolicyWithGate(
        backbone=backbone,
        detector=detector,
        gate_model=gate_model,
        use_learned_gate=True,
        log_metrics=False
    )
    
    obs = create_test_observation()
    
    # Force an escalation by manipulating features
    # (In practice, would need to set up conditions for escalation)
    
    # First step might escalate
    output1 = policy.step(obs)
    escalated1 = output1['dygrav_triggered']
    
    if escalated1:
        # Next steps should be suppressed due to cool-down
        output2 = policy.step(obs)
        output3 = policy.step(obs)
        
        # At least one should be suppressed
        assert not output2['dygrav_triggered'] or not output3['dygrav_triggered']


def test_cost_tracking():
    """Test cost tracking across episode."""
    backbone = MockBackbone()
    detector = SimpleDetector()
    gate_model = GateModel(input_dim=10, hidden_dim=64)
    
    policy = EnhancedPolicyWithGate(
        backbone=backbone,
        detector=detector,
        gate_model=gate_model,
        use_learned_gate=True,
        log_metrics=False
    )
    
    obs = create_test_observation()
    
    # Run multiple steps
    total_cost = 0
    for _ in range(10):
        output = policy.step(obs)
        total_cost += output['dygrav_cost']
    
    # Cost should be bounded
    assert total_cost >= 0
    assert total_cost < 1000  # Reasonable upper bound


def test_scene_graph_with_hypothesis():
    """Test enhanced scene graph with hypothesis nodes."""
    from dygrav.modules.scene_graph import SceneGraphBuilder
    
    builder = SceneGraphBuilder(
        k_cap=6,
        ttl_steps=3,
        enable_hypothesis=True
    )
    
    # Create regions with ambiguous labels
    regions = [
        Region(100, 100, 200, 200, 'chair', 0.9),
        Region(300, 100, 400, 200, 'chair', 0.85),
        Region(200, 300, 350, 450, 'table', 0.95)
    ]
    
    clip_scores = [0.9, 0.85, 0.7]
    instruction_spans = ['left', 'chair']
    
    # Build graph
    edges = builder.build(
        regions,
        clip_scores=clip_scores,
        instruction_spans=instruction_spans
    )
    
    # Check graph was built
    assert len(edges) > 0
    assert builder.current_graph is not None
    assert len(builder.current_graph.nodes) <= builder.k_cap
    
    # Check TTL
    edges2 = builder.build(regions)  # Should reuse graph
    assert builder.steps_since_creation == 1


def test_ver_slice_expert():
    """Test VER-slice expert execution."""
    from dygrav.modules.ver_slice import create_ver_slice_expert
    
    expert = create_ver_slice_expert(budget=1.0)
    
    # Create test inputs
    depth_map = torch.randn(480, 640) * 5.0
    instruction = "Go up the stairs"
    
    # Execute expert
    bias, analysis = expert(depth_map, instruction)
    
    # Check outputs
    assert bias.shape[0] == 6  # Action space size
    assert hasattr(analysis, 'has_stairs')
    assert hasattr(analysis, 'navigable_regions')


def test_llm_tools_expert():
    """Test LLM-tools expert execution."""
    from dygrav.modules.llm_tools import create_llm_tools_expert
    
    expert = create_llm_tools_expert(token_budget=16)
    
    # Parse instruction
    instruction = "Count the doors on the left"
    context = {'regions': [
        Region(100, 100, 200, 300, 'door', 0.9),
        Region(50, 100, 150, 300, 'door', 0.85)
    ]}
    
    tool_calls = expert.parse_instruction(instruction, context)
    
    # Should identify count tool
    assert len(tool_calls) > 0
    assert any(call.tool.value == 'count' for call in tool_calls)
    
    # Execute tool
    if tool_calls:
        result = expert.execute_tool(
            tool_calls[0],
            context,
            context['regions']
        )
        
        assert result.success
        assert result.fallback_used  # Since no LLM configured


def test_training_module():
    """Test cost-aware training module initialization."""
    config = TrainingConfig(
        backbone_type="tiny",
        num_epochs=10,
        target_budget=50.0
    )
    
    module = CostAwareNavModule(config)
    
    assert module is not None
    assert module.gate_model is not None
    assert module.dual_optimizer is not None
    
    # Test forward pass
    obs = {
        'rgb': torch.randn(3, 480, 640),
        'instruction': 'test instruction',
        'depth': torch.randn(480, 640),
        'regions': []
    }
    
    policy_out, gate_out = module.forward(obs)
    
    assert 'logits' in policy_out
    assert gate_out.module in ['none', 'micro_graph', 'ver_slice', 'llm_tools']


def test_calibration():
    """Test calibration computation."""
    from dygrav.training.calibration import compute_ece, TemperatureScaling
    
    # Create synthetic predictions
    n_samples = 100
    n_classes = 6
    
    logits = torch.randn(n_samples, n_classes)
    labels = torch.randint(0, n_classes, (n_samples,))
    
    # Compute ECE before calibration
    probs = torch.softmax(logits, dim=-1)
    metrics_before = compute_ece(probs, labels, n_bins=10)
    
    assert 0 <= metrics_before.ece <= 1
    assert 0 <= metrics_before.mce <= 1
    
    # Apply temperature scaling
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(logits, labels)
    
    # Compute ECE after calibration
    calibrated_logits = logits / optimal_temp
    calibrated_probs = torch.softmax(calibrated_logits, dim=-1)
    metrics_after = compute_ece(calibrated_probs, labels, n_bins=10)
    
    # ECE should not increase (ideally decrease)
    assert metrics_after.ece <= metrics_before.ece + 0.1


def test_full_episode_simulation():
    """Test full episode simulation with all components."""
    # Initialize policy with all components
    backbone = MockBackbone()
    detector = SimpleDetector()
    gate_model = GateModel(input_dim=10, hidden_dim=64)
    
    policy = EnhancedPolicyWithGate(
        backbone=backbone,
        detector=detector,
        gate_model=gate_model,
        use_learned_gate=True,
        log_metrics=False
    )
    
    # Simulate episode
    episode_length = 20
    observations = [create_test_observation() for _ in range(episode_length)]
    
    actions = []
    total_cost = 0
    triggers = 0
    
    for obs in observations:
        output = policy.step(obs)
        actions.append(output['action'])
        total_cost += output['dygrav_cost']
        if output['dygrav_triggered']:
            triggers += 1
    
    # Reset for new episode
    policy.reset_episode()
    
    # Check results
    assert len(actions) == episode_length
    assert all(a in ['forward', 'turn_left', 'turn_right', 'stop', 'up', 'down'] for a in actions)
    assert total_cost >= 0
    assert 0 <= triggers <= episode_length
    
    # Trigger rate should be reasonable
    trigger_rate = triggers / episode_length
    assert 0 <= trigger_rate <= 0.5  # Should not trigger too often


def test_budget_constraint():
    """Test that budget constraints are respected."""
    gate_model = GateModel(input_dim=10, hidden_dim=64)
    
    # Track costs over many samples
    costs = []
    for _ in range(100):
        features = torch.randn(10)
        output = gate_model(features, training=False)
        costs.append(output.cost)
    
    # Average cost should be reasonable
    avg_cost = np.mean(costs)
    assert 0 <= avg_cost <= 50  # Within expected range
    
    # Check module-specific budgets
    for module_name, config in gate_model.module_configs.items():
        assert all(b >= 0 for b in config.budgets)
        assert all(l >= 0 for l in config.latency_lut.values())


if __name__ == "__main__":
    # Run all tests
    test_gate_model_initialization()
    test_gate_forward_pass()
    test_novelty_tracker()
    test_skill_classifier()
    test_enhanced_policy_initialization()
    test_enhanced_policy_step()
    test_cool_down_mechanism()
    test_cost_tracking()
    test_scene_graph_with_hypothesis()
    test_ver_slice_expert()
    test_llm_tools_expert()
    test_training_module()
    test_calibration()
    test_full_episode_simulation()
    test_budget_constraint()
    
    print("All integration tests passed!")
