"""
Enhanced policy with full gate model integration and cost tracking.
Implements the complete inference loop from the specification.
"""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from .types import DygravSignal, Region, VLMResult
from ..modules.ambiguity import AmbiguityDetector
from ..modules.vlm_query import VLMQuery
from ..modules.scene_graph import SceneGraphBuilder, GraphNode
from ..modules.fusion import DygravFusion
from ..modules.gate_model import GateModel, GateOutput
from ..modules.gate_features import (
    extract_gate_features,
    NoveltyTracker,
    SkillClassifier,
    RuleBasedSkillClassifier,
    compute_clip_margin
)
from ..modules.ver_slice import create_ver_slice_expert
from ..modules.llm_tools import create_llm_tools_expert
from ..utils.caching import ProposalCache, ScoreCache


@dataclass
class InferenceMetrics:
    """Metrics collected during inference."""
    step: int
    module_selected: str
    budget: float
    latency_ms: float
    energy_j: float
    total_cost: float
    entropy: float
    logit_gap: float
    clip_margin: float
    novelty: float
    skill_probs: List[float]
    action_taken: str
    trigger_reason: str


class EnhancedPolicyWithGate:
    """
    Enhanced policy implementing the full specification.
    Includes learned gate, all experts, and complete feature extraction.
    """
    
    def __init__(
        self,
        backbone,
        detector,
        gate_model: Optional[GateModel] = None,
        use_learned_gate: bool = True,
        skill_classifier_type: str = "rule",
        cache_cfg: Optional[Dict] = None,
        log_metrics: bool = True,
        metrics_path: Optional[Path] = None
    ):
        self.backbone = backbone
        self.detector = detector
        
        # Gate model (learned or rule-based fallback)
        self.use_learned_gate = use_learned_gate and gate_model is not None
        if self.use_learned_gate:
            self.gate_model = gate_model
        else:
            # Fallback to rule-based ambiguity detector
            self.amb_detector = AmbiguityDetector(
                tau_conf=0.55,
                tau_entropy=1.25,
                max_candidates=4
            )
        
        # Initialize experts
        self.vlm = VLMQuery(
            model_name="ViT-L-14",
            pretrained="openai",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.scene_graph = SceneGraphBuilder(
            k_cap=12,
            ttl_steps=5,
            enable_hypothesis=True
        )
        
        self.ver_slice_experts = {
            0.5: create_ver_slice_expert(0.5),
            1.0: create_ver_slice_expert(1.0),
            1.5: create_ver_slice_expert(1.5)
        }
        
        self.llm_experts = {
            16: create_llm_tools_expert(16),
            48: create_llm_tools_expert(48)
        }
        
        self.fusion = DygravFusion()
        
        # Feature extractors
        self.novelty_tracker = NoveltyTracker(memory_size=10)
        
        if skill_classifier_type == "neural":
            self.skill_classifier = SkillClassifier()
        else:
            self.skill_classifier = RuleBasedSkillClassifier()
        
        # Caching
        cache_cfg = cache_cfg or {}
        self.proposal_cache = ProposalCache(
            max_size=cache_cfg.get("proposal_cache_size", 1000)
        )
        self.score_cache = ScoreCache(
            max_size=cache_cfg.get("score_cache_size", 5000)
        )
        
        # Metrics logging
        self.log_metrics = log_metrics
        self.metrics_path = metrics_path or Path("inference_metrics.jsonl")
        self.step_count = 0
        self.episode_metrics = []
        
        # Cool-down tracking
        self.steps_since_escalation = 0
        self.cool_down_period = 2
    
    def step(self, obs: Dict) -> Dict:
        """
        Complete inference step following the specification.
        
        Algorithm:
        1. Perception: get z_t from π_0, compute features
        2. Gate: compute (m_t, d_t) = g_θ(u_t)
        3. If m_t = ∅: act with argmax p_t
        4. Else: call expert π_{m_t}, get â_t or b_t
        5. Execute a_t and log cost
        """
        start_time = time.time()
        self.step_count += 1
        
        # Step 1: Perception
        policy_out = self.backbone.step(obs)
        
        # Extract features for gate
        features = self._extract_features(obs, policy_out)
        
        # Step 2: Gate decision
        if self.use_learned_gate:
            gate_output = self._learned_gate_decision(features)
        else:
            gate_output = self._rule_based_gate_decision(features, obs)
        
        # Initialize metrics
        metrics = InferenceMetrics(
            step=self.step_count,
            module_selected=gate_output.module,
            budget=gate_output.depth,
            latency_ms=0.0,
            energy_j=0.0,
            total_cost=gate_output.cost,
            entropy=features['entropy'],
            logit_gap=features['logit_gap'],
            clip_margin=features['clip_margin'],
            novelty=features['novelty'],
            skill_probs=features['skill_probs'].tolist(),
            action_taken="",
            trigger_reason=""
        )
        
        # Step 3/4: Execute expert if triggered
        if gate_output.module != 'none':
            expert_start = time.time()
            
            expert_output = self._execute_expert(
                gate_output.module,
                gate_output.depth,
                obs,
                policy_out,
                features
            )
            
            expert_latency = (time.time() - expert_start) * 1000
            metrics.latency_ms = expert_latency
            
            # Apply expert output
            if expert_output:
                if 'action_override' in expert_output:
                    policy_out['action'] = expert_output['action_override']
                    metrics.trigger_reason = "action_override"
                elif 'bias' in expert_output:
                    # Apply bias to logits
                    if 'logits' in policy_out:
                        policy_out['logits'] = policy_out['logits'] + expert_output['bias']
                        policy_out['biased'] = True
                    metrics.trigger_reason = "logit_bias"
            
            # Update cool-down
            self.steps_since_escalation = 0
        else:
            self.steps_since_escalation += 1
        
        # Step 5: Determine final action
        if 'action' not in policy_out and 'logits' in policy_out:
            if policy_out['logits'].dim() > 1:
                probs = F.softmax(policy_out['logits'], dim=-1)
                action_idx = probs.argmax(dim=-1)
            else:
                probs = F.softmax(policy_out['logits'], dim=0)
                action_idx = probs.argmax()
            
            # Map to action (simplified)
            action_map = ['forward', 'turn_left', 'turn_right', 'stop', 'up', 'down']
            policy_out['action'] = action_map[action_idx % len(action_map)]
        
        metrics.action_taken = policy_out.get('action', 'none')
        
        # Log metrics
        total_latency = (time.time() - start_time) * 1000
        metrics.latency_ms = total_latency
        
        if self.log_metrics:
            self._log_metrics(metrics)
        
        # Add debug info
        policy_out['dygrav_triggered'] = gate_output.module != 'none'
        policy_out['dygrav_module'] = gate_output.module
        policy_out['dygrav_cost'] = gate_output.cost
        policy_out['inference_metrics'] = metrics
        
        return policy_out
    
    def _extract_features(self, obs: Dict, policy_out: Dict) -> Dict:
        """Extract all features needed for gate decision."""
        # Get logits and compute uncertainty features
        logits = policy_out.get('logits')
        if logits is not None:
            if isinstance(logits, torch.Tensor):
                probs = F.softmax(logits, dim=-1 if logits.dim() > 1 else 0)
                entropy = -(probs * probs.clamp(min=1e-9).log()).sum(dim=-1 if logits.dim() > 1 else 0)
                entropy = entropy.mean().item() if logits.dim() > 1 else entropy.item()
                
                # Logit gap
                top2 = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1 if logits.dim() > 1 else 0)
                if top2.values.shape[-1] >= 2:
                    logit_gap = (top2.values[..., 0] - top2.values[..., 1]).mean().item()
                else:
                    logit_gap = float('inf')
            else:
                entropy = 0.0
                logit_gap = 0.0
        else:
            entropy = 0.0
            logit_gap = 0.0
        
        # Get VLM scores and compute CLIP margin
        vlm_scores = policy_out.get('vlm_scores', [])
        clip_margin = compute_clip_margin(vlm_scores) if vlm_scores else 0.0
        
        # Visual novelty
        visual_features = policy_out.get('visual_features')
        if visual_features is not None:
            novelty = self.novelty_tracker.compute_novelty(visual_features)
            self.novelty_tracker.update_memory(visual_features)
        else:
            novelty = 0.5
        
        # Skill classification
        instruction = obs.get('instruction', '')
        if isinstance(self.skill_classifier, SkillClassifier):
            # Neural classifier needs tokenized input
            skill_probs = np.ones(5) * 0.5  # Placeholder
        else:
            skill_probs = self.skill_classifier.predict_skills(instruction)
        
        # Candidate count
        regions = obs.get('regions', [])
        candidate_count = len(regions)
        
        return {
            'entropy': entropy,
            'logit_gap': logit_gap,
            'clip_margin': clip_margin,
            'candidate_count': candidate_count,
            'novelty': novelty,
            'skill_probs': skill_probs,
            'confidence': policy_out.get('confidence', 1.0),
            'attn_entropy': policy_out.get('attn_entropy', 0.0)
        }
    
    def _learned_gate_decision(self, features: Dict) -> GateOutput:
        """Make gate decision using learned model."""
        # Convert features to tensor
        feature_tensor = torch.tensor([
            features['entropy'],
            features['logit_gap'],
            features['clip_margin'],
            float(features['candidate_count']),
            features['novelty'],
            *features['skill_probs'].tolist()
        ], dtype=torch.float32)
        
        # Apply cool-down if needed
        if self.steps_since_escalation < self.cool_down_period:
            # Suppress escalation during cool-down
            force_module = 0  # Force 'none'
        else:
            force_module = None
        
        # Get gate decision
        gate_output = self.gate_model(
            feature_tensor,
            training=False,
            force_module=force_module
        )
        
        return gate_output
    
    def _rule_based_gate_decision(self, features: Dict, obs: Dict) -> GateOutput:
        """Fallback rule-based gate decision."""
        # Use ambiguity detector
        regions = obs.get('regions', [])
        dec = self.amb_detector(
            features['confidence'],
            features['attn_entropy'],
            True,  # Assume attribute present
            regions
        )
        
        # Convert to GateOutput format
        if dec.trigger:
            # Simple heuristic for module selection based on skills
            skill_probs = features['skill_probs']
            
            if skill_probs[3] > 0.5:  # Vertical skill
                module = 'ver_slice'
                depth = 1.0
            elif skill_probs[4] > 0.5:  # Numeric skill
                module = 'llm_tools'
                depth = 16.0
            else:
                module = 'micro_graph'
                depth = 6.0
            
            # Estimate cost
            cost = depth * 10  # Simplified
        else:
            module = 'none'
            depth = 0.0
            cost = 0.0
        
        return GateOutput(
            module=module,
            depth=depth,
            module_probs=torch.zeros(4),  # Placeholder
            module_idx=0,
            depth_idx=0,
            cost=cost
        )
    
    def _execute_expert(
        self,
        module: str,
        budget: float,
        obs: Dict,
        policy_out: Dict,
        features: Dict
    ) -> Optional[Dict]:
        """Execute the selected expert module."""
        
        if module == 'micro_graph':
            # Execute Micro-Graph expert
            regions = obs.get('regions', [])
            vlm_scores = []
            
            # Get VLM scores for regions
            image = obs.get('rgb')
            if image is not None and regions:
                for region in regions[:int(budget)]:  # K-cap from budget
                    vlm_result = self.vlm.score(image, [region], obs.get('instruction', ''))
                    if vlm_result:
                        vlm_scores.extend(vlm_result)
            
            # Build scene graph
            clip_scores = [v.score for v in vlm_scores] if vlm_scores else None
            edges = self.scene_graph.build(
                regions,
                clip_scores=clip_scores,
                instruction_spans=obs.get('instruction', '').split()
            )
            
            # Get best node
            if self.scene_graph.current_graph and self.scene_graph.current_graph.nodes:
                best_node = max(
                    self.scene_graph.current_graph.nodes,
                    key=lambda n: n.clip_score
                )
                action = self.scene_graph.get_action_override(best_node, obs.get('instruction', ''))
                if action:
                    return {'action_override': action}
            
            return {'edges': edges}
        
        elif module == 'ver_slice':
            # Execute VER-slice expert
            if 'depth' in obs:
                # Select expert based on budget
                radius = min(budget, 1.5)
                if radius not in self.ver_slice_experts:
                    radius = 1.0
                
                expert = self.ver_slice_experts[radius]
                bias, analysis = expert(
                    obs['depth'],
                    obs.get('instruction', ''),
                    obs.get('rgb'),
                    radius=radius
                )
                
                return {'bias': bias, 'analysis': analysis}
        
        elif module == 'llm_tools':
            # Execute LLM-tools expert
            token_budget = int(budget)
            if token_budget not in self.llm_experts:
                token_budget = 48
            
            expert = self.llm_experts[token_budget]
            
            # Parse instruction for tool calls
            tool_calls = expert.parse_instruction(
                obs.get('instruction', ''),
                {'regions': obs.get('regions', [])}
            )
            
            # Execute tool calls
            results = []
            for call in tool_calls:
                if call.max_tokens <= token_budget:
                    result = expert.execute_tool(
                        call,
                        obs,
                        obs.get('regions', [])
                    )
                    results.append(result)
                    token_budget -= call.max_tokens
                    
                    if token_budget <= 0:
                        break
            
            # Generate action bias
            if results:
                bias = expert.generate_action_bias(results)
                return {'bias': bias, 'tool_results': results}
        
        return None
    
    def _log_metrics(self, metrics: InferenceMetrics):
        """Log metrics to file."""
        self.episode_metrics.append(metrics)
        
        # Write to JSONL file
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(asdict(metrics)) + '\n')
    
    def reset_episode(self):
        """Reset for new episode."""
        self.step_count = 0
        self.steps_since_escalation = 0
        self.scene_graph.reset()
        self.novelty_tracker = NoveltyTracker()  # Reset memory
        
        if self.use_learned_gate:
            self.gate_model.reset_cool_down()
        
        # Compute episode statistics
        if self.episode_metrics:
            total_cost = sum(m.total_cost for m in self.episode_metrics)
            avg_latency = np.mean([m.latency_ms for m in self.episode_metrics])
            trigger_rate = sum(1 for m in self.episode_metrics if m.module_selected != 'none') / len(self.episode_metrics)
            
            print(f"Episode stats: total_cost={total_cost:.2f}, avg_latency={avg_latency:.2f}ms, trigger_rate={trigger_rate:.2%}")
        
        self.episode_metrics = []
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "proposal_cache": self.proposal_cache.stats(),
            "score_cache": self.score_cache.stats(),
        }
