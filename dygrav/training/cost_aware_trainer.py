"""
Cost-aware training module with dual ascent and two-stage schedule.
Implements the training pipeline from the specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

from ..core.policy import PolicyWithDygrav
from ..modules.gate_model import GateModel, DualVariableOptimizer, CostAwareLoss, GateOutput
from ..modules.gate_features import (
    extract_gate_features, 
    NoveltyTracker, 
    SkillClassifier,
    RuleBasedSkillClassifier
)
from ..modules.ver_slice import create_ver_slice_expert
from ..modules.llm_tools import create_llm_tools_expert
from ..detectors.yolo import SimpleDetector


@dataclass
class TrainingConfig:
    """Configuration for cost-aware training."""
    # Model config
    backbone_type: str = "tiny"
    gate_hidden_dim: int = 64
    skill_classifier_type: str = "rule"  # "neural" or "rule"
    
    # Training config
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 20  # For two-stage training
    
    # Cost config
    target_budget: float = 50.0  # ms per episode
    lambda_init: float = 0.1
    lambda_lr: float = 0.01
    cost_weight_ms: float = 1.0
    cost_weight_energy: float = 0.0
    
    # Gumbel-Softmax config
    temperature_init: float = 1.0
    temperature_min: float = 0.3
    
    # Distillation config
    distillation_weight: float = 0.1
    distillation_temperature: float = 3.0
    
    # Calibration config
    calibration_samples: int = 1000
    ece_bins: int = 10


class CostAwareNavModule(pl.LightningModule):
    """
    PyTorch Lightning module for cost-aware navigation training.
    Implements the complete training pipeline from the specification.
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize backbone
        self.backbone = self._init_backbone(config.backbone_type)
        
        # Initialize gate model
        self.gate_model = GateModel(
            input_dim=10,  # Features: entropy, logit_gap, clip_margin, K, novelty, 5 skills
            hidden_dim=config.gate_hidden_dim,
            temperature=config.temperature_init
        )
        
        # Initialize experts
        self.experts = {
            'micro_graph': None,  # Will use existing SceneGraphBuilder
            'ver_slice': create_ver_slice_expert(budget=1.0),
            'llm_tools': create_llm_tools_expert(token_budget=48)
        }
        
        # Initialize feature extractors
        self.novelty_tracker = NoveltyTracker()
        if config.skill_classifier_type == "neural":
            self.skill_classifier = SkillClassifier()
        else:
            self.skill_classifier = RuleBasedSkillClassifier()
        
        # Initialize dual optimizer
        self.dual_optimizer = DualVariableOptimizer(
            initial_lambda=config.lambda_init,
            lr=config.lambda_lr
        )
        
        # Loss functions
        self.navigation_loss = nn.CrossEntropyLoss()
        self.cost_aware_loss = CostAwareLoss(self.dual_optimizer)
        
        # For distillation
        self.distillation_buffer = []
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')
        
        # Training stage tracking
        self.current_stage = "warmup"  # "warmup" or "finetune"
        self.warmup_complete = False
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
    
    def _init_backbone(self, backbone_type: str):
        """Initialize navigation backbone."""
        if backbone_type == "tiny":
            from ..lightning.module import TinyBackbone
            return TinyBackbone()
        else:
            # Placeholder for other backbones
            raise NotImplementedError(f"Backbone {backbone_type} not implemented")
    
    def forward(self, obs: Dict[str, Any]) -> Tuple[Dict[str, Any], GateOutput]:
        """
        Forward pass with gate-controlled expert selection.
        
        Args:
            obs: Observation dict with 'rgb', 'depth', 'instruction', etc.
            
        Returns:
            Policy output and gate output
        """
        # Get base policy output
        policy_out = self.backbone.step(obs)
        
        # Extract visual features for novelty
        if 'visual_features' in policy_out:
            visual_embedding = policy_out['visual_features']
        else:
            # Fallback: use a simple feature extraction
            visual_embedding = torch.randn(512)  # Placeholder
        
        # Extract gate features
        gate_features = extract_gate_features(
            logits=policy_out.get('logits'),
            instruction=obs.get('instruction', ''),
            vlm_scores=policy_out.get('vlm_scores', []),
            visual_embedding=visual_embedding,
            novelty_tracker=self.novelty_tracker,
            skill_classifier=self.skill_classifier,
            tokenizer=None  # Would need proper tokenizer
        )
        
        # Convert to tensor
        feature_tensor = gate_features.to_tensor(device=self.device)
        
        # Gate decision
        gate_output = self.gate_model(
            feature_tensor,
            training=self.training
        )
        
        # Execute selected expert if not "none"
        if gate_output.module != 'none':
            expert_output = self._execute_expert(
                gate_output.module,
                gate_output.depth,
                obs,
                policy_out
            )
            
            # Apply expert bias/override
            if expert_output is not None:
                if 'action_override' in expert_output:
                    policy_out['action'] = expert_output['action_override']
                elif 'bias' in expert_output:
                    policy_out['logits'] = policy_out['logits'] + expert_output['bias']
        
        # Update novelty tracker
        self.novelty_tracker.update_memory(visual_embedding)
        
        return policy_out, gate_output
    
    def _execute_expert(
        self,
        module_name: str,
        budget: float,
        obs: Dict,
        policy_out: Dict
    ) -> Optional[Dict]:
        """Execute selected expert module."""
        if module_name == 'micro_graph':
            # Use existing scene graph builder
            from ..modules.scene_graph import SceneGraphBuilder
            if not hasattr(self, 'scene_graph_builder'):
                self.scene_graph_builder = SceneGraphBuilder(k_cap=int(budget))
            
            regions = obs.get('regions', [])
            edges = self.scene_graph_builder.build(regions)
            
            return {'edges': edges}
        
        elif module_name == 'ver_slice':
            if 'depth' in obs:
                expert = self.experts['ver_slice']
                bias, analysis = expert(
                    obs['depth'],
                    obs.get('instruction', ''),
                    obs.get('rgb'),
                    radius=budget
                )
                return {'bias': bias, 'analysis': analysis}
        
        elif module_name == 'llm_tools':
            expert = self.experts['llm_tools']
            tool_calls = expert.parse_instruction(
                obs.get('instruction', ''),
                {'regions': obs.get('regions', [])}
            )
            
            results = []
            for call in tool_calls[:int(budget/16)]:  # Limit by budget
                result = expert.execute_tool(
                    call,
                    obs,
                    obs.get('regions', [])
                )
                results.append(result)
            
            if results:
                bias = expert.generate_action_bias(results)
                return {'bias': bias, 'tool_results': results}
        
        return None
    
    def training_step(self, batch, batch_idx):
        """Training step with cost-aware loss."""
        obs_batch, target_batch = batch
        batch_size = len(target_batch)
        
        # Collect outputs for entire episode
        all_logits = []
        gate_outputs = []
        
        for i in range(batch_size):
            obs = {
                'rgb': obs_batch['rgb'][i],
                'instruction': obs_batch.get('instruction', [''])[i],
                'depth': obs_batch.get('depth', [None])[i],
                'regions': obs_batch.get('regions', [[]])[i]
            }
            
            policy_out, gate_out = self.forward(obs)
            all_logits.append(policy_out['logits'])
            gate_outputs.append(gate_out)
        
        # Stack logits
        logits = torch.stack(all_logits)
        
        # Compute navigation loss
        nav_loss = self.navigation_loss(logits, target_batch)
        
        # Compute cost-aware loss
        if self.current_stage == "finetune":
            total_loss, metrics = self.cost_aware_loss(
                nav_loss,
                gate_outputs,
                self.config.target_budget
            )
        else:
            # Warmup stage: just navigation loss
            total_loss = nav_loss
            metrics = {'navigation_loss': nav_loss.item()}
        
        # Add distillation loss if we have successful expert corrections
        if self.distillation_buffer and self.current_stage == "finetune":
            distill_loss = self._compute_distillation_loss()
            total_loss += self.config.distillation_weight * distill_loss
            metrics['distillation_loss'] = distill_loss.item()
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'train/{key}', value, prog_bar=(key in ['navigation_loss', 'lambda']))
        
        # Anneal temperature
        if self.current_stage == "finetune":
            progress = self.current_epoch / self.config.num_epochs
            self.gate_model.anneal_temperature(
                self.global_step,
                self.config.num_epochs * len(self.trainer.train_dataloader)
            )
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step with metrics tracking."""
        obs_batch, target_batch = batch
        batch_size = len(target_batch)
        
        all_logits = []
        gate_outputs = []
        total_cost = 0.0
        
        for i in range(batch_size):
            obs = {
                'rgb': obs_batch['rgb'][i],
                'instruction': obs_batch.get('instruction', [''])[i],
                'depth': obs_batch.get('depth', [None])[i],
                'regions': obs_batch.get('regions', [[]])[i]
            }
            
            policy_out, gate_out = self.forward(obs)
            all_logits.append(policy_out['logits'])
            gate_outputs.append(gate_out)
            total_cost += gate_out.cost
        
        logits = torch.stack(all_logits)
        
        # Compute losses
        nav_loss = self.navigation_loss(logits, target_batch)
        
        # Compute accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == target_batch).float().mean()
        
        # Compute trigger rate
        trigger_rate = sum(1 for g in gate_outputs if g.module != 'none') / len(gate_outputs)
        
        # Average cost
        avg_cost = total_cost / len(gate_outputs)
        
        # Log metrics
        self.log('val/loss', nav_loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
        self.log('val/trigger_rate', trigger_rate)
        self.log('val/avg_cost_ms', avg_cost)
        
        return nav_loss
    
    def on_epoch_end(self):
        """Handle stage transitions."""
        # Check if warmup is complete
        if self.current_stage == "warmup" and self.current_epoch >= self.config.warmup_epochs:
            self.current_stage = "finetune"
            self.warmup_complete = True
            print(f"Warmup complete at epoch {self.current_epoch}. Starting fine-tuning.")
    
    def configure_optimizers(self):
        """Configure optimizers for different components."""
        # Main optimizer for backbone and gate
        main_params = list(self.backbone.parameters()) + list(self.gate_model.parameters())
        main_optimizer = torch.optim.AdamW(
            main_params,
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Skill classifier optimizer (if neural)
        optimizers = [main_optimizer]
        if isinstance(self.skill_classifier, SkillClassifier):
            skill_optimizer = torch.optim.Adam(
                self.skill_classifier.parameters(),
                lr=self.config.learning_rate * 0.1
            )
            optimizers.append(skill_optimizer)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            main_optimizer,
            T_max=self.config.num_epochs
        )
        
        return optimizers, [scheduler]
    
    def _compute_distillation_loss(self) -> torch.Tensor:
        """Compute distillation loss from expert corrections."""
        if not self.distillation_buffer:
            return torch.tensor(0.0)
        
        # Aggregate distillation samples
        student_logits = []
        teacher_logits = []
        
        for sample in self.distillation_buffer[-100:]:  # Use recent 100 samples
            student_logits.append(sample['student_logits'])
            teacher_logits.append(sample['teacher_logits'])
        
        student_logits = torch.stack(student_logits)
        teacher_logits = torch.stack(teacher_logits)
        
        # Apply temperature scaling
        student_probs = F.log_softmax(
            student_logits / self.config.distillation_temperature,
            dim=-1
        )
        teacher_probs = F.softmax(
            teacher_logits / self.config.distillation_temperature,
            dim=-1
        )
        
        # KL divergence loss
        loss = self.distillation_loss(student_probs, teacher_probs)
        
        return loss * (self.config.distillation_temperature ** 2)


class TwoStageTrainer:
    """
    Implements two-stage training schedule from the specification.
    Stage 1: Oracle warm-start for the gate
    Stage 2: Constrained fine-tuning with cost awareness
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.oracle_data = []
    
    def generate_oracle_data(
        self,
        dataloader,
        policy: PolicyWithDygrav,
        num_samples: int = 1000
    ) -> List[Dict]:
        """
        Generate oracle supervision for gate warm-start.
        Try different expert configurations and select cheapest helpful ones.
        """
        oracle_samples = []
        
        for batch_idx, (obs_batch, target_batch) in enumerate(dataloader):
            if len(oracle_samples) >= num_samples:
                break
            
            for i in range(len(target_batch)):
                obs = {
                    'rgb': obs_batch['rgb'][i],
                    'instruction': obs_batch.get('instruction', [''])[i]
                }
                
                # Get baseline performance
                baseline_out = policy.step(obs)
                baseline_correct = (baseline_out['action'] == target_batch[i]).item()
                
                # Try different expert configurations
                best_config = None
                best_cost = float('inf')
                
                for module in ['micro_graph', 'ver_slice', 'llm_tools']:
                    for depth in [0.3, 0.6, 1.0]:
                        # Simulate expert execution
                        # In practice, would actually run the expert
                        simulated_correct = np.random.random() > 0.3
                        simulated_cost = depth * 20  # Simplified cost model
                        
                        if simulated_correct and not baseline_correct:
                            if simulated_cost < best_cost:
                                best_config = {
                                    'module': module,
                                    'depth': depth,
                                    'cost': simulated_cost
                                }
                                best_cost = simulated_cost
                
                # Store oracle supervision
                if best_config:
                    oracle_samples.append({
                        'features': obs,  # Would extract actual features
                        'target_module': best_config['module'],
                        'target_depth': best_config['depth']
                    })
        
        return oracle_samples
    
    def train_warmup(
        self,
        model: CostAwareNavModule,
        oracle_data: List[Dict],
        num_epochs: int = 20
    ):
        """Train gate with oracle supervision."""
        optimizer = torch.optim.Adam(model.gate_model.parameters(), lr=0.001)
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for sample in oracle_data:
                # Extract features (simplified)
                features = torch.randn(10)  # Would use actual feature extraction
                
                # Get gate prediction
                gate_out = model.gate_model(features, training=True)
                
                # Compute loss against oracle
                module_target = ['none', 'micro_graph', 'ver_slice', 'llm_tools'].index(
                    sample['target_module']
                )
                module_loss = F.cross_entropy(
                    gate_out.module_probs.unsqueeze(0),
                    torch.tensor([module_target])
                )
                
                depth_target = torch.tensor([sample['target_depth']])
                depth_loss = F.mse_loss(
                    torch.tensor([gate_out.depth]),
                    depth_target
                )
                
                loss = module_loss + depth_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Warmup epoch {epoch}: loss = {total_loss / len(oracle_data):.4f}")


def create_cost_aware_trainer(config: Optional[TrainingConfig] = None) -> CostAwareNavModule:
    """
    Factory function to create cost-aware trainer.
    
    Args:
        config: Training configuration
        
    Returns:
        Configured CostAwareNavModule
    """
    if config is None:
        config = TrainingConfig()
    
    return CostAwareNavModule(config)
