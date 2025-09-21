"""
Learned gate model with Gumbel-Softmax for module selection.
Implements the cost-aware, budgeted gate from the specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import numpy as np


@dataclass
class ModuleConfig:
    """Configuration for an expert module."""
    name: str  # Module identifier
    budgets: List[float]  # Available budget levels
    latency_lut: Dict[float, float]  # Budget -> latency (ms)
    energy_lut: Dict[float, float]  # Budget -> energy (J)


@dataclass
class GateOutput:
    """Output from the gate model."""
    module: str  # Selected module name
    depth: float  # Depth/budget level
    module_probs: torch.Tensor  # Soft module probabilities (for training)
    module_idx: int  # Hard module index
    depth_idx: int  # Hard depth index
    cost: float  # Computed cost for this selection


class GateModel(nn.Module):
    """
    Learned gate model g_θ that selects modules and budgets.
    Uses Gumbel-Softmax for differentiable discrete selection.
    """
    
    def __init__(
        self,
        input_dim: int = 10,  # Size of feature vector u_t
        hidden_dim: int = 64,
        modules: Optional[List[str]] = None,
        depth_bins: int = 3,
        temperature: float = 1.0,
        cool_down_steps: int = 2
    ):
        super().__init__()
        
        # Module configuration
        self.modules = modules or ['none', 'micro_graph', 'ver_slice', 'llm_tools']
        self.num_modules = len(self.modules)
        self.depth_bins = depth_bins
        
        # MLP for gate decisions
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_modules + 1)  # +1 for depth
        )
        
        # Gumbel-Softmax temperature (will be annealed during training)
        self.register_buffer('temperature', torch.tensor(temperature))
        self.min_temperature = 0.3
        
        # Cool-down mechanism
        self.cool_down_steps = cool_down_steps
        self.steps_since_escalation = 0
        self.last_module_idx = 0  # 'none' module
        
        # Module configurations with cost LUTs
        self._init_module_configs()
    
    def _init_module_configs(self):
        """Initialize module configurations with cost look-up tables."""
        # These are example values - should be measured on actual hardware
        self.module_configs = {
            'none': ModuleConfig(
                name='none',
                budgets=[0.0],
                latency_lut={0.0: 0.0},
                energy_lut={0.0: 0.0}
            ),
            'micro_graph': ModuleConfig(
                name='micro_graph',
                budgets=[6.0, 12.0],  # K values
                latency_lut={6.0: 5.0, 12.0: 8.0},  # ms
                energy_lut={6.0: 0.01, 12.0: 0.02}  # J
            ),
            'ver_slice': ModuleConfig(
                name='ver_slice',
                budgets=[0.5, 1.0, 1.5],  # radius in meters
                latency_lut={0.5: 10.0, 1.0: 15.0, 1.5: 22.0},  # ms
                energy_lut={0.5: 0.02, 1.0: 0.03, 1.5: 0.05}  # J
            ),
            'llm_tools': ModuleConfig(
                name='llm_tools',
                budgets=[16.0, 48.0],  # token budgets
                latency_lut={16.0: 20.0, 48.0: 45.0},  # ms
                energy_lut={16.0: 0.05, 48.0: 0.12}  # J
            )
        }
    
    def forward(
        self,
        features: torch.Tensor,
        training: bool = False,
        force_module: Optional[int] = None
    ) -> GateOutput:
        """
        Forward pass of the gate model.
        
        Args:
            features: Input features u_t [batch_size, input_dim] or [input_dim]
            training: Whether in training mode (uses Gumbel-Softmax)
            force_module: Optional module index to force selection (for ablations)
            
        Returns:
            GateOutput with module selection and depth
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Pass through MLP
        gate_output = self.gate_mlp(features)
        
        # Split into module logits and depth
        module_logits = gate_output[:, :self.num_modules]
        depth_logit = gate_output[:, -1]
        
        # Apply cool-down if needed
        if not training and self.steps_since_escalation < self.cool_down_steps:
            if self.last_module_idx != 0:  # If we escalated before
                # Suppress non-none modules unless uncertainty remains very high
                module_logits[:, 1:] -= 5.0  # Strong suppression
        
        # Module selection
        if force_module is not None:
            # Force specific module (for ablations)
            module_idx = force_module
            module_probs = F.one_hot(torch.tensor(module_idx), self.num_modules).float()
        elif training:
            # Gumbel-Softmax for differentiable selection
            module_probs = self.gumbel_softmax(module_logits, self.temperature)
            module_idx = module_probs.argmax(dim=-1).item()
        else:
            # Hard argmax for inference
            module_probs = F.softmax(module_logits, dim=-1)
            module_idx = module_logits.argmax(dim=-1).item()
        
        # Depth selection (continuous in [0, 1], then discretized)
        depth_continuous = torch.sigmoid(depth_logit).item()
        depth_idx = self.discretize_depth(depth_continuous)
        
        # Get selected module name and budget
        module_name = self.modules[module_idx]
        module_config = self.module_configs[module_name]
        
        # Select actual budget value
        if len(module_config.budgets) > 1:
            budget_idx = min(depth_idx, len(module_config.budgets) - 1)
            budget = module_config.budgets[budget_idx]
        else:
            budget = module_config.budgets[0]
        
        # Compute cost
        cost = self.compute_cost(module_name, budget)
        
        # Update cool-down tracker
        if not training:
            if module_idx != 0 and self.last_module_idx == 0:
                # Escalation occurred
                self.steps_since_escalation = 0
            else:
                self.steps_since_escalation += 1
            self.last_module_idx = module_idx
        
        return GateOutput(
            module=module_name,
            depth=budget,
            module_probs=module_probs.squeeze(0),
            module_idx=module_idx,
            depth_idx=depth_idx,
            cost=cost
        )
    
    def gumbel_softmax(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """
        Gumbel-Softmax for differentiable discrete sampling.
        
        Args:
            logits: Module logits [batch_size, num_modules]
            temperature: Temperature parameter (lower = more discrete)
            
        Returns:
            Soft one-hot vectors [batch_size, num_modules]
        """
        # Sample Gumbel noise
        U = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(U + 1e-20) + 1e-20)
        
        # Add noise and apply softmax with temperature
        y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        # Straight-through estimator: use hard samples in forward, soft in backward
        if self.training:
            # Get hard samples
            _, max_idx = y.max(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y).scatter_(-1, max_idx, 1.0)
            
            # Use hard in forward, soft in backward
            y = (y_hard - y).detach() + y
        
        return y
    
    def discretize_depth(self, depth_continuous: float) -> int:
        """
        Discretize continuous depth to budget bins.
        
        Args:
            depth_continuous: Continuous depth in [0, 1]
            
        Returns:
            Discrete depth index
        """
        # Map [0, 1] to discrete bins
        depth_idx = int(depth_continuous * self.depth_bins)
        depth_idx = min(depth_idx, self.depth_bins - 1)
        return depth_idx
    
    def compute_cost(self, module: str, budget: float, w_ms: float = 1.0, w_j: float = 0.0) -> float:
        """
        Compute cost C(m_t, d_t) from LUTs.
        
        Args:
            module: Module name
            budget: Budget level
            w_ms: Weight for latency
            w_j: Weight for energy
            
        Returns:
            Weighted cost
        """
        config = self.module_configs[module]
        latency = config.latency_lut.get(budget, 0.0)
        energy = config.energy_lut.get(budget, 0.0)
        
        cost = w_ms * latency + w_j * energy
        return cost
    
    def anneal_temperature(self, step: int, total_steps: int):
        """
        Anneal Gumbel-Softmax temperature during training.
        
        Args:
            step: Current training step
            total_steps: Total training steps
        """
        # Linear annealing from initial temperature to min_temperature
        progress = min(step / total_steps, 1.0)
        new_temp = self.temperature * (1 - progress) + self.min_temperature * progress
        self.temperature.copy_(torch.tensor(new_temp))
    
    def reset_cool_down(self):
        """Reset cool-down state (call at episode boundaries)."""
        self.steps_since_escalation = 0
        self.last_module_idx = 0


class DualVariableOptimizer:
    """
    Dual variable optimizer for constrained optimization.
    Updates λ to enforce budget constraints.
    """
    
    def __init__(
        self,
        initial_lambda: float = 0.1,
        lr: float = 0.01,
        ema_decay: float = 0.9
    ):
        self.lambda_value = initial_lambda
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema_cost = 0.0
    
    def update(self, episode_cost: float, budget: float) -> float:
        """
        Update dual variable based on budget violation.
        
        Args:
            episode_cost: Total cost for the episode
            budget: Target budget B
            
        Returns:
            Updated lambda value
        """
        # Update EMA of costs
        self.ema_cost = self.ema_decay * self.ema_cost + (1 - self.ema_decay) * episode_cost
        
        # Dual ascent update
        violation = self.ema_cost - budget
        self.lambda_value = max(0.0, self.lambda_value + self.lr * violation)
        
        return self.lambda_value
    
    def get_lambda(self) -> float:
        """Get current lambda value."""
        return self.lambda_value


class CostAwareLoss(nn.Module):
    """
    Cost-aware loss function for gate training.
    L = E[Σ_t ℓ(a_t) + λ·C(m_t, d_t)]
    """
    
    def __init__(self, dual_optimizer: DualVariableOptimizer):
        super().__init__()
        self.dual_optimizer = dual_optimizer
    
    def forward(
        self,
        navigation_loss: torch.Tensor,
        gate_outputs: List[GateOutput],
        budget: float
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute cost-aware loss.
        
        Args:
            navigation_loss: Base navigation loss ℓ(a_t)
            gate_outputs: Gate outputs for each timestep
            budget: Target budget B
            
        Returns:
            Total loss and metrics dict
        """
        # Compute total cost for the episode
        total_cost = sum(g.cost for g in gate_outputs)
        
        # Get current lambda
        lambda_value = self.dual_optimizer.get_lambda()
        
        # Cost-aware loss
        cost_term = lambda_value * total_cost
        total_loss = navigation_loss + cost_term
        
        # Update dual variable
        new_lambda = self.dual_optimizer.update(total_cost, budget)
        
        metrics = {
            'navigation_loss': navigation_loss.item(),
            'cost_term': cost_term,
            'total_cost': total_cost,
            'lambda': new_lambda,
            'budget_violation': total_cost - budget
        }
        
        return total_loss, metrics
