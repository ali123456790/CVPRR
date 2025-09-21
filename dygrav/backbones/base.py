"""Base backbone wrapper contract for VLN models."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class BackboneWrapper(nn.Module, ABC):
    """
    Abstract base class for VLN backbone wrappers.
    
    This class defines the contract that all backbone wrappers must implement
    to work with the DyGRAV system. The contract ensures consistent interfaces
    while allowing different backbone architectures (HAMT, RvLN-BERT, etc.).
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize backbone wrapper.
        
        Args:
            model_config: Configuration dictionary for the backbone model
        """
        super().__init__()
        self.model_config = model_config
        self.model = None
        
    @abstractmethod
    def forward(
        self,
        pano_feats: torch.FloatTensor,
        instr_tokens: torch.LongTensor,
        instr_mask: torch.BoolTensor,
        hist: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the backbone model.
        
        Args:
            pano_feats: Panoramic visual features [B, 36, D]
            instr_tokens: Instruction token IDs [B, L]
            instr_mask: Instruction attention mask [B, L]
            hist: Optional history/state dictionary
        
        Returns:
            Dictionary containing:
                - "logits": Action logits [B, A] where A is number of actions
                - "stop": Stop probability [B]
                - "attn": Attention weights for visualization
                - Other model-specific outputs
        """
        pass
    
    @abstractmethod
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single step inference for policy integration.
        
        Args:
            obs: Observation dictionary containing visual and textual inputs
        
        Returns:
            Dictionary with policy outputs including confidence and entropy
        """
        pass
    
    @abstractmethod
    def bias(self, policy_out: Dict[str, Any], dygrav_signal: Any) -> Dict[str, Any]:
        """
        Apply DyGRAV bias to policy outputs.
        
        Args:
            policy_out: Original policy outputs
            dygrav_signal: DyGRAV grounding signal
        
        Returns:
            Modified policy outputs with DyGRAV bias applied
        """
        pass
    
    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        pass
    
    @abstractmethod
    def get_attention_weights(self, last_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract attention weights for visualization.
        
        Args:
            last_output: Output from last forward pass
        
        Returns:
            Attention weights tensor for heatmap visualization
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        return {
            "model_type": self.__class__.__name__,
            "config": self.model_config,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
    
    def freeze_backbone(self) -> None:
        """Freeze backbone parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
