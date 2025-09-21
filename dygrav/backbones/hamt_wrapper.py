"""HAMT (History Aware Multimodal Transformer) backbone wrapper for DyGRAV."""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackboneWrapper


class HAMTWrapper(BackboneWrapper):
    """
    HAMT (History Aware Multimodal Transformer) wrapper for DyGRAV.
    
    This wrapper implements the HAMT architecture for vision-language navigation,
    providing the standard BackboneWrapper interface while maintaining compatibility
    with pre-trained HAMT models.
    
    HAMT features:
    - Cross-modal attention between vision and language
    - History-aware navigation with recurrent memory
    - Multi-head attention for robust feature fusion
    - Action prediction with stop token support
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        vocab_size: int = 50000,
        feature_dim: int = 2048,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_actions: int = 4,  # forward, left, right, stop
        use_history: bool = True,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize HAMT wrapper.
        
        Args:
            model_config: Configuration dictionary
            vocab_size: Size of instruction vocabulary
            feature_dim: Dimension of visual features
            hidden_dim: Hidden dimension of transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout probability
            max_seq_len: Maximum sequence length
            num_actions: Number of navigation actions
            use_history: Whether to use history mechanism
            checkpoint_path: Path to pre-trained checkpoint
        """
        super().__init__(model_config)
        
        self.vocab_size = vocab_size
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.num_actions = num_actions
        self.use_history = use_history
        
        # Build HAMT model
        self._build_model()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Cache for attention weights
        self._last_attention = None
        self._last_output = None
    
    def _build_model(self):
        """Build the HAMT model architecture."""
        
        # Instruction encoder
        self.instruction_encoder = nn.Sequential(
            nn.Embedding(self.vocab_size, self.hidden_dim, padding_idx=0),
            nn.Dropout(self.dropout),
        )
        
        # Visual feature projection
        self.visual_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
        # Positional encoding for visual features (36 viewpoints)
        self.visual_pos_encoding = nn.Parameter(
            torch.randn(1, 36, self.hidden_dim) * 0.02
        )
        
        # Positional encoding for instructions
        self.register_buffer(
            'instruction_pos_encoding',
            self._create_positional_encoding(self.max_seq_len, self.hidden_dim)
        )
        
        # Cross-modal transformer layers
        self.transformer_layers = nn.ModuleList([
            HAMTTransformerLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])
        
        # History mechanism
        if self.use_history:
            self.history_encoder = nn.LSTM(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=self.dropout if self.num_layers > 1 else 0,
            )
        
        # Output heads
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_actions),
        )
        
        self.stop_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        
        # Layer normalization
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, max_len, d_model]
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        pano_feats: torch.FloatTensor,
        instr_tokens: torch.LongTensor,
        instr_mask: torch.BoolTensor,
        hist: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HAMT model.
        
        Args:
            pano_feats: Panoramic visual features [B, 36, D]
            instr_tokens: Instruction token IDs [B, L]
            instr_mask: Instruction attention mask [B, L]
            hist: Optional history dictionary
        
        Returns:
            Dictionary with model outputs
        """
        batch_size = pano_feats.size(0)
        
        # Encode instructions
        instr_embeds = self.instruction_encoder(instr_tokens)  # [B, L, H]
        seq_len = instr_embeds.size(1)
        
        # Add positional encoding to instructions
        pos_encoding = self.instruction_pos_encoding[:, :seq_len, :]
        instr_embeds = instr_embeds + pos_encoding
        
        # Project visual features
        visual_embeds = self.visual_projection(pano_feats)  # [B, 36, H]
        visual_embeds = visual_embeds + self.visual_pos_encoding
        
        # Create attention masks
        # Instruction mask: [B, L] -> [B, L, L]
        instr_attn_mask = self._create_attention_mask(instr_mask)
        
        # Visual mask (all viewpoints are valid)
        visual_mask = torch.ones(batch_size, 36, device=pano_feats.device, dtype=torch.bool)
        visual_attn_mask = self._create_attention_mask(visual_mask)
        
        # Cross-modal attention through transformer layers
        all_attention_weights = []
        
        for layer in self.transformer_layers:
            instr_embeds, visual_embeds, attn_weights = layer(
                instr_embeds, visual_embeds,
                instr_attn_mask, visual_attn_mask,
                instr_mask, visual_mask
            )
            all_attention_weights.append(attn_weights)
        
        # Global pooling of visual features
        visual_pooled = self._global_pool(visual_embeds, visual_mask)  # [B, H]
        
        # History integration
        if self.use_history and hist is not None:
            visual_pooled = self._integrate_history(visual_pooled, hist)
        
        # Generate outputs
        visual_pooled = self.output_norm(visual_pooled)
        
        action_logits = self.action_head(visual_pooled)  # [B, A]
        stop_logits = self.stop_head(visual_pooled)      # [B, 1]
        
        # Combine attention weights for visualization
        combined_attention = torch.stack(all_attention_weights, dim=1).mean(dim=1)  # [B, 36, L]
        
        # Cache for step() method
        self._last_attention = combined_attention
        self._last_output = {
            "action_logits": action_logits,
            "stop_logits": stop_logits,
            "visual_features": visual_pooled,
            "instruction_features": instr_embeds,
        }
        
        return {
            "logits": action_logits,
            "stop": stop_logits.squeeze(-1),
            "attn": combined_attention,
            "visual_features": visual_pooled,
            "instruction_features": instr_embeds,
        }
    
    def _create_attention_mask(self, mask: torch.BoolTensor) -> torch.FloatTensor:
        """Create attention mask for transformer."""
        batch_size, seq_len = mask.size()
        attn_mask = mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)
        attn_mask = attn_mask.float()
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)
        return attn_mask
    
    def _global_pool(self, features: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Global pooling with mask."""
        # Mask out invalid positions
        mask_expanded = mask.unsqueeze(-1).expand_as(features)
        features_masked = features * mask_expanded.float()
        
        # Average pooling
        pooled = features_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
        return pooled
    
    def _integrate_history(self, current_features: torch.Tensor, hist: Dict[str, Any]) -> torch.Tensor:
        """Integrate history information."""
        if "features" not in hist:
            return current_features
        
        # Get history features
        hist_features = hist["features"]  # [B, T, H]
        
        # Encode history with LSTM
        hist_encoded, _ = self.history_encoder(hist_features)  # [B, T, H]
        
        # Pool history
        hist_pooled = hist_encoded.mean(dim=1)  # [B, H]
        
        # Combine with current features
        combined = current_features + hist_pooled
        return combined
    
    def step(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Single step inference for policy integration.
        
        Args:
            obs: Observation dictionary
        
        Returns:
            Policy outputs with confidence and entropy
        """
        # Extract inputs from observation
        pano_feats = obs.get("pano_feats")
        instr_tokens = obs.get("instr_tokens") 
        instr_mask = obs.get("instr_mask")
        hist = obs.get("hist")
        
        # Handle missing inputs with defaults
        if pano_feats is None:
            # Create dummy panoramic features
            batch_size = 1
            pano_feats = torch.randn(batch_size, 36, self.feature_dim)
            
        if instr_tokens is None:
            # Create dummy instruction
            batch_size = pano_feats.size(0)
            instr_tokens = torch.ones(batch_size, 10, dtype=torch.long)
            instr_mask = torch.ones(batch_size, 10, dtype=torch.bool)
        
        # Ensure correct device
        device = next(self.parameters()).device
        pano_feats = pano_feats.to(device)
        instr_tokens = instr_tokens.to(device)
        instr_mask = instr_mask.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.forward(pano_feats, instr_tokens, instr_mask, hist)
        
        # Calculate confidence and entropy
        action_probs = F.softmax(outputs["logits"], dim=-1)
        confidence = action_probs.max(dim=-1).values.mean().item()
        
        # Calculate entropy
        log_probs = F.log_softmax(outputs["logits"], dim=-1)
        entropy = -(action_probs * log_probs).sum(dim=-1).mean().item()
        
        # Get current phrase (simplified)
        current_phrase = obs.get("current_phrase", "navigate forward")
        
        return {
            "logits": outputs["logits"],
            "confidence": confidence,
            "attn_entropy": entropy,
            "current_phrase": current_phrase,
            "stop_prob": torch.sigmoid(outputs["stop"]).item(),
            "attention_weights": outputs["attn"],
        }
    
    def bias(self, policy_out: Dict[str, Any], dygrav_signal: Any) -> Dict[str, Any]:
        """
        Apply DyGRAV bias to policy outputs.
        
        Args:
            policy_out: Original policy outputs
            dygrav_signal: DyGRAV grounding signal
        
        Returns:
            Modified policy outputs
        """
        if dygrav_signal is None or dygrav_signal.chosen_region is None:
            return policy_out
        
        # Apply bias based on chosen region
        logits = policy_out["logits"]
        
        # Simple bias: boost forward action when grounding is confident
        if hasattr(dygrav_signal, 'vlm_scores') and dygrav_signal.vlm_scores:
            max_vlm_score = max(score.score for score in dygrav_signal.vlm_scores)
            bias_strength = max_vlm_score * 0.5  # Scale bias by VLM confidence
            
            # Boost forward action (assuming index 0)
            bias_vector = torch.tensor([bias_strength, 0, 0, -bias_strength/2], 
                                     device=logits.device, dtype=logits.dtype)
            
            if len(logits.shape) > 1:
                bias_vector = bias_vector.unsqueeze(0).expand(logits.size(0), -1)
            
            policy_out["logits"] = logits + bias_vector
            policy_out["biased"] = True
            policy_out["bias_strength"] = bias_strength
        
        return policy_out
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict with strict=False to handle missing keys
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            print(f"✓ Loaded HAMT checkpoint from {checkpoint_path}")
            
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            raise
    
    def get_attention_weights(self, last_output: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract attention weights for visualization."""
        if self._last_attention is not None:
            return self._last_attention
        elif "attn" in last_output:
            return last_output["attn"]
        else:
            # Return dummy attention weights
            batch_size = last_output.get("logits", torch.tensor([[0]])).size(0)
            return torch.ones(batch_size, 36, 10)  # [B, 36, L]


class HAMTTransformerLayer(nn.Module):
    """Single transformer layer for HAMT with cross-modal attention."""
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Self-attention for instructions
        self.instr_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Self-attention for visual features
        self.visual_self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: visual attending to instructions
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Feed-forward networks
        self.instr_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        self.visual_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        
        # Layer normalization
        self.instr_norm1 = nn.LayerNorm(hidden_dim)
        self.instr_norm2 = nn.LayerNorm(hidden_dim)
        self.visual_norm1 = nn.LayerNorm(hidden_dim)
        self.visual_norm2 = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        instr_embeds: torch.Tensor,
        visual_embeds: torch.Tensor,
        instr_attn_mask: torch.Tensor,
        visual_attn_mask: torch.Tensor,
        instr_mask: torch.BoolTensor,
        visual_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through transformer layer."""
        
        # Instruction self-attention
        instr_residual = instr_embeds
        instr_embeds, _ = self.instr_self_attn(
            instr_embeds, instr_embeds, instr_embeds,
            key_padding_mask=~instr_mask,
        )
        instr_embeds = self.instr_norm1(instr_residual + self.dropout(instr_embeds))
        
        # Visual self-attention
        visual_residual = visual_embeds
        visual_embeds, _ = self.visual_self_attn(
            visual_embeds, visual_embeds, visual_embeds,
            key_padding_mask=~visual_mask,
        )
        visual_embeds = self.visual_norm1(visual_residual + self.dropout(visual_embeds))
        
        # Cross-attention: visual queries attending to instruction keys/values
        cross_residual = visual_embeds
        visual_cross, cross_attn_weights = self.cross_attn(
            visual_embeds, instr_embeds, instr_embeds,
            key_padding_mask=~instr_mask,
        )
        visual_embeds = self.cross_norm(cross_residual + self.dropout(visual_cross))
        
        # Feed-forward networks
        instr_residual = instr_embeds
        instr_embeds = self.instr_ffn(instr_embeds)
        instr_embeds = self.instr_norm2(instr_residual + self.dropout(instr_embeds))
        
        visual_residual = visual_embeds
        visual_embeds = self.visual_ffn(visual_embeds)
        visual_embeds = self.visual_norm2(visual_residual + self.dropout(visual_embeds))
        
        return instr_embeds, visual_embeds, cross_attn_weights
