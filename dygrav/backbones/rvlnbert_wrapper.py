"""RvLN-BERT (Recurrent VLN-BERT) backbone wrapper for DyGRAV."""

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BackboneWrapper

try:
    from transformers import BertModel, BertConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class RvLNBERTWrapper(BackboneWrapper):
    """
    RvLN-BERT (Recurrent VLN-BERT) wrapper for DyGRAV.
    
    This wrapper implements a BERT-based architecture for vision-language navigation,
    combining pre-trained BERT with visual processing and recurrent memory for
    navigation history.
    
    RvLN-BERT features:
    - Pre-trained BERT encoder for instruction understanding
    - Visual-textual cross-modal fusion
    - Recurrent memory for navigation history
    - Lightweight alternative to full transformer architectures
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        bert_model_name: str = "bert-base-uncased",
        feature_dim: int = 2048,
        hidden_dim: int = 768,
        num_recurrent_layers: int = 2,
        dropout: float = 0.1,
        num_actions: int = 4,
        freeze_bert: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize RvLN-BERT wrapper.
        
        Args:
            model_config: Configuration dictionary
            bert_model_name: Pre-trained BERT model name
            feature_dim: Dimension of visual features
            hidden_dim: Hidden dimension (should match BERT)
            num_recurrent_layers: Number of LSTM layers for history
            dropout: Dropout probability
            num_actions: Number of navigation actions
            freeze_bert: Whether to freeze BERT parameters
            checkpoint_path: Path to pre-trained checkpoint
        """
        super().__init__(model_config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for RvLN-BERT. Install with: pip install transformers")
        
        self.bert_model_name = bert_model_name
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_recurrent_layers = num_recurrent_layers
        self.dropout = dropout
        self.num_actions = num_actions
        self.freeze_bert = freeze_bert
        
        # Build model
        self._build_model()
        
        # Freeze BERT if requested
        if self.freeze_bert:
            self._freeze_bert()
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        # Cache for attention weights
        self._last_attention = None
        self._last_output = None
    
    def _build_model(self):
        """Build the RvLN-BERT model architecture."""
        
        # BERT encoder for instructions
        bert_config = BertConfig.from_pretrained(self.bert_model_name)
        self.bert = BertModel.from_pretrained(self.bert_model_name, config=bert_config)
        
        # Ensure hidden dimensions match
        bert_hidden_size = self.bert.config.hidden_size
        if self.hidden_dim != bert_hidden_size:
            print(f"Warning: Adjusting hidden_dim from {self.hidden_dim} to {bert_hidden_size} to match BERT")
            self.hidden_dim = bert_hidden_size
        
        # Visual feature processing
        self.visual_projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Positional encoding for visual features (36 viewpoints)
        self.visual_pos_encoding = nn.Parameter(
            torch.randn(1, 36, self.hidden_dim) * 0.02
        )
        
        # Cross-modal fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            dropout=self.dropout,
            batch_first=True,
        )
        
        # Visual-textual fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_dim),
        )
        
        # Recurrent memory for navigation history
        self.history_lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim // 2,
            num_layers=self.num_recurrent_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout if self.num_recurrent_layers > 1 else 0,
        )
        
        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_actions),
        )
        
        # Stop prediction head
        self.stop_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        
        # Output normalization
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        
        # Initialize non-BERT weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights (excluding BERT)."""
        for module in [self.visual_projection, self.fusion_layer, 
                      self.action_head, self.stop_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0)
    
    def _freeze_bert(self):
        """Freeze BERT parameters."""
        for param in self.bert.parameters():
            param.requires_grad = False
        print("✓ BERT parameters frozen")
    
    def forward(
        self,
        pano_feats: torch.FloatTensor,
        instr_tokens: torch.LongTensor,
        instr_mask: torch.BoolTensor,
        hist: Optional[Dict[str, Any]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through RvLN-BERT model.
        
        Args:
            pano_feats: Panoramic visual features [B, 36, D]
            instr_tokens: Instruction token IDs [B, L]
            instr_mask: Instruction attention mask [B, L]
            hist: Optional history dictionary
        
        Returns:
            Dictionary with model outputs
        """
        batch_size = pano_feats.size(0)
        
        # Encode instructions with BERT
        bert_outputs = self.bert(
            input_ids=instr_tokens,
            attention_mask=instr_mask.long(),
            return_dict=True,
        )
        instr_features = bert_outputs.last_hidden_state  # [B, L, H]
        instr_pooled = bert_outputs.pooler_output        # [B, H]
        
        # Process visual features
        visual_features = self.visual_projection(pano_feats)  # [B, 36, H]
        visual_features = visual_features + self.visual_pos_encoding
        
        # Cross-modal attention: visual attending to instructions
        visual_attended, cross_attn_weights = self.cross_modal_attention(
            query=visual_features,
            key=instr_features,
            value=instr_features,
            key_padding_mask=~instr_mask,
        )
        
        # Pool visual features
        visual_pooled = visual_attended.mean(dim=1)  # [B, H]
        
        # Fuse visual and textual features
        fused_features = torch.cat([visual_pooled, instr_pooled], dim=-1)  # [B, 2H]
        fused_features = self.fusion_layer(fused_features)  # [B, H]
        
        # Integrate history if available
        if hist is not None and "features" in hist:
            fused_features = self._integrate_history(fused_features, hist)
        
        # Generate outputs
        fused_features = self.output_norm(fused_features)
        
        action_logits = self.action_head(fused_features)  # [B, A]
        stop_logits = self.stop_head(fused_features)      # [B, 1]
        
        # Cache for step() method
        self._last_attention = cross_attn_weights
        self._last_output = {
            "action_logits": action_logits,
            "stop_logits": stop_logits,
            "fused_features": fused_features,
            "instruction_features": instr_features,
            "visual_features": visual_attended,
        }
        
        return {
            "logits": action_logits,
            "stop": stop_logits.squeeze(-1),
            "attn": cross_attn_weights,
            "fused_features": fused_features,
            "instruction_features": instr_features,
            "visual_features": visual_attended,
        }
    
    def _integrate_history(self, current_features: torch.Tensor, hist: Dict[str, Any]) -> torch.Tensor:
        """Integrate navigation history using LSTM."""
        if "features" not in hist:
            return current_features
        
        # Get history features
        hist_features = hist["features"]  # [B, T, H]
        
        # Process through LSTM
        hist_encoded, _ = self.history_lstm(hist_features)  # [B, T, H]
        
        # Pool history representation
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
            batch_size = 1
            pano_feats = torch.randn(batch_size, 36, self.feature_dim)
        
        if instr_tokens is None:
            batch_size = pano_feats.size(0)
            # Create dummy instruction tokens (CLS + SEP)
            instr_tokens = torch.tensor([[101, 102] + [0] * 8], dtype=torch.long)  # BERT tokens
            instr_mask = torch.tensor([[True, True] + [False] * 8], dtype=torch.bool)
            instr_tokens = instr_tokens.expand(batch_size, -1)
            instr_mask = instr_mask.expand(batch_size, -1)
        
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
        current_phrase = obs.get("current_phrase", "navigate to destination")
        
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
        
        # Apply bias based on grounding confidence
        logits = policy_out["logits"]
        
        # Calculate bias strength from VLM scores
        if hasattr(dygrav_signal, 'vlm_scores') and dygrav_signal.vlm_scores:
            max_vlm_score = max(score.score for score in dygrav_signal.vlm_scores)
            bias_strength = max_vlm_score * 0.3  # Conservative bias for BERT model
            
            # Apply directional bias based on region position
            region = dygrav_signal.chosen_region
            region_center_x = (region.xyxy[0] + region.xyxy[2]) / 2
            
            # Assume image width of 640 for normalization
            normalized_x = region_center_x / 640.0
            
            if normalized_x < 0.33:  # Left side
                bias_vector = torch.tensor([0, bias_strength, -bias_strength/2, 0])
            elif normalized_x > 0.67:  # Right side
                bias_vector = torch.tensor([0, -bias_strength/2, bias_strength, 0])
            else:  # Center
                bias_vector = torch.tensor([bias_strength, 0, 0, 0])
            
            bias_vector = bias_vector.to(logits.device, dtype=logits.dtype)
            
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
            
            # Load state dict with strict=False to handle BERT parameter mismatches
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
            
            print(f"✓ Loaded RvLN-BERT checkpoint from {checkpoint_path}")
            
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
    
    def get_bert_embeddings(self, input_ids: torch.LongTensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """Get BERT embeddings for analysis."""
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask.long())
            return outputs.last_hidden_state
    
    def unfreeze_bert(self):
        """Unfreeze BERT parameters for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True
        print("✓ BERT parameters unfrozen")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information."""
        base_info = super().get_model_info()
        
        # Add BERT-specific info
        bert_params = sum(p.numel() for p in self.bert.parameters())
        non_bert_params = sum(p.numel() for p in self.parameters()) - bert_params
        
        base_info.update({
            "bert_model": self.bert_model_name,
            "bert_parameters": bert_params,
            "non_bert_parameters": non_bert_params,
            "bert_frozen": not any(p.requires_grad for p in self.bert.parameters()),
            "hidden_dim": self.hidden_dim,
        })
        
        return base_info
