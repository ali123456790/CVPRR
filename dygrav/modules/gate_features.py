"""
Gate input features computation for the learned gate model.
Implements all features from the mathematical specification.
"""

from typing import List, Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from collections import deque

from ..core.types import Region, VLMResult


@dataclass
class GateFeatures:
    """Complete feature vector for gate model input."""
    entropy: float  # H(π_t)
    logit_gap: float  # Δz_t = z_max - z_second
    clip_margin: float  # δ_CLIP = c_1 - c_2
    candidate_count: int  # K = number of candidates
    novelty: float  # visual novelty score
    skill_logits: np.ndarray  # s_t = [p_direction, p_landmark, p_region, p_vertical, p_numeric]
    
    def to_tensor(self, device: str = 'cpu') -> torch.Tensor:
        """Convert to normalized tensor for gate input."""
        features = torch.tensor([
            self.entropy,
            self.logit_gap,
            self.clip_margin,
            float(self.candidate_count),
            self.novelty,
            *self.skill_logits.tolist()
        ], dtype=torch.float32, device=device)
        
        # Normalize features (simple standardization, can be improved with learned stats)
        # These are rough estimates for normalization ranges
        norm_factors = torch.tensor([
            2.0,   # entropy (typically 0-2)
            10.0,  # logit gap (typically 0-10)
            1.0,   # CLIP margin (already 0-1)
            10.0,  # candidate count (typically 0-10)
            1.0,   # novelty (already 0-1)
            1.0, 1.0, 1.0, 1.0, 1.0  # skill logits (already 0-1 from sigmoid)
        ], device=device)
        
        return features / norm_factors


class NoveltyTracker:
    """
    Visual memory bank for novelty detection.
    Maintains top-k visual embeddings with EMA decay.
    """
    
    def __init__(self, memory_size: int = 10, decay_rate: float = 0.95):
        self.memory_size = memory_size
        self.decay_rate = decay_rate
        self.memory_bank: deque = deque(maxlen=memory_size)
        self.memory_weights: deque = deque(maxlen=memory_size)
    
    def compute_novelty(self, visual_embedding: torch.Tensor) -> float:
        """
        Compute novelty as 1 - max cosine similarity with memory bank.
        
        Args:
            visual_embedding: Current visual feature vector
            
        Returns:
            Novelty score in [0, 1]
        """
        if len(self.memory_bank) == 0:
            # First observation is maximally novel
            return 1.0
        
        # Normalize embedding
        if visual_embedding.dim() > 1:
            visual_embedding = visual_embedding.flatten()
        v_norm = F.normalize(visual_embedding, dim=0)
        
        # Compute cosine similarities with memory bank
        max_similarity = 0.0
        for mem_embedding, weight in zip(self.memory_bank, self.memory_weights):
            m_norm = F.normalize(mem_embedding, dim=0)
            similarity = (v_norm @ m_norm).item()
            weighted_sim = similarity * weight
            max_similarity = max(max_similarity, weighted_sim)
        
        novelty = 1.0 - max_similarity
        return float(novelty)
    
    def update_memory(self, visual_embedding: torch.Tensor):
        """Add new embedding to memory bank with EMA decay on weights."""
        if visual_embedding.dim() > 1:
            visual_embedding = visual_embedding.flatten()
        
        # Decay existing weights
        for i in range(len(self.memory_weights)):
            self.memory_weights[i] *= self.decay_rate
        
        # Add new embedding with full weight
        self.memory_bank.append(visual_embedding.detach().clone())
        self.memory_weights.append(1.0)
        
        # Remove embeddings with very low weights
        min_weight_threshold = 0.01
        while self.memory_weights and self.memory_weights[0] < min_weight_threshold:
            self.memory_bank.popleft()
            self.memory_weights.popleft()


class SkillClassifier(nn.Module):
    """
    Multi-label classifier for instruction skills.
    Outputs calibrated probabilities for each skill type.
    """
    
    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 5)  # 5 skills
        )
        self.temperature = nn.Parameter(torch.ones(1))  # For calibration
        self.skill_names = ['direction', 'landmark', 'region', 'vertical', 'numeric']
    
    def forward(self, instruction_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            instruction_tokens: [batch_size, seq_len] tokenized instruction
            
        Returns:
            [batch_size, 5] multi-label probabilities (post-sigmoid)
        """
        x = self.embedding(instruction_tokens)
        lstm_out, _ = self.lstm(x)
        
        # Use last hidden state
        pooled = lstm_out[:, -1, :]
        
        # Get logits and apply temperature scaling for calibration
        logits = self.classifier(pooled)
        calibrated_logits = logits / self.temperature
        
        # Multi-label sigmoid (not softmax!)
        probs = torch.sigmoid(calibrated_logits)
        return probs
    
    def predict_skills(self, instruction: str, tokenizer) -> np.ndarray:
        """
        Predict skill probabilities for a text instruction.
        
        Args:
            instruction: Text instruction
            tokenizer: Tokenizer to convert text to tokens
            
        Returns:
            numpy array of shape [5] with skill probabilities
        """
        tokens = tokenizer(instruction)
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).unsqueeze(0)
        
        with torch.no_grad():
            probs = self.forward(tokens)
        
        return probs.squeeze(0).cpu().numpy()


class RuleBasedSkillClassifier:
    """
    Fallback rule-based skill classifier when neural model isn't available.
    Uses keyword matching to identify skill requirements.
    """
    
    def __init__(self):
        self.skill_keywords = {
            'direction': ['left', 'right', 'forward', 'back', 'turn', 'face', 'straight'],
            'landmark': ['door', 'chair', 'table', 'window', 'painting', 'plant', 'lamp'],
            'region': ['room', 'hallway', 'kitchen', 'bedroom', 'bathroom', 'living'],
            'vertical': ['stairs', 'up', 'down', 'floor', 'level', 'landing', 'ramp'],
            'numeric': ['first', 'second', 'third', 'two', 'three', 'count', 'number']
        }
    
    def predict_skills(self, instruction: str) -> np.ndarray:
        """
        Predict skill probabilities based on keyword presence.
        
        Args:
            instruction: Text instruction
            
        Returns:
            numpy array of shape [5] with skill probabilities
        """
        instruction_lower = instruction.lower()
        probs = np.zeros(5)
        
        for i, (skill, keywords) in enumerate(self.skill_keywords.items()):
            # Check if any keyword is present
            matches = sum(1 for kw in keywords if kw in instruction_lower)
            # Convert to probability (saturates at 1.0)
            probs[i] = min(1.0, matches * 0.3)
        
        return probs


def compute_logit_gap(logits: torch.Tensor) -> float:
    """
    Compute the gap between top-1 and top-2 logits.
    
    Args:
        logits: Action logits [batch_size, num_actions] or [num_actions]
        
    Returns:
        Logit gap Δz_t = z_max - z_second
    """
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    
    # Get top-2 logits
    top2_values, _ = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1)
    
    if top2_values.shape[-1] < 2:
        # Only one action available
        return float('inf')
    
    gap = (top2_values[:, 0] - top2_values[:, 1]).mean().item()
    return float(gap)


def compute_entropy(probs: torch.Tensor) -> float:
    """
    Compute entropy of action distribution.
    
    Args:
        probs: Action probabilities [batch_size, num_actions] or [num_actions]
        
    Returns:
        Entropy H(π_t)
    """
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    
    # Clamp for numerical stability
    probs = probs.clamp(min=1e-9)
    entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
    return float(entropy)


def compute_clip_margin(vlm_scores: List[VLMResult]) -> float:
    """
    Compute CLIP margin between top-1 and top-2 candidates.
    
    Args:
        vlm_scores: List of VLM scoring results
        
    Returns:
        CLIP margin δ_CLIP = c_1 - c_2
    """
    if not vlm_scores:
        return 0.0
    
    if len(vlm_scores) == 1:
        # Only one candidate, maximum margin
        return 1.0
    
    # Sort by score
    sorted_scores = sorted(vlm_scores, key=lambda x: x.score, reverse=True)
    margin = sorted_scores[0].score - sorted_scores[1].score
    
    return float(margin)


def extract_gate_features(
    logits: Optional[torch.Tensor],
    instruction: str,
    vlm_scores: List[VLMResult],
    visual_embedding: Optional[torch.Tensor],
    novelty_tracker: Optional[NoveltyTracker],
    skill_classifier,
    tokenizer=None
) -> GateFeatures:
    """
    Extract all features needed for the gate model.
    
    Args:
        logits: Action logits from backbone
        instruction: Current instruction text
        vlm_scores: VLM scoring results
        visual_embedding: Current visual features for novelty
        novelty_tracker: Novelty tracking object
        skill_classifier: Skill classifier (neural or rule-based)
        tokenizer: Optional tokenizer for skill classifier
        
    Returns:
        Complete GateFeatures object
    """
    # Compute action distribution features
    if logits is not None:
        probs = F.softmax(logits, dim=-1)
        entropy = compute_entropy(probs)
        logit_gap = compute_logit_gap(logits)
    else:
        entropy = 0.0
        logit_gap = 0.0
    
    # Compute CLIP margin
    clip_margin = compute_clip_margin(vlm_scores)
    
    # Candidate count
    candidate_count = len(vlm_scores)
    
    # Compute novelty
    if novelty_tracker and visual_embedding is not None:
        novelty = novelty_tracker.compute_novelty(visual_embedding)
    else:
        novelty = 0.5  # Default neutral novelty
    
    # Compute skill logits
    if hasattr(skill_classifier, 'predict_skills'):
        if isinstance(skill_classifier, SkillClassifier):
            # Neural classifier
            skill_logits = skill_classifier.predict_skills(instruction, tokenizer)
        else:
            # Rule-based classifier
            skill_logits = skill_classifier.predict_skills(instruction)
    else:
        # Fallback: all skills equally likely
        skill_logits = np.ones(5) * 0.5
    
    return GateFeatures(
        entropy=entropy,
        logit_gap=logit_gap,
        clip_margin=clip_margin,
        candidate_count=candidate_count,
        novelty=novelty,
        skill_logits=skill_logits
    )
