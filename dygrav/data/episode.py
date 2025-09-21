"""Normalized Episode Schema for standardized data representation across all datasets."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import torch
    Tensor = torch.Tensor
except ImportError:  # pragma: no cover
    Tensor = Any


@dataclass
class ViewPoint:
    """Single viewpoint in the environment."""
    viewpoint_id: str
    position: Tuple[float, float, float]  # (x, y, z)
    heading: float  # radians
    elevation: float  # radians
    image_path: Optional[str] = None
    features: Optional[Tensor] = None  # Pre-computed visual features
    objects: Optional[List[Dict[str, Any]]] = None  # Object detections/annotations


@dataclass
class NavigationAction:
    """Single navigation action."""
    action_type: str  # "forward", "turn_left", "turn_right", "stop", "teleport"
    target_viewpoint: Optional[str] = None  # For teleport actions
    distance: Optional[float] = None  # For forward actions
    angle: Optional[float] = None  # For turn actions


@dataclass
class GroundingAnnotation:
    """Ground truth grounding annotation for a specific instruction phrase."""
    phrase: str
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    viewpoint_id: str
    step_idx: int
    category: str  # "attribute", "relation", "object"
    confidence: float = 1.0
    object_id: Optional[str] = None


@dataclass
class Episode:
    """Normalized episode representation for all VLN datasets."""
    
    # Core identifiers
    episode_id: str
    dataset: str  # "r2r", "rxr", "rxr_fg"
    split: str  # "train", "val_seen", "val_unseen", "test"
    
    # Navigation data
    instruction: str
    path: List[ViewPoint]
    actions: List[NavigationAction]
    start_position: Tuple[float, float, float]
    goal_position: Tuple[float, float, float]
    
    # Ground truth data
    success: bool = False
    shortest_path_length: float = 0.0
    trajectory_length: float = 0.0
    
    # Grounding annotations (for fine-grained datasets)
    grounding_annotations: List[GroundingAnnotation] = field(default_factory=list)
    
    # Additional metadata
    scan_id: str = ""
    language: str = "en"
    annotator_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate episode data after initialization."""
        if len(self.path) == 0:
            raise ValueError(f"Episode {self.episode_id} has empty path")
        
        if len(self.actions) > 0 and len(self.actions) != len(self.path) - 1:
            # Actions should be one less than path length (no action for final state)
            pass  # Some datasets might have different conventions
    
    @property
    def num_steps(self) -> int:
        """Number of steps in the episode."""
        return len(self.path)
    
    @property
    def has_grounding_annotations(self) -> bool:
        """Check if episode has grounding annotations."""
        return len(self.grounding_annotations) > 0
    
    @property
    def grounding_categories(self) -> List[str]:
        """Get unique grounding categories in this episode."""
        return list(set(ann.category for ann in self.grounding_annotations))
    
    def get_images(self, load_images: bool = True) -> List[Optional[Image.Image]]:
        """Load images for all viewpoints."""
        images = []
        for viewpoint in self.path:
            if viewpoint.image_path and load_images:
                try:
                    img = Image.open(viewpoint.image_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load image {viewpoint.image_path}: {e}")
                    images.append(None)
            else:
                images.append(None)
        return images
    
    def get_grounding_for_step(self, step_idx: int) -> List[GroundingAnnotation]:
        """Get grounding annotations for a specific step."""
        return [ann for ann in self.grounding_annotations if ann.step_idx == step_idx]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary for serialization."""
        return {
            'episode_id': self.episode_id,
            'dataset': self.dataset,
            'split': self.split,
            'instruction': self.instruction,
            'start_position': self.start_position,
            'goal_position': self.goal_position,
            'success': self.success,
            'shortest_path_length': self.shortest_path_length,
            'trajectory_length': self.trajectory_length,
            'scan_id': self.scan_id,
            'language': self.language,
            'annotator_id': self.annotator_id,
            'metadata': self.metadata,
            'num_steps': self.num_steps,
            'has_grounding': self.has_grounding_annotations,
            'grounding_categories': self.grounding_categories,
        }


@dataclass
class EpisodeBatch:
    """Batched episode data for efficient processing."""
    
    # Core data
    episode_ids: List[str]
    instructions: List[str]
    datasets: List[str]
    splits: List[str]
    
    # Visual data
    images: List[List[Optional[Image.Image]]]  # [batch_size, max_steps, ...]
    
    # Navigation data
    actions: List[List[NavigationAction]]
    positions: List[List[Tuple[float, float, float]]]
    
    # Ground truth
    successes: List[bool]
    path_lengths: List[float]
    shortest_paths: List[float]
    
    # Grounding data
    grounding_annotations: List[List[GroundingAnnotation]]
    
    # Padding info
    sequence_lengths: List[int]  # Actual length of each episode
    max_length: int
    
    # Optional fields with defaults
    features: Optional[Tensor] = None  # [batch_size, max_steps, feature_dim]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return len(self.episode_ids)
    
    def to_device(self, device: Union[str, Any]) -> 'EpisodeBatch':
        """Move tensors to specified device."""
        if self.features is not None:
            try:
                import torch
                if isinstance(device, str):
                    device = torch.device(device)
                self.features = self.features.to(device)
            except ImportError:
                pass
        return self


def collate_episodes(episodes: List[Episode], 
                    load_images: bool = True,
                    max_length: Optional[int] = None,
                    pad_token: int = 0) -> EpisodeBatch:
    """Collate a list of episodes into a batch with proper padding."""
    
    if not episodes:
        raise ValueError("Cannot collate empty list of episodes")
    
    batch_size = len(episodes)
    sequence_lengths = [ep.num_steps for ep in episodes]
    actual_max_length = max(sequence_lengths)
    
    if max_length is not None:
        actual_max_length = min(actual_max_length, max_length)
        # Truncate episodes if they exceed max_length
        sequence_lengths = [min(length, max_length) for length in sequence_lengths]
    
    # Collect basic data
    episode_ids = [ep.episode_id for ep in episodes]
    instructions = [ep.instruction for ep in episodes]
    datasets = [ep.dataset for ep in episodes]
    splits = [ep.split for ep in episodes]
    successes = [ep.success for ep in episodes]
    path_lengths = [ep.trajectory_length for ep in episodes]
    shortest_paths = [ep.shortest_path_length for ep in episodes]
    
    # Collect images (with padding)
    batched_images = []
    batched_actions = []
    batched_positions = []
    batched_grounding = []
    
    for ep in episodes:
        # Get images for this episode
        ep_images = ep.get_images(load_images=load_images)
        ep_actions = ep.actions
        ep_positions = [vp.position for vp in ep.path]
        ep_grounding = ep.grounding_annotations
        
        # Truncate if necessary
        ep_length = min(len(ep_images), actual_max_length)
        ep_images = ep_images[:ep_length]
        ep_actions = ep_actions[:ep_length-1] if ep_actions else []
        ep_positions = ep_positions[:ep_length]
        
        # Pad images with None
        while len(ep_images) < actual_max_length:
            ep_images.append(None)
        
        # Pad actions with stop action
        while len(ep_actions) < actual_max_length - 1:
            ep_actions.append(NavigationAction("stop"))
        
        # Pad positions with last known position
        last_pos = ep_positions[-1] if ep_positions else (0.0, 0.0, 0.0)
        while len(ep_positions) < actual_max_length:
            ep_positions.append(last_pos)
        
        batched_images.append(ep_images)
        batched_actions.append(ep_actions)
        batched_positions.append(ep_positions)
        batched_grounding.append(ep_grounding)
    
    return EpisodeBatch(
        episode_ids=episode_ids,
        instructions=instructions,
        datasets=datasets,
        splits=splits,
        images=batched_images,
        actions=batched_actions,
        positions=batched_positions,
        successes=successes,
        path_lengths=path_lengths,
        shortest_paths=shortest_paths,
        grounding_annotations=batched_grounding,
        sequence_lengths=sequence_lengths,
        max_length=actual_max_length,
    )
