"""Normalized Episode Schema for standardized data representation across all datasets."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image
try:
    import torch
except Exception:
    torch = None  # pragma: no cover

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
    depth_path: Optional[str] = None  # Optional depth map path


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
    bev_features: Optional[Tensor] = None  # [batch_size, max_steps, bev_dim]
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
    
    # Collect images (with padding) and optional depth→BEV
    batched_images = []
    batched_actions = []
    batched_positions = []
    batched_grounding = []
    # Optional visual features: pack to [B, T, 36, D] if available
    feature_slices: List[List[Optional[np.ndarray]]] = []
    # Optional BEV vectors per step
    bev_slices: List[List[Optional[np.ndarray]]] = []
    # Lazy import BEVBuilder
    try:
        from ..modules.bev import BEVBuilder  # type: ignore
        bev_builder = BEVBuilder()
    except Exception:
        bev_builder = None
    
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
        # Collect per-step features if present on viewpoints
        ep_feats: List[Optional[np.ndarray]] = []
        ep_bev: List[Optional[np.ndarray]] = []
        for idx_vp in range(ep_length):
            vp = ep.path[idx_vp]
            if getattr(vp, "features", None) is not None:
                feat = vp.features
                # Accept torch.Tensor or np.ndarray
                if hasattr(feat, "detach"):
                    feat = feat.detach().cpu().numpy()
                ep_feats.append(np.asarray(feat))
            else:
                ep_feats.append(None)
            # Depth→BEV if available
            if bev_builder is not None and getattr(vp, "depth_path", None):
                try:
                    import numpy as _np
                    depth = None
                    if vp.depth_path.endswith('.npy'):
                        depth = _np.load(vp.depth_path)
                    if depth is not None:
                        import torch as _torch
                        depth_t = _torch.from_numpy(depth).float()
                        bev_vec = bev_builder.build(depth_t)
                        ep_bev.append(bev_vec.detach().cpu().numpy())
                    else:
                        ep_bev.append(None)
                except Exception:
                    ep_bev.append(None)
            else:
                ep_bev.append(None)
        
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
        
        # Pad per-step features list to match length
        while len(ep_feats) < actual_max_length:
            ep_feats.append(None)
        while len(ep_bev) < actual_max_length:
            ep_bev.append(None)

        batched_images.append(ep_images)
        batched_actions.append(ep_actions)
        batched_positions.append(ep_positions)
        batched_grounding.append(ep_grounding)
        feature_slices.append(ep_feats)
        bev_slices.append(ep_bev)
    
    # Build packed feature tensor if any episode has features
    packed_features = None
    packed_bev = None
    try:
        if any(any(f is not None for f in ep_feats) for ep_feats in feature_slices) and torch is not None:
            # Determine feature shape (36, D)
            feat_shape = None
            for ep_feats in feature_slices:
                for f in ep_feats:
                    if f is not None:
                        if f.ndim == 1:
                            # Some stores may provide [D] per-view; expand to (36, D) later
                            feat_shape = (36, f.shape[-1])
                        elif f.ndim == 2:
                            feat_shape = (f.shape[0], f.shape[1])
                        break
                if feat_shape is not None:
                    break
            if feat_shape is None:
                feat_shape = (36, 2048)
            B = batch_size
            T = actual_max_length
            V, D = feat_shape
            packed = np.zeros((B, T, V, D), dtype=np.float32)
            for b, ep_feats in enumerate(feature_slices):
                for t, f in enumerate(ep_feats):
                    if f is None:
                        continue
                    f_arr = np.asarray(f, dtype=np.float32)
                    if f_arr.ndim == 1:
                        # If single-view feature, tile or leave zeros beyond available views
                        f_arr = np.repeat(f_arr[None, :], V, axis=0)
                    # Adjust number of views to V=36 by pad/truncate
                    if f_arr.shape[0] < V:
                        pad_views = V - f_arr.shape[0]
                        f_arr = np.vstack([f_arr, np.repeat(f_arr[-1][None], pad_views, axis=0)])
                    elif f_arr.shape[0] > V:
                        f_arr = f_arr[:V]
                    packed[b, t] = f_arr.astype(np.float32)
            packed_features = torch.from_numpy(packed)
    except Exception:
        packed_features = None

    # Pack BEV features [B, T, Dbev]
    try:
        if any(any(v is not None for v in ep_bev) for ep_bev in bev_slices) and torch is not None:
            # Determine bev dim
            bev_dim = None
            for ep_bev in bev_slices:
                for v in ep_bev:
                    if v is not None:
                        bev_dim = v.shape[-1]
                        break
                if bev_dim is not None:
                    break
            bev_dim = bev_dim or 1152
            B = batch_size
            T = actual_max_length
            packedb = np.zeros((B, T, bev_dim), dtype=np.float32)
            for b, ep_bev in enumerate(bev_slices):
                for t, v in enumerate(ep_bev):
                    if v is None:
                        continue
                    vv = np.asarray(v, dtype=np.float32)
                    if vv.shape[-1] != bev_dim:
                        # truncate or pad
                        if vv.shape[-1] > bev_dim:
                            vv = vv[:bev_dim]
                        else:
                            pad = np.zeros((bev_dim,), dtype=np.float32)
                            pad[:vv.shape[-1]] = vv
                            vv = pad
                    packedb[b, t] = vv
            packed_bev = torch.from_numpy(packedb)
    except Exception:
        packed_bev = None

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
        features=packed_features,
        bev_features=packed_bev,
    )
