"""Room-to-Room (R2R) Dataset Implementation."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .episode import Episode, ViewPoint, NavigationAction, EpisodeBatch, collate_episodes
from .r2r_feature_store import R2RFeatureStore

FEAT_TSV = "dygrav/data/r2r/img_features/ResNet-152-imagenet.tsv"


class R2RDataset(Dataset):
    """
    Room-to-Room (R2R) dataset loader.
    
    The R2R dataset contains navigation episodes in Matterport3D environments
    with natural language instructions.
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        max_episodes: Optional[int] = None,
        load_images: bool = True,
        image_size: Tuple[int, int] = (640, 480),
        max_path_length: Optional[int] = None,
        cache_images: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize R2R dataset.
        
        Args:
            data_root: Root directory containing R2R data
            split: Dataset split ("train", "val_seen", "val_unseen", "test")
            max_episodes: Maximum number of episodes to load (for debugging)
            load_images: Whether to load actual images
            image_size: Target image size (width, height)
            max_path_length: Maximum path length to consider
            cache_images: Whether to cache loaded images in memory
            verbose: Whether to print loading progress
        """
        self.data_root = Path(data_root)
        self.split = split
        self.max_episodes = max_episodes
        self.load_images = load_images
        self.image_size = image_size
        self.max_path_length = max_path_length
        self.cache_images = cache_images
        self.verbose = verbose
        
        self.feats = R2RFeatureStore(FEAT_TSV)
        
        # Paths
        self.annotations_path = self.data_root / "annotations" / f"R2R_{split}.json"
        self.connectivity_path = self.data_root / "connectivity"
        self.images_path = self.data_root / "images"
        self.depth_path = self.data_root / "depth"  # optional
        
        # Cache
        self._image_cache = {} if cache_images else None
        self._connectivity_cache = {}
        
        # Load data
        self.episodes = self._load_episodes()
        
        if self.verbose:
            print(f"Loaded {len(self.episodes)} episodes from R2R {split} split")
    
    def _load_episodes(self) -> List[Episode]:
        """Load episodes from R2R annotations."""
        
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"R2R annotations not found: {self.annotations_path}")
        
        with open(self.annotations_path, 'r') as f:
            raw_data = json.load(f)
        
        episodes = []
        
        for idx, item in enumerate(raw_data):
            if self.max_episodes and len(episodes) >= self.max_episodes:
                break
                
            try:
                episode = self._parse_episode(item, idx)
                if episode:
                    episodes.append(episode)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to parse episode {idx}: {e}")
                continue
        
        return episodes
    
    def _parse_episode(self, item: Dict[str, Any], idx: int) -> Optional[Episode]:
        """Parse a single episode from R2R format."""
        
        # Extract basic info
        episode_id = f"r2r_{self.split}_{idx}_{item.get('path_id', idx)}"
        instruction = item['instructions'][0]  # Take first instruction
        path_ids = item['path']
        scan_id = item['scan']
        
        # Load connectivity for this scan
        connectivity = self._load_connectivity(scan_id)
        if not connectivity:
            return None
        
        # Build path of viewpoints
        path = []
        for step_idx, viewpoint_id in enumerate(path_ids):
            if viewpoint_id not in connectivity:
                if self.verbose:
                    print(f"Warning: Viewpoint {viewpoint_id} not in connectivity graph")
                continue
            
            vp_info = connectivity[viewpoint_id]
            position = (vp_info['pose'][3], vp_info['pose'][7], vp_info['pose'][11])
            heading = item.get('heading', 0.0) if step_idx == 0 else 0.0
            
            features = self.feats.get(scan_id, viewpoint_id)
            
            # Optional depth map path
            dpath = None
            if self.depth_path.exists():
                cand = self.depth_path / scan_id / f"{viewpoint_id}.npy"
                if cand.exists():
                    dpath = str(cand)

            viewpoint = ViewPoint(
                viewpoint_id=viewpoint_id,
                position=position,
                heading=heading,
                elevation=item.get('elevation', 0.0) if step_idx == 0 else 0.0,
                features=features,
                depth_path=dpath,
            )
            path.append(viewpoint)
        
        if len(path) == 0:
            return None
        
        # Truncate path if too long
        if self.max_path_length and len(path) > self.max_path_length:
            path = path[:self.max_path_length]
        
        # Build actions (simple forward/stop actions for R2R)
        actions = []
        for i in range(len(path) - 1):
            actions.append(NavigationAction("forward", target_viewpoint=path[i+1].viewpoint_id))
        
        # Calculate path metrics
        trajectory_length = self._calculate_path_length(path)
        
        # Start and goal positions
        start_pos = path[0].position
        goal_pos = path[-1].position
        
        episode = Episode(
            episode_id=episode_id,
            dataset="r2r",
            split=self.split,
            instruction=instruction,
            path=path,
            actions=actions,
            start_position=start_pos,
            goal_position=goal_pos,
            trajectory_length=trajectory_length,
            shortest_path_length=trajectory_length,  # R2R doesn't provide shortest path
            scan_id=scan_id,
            metadata={
                'path_id': item.get('path_id', idx),
                'distance': item.get('distance', trajectory_length),
                'all_instructions': item.get('instructions', [instruction]),
            }
        )
        
        return episode
    
    def _load_connectivity(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Load connectivity graph for a scan."""
        
        if scan_id in self._connectivity_cache:
            return self._connectivity_cache[scan_id]
        
        connectivity_file = self.connectivity_path / f"{scan_id}_connectivity.json"
        if not connectivity_file.exists():
            return None
        
        try:
            with open(connectivity_file, 'r') as f:
                connectivity = json.load(f)
            
            # Convert to dict keyed by viewpoint_id
            connectivity_dict = {item['image_id']: item for item in connectivity}
            self._connectivity_cache[scan_id] = connectivity_dict
            return connectivity_dict
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load connectivity for {scan_id}: {e}")
            return None
    
    def _calculate_path_length(self, path: List[ViewPoint]) -> float:
        """Calculate total path length."""
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            pos1 = path[i].position
            pos2 = path[i+1].position
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
            total_length += distance
        
        return total_length
    
    def __len__(self) -> int:
        """Get dataset size."""
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Episode:
        """Get episode by index."""
        return self.episodes[idx]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.episodes:
            return {}
        
        path_lengths = [len(ep.path) for ep in self.episodes]
        trajectory_lengths = [ep.trajectory_length for ep in self.episodes]
        instruction_lengths = [len(ep.instruction.split()) for ep in self.episodes]
        
        scans = list(set(ep.scan_id for ep in self.episodes))
        
        return {
            'num_episodes': len(self.episodes),
            'num_scans': len(scans),
            'avg_path_length': np.mean(path_lengths),
            'max_path_length': np.max(path_lengths),
            'avg_trajectory_length': np.mean(trajectory_lengths),
            'avg_instruction_length': np.mean(instruction_lengths),
            'scans': scans,
        }
    
    def collate_fn(self, episodes: List[Episode]) -> EpisodeBatch:
        """Custom collate function for R2R episodes."""
        return collate_episodes(
            episodes,
            load_images=self.load_images,
            max_length=self.max_path_length,
        )


def create_r2r_dataset(
    data_root: Union[str, Path],
    split: str = "train",
    **kwargs
) -> R2RDataset:
    """
    Factory function to create R2R dataset.
    
    Args:
        data_root: Root directory containing R2R data
        split: Dataset split
        **kwargs: Additional arguments for R2RDataset
    
    Returns:
        R2RDataset instance
    """
    return R2RDataset(data_root=data_root, split=split, **kwargs)


# Convenience functions for common splits
def create_r2r_train(data_root: Union[str, Path], **kwargs) -> R2RDataset:
    """Create R2R training dataset."""
    return create_r2r_dataset(data_root, split="train", **kwargs)


def create_r2r_val_seen(data_root: Union[str, Path], **kwargs) -> R2RDataset:
    """Create R2R validation seen dataset."""
    return create_r2r_dataset(data_root, split="val_seen", **kwargs)


def create_r2r_val_unseen(data_root: Union[str, Path], **kwargs) -> R2RDataset:
    """Create R2R validation unseen dataset."""
    return create_r2r_dataset(data_root, split="val_unseen", **kwargs)


def create_r2r_test(data_root: Union[str, Path], **kwargs) -> R2RDataset:
    """Create R2R test dataset."""
    return create_r2r_dataset(data_root, split="test", **kwargs)
