"""Room-across-Room (RxR) Dataset Implementation."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .episode import Episode, ViewPoint, NavigationAction, GroundingAnnotation, EpisodeBatch, collate_episodes


class RxRDataset(Dataset):
    """
    Room-across-Room (RxR) dataset loader.
    
    RxR extends R2R with multilingual instructions and more detailed annotations.
    This implementation focuses on English instructions but can be extended for multilingual support.
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        language: str = "en",
        max_episodes: Optional[int] = None,
        load_images: bool = True,
        image_size: Tuple[int, int] = (640, 480),
        max_path_length: Optional[int] = None,
        cache_images: bool = False,
        include_grounding: bool = False,
        verbose: bool = False,
    ):
        """
        Initialize RxR dataset.
        
        Args:
            data_root: Root directory containing RxR data
            split: Dataset split ("train", "val_seen", "val_unseen", "test")
            language: Language code ("en", "hi", "te")
            max_episodes: Maximum number of episodes to load
            load_images: Whether to load actual images
            image_size: Target image size (width, height)
            max_path_length: Maximum path length to consider
            cache_images: Whether to cache loaded images in memory
            include_grounding: Whether to include grounding annotations (if available)
            verbose: Whether to print loading progress
        """
        self.data_root = Path(data_root)
        self.split = split
        self.language = language
        self.max_episodes = max_episodes
        self.load_images = load_images
        self.image_size = image_size
        self.max_path_length = max_path_length
        self.cache_images = cache_images
        self.include_grounding = include_grounding
        self.verbose = verbose
        
        # Paths
        self.annotations_path = self.data_root / "annotations" / f"rxr_{split}_{language}.jsonl"
        self.connectivity_path = self.data_root / "connectivity"
        self.images_path = self.data_root / "images"
        self.grounding_path = self.data_root / "grounding" / f"rxr_{split}_{language}_grounding.jsonl"
        
        # Cache
        self._image_cache = {} if cache_images else None
        self._connectivity_cache = {}
        self._grounding_cache = {}
        
        # Load data
        self.episodes = self._load_episodes()
        
        # Load grounding annotations if available
        if self.include_grounding:
            self._load_grounding_annotations()
        
        if self.verbose:
            print(f"Loaded {len(self.episodes)} episodes from RxR {split} split ({language})")
    
    def _load_episodes(self) -> List[Episode]:
        """Load episodes from RxR annotations."""
        
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"RxR annotations not found: {self.annotations_path}")
        
        episodes = []
        
        with open(self.annotations_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if self.max_episodes and len(episodes) >= self.max_episodes:
                    break
                
                try:
                    item = json.loads(line.strip())
                    episode = self._parse_episode(item, idx)
                    if episode:
                        episodes.append(episode)
                except Exception as e:
                    if self.verbose:
                        print(f"Warning: Failed to parse episode {idx}: {e}")
                    continue
        
        return episodes
    
    def _parse_episode(self, item: Dict[str, Any], idx: int) -> Optional[Episode]:
        """Parse a single episode from RxR format."""
        
        # Extract basic info
        instruction_id = item.get('instruction_id', f'rxr_{self.split}_{idx}')
        episode_id = f"rxr_{self.split}_{self.language}_{instruction_id}"
        instruction = item['instruction']
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
            
            # Get heading/elevation from path or use defaults
            heading = item.get('start_pano', {}).get('heading', 0.0) if step_idx == 0 else 0.0
            elevation = item.get('start_pano', {}).get('elevation', 0.0) if step_idx == 0 else 0.0
            
            # Image path
            image_path = None
            if self.load_images:
                image_path = self.images_path / scan_id / f"{viewpoint_id}.jpg"
                if not image_path.exists():
                    # Try alternative naming conventions
                    for ext in ['.png', '.jpeg']:
                        alt_path = self.images_path / scan_id / f"{viewpoint_id}{ext}"
                        if alt_path.exists():
                            image_path = alt_path
                            break
                    else:
                        image_path = None
            
            viewpoint = ViewPoint(
                viewpoint_id=viewpoint_id,
                position=position,
                heading=heading,
                elevation=elevation,
                image_path=str(image_path) if image_path else None,
            )
            path.append(viewpoint)
        
        if len(path) == 0:
            return None
        
        # Truncate path if too long
        if self.max_path_length and len(path) > self.max_path_length:
            path = path[:self.max_path_length]
        
        # Build actions
        actions = []
        for i in range(len(path) - 1):
            actions.append(NavigationAction("forward", target_viewpoint=path[i+1].viewpoint_id))
        
        # Calculate path metrics
        trajectory_length = self._calculate_path_length(path)
        
        # Start and goal positions
        start_pos = path[0].position
        goal_pos = path[-1].position
        
        # Success information (if available)
        success = item.get('success', False)
        
        episode = Episode(
            episode_id=episode_id,
            dataset="rxr",
            split=self.split,
            instruction=instruction,
            path=path,
            actions=actions,
            start_position=start_pos,
            goal_position=goal_pos,
            success=success,
            trajectory_length=trajectory_length,
            shortest_path_length=item.get('distance', trajectory_length),
            scan_id=scan_id,
            language=self.language,
            annotator_id=item.get('annotator_id'),
            metadata={
                'instruction_id': instruction_id,
                'path_id': item.get('path_id'),
                'guide_id': item.get('guide_id'),
                'start_pano': item.get('start_pano', {}),
                'end_pano': item.get('end_pano', {}),
                'pose_trace': item.get('pose_trace', []),
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
    
    def _load_grounding_annotations(self):
        """Load grounding annotations if available."""
        
        if not self.grounding_path.exists():
            if self.verbose:
                print(f"No grounding annotations found at {self.grounding_path}")
            return
        
        grounding_data = {}
        
        try:
            with open(self.grounding_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    instruction_id = item.get('instruction_id')
                    if instruction_id:
                        grounding_data[instruction_id] = item
        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not load grounding annotations: {e}")
            return
        
        # Add grounding annotations to episodes
        for episode in self.episodes:
            instruction_id = episode.metadata.get('instruction_id')
            if instruction_id in grounding_data:
                grounding_item = grounding_data[instruction_id]
                annotations = self._parse_grounding_annotations(grounding_item, episode)
                episode.grounding_annotations.extend(annotations)
    
    def _parse_grounding_annotations(
        self, 
        grounding_item: Dict[str, Any], 
        episode: Episode
    ) -> List[GroundingAnnotation]:
        """Parse grounding annotations for an episode."""
        
        annotations = []
        
        # Parse phrase-level grounding
        phrases = grounding_item.get('phrases', [])
        for phrase_data in phrases:
            phrase = phrase_data.get('phrase', '')
            step_idx = phrase_data.get('step', 0)
            bbox = phrase_data.get('bbox', [0, 0, 100, 100])  # [x1, y1, x2, y2]
            category = phrase_data.get('category', 'object')
            confidence = phrase_data.get('confidence', 1.0)
            
            # Validate step index
            if step_idx >= len(episode.path):
                continue
            
            viewpoint_id = episode.path[step_idx].viewpoint_id
            
            annotation = GroundingAnnotation(
                phrase=phrase,
                bbox=tuple(bbox),
                viewpoint_id=viewpoint_id,
                step_idx=step_idx,
                category=category,
                confidence=confidence,
                object_id=phrase_data.get('object_id'),
            )
            annotations.append(annotation)
        
        return annotations
    
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
        
        # Grounding stats
        grounding_stats = {}
        if self.include_grounding:
            episodes_with_grounding = [ep for ep in self.episodes if ep.has_grounding_annotations]
            if episodes_with_grounding:
                total_annotations = sum(len(ep.grounding_annotations) for ep in episodes_with_grounding)
                categories = []
                for ep in episodes_with_grounding:
                    categories.extend(ep.grounding_categories)
                
                grounding_stats = {
                    'episodes_with_grounding': len(episodes_with_grounding),
                    'total_annotations': total_annotations,
                    'avg_annotations_per_episode': total_annotations / len(episodes_with_grounding),
                    'categories': list(set(categories)),
                }
        
        return {
            'num_episodes': len(self.episodes),
            'num_scans': len(scans),
            'language': self.language,
            'avg_path_length': np.mean(path_lengths),
            'max_path_length': np.max(path_lengths),
            'avg_trajectory_length': np.mean(trajectory_lengths),
            'avg_instruction_length': np.mean(instruction_lengths),
            'scans': scans,
            'grounding': grounding_stats,
        }
    
    def collate_fn(self, episodes: List[Episode]) -> EpisodeBatch:
        """Custom collate function for RxR episodes."""
        return collate_episodes(
            episodes,
            load_images=self.load_images,
            max_length=self.max_path_length,
        )


class RxRFGDataset(RxRDataset):
    """
    RxR Fine-Grained (RxR-FG) dataset - a curated subset of RxR with detailed grounding annotations.
    
    This dataset focuses on episodes with rich attribute and relation descriptions.
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        language: str = "en",
        criteria_file: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize RxR-FG dataset.
        
        Args:
            data_root: Root directory containing RxR data
            split: Dataset split
            language: Language code
            criteria_file: Path to criteria file for filtering episodes
            **kwargs: Additional arguments for RxRDataset
        """
        self.criteria_file = Path(criteria_file) if criteria_file else None
        self.filter_criteria = self._load_criteria()
        
        # Always include grounding for RxR-FG
        kwargs['include_grounding'] = True
        
        super().__init__(
            data_root=data_root,
            split=split,
            language=language,
            **kwargs
        )
        
        # Apply filtering
        self.episodes = self._filter_episodes(self.episodes)
        
        if self.verbose:
            print(f"Filtered to {len(self.episodes)} episodes for RxR-FG")
    
    def _load_criteria(self) -> Dict[str, Any]:
        """Load filtering criteria for RxR-FG."""
        
        if self.criteria_file and self.criteria_file.exists():
            try:
                with open(self.criteria_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not load criteria file: {e}")
        
        # Default criteria
        return {
            "relations": ["between", "next to", "on", "under", "left of", "right of"],
            "attributes": ["ceramic", "wooden", "smallest", "largest", "red", "blue", "white", "black"],
            "min_grounding_annotations": 1,
            "max_path_length": 20,
        }
    
    def _filter_episodes(self, episodes: List[Episode]) -> List[Episode]:
        """Filter episodes based on RxR-FG criteria."""
        
        filtered = []
        
        for episode in episodes:
            if self._meets_criteria(episode):
                # Mark as RxR-FG dataset
                episode.dataset = "rxr_fg"
                filtered.append(episode)
        
        return filtered
    
    def _meets_criteria(self, episode: Episode) -> bool:
        """Check if episode meets RxR-FG criteria."""
        
        # Check path length
        max_length = self.filter_criteria.get('max_path_length', 20)
        if len(episode.path) > max_length:
            return False
        
        # Check for grounding annotations
        min_annotations = self.filter_criteria.get('min_grounding_annotations', 1)
        if len(episode.grounding_annotations) < min_annotations:
            return False
        
        # Check for target relations and attributes
        target_relations = self.filter_criteria.get('relations', [])
        target_attributes = self.filter_criteria.get('attributes', [])
        
        instruction_lower = episode.instruction.lower()
        
        has_relation = any(rel in instruction_lower for rel in target_relations)
        has_attribute = any(attr in instruction_lower for attr in target_attributes)
        
        # Require at least one relation or attribute
        if not (has_relation or has_attribute):
            return False
        
        return True


# Factory functions
def create_rxr_dataset(
    data_root: Union[str, Path],
    split: str = "train",
    language: str = "en",
    **kwargs
) -> RxRDataset:
    """Factory function to create RxR dataset."""
    return RxRDataset(data_root=data_root, split=split, language=language, **kwargs)


def create_rxr_fg_dataset(
    data_root: Union[str, Path],
    split: str = "train",
    language: str = "en",
    **kwargs
) -> RxRFGDataset:
    """Factory function to create RxR-FG dataset."""
    return RxRFGDataset(data_root=data_root, split=split, language=language, **kwargs)


# Convenience functions for common splits
def create_rxr_train(data_root: Union[str, Path], language: str = "en", **kwargs) -> RxRDataset:
    """Create RxR training dataset."""
    return create_rxr_dataset(data_root, split="train", language=language, **kwargs)


def create_rxr_val_seen(data_root: Union[str, Path], language: str = "en", **kwargs) -> RxRDataset:
    """Create RxR validation seen dataset."""
    return create_rxr_dataset(data_root, split="val_seen", language=language, **kwargs)


def create_rxr_val_unseen(data_root: Union[str, Path], language: str = "en", **kwargs) -> RxRDataset:
    """Create RxR validation unseen dataset."""
    return create_rxr_dataset(data_root, split="val_unseen", language=language, **kwargs)
