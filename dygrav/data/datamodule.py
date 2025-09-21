from __future__ import annotations
from typing import Any, Dict, Tuple, List, Optional, Union
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .episode import Episode, EpisodeBatch, collate_episodes
from .r2r_dataset import R2RDataset, create_r2r_dataset
from .rxr_dataset import RxRDataset, RxRFGDataset, create_rxr_dataset, create_rxr_fg_dataset


def custom_collate_fn(batch: List[Tuple[Dict[str, Any], torch.Tensor]]):
    """Custom collate function to handle PIL Images (for synthetic data)."""
    obs_batch = []
    labels = []
    
    for obs, label in batch:
        # Keep PIL image as-is, stack features
        obs_batch.append(obs)
        labels.append(label)
    
    # Stack features into batch tensor
    feats = torch.stack([obs["feat"] for obs in obs_batch])
    # Keep images as list (they'll be handled individually by the policy)
    images = [obs["rgb"] for obs in obs_batch]
    
    # Create batched observation dict
    batched_obs = {"feat": feats, "rgb": images}
    batched_labels = torch.stack(labels)
    
    return batched_obs, batched_labels


def enhanced_collate_fn(batch: List[Episode]) -> EpisodeBatch:
    """Enhanced collate function with bucketing and padding for real datasets."""
    if not batch:
        raise ValueError("Cannot collate empty batch")
    
    # Sort batch by sequence length for better GPU utilization
    batch = sorted(batch, key=lambda ep: len(ep.path), reverse=True)
    
    # Use the episode collate function with enhanced features
    return collate_episodes(
        batch,
        load_images=True,
        max_length=None,  # Will use the longest sequence in batch
    )


class SyntheticVLNDataset(Dataset):
    """Synthetic VLN dataset for testing and development."""
    
    def __init__(self, n: int = 256, seed: int = 17):
        g = torch.Generator().manual_seed(seed)
        self.feats = torch.randn(n, 8, generator=g)
        self.labels = torch.randint(0, 2, (n,), generator=g)

    def __len__(self): return self.feats.shape[0]

    def __getitem__(self, idx) -> Tuple[Dict[str, Any], torch.Tensor]:
        feat = self.feats[idx]
        img = Image.new("RGB", (200, 100), color=(255, 255, 255))
        obs = {"feat": feat, "rgb": img}
        return obs, self.labels[idx]


class VLNDataModule:
    """
    Unified data module for VLN datasets (R2R, RxR, RxR-FG, Synthetic).
    
    This class manages dataset creation, loading, and batching for different
    VLN datasets with a consistent interface.
    """
    
    def __init__(
        self,
        dataset_name: str = "synthetic",
        data_root: Optional[Union[str, Path]] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        max_episodes: Optional[int] = None,
        max_path_length: Optional[int] = None,
        load_images: bool = True,
        cache_images: bool = False,
        language: str = "en",
        splits: Optional[Dict[str, str]] = None,
        seed: int = 17,
        verbose: bool = False,
        **dataset_kwargs,
    ):
        """
        Initialize VLN data module.
        
        Args:
            dataset_name: Name of dataset ("r2r", "rxr", "rxr_fg", "synthetic")
            data_root: Root directory of dataset
            batch_size: Batch size for data loaders
            num_workers: Number of workers for data loading
            max_episodes: Maximum episodes to load (for debugging)
            max_path_length: Maximum path length to consider
            load_images: Whether to load actual images
            cache_images: Whether to cache images in memory
            language: Language for multilingual datasets
            splits: Custom split names mapping
            seed: Random seed
            verbose: Whether to print progress
            **dataset_kwargs: Additional arguments for dataset creation
        """
        self.dataset_name = dataset_name
        self.data_root = Path(data_root) if data_root else None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_episodes = max_episodes
        self.max_path_length = max_path_length
        self.load_images = load_images
        self.cache_images = cache_images
        self.language = language
        self.seed = seed
        self.verbose = verbose
        self.dataset_kwargs = dataset_kwargs
        
        # Default split mapping
        self.splits = splits or {
            "train": "train",
            "val": "val_seen", 
            "val_unseen": "val_unseen",
            "test": "test",
        }
        
        # Datasets cache
        self._datasets = {}
        
        # Validate configuration
        self._validate_config()
    
    def _validate_config(self):
        """Validate data module configuration."""
        if self.dataset_name not in ["r2r", "rxr", "rxr_fg", "synthetic"]:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
        
        if self.dataset_name != "synthetic" and not self.data_root:
            raise ValueError(f"data_root required for dataset: {self.dataset_name}")
        
        if self.dataset_name != "synthetic" and not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")
    
    def _create_dataset(self, split: str) -> Dataset:
        """Create dataset for given split."""
        
        if self.dataset_name == "synthetic":
            return SyntheticVLNDataset(n=self.max_episodes or 256, seed=self.seed)
        
        # Common arguments for real datasets
        common_args = {
            "data_root": self.data_root,
            "split": self.splits.get(split, split),
            "max_episodes": self.max_episodes,
            "load_images": self.load_images,
            "max_path_length": self.max_path_length,
            "cache_images": self.cache_images,
            "verbose": self.verbose,
            **self.dataset_kwargs,
        }
        
        if self.dataset_name == "r2r":
            return create_r2r_dataset(**common_args)
        
        elif self.dataset_name == "rxr":
            return create_rxr_dataset(language=self.language, **common_args)
        
        elif self.dataset_name == "rxr_fg":
            criteria_file = self.data_root / "rxr_fg_criteria.json"
            return create_rxr_fg_dataset(
                language=self.language,
                criteria_file=criteria_file if criteria_file.exists() else None,
                **common_args,
            )
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _get_collate_fn(self):
        """Get appropriate collate function for dataset."""
        if self.dataset_name == "synthetic":
            return custom_collate_fn
        else:
            return enhanced_collate_fn
    
    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        if "train" not in self._datasets:
            self._datasets["train"] = self._create_dataset("train")
        
        return DataLoader(
            self._datasets["train"],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=True,
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        if "val" not in self._datasets:
            self._datasets["val"] = self._create_dataset("val")
        
        return DataLoader(
            self._datasets["val"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=True,
        )
    
    def val_unseen_dataloader(self) -> DataLoader:
        """Create validation unseen data loader."""
        if "val_unseen" not in self._datasets:
            self._datasets["val_unseen"] = self._create_dataset("val_unseen")
        
        return DataLoader(
            self._datasets["val_unseen"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=True,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        if "test" not in self._datasets:
            self._datasets["test"] = self._create_dataset("test")
        
        return DataLoader(
            self._datasets["test"],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._get_collate_fn(),
            pin_memory=True,
        )
    
    def get_dataset(self, split: str) -> Dataset:
        """Get dataset for specific split."""
        if split not in self._datasets:
            self._datasets[split] = self._create_dataset(split)
        return self._datasets[split]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all loaded datasets."""
        stats = {
            "dataset_name": self.dataset_name,
            "data_root": str(self.data_root) if self.data_root else None,
            "language": self.language,
            "batch_size": self.batch_size,
            "load_images": self.load_images,
        }
        
        # Add per-split statistics
        for split_name in ["train", "val", "val_unseen", "test"]:
            try:
                dataset = self.get_dataset(split_name)
                if hasattr(dataset, 'get_stats'):
                    stats[f"{split_name}_stats"] = dataset.get_stats()
                else:
                    stats[f"{split_name}_stats"] = {"num_episodes": len(dataset)}
            except Exception:
                stats[f"{split_name}_stats"] = None
        
        return stats


# Backward compatibility
class SyntheticDataModule(VLNDataModule):
    """Backward compatible synthetic data module."""
    
    def __init__(self, batch_size: int = 16, **kwargs):
        super().__init__(
            dataset_name="synthetic",
            batch_size=batch_size,
            **kwargs
        )
