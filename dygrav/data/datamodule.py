from __future__ import annotations
from typing import Any, Dict, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

def custom_collate_fn(batch: List[Tuple[Dict[str, Any], torch.Tensor]]):
    """Custom collate function to handle PIL Images."""
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

class SyntheticVLNDataset(Dataset):
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

class SyntheticDataModule:
    def __init__(self, batch_size: int = 16):
        self.batch_size = batch_size
    def train_dataloader(self):
        return DataLoader(
            SyntheticVLNDataset(256), 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=custom_collate_fn
        )
