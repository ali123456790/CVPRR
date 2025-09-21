"""Unit tests for data loaders and episode handling."""

import pytest
import tempfile
import json
from pathlib import Path
from PIL import Image
import numpy as np

from dygrav.data.episode import (
    Episode, ViewPoint, NavigationAction, GroundingAnnotation, 
    EpisodeBatch, collate_episodes
)
from dygrav.data.datamodule import VLNDataModule, enhanced_collate_fn
from dygrav.data.r2r_dataset import R2RDataset
from dygrav.data.rxr_dataset import RxRDataset, RxRFGDataset


class TestEpisodeSchema:
    """Test the normalized episode schema."""
    
    def test_episode_creation(self):
        """Test basic episode creation."""
        viewpoints = [
            ViewPoint("vp1", (0, 0, 0), 0.0, 0.0),
            ViewPoint("vp2", (1, 0, 0), 0.0, 0.0),
            ViewPoint("vp3", (2, 0, 0), 0.0, 0.0),
        ]
        
        actions = [
            NavigationAction("forward", target_viewpoint="vp2"),
            NavigationAction("forward", target_viewpoint="vp3"),
        ]
        
        episode = Episode(
            episode_id="test_001",
            dataset="test",
            split="train",
            instruction="Go forward twice",
            path=viewpoints,
            actions=actions,
            start_position=(0, 0, 0),
            goal_position=(2, 0, 0),
        )
        
        assert episode.episode_id == "test_001"
        assert episode.num_steps == 3
        assert len(episode.actions) == 2
        assert not episode.has_grounding_annotations
    
    def test_episode_with_grounding(self):
        """Test episode with grounding annotations."""
        viewpoints = [ViewPoint("vp1", (0, 0, 0), 0.0, 0.0)]
        
        grounding = [
            GroundingAnnotation(
                phrase="red chair",
                bbox=(10, 10, 50, 50),
                viewpoint_id="vp1",
                step_idx=0,
                category="attribute"
            )
        ]
        
        episode = Episode(
            episode_id="test_002",
            dataset="test",
            split="train",
            instruction="Find the red chair",
            path=viewpoints,
            actions=[],
            start_position=(0, 0, 0),
            goal_position=(0, 0, 0),
            grounding_annotations=grounding,
        )
        
        assert episode.has_grounding_annotations
        assert "attribute" in episode.grounding_categories
        assert len(episode.get_grounding_for_step(0)) == 1
    
    def test_episode_validation(self):
        """Test episode validation."""
        # Empty path should raise error
        with pytest.raises(ValueError):
            Episode(
                episode_id="invalid",
                dataset="test",
                split="train",
                instruction="Invalid episode",
                path=[],
                actions=[],
                start_position=(0, 0, 0),
                goal_position=(0, 0, 0),
            )


class TestEpisodeBatching:
    """Test episode batching and collation."""
    
    def create_test_episodes(self, n: int = 3) -> list[Episode]:
        """Create test episodes with varying lengths."""
        episodes = []
        
        for i in range(n):
            # Varying path lengths
            path_length = 3 + i
            viewpoints = [
                ViewPoint(f"vp{j}", (j, 0, 0), 0.0, 0.0)
                for j in range(path_length)
            ]
            
            actions = [
                NavigationAction("forward", target_viewpoint=f"vp{j+1}")
                for j in range(path_length - 1)
            ]
            
            episodes.append(Episode(
                episode_id=f"test_{i}",
                dataset="test",
                split="train",
                instruction=f"Test instruction {i}",
                path=viewpoints,
                actions=actions,
                start_position=(0, 0, 0),
                goal_position=(path_length-1, 0, 0),
                trajectory_length=float(path_length-1),
            ))
        
        return episodes
    
    def test_collate_episodes(self):
        """Test episode collation."""
        episodes = self.create_test_episodes(3)
        
        batch = collate_episodes(episodes, load_images=False)
        
        assert isinstance(batch, EpisodeBatch)
        assert batch.batch_size == 3
        assert len(batch.episode_ids) == 3
        assert len(batch.instructions) == 3
        assert batch.max_length == 5  # Longest episode has 5 steps
        
        # Check sequence lengths
        expected_lengths = [3, 4, 5]
        assert batch.sequence_lengths == expected_lengths
        
        # Check padding
        assert len(batch.images) == 3  # batch size
        assert all(len(ep_images) == 5 for ep_images in batch.images)  # max length
    
    def test_enhanced_collate_fn(self):
        """Test enhanced collate function."""
        episodes = self.create_test_episodes(4)
        
        batch = enhanced_collate_fn(episodes)
        
        assert isinstance(batch, EpisodeBatch)
        assert batch.batch_size == 4
        
        # Should be sorted by length (descending)
        lengths = batch.sequence_lengths
        assert lengths == sorted(lengths, reverse=True)
    
    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        with pytest.raises(ValueError):
            collate_episodes([])
        
        with pytest.raises(ValueError):
            enhanced_collate_fn([])
    
    def test_max_length_truncation(self):
        """Test truncation with max_length."""
        episodes = self.create_test_episodes(3)
        
        batch = collate_episodes(episodes, load_images=False, max_length=3)
        
        assert batch.max_length == 3
        assert all(length <= 3 for length in batch.sequence_lengths)


class TestVLNDataModule:
    """Test the unified VLN data module."""
    
    def test_synthetic_datamodule(self):
        """Test synthetic data module."""
        dm = VLNDataModule(
            dataset_name="synthetic",
            batch_size=8,
            max_episodes=32,
        )
        
        train_loader = dm.train_dataloader()
        assert train_loader.batch_size == 8
        
        # Test batch
        batch = next(iter(train_loader))
        assert len(batch) == 2  # obs, labels for synthetic
    
    def test_datamodule_validation(self):
        """Test data module validation."""
        # Unknown dataset should raise error
        with pytest.raises(ValueError):
            VLNDataModule(dataset_name="unknown")
        
        # Missing data_root for real datasets should raise error
        with pytest.raises(ValueError):
            VLNDataModule(dataset_name="r2r")
    
    def test_datamodule_stats(self):
        """Test data module statistics."""
        dm = VLNDataModule(
            dataset_name="synthetic",
            max_episodes=16,
        )
        
        stats = dm.get_stats()
        
        assert stats["dataset_name"] == "synthetic"
        assert stats["batch_size"] == 16
        assert "train_stats" in stats


class TestDatasetLoading:
    """Test dataset loading with mock data."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with mock files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create directory structure
            (temp_path / "annotations").mkdir()
            (temp_path / "connectivity").mkdir()
            (temp_path / "images").mkdir()
            
            # Create mock R2R annotation
            r2r_data = [
                {
                    "path_id": 1,
                    "scan": "test_scan",
                    "path": ["vp1", "vp2", "vp3"],
                    "instructions": ["Go to the kitchen"],
                    "heading": 0.0,
                    "elevation": 0.0,
                }
            ]
            
            with open(temp_path / "annotations" / "R2R_train.json", 'w') as f:
                json.dump(r2r_data, f)
            
            # Create mock connectivity
            connectivity_data = [
                {
                    "image_id": "vp1",
                    "pose": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],  # Identity matrix
                },
                {
                    "image_id": "vp2", 
                    "pose": [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
                },
                {
                    "image_id": "vp3",
                    "pose": [1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0],
                },
            ]
            
            with open(temp_path / "connectivity" / "test_scan_connectivity.json", 'w') as f:
                json.dump(connectivity_data, f)
            
            # Create mock RxR annotation
            rxr_data = {
                "instruction_id": "test_rxr_1",
                "scan": "test_scan",
                "path": ["vp1", "vp2"],
                "instruction": "Walk to the blue chair",
                "start_pano": {"heading": 0.0, "elevation": 0.0},
                "distance": 1.0,
            }
            
            with open(temp_path / "annotations" / "rxr_train_en.jsonl", 'w') as f:
                json.dump(rxr_data, f)
                f.write('\n')
            
            yield temp_path
    
    def test_r2r_dataset_loading(self, temp_data_dir):
        """Test R2R dataset loading."""
        dataset = R2RDataset(
            data_root=temp_data_dir,
            split="train",
            load_images=False,
            verbose=False,
        )
        
        assert len(dataset) > 0
        
        episode = dataset[0]
        assert episode.dataset == "r2r"
        assert episode.split == "train"
        assert len(episode.path) == 3
        assert episode.scan_id == "test_scan"
        
        stats = dataset.get_stats()
        assert stats["num_episodes"] == 1
        assert stats["num_scans"] == 1
    
    def test_rxr_dataset_loading(self, temp_data_dir):
        """Test RxR dataset loading."""
        dataset = RxRDataset(
            data_root=temp_data_dir,
            split="train",
            language="en",
            load_images=False,
            verbose=False,
        )
        
        assert len(dataset) > 0
        
        episode = dataset[0]
        assert episode.dataset == "rxr"
        assert episode.language == "en"
        assert len(episode.path) == 2
        
        stats = dataset.get_stats()
        assert stats["language"] == "en"


class TestDataShapesAndDeterminism:
    """Test data shapes and deterministic behavior."""
    
    def test_deterministic_shuffling(self):
        """Test that shuffling is deterministic with fixed seed."""
        dm1 = VLNDataModule(
            dataset_name="synthetic",
            batch_size=4,
            max_episodes=16,
            seed=42,
        )
        
        dm2 = VLNDataModule(
            dataset_name="synthetic", 
            batch_size=4,
            max_episodes=16,
            seed=42,
        )
        
        # Get first batch from each
        batch1 = next(iter(dm1.train_dataloader()))
        batch2 = next(iter(dm2.train_dataloader()))
        
        # Should be identical with same seed
        if hasattr(batch1[0], 'feat') and hasattr(batch2[0], 'feat'):
            np.testing.assert_array_equal(
                batch1[0]['feat'].numpy(),
                batch2[0]['feat'].numpy()
            )
    
    def test_batch_shapes_consistency(self):
        """Test that batch shapes are consistent."""
        dm = VLNDataModule(
            dataset_name="synthetic",
            batch_size=8,
            max_episodes=32,
        )
        
        loader = dm.train_dataloader()
        
        # Test multiple batches
        for i, batch in enumerate(loader):
            if i >= 3:  # Test first 3 batches
                break
            
            obs, labels = batch
            
            # Check shapes
            assert obs['feat'].shape[0] == 8  # batch size
            assert obs['feat'].shape[1] == 8  # feature dim
            assert labels.shape[0] == 8  # batch size
            assert len(obs['rgb']) == 8  # image list length
    
    def test_padding_correctness(self):
        """Test that padding is applied correctly."""
        episodes = []
        
        # Create episodes with different lengths
        for i, length in enumerate([2, 4, 3]):
            viewpoints = [
                ViewPoint(f"vp{j}", (j, 0, 0), 0.0, 0.0)
                for j in range(length)
            ]
            
            episodes.append(Episode(
                episode_id=f"test_{i}",
                dataset="test",
                split="train", 
                instruction=f"Test {i}",
                path=viewpoints,
                actions=[],
                start_position=(0, 0, 0),
                goal_position=(length-1, 0, 0),
            ))
        
        batch = collate_episodes(episodes, load_images=False)
        
        # Max length should be 4
        assert batch.max_length == 4
        
        # All sequences should be padded to max length
        assert all(len(pos_seq) == 4 for pos_seq in batch.positions)
        assert all(len(img_seq) == 4 for img_seq in batch.images)
        
        # Sequence lengths should track original lengths
        assert batch.sequence_lengths == [2, 4, 3]


if __name__ == "__main__":
    pytest.main([__file__])
