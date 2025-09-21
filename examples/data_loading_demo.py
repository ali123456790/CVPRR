#!/usr/bin/env python
"""
Demonstration of the new real data loading infrastructure for DyGRAV.

This script shows how to use the R2R, RxR, and RxR-FG datasets with the
unified VLNDataModule interface.
"""

from pathlib import Path
from dygrav.data import VLNDataModule, Episode, create_r2r_dataset, create_rxr_fg_dataset


def demo_synthetic_data():
    """Demonstrate synthetic data loading (works without external data)."""
    print("="*60)
    print("SYNTHETIC DATA DEMO")
    print("="*60)
    
    # Create synthetic data module
    dm = VLNDataModule(
        dataset_name="synthetic",
        batch_size=4,
        max_episodes=16,
        seed=42,
    )
    
    # Get statistics
    stats = dm.get_stats()
    print(f"Dataset: {stats['dataset_name']}")
    print(f"Batch size: {stats['batch_size']}")
    
    # Create data loaders
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    obs, labels = batch
    print(f"Batch shapes - Features: {obs['feat'].shape}, Labels: {labels.shape}")
    print(f"Images in batch: {len(obs['rgb'])}")
    
    print("✓ Synthetic data loading successful\n")


def demo_real_data_structure():
    """Demonstrate the normalized episode structure."""
    print("="*60)
    print("NORMALIZED EPISODE SCHEMA DEMO")
    print("="*60)
    
    from dygrav.data import ViewPoint, NavigationAction, GroundingAnnotation
    
    # Create example episode
    viewpoints = [
        ViewPoint("vp_001", (0.0, 0.0, 0.0), 0.0, 0.0, image_path="scan1/vp_001.jpg"),
        ViewPoint("vp_002", (1.5, 0.0, 0.0), 1.57, 0.0, image_path="scan1/vp_002.jpg"),
        ViewPoint("vp_003", (3.0, 1.0, 0.0), 3.14, 0.0, image_path="scan1/vp_003.jpg"),
    ]
    
    actions = [
        NavigationAction("forward", target_viewpoint="vp_002"),
        NavigationAction("turn_left"),
        NavigationAction("forward", target_viewpoint="vp_003"),
    ]
    
    grounding = [
        GroundingAnnotation(
            phrase="red chair",
            bbox=(120, 80, 200, 160),
            viewpoint_id="vp_002",
            step_idx=1,
            category="attribute",
            confidence=0.95
        )
    ]
    
    episode = Episode(
        episode_id="demo_001",
        dataset="demo",
        split="train",
        instruction="Walk forward and turn left to find the red chair",
        path=viewpoints,
        actions=actions,
        start_position=(0.0, 0.0, 0.0),
        goal_position=(3.0, 1.0, 0.0),
        trajectory_length=4.5,
        shortest_path_length=4.2,
        grounding_annotations=grounding,
        scan_id="scan_001",
    )
    
    print(f"Episode ID: {episode.episode_id}")
    print(f"Instruction: {episode.instruction}")
    print(f"Number of steps: {episode.num_steps}")
    print(f"Has grounding: {episode.has_grounding_annotations}")
    print(f"Grounding categories: {episode.grounding_categories}")
    print(f"Trajectory length: {episode.trajectory_length:.1f}m")
    
    # Test episode methods
    step_1_grounding = episode.get_grounding_for_step(1)
    print(f"Grounding at step 1: {len(step_1_grounding)} annotations")
    
    episode_dict = episode.to_dict()
    print(f"Episode dict keys: {list(episode_dict.keys())}")
    
    print("✓ Episode schema demonstration successful\n")


def demo_batch_collation():
    """Demonstrate episode batching and collation."""
    print("="*60)
    print("BATCH COLLATION DEMO")
    print("="*60)
    
    from dygrav.data import ViewPoint, NavigationAction, collate_episodes
    
    # Create episodes with different lengths
    episodes = []
    for i, length in enumerate([3, 5, 4]):
        viewpoints = [
            ViewPoint(f"vp_{j}", (float(j), 0.0, 0.0), 0.0, 0.0)
            for j in range(length)
        ]
        
        actions = [
            NavigationAction("forward", target_viewpoint=f"vp_{j+1}")
            for j in range(length - 1)
        ]
        
        episode = Episode(
            episode_id=f"batch_demo_{i}",
            dataset="demo",
            split="train",
            instruction=f"Navigate through {length} viewpoints",
            path=viewpoints,
            actions=actions,
            start_position=(0.0, 0.0, 0.0),
            goal_position=(float(length-1), 0.0, 0.0),
            trajectory_length=float(length-1),
        )
        episodes.append(episode)
    
    # Collate episodes into batch
    batch = collate_episodes(episodes, load_images=False)
    
    print(f"Batch size: {batch.batch_size}")
    print(f"Max length: {batch.max_length}")
    print(f"Sequence lengths: {batch.sequence_lengths}")
    print(f"Episode IDs: {batch.episode_ids}")
    
    # Check padding
    print(f"All image sequences length: {[len(img_seq) for img_seq in batch.images]}")
    print(f"All position sequences length: {[len(pos_seq) for pos_seq in batch.positions]}")
    
    print("✓ Batch collation demonstration successful\n")


def demo_data_module_usage():
    """Demonstrate VLNDataModule usage patterns."""
    print("="*60)
    print("VLN DATA MODULE USAGE DEMO")
    print("="*60)
    
    # Example configurations for different datasets
    configs = {
        "synthetic": {
            "dataset_name": "synthetic",
            "batch_size": 8,
            "max_episodes": 32,
        },
        "r2r": {
            "dataset_name": "r2r",
            "data_root": "./data/r2r",  # Would need actual data
            "batch_size": 4,
            "max_episodes": 100,
            "max_path_length": 20,
            "load_images": True,
        },
        "rxr": {
            "dataset_name": "rxr",
            "data_root": "./data/rxr",  # Would need actual data
            "batch_size": 4,
            "language": "en",
            "include_grounding": False,
            "max_episodes": 50,
        },
        "rxr_fg": {
            "dataset_name": "rxr_fg",
            "data_root": "./data/rxr",  # Would need actual data
            "batch_size": 4,
            "language": "en",
            "include_grounding": True,
            "max_episodes": 30,
        }
    }
    
    # Show configuration for each dataset type
    for name, config in configs.items():
        print(f"\n{name.upper()} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        # Only create data module for synthetic (doesn't require external data)
        if name == "synthetic":
            try:
                dm = VLNDataModule(**config)
                stats = dm.get_stats()
                print(f"  ✓ Created successfully - {stats['train_stats']['num_episodes']} episodes")
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print("\n✓ Data module usage demonstration complete\n")


def demo_rxr_fg_builder():
    """Demonstrate RxR-FG dataset building."""
    print("="*60)
    print("RxR-FG BUILDER DEMO")
    print("="*60)
    
    print("The RxR-FG builder can be used to create fine-grained subsets:")
    print()
    print("Command line usage:")
    print("  python -m dygrav.tools.build_rxr_fg \\")
    print("    --rxr_root /path/to/rxr/data \\")
    print("    --output_dir /path/to/rxr_fg/output \\")
    print("    --splits train val_seen val_unseen \\")
    print("    --languages en \\")
    print("    --verbose")
    print()
    print("Programmatic usage:")
    print("  from dygrav.tools.build_rxr_fg import RxRFGBuilder")
    print("  builder = RxRFGBuilder(rxr_root, output_dir)")
    print("  results = builder.build_dataset(['train'], ['en'])")
    print()
    print("Filtering criteria include:")
    print("  - Spatial relations: 'between', 'next to', 'left of', etc.")
    print("  - Visual attributes: 'red', 'wooden', 'smallest', etc.")
    print("  - Quality filters: instruction length, spatial diversity")
    print("  - Path constraints: min/max length limits")
    
    print("✓ RxR-FG builder demonstration complete\n")


def demo_evaluation_integration():
    """Demonstrate integration with evaluation system."""
    print("="*60)
    print("EVALUATION INTEGRATION DEMO")
    print("="*60)
    
    from dygrav.eval.evaluator import EvalConfig, run_eval
    
    # Show how to configure evaluation with real datasets
    configs = {
        "synthetic": EvalConfig(
            dataset="synthetic",
            num_episodes=20,
            use_dygrav=True,
            verbose=True,
        ),
        "r2r": EvalConfig(
            dataset="r2r",
            data_path="./data/r2r",
            num_episodes=50,
            use_dygrav=True,
            tau_conf=0.55,
            tau_entropy=1.25,
        ),
        "rxr_fg": EvalConfig(
            dataset="rxr_fg", 
            data_path="./data/rxr",
            num_episodes=30,
            use_dygrav=True,
            iou_threshold=0.5,
            save_predictions=True,
        )
    }
    
    print("Evaluation configurations:")
    for name, config in configs.items():
        print(f"\n{name.upper()}:")
        print(f"  Dataset: {config.dataset}")
        print(f"  Episodes: {config.num_episodes}")
        print(f"  DyGRAV enabled: {config.use_dygrav}")
        if hasattr(config, 'data_path') and config.data_path:
            print(f"  Data path: {config.data_path}")
    
    # Run synthetic evaluation (works without external data)
    print(f"\nRunning synthetic evaluation...")
    try:
        results = run_eval(n=10, config=configs["synthetic"])
        print(f"✓ Evaluation complete:")
        print(f"  Trigger rate: {results['trigger_rate']:.3f}")
        print(f"  Grounding accuracy: {results['GA']:.3f}")
        print(f"  Success rate: {results['SR']:.3f}")
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
    
    print("✓ Evaluation integration demonstration complete\n")


def main():
    """Run all demonstrations."""
    print("DyGRAV REAL DATA LOADERS DEMONSTRATION")
    print("=" * 80)
    print()
    
    try:
        demo_synthetic_data()
        demo_real_data_structure()
        demo_batch_collation()
        demo_data_module_usage()
        demo_rxr_fg_builder()
        demo_evaluation_integration()
        
        print("=" * 80)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Obtain R2R/RxR datasets and place in ./data/ directory")
        print("2. Run RxR-FG builder to create fine-grained subset")
        print("3. Update config files with actual data paths")
        print("4. Run training/evaluation with real data")
        print()
        print("For more information, see:")
        print("- dygrav/data/ - Dataset implementations")
        print("- configs/data/ - Dataset configurations") 
        print("- tests/test_data_loaders.py - Unit tests")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
