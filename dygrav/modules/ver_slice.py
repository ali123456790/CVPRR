"""
VER-slice expert for handling vertical geometry and spatial reasoning.
Uses voxel grids for 3D understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.types import Region


@dataclass
class VoxelGrid:
    """3D voxel grid representation."""
    data: torch.Tensor  # [channels, depth, height, width]
    origin: np.ndarray  # [3] world coordinates of grid origin
    voxel_size: float  # Size of each voxel in meters
    radius: float  # Grid radius in meters


@dataclass
class VerticalAnalysis:
    """Results from vertical geometry analysis."""
    has_stairs: bool
    stair_direction: Optional[str]  # 'up', 'down', or None
    num_steps: int
    has_ramp: bool
    ramp_angle: float  # degrees
    floor_changes: List[float]  # Height changes detected
    navigable_regions: List[Tuple[int, int, int]]  # Voxel indices


class VERSliceExpert(nn.Module):
    """
    VER-slice expert for vertical and geometric reasoning.
    Builds frustum-aligned voxel grids for local 3D analysis.
    """
    
    def __init__(
        self,
        voxel_size: float = 0.075,  # 7.5cm voxels
        default_radius: float = 1.0,  # 1 meter default radius
        height_levels: int = 32,
        channels: int = 4  # occupancy, height, normal_z, semantics
    ):
        super().__init__()
        
        self.voxel_size = voxel_size
        self.default_radius = default_radius
        self.height_levels = height_levels
        self.channels = channels
        
        # Learned components for vertical understanding
        self.vertical_encoder = nn.Sequential(
            nn.Conv3d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Stair/ramp detector
        self.geometry_classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 128),  # Assuming 8x8x8 after pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 6)  # has_stairs, stair_up, stair_down, has_ramp, num_steps, ramp_angle
        )
        
        # Action bias predictor
        self.action_bias = nn.Linear(32, 6)  # Bias for navigation actions
    
    def build_voxel_grid(
        self,
        depth_map: torch.Tensor,
        rgb_image: Optional[torch.Tensor] = None,
        camera_pose: Optional[np.ndarray] = None,
        radius: float = None
    ) -> VoxelGrid:
        """
        Build a frustum-aligned voxel grid from depth and RGB.
        
        Args:
            depth_map: [H, W] depth map in meters
            rgb_image: Optional [3, H, W] RGB image
            camera_pose: Optional [4, 4] camera pose matrix
            radius: Grid radius in meters
            
        Returns:
            VoxelGrid object
        """
        if radius is None:
            radius = self.default_radius
        
        H, W = depth_map.shape
        device = depth_map.device
        
        # Compute grid dimensions
        grid_size = int(2 * radius / self.voxel_size)
        
        # Initialize voxel grid
        voxel_data = torch.zeros(
            self.channels,
            self.height_levels,
            grid_size,
            grid_size,
            device=device
        )
        
        # Generate 3D points from depth map
        points_3d = self._depth_to_points(depth_map, camera_pose)
        
        # Filter points within radius
        mask = torch.norm(points_3d[:, :2], dim=1) < radius
        points_3d = points_3d[mask]
        
        if rgb_image is not None and mask.sum() > 0:
            # Get corresponding colors
            rgb_flat = rgb_image.reshape(3, -1).T
            colors = rgb_flat[mask]
        else:
            colors = None
        
        # Voxelize points
        self._voxelize_points(voxel_data, points_3d, colors, radius)
        
        # Compute additional channels
        self._compute_normals(voxel_data)
        self._compute_height_map(voxel_data)
        
        origin = np.array([-radius, -radius, 0.0])
        if camera_pose is not None:
            origin = camera_pose[:3, 3] + origin
        
        return VoxelGrid(
            data=voxel_data,
            origin=origin,
            voxel_size=self.voxel_size,
            radius=radius
        )
    
    def _depth_to_points(
        self,
        depth_map: torch.Tensor,
        camera_pose: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Convert depth map to 3D points."""
        H, W = depth_map.shape
        device = depth_map.device
        
        # Camera intrinsics (simplified - should be provided)
        fx = fy = W / 2.0
        cx = W / 2.0
        cy = H / 2.0
        
        # Create pixel grid
        xx, yy = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing='xy'
        )
        
        # Back-project to 3D
        z = depth_map
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        
        # Stack to get points
        points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        
        # Apply camera pose if provided
        if camera_pose is not None:
            R = torch.tensor(camera_pose[:3, :3], device=device, dtype=torch.float32)
            t = torch.tensor(camera_pose[:3, 3], device=device, dtype=torch.float32)
            points = points @ R.T + t
        
        return points
    
    def _voxelize_points(
        self,
        voxel_data: torch.Tensor,
        points: torch.Tensor,
        colors: Optional[torch.Tensor],
        radius: float
    ):
        """Convert 3D points to voxel occupancy."""
        # Convert points to voxel indices
        voxel_indices = ((points + radius) / self.voxel_size).long()
        
        # Filter valid indices
        valid_mask = (
            (voxel_indices >= 0).all(dim=1) &
            (voxel_indices[:, 0] < voxel_data.shape[3]) &
            (voxel_indices[:, 1] < voxel_data.shape[2]) &
            (voxel_indices[:, 2] < voxel_data.shape[1])
        )
        
        voxel_indices = voxel_indices[valid_mask]
        
        # Set occupancy
        for idx in voxel_indices:
            z, y, x = idx
            voxel_data[0, z, y, x] = 1.0  # Occupancy channel
            
            # Add color/semantic info if available
            if colors is not None and valid_mask.sum() > 0:
                # Simple semantic channel based on color (placeholder)
                color_idx = valid_mask.nonzero()[0]
                if color_idx < len(colors):
                    voxel_data[3, z, y, x] = colors[color_idx].mean()
    
    def _compute_normals(self, voxel_data: torch.Tensor):
        """Compute surface normals from occupancy."""
        occupancy = voxel_data[0]
        
        # Compute gradients
        dz = F.conv3d(
            occupancy.unsqueeze(0).unsqueeze(0),
            torch.tensor([[[[-1], [0], [1]]]], device=occupancy.device, dtype=torch.float32),
            padding=(1, 0, 0)
        ).squeeze()
        
        # Store vertical component of normal
        voxel_data[2] = torch.abs(dz)
    
    def _compute_height_map(self, voxel_data: torch.Tensor):
        """Compute height map from occupancy."""
        occupancy = voxel_data[0]
        
        # For each x,y position, find the highest occupied voxel
        height_map = torch.zeros_like(occupancy[0])
        for y in range(occupancy.shape[1]):
            for x in range(occupancy.shape[2]):
                column = occupancy[:, y, x]
                if column.sum() > 0:
                    highest_z = column.nonzero()[-1].item()
                    height_map[y, x] = highest_z * self.voxel_size
        
        # Store in channel 1
        voxel_data[1, 0] = height_map
    
    def analyze_vertical_geometry(self, voxel_grid: VoxelGrid) -> VerticalAnalysis:
        """
        Analyze voxel grid for vertical structures.
        
        Args:
            voxel_grid: VoxelGrid to analyze
            
        Returns:
            VerticalAnalysis with detected features
        """
        # Pass through encoder
        features = self.vertical_encoder(voxel_grid.data.unsqueeze(0))
        features_flat = features.flatten(1)
        
        # Classify geometry
        predictions = self.geometry_classifier(features_flat)
        
        # Parse predictions
        has_stairs = torch.sigmoid(predictions[0, 0]) > 0.5
        stair_up_prob = torch.sigmoid(predictions[0, 1])
        stair_down_prob = torch.sigmoid(predictions[0, 2])
        has_ramp = torch.sigmoid(predictions[0, 3]) > 0.5
        num_steps = int(torch.relu(predictions[0, 4]).item())
        ramp_angle = predictions[0, 5].item() * 45.0  # Scale to degrees
        
        # Determine stair direction
        stair_direction = None
        if has_stairs:
            if stair_up_prob > stair_down_prob:
                stair_direction = 'up'
            else:
                stair_direction = 'down'
        
        # Detect floor changes from height map
        height_map = voxel_grid.data[1, 0]
        floor_changes = self._detect_floor_changes(height_map)
        
        # Find navigable regions (simplified)
        navigable = self._find_navigable_regions(voxel_grid.data[0])
        
        return VerticalAnalysis(
            has_stairs=has_stairs.item(),
            stair_direction=stair_direction,
            num_steps=num_steps,
            has_ramp=has_ramp.item(),
            ramp_angle=ramp_angle,
            floor_changes=floor_changes,
            navigable_regions=navigable
        )
    
    def _detect_floor_changes(self, height_map: torch.Tensor) -> List[float]:
        """Detect significant height changes in the height map."""
        # Simple edge detection for floor changes
        height_grad = torch.abs(torch.diff(height_map, dim=0))
        significant_changes = (height_grad > 0.15).nonzero()  # 15cm threshold
        
        changes = []
        for idx in significant_changes:
            y, x = idx
            height_diff = height_map[y+1, x] - height_map[y, x]
            changes.append(height_diff.item())
        
        return changes
    
    def _find_navigable_regions(self, occupancy: torch.Tensor) -> List[Tuple[int, int, int]]:
        """Find navigable voxel regions."""
        # Simple approach: find unoccupied voxels near the ground
        navigable = []
        
        for z in range(min(5, occupancy.shape[0])):  # Check lower levels
            for y in range(occupancy.shape[1]):
                for x in range(occupancy.shape[2]):
                    if occupancy[z, y, x] < 0.1:  # Unoccupied
                        # Check if there's ground below
                        if z > 0 and occupancy[z-1, y, x] > 0.5:
                            navigable.append((z, y, x))
        
        return navigable
    
    def forward(
        self,
        depth_map: torch.Tensor,
        instruction: str,
        rgb_image: Optional[torch.Tensor] = None,
        radius: float = None
    ) -> Tuple[torch.Tensor, VerticalAnalysis]:
        """
        Process input and return action bias.
        
        Args:
            depth_map: Depth map
            instruction: Navigation instruction
            rgb_image: Optional RGB image
            radius: Analysis radius
            
        Returns:
            Action bias tensor and vertical analysis
        """
        # Build voxel grid
        voxel_grid = self.build_voxel_grid(depth_map, rgb_image, radius=radius)
        
        # Analyze geometry
        analysis = self.analyze_vertical_geometry(voxel_grid)
        
        # Compute action bias based on analysis
        features = self.vertical_encoder(voxel_grid.data.unsqueeze(0))
        features_flat = features.flatten(1)
        
        # Get intermediate features for bias
        intermediate = self.geometry_classifier[:3](features_flat)  # Use first layers
        bias = self.action_bias(intermediate)
        
        # Adjust bias based on detected features
        if analysis.has_stairs:
            if analysis.stair_direction == 'up':
                bias[2] += 1.0  # Bias towards "go up" action
            elif analysis.stair_direction == 'down':
                bias[3] += 1.0  # Bias towards "go down" action
        
        return bias.squeeze(0), analysis


def create_ver_slice_expert(budget: float = 1.0) -> VERSliceExpert:
    """
    Factory function to create VER-slice expert with specified budget.
    
    Args:
        budget: Radius budget in meters (0.5, 1.0, or 1.5)
        
    Returns:
        Configured VERSliceExpert
    """
    # Adjust voxel size based on budget
    if budget <= 0.5:
        voxel_size = 0.10  # Coarser for smaller radius
    elif budget <= 1.0:
        voxel_size = 0.075  # Default
    else:
        voxel_size = 0.05  # Finer for larger radius
    
    return VERSliceExpert(
        voxel_size=voxel_size,
        default_radius=budget
    )
