"""
Calibration methods for skill logits and uncertainty features.
Implements temperature scaling and ECE computation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class CalibrationMetrics:
    """Metrics for model calibration."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    temperature: float  # Optimal temperature
    reliability_diagram: Optional[np.ndarray] = None
    bin_accuracies: Optional[np.ndarray] = None
    bin_confidences: Optional[np.ndarray] = None
    bin_counts: Optional[np.ndarray] = None


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibration.
    Learns a single temperature parameter to calibrate predictions.
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Fit temperature using validation data.
        
        Args:
            logits: Model logits [N, C]
            labels: True labels [N]
            lr: Learning rate
            max_iter: Maximum iterations
            
        Returns:
            Optimal temperature value
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = self.forward(logits)
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        return self.temperature.item()


class MultiLabelCalibrator:
    """
    Calibration for multi-label classification (skill logits).
    Calibrates each label independently using isotonic regression or temperature scaling.
    """
    
    def __init__(self, num_labels: int = 5):
        self.num_labels = num_labels
        self.temperatures = [1.0] * num_labels
        self.calibrators = []
    
    def fit(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        method: str = "temperature"
    ):
        """
        Fit calibration for multi-label predictions.
        
        Args:
            logits: Model logits [N, L] where L is number of labels
            labels: True multi-labels [N, L] (binary)
            method: Calibration method ("temperature" or "isotonic")
        """
        for i in range(self.num_labels):
            label_logits = logits[:, i]
            label_targets = labels[:, i]
            
            if method == "temperature":
                # Fit temperature for this label
                temp_scaler = TemperatureScaling()
                optimal_temp = temp_scaler.fit(
                    label_logits.unsqueeze(1),
                    label_targets.long(),
                    lr=0.01,
                    max_iter=50
                )
                self.temperatures[i] = optimal_temp
            else:
                # Isotonic regression (requires sklearn)
                try:
                    from sklearn.isotonic import IsotonicRegression
                    probs = torch.sigmoid(label_logits).cpu().numpy()
                    targets = label_targets.cpu().numpy()
                    
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(probs, targets)
                    self.calibrators.append(calibrator)
                except ImportError:
                    print("sklearn not available, falling back to temperature scaling")
                    self.temperatures[i] = 1.0
    
    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to multi-label logits.
        
        Args:
            logits: Uncalibrated logits [N, L]
            
        Returns:
            Calibrated probabilities [N, L]
        """
        calibrated = torch.zeros_like(logits)
        
        for i in range(self.num_labels):
            if self.calibrators and i < len(self.calibrators):
                # Apply isotonic regression
                probs = torch.sigmoid(logits[:, i])
                calibrated_probs = self.calibrators[i].transform(probs.cpu().numpy())
                calibrated[:, i] = torch.tensor(calibrated_probs, device=logits.device)
            else:
                # Apply temperature scaling
                calibrated[:, i] = torch.sigmoid(logits[:, i] / self.temperatures[i])
        
        return calibrated


def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 10
) -> CalibrationMetrics:
    """
    Compute Expected Calibration Error (ECE) and related metrics.
    
    Args:
        probs: Predicted probabilities [N, C] or [N] for binary
        labels: True labels [N]
        n_bins: Number of bins for ECE
        
    Returns:
        CalibrationMetrics with ECE, MCE, and reliability diagram
    """
    if probs.dim() == 2:
        # Multi-class: use confidence of predicted class
        confidences, predictions = probs.max(dim=1)
    else:
        # Binary
        confidences = probs
        predictions = (probs > 0.5).long()
    
    accuracies = (predictions == labels).float()
    
    # Create bins
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            # Compute accuracy and confidence in bin
            accuracy_in_bin = accuracies[in_bin].mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            
            # ECE contribution
            ece += prop_in_bin * abs(avg_confidence_in_bin - accuracy_in_bin)
            
            # MCE update
            mce = max(mce, abs(avg_confidence_in_bin - accuracy_in_bin))
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum().item())
        else:
            bin_accuracies.append(0)
            bin_confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    return CalibrationMetrics(
        ece=ece.item() if torch.is_tensor(ece) else ece,
        mce=mce,
        temperature=1.0,  # Not computed here
        bin_accuracies=np.array(bin_accuracies),
        bin_confidences=np.array(bin_confidences),
        bin_counts=np.array(bin_counts)
    )


def plot_reliability_diagram(
    metrics: CalibrationMetrics,
    save_path: Optional[Path] = None
):
    """
    Plot reliability diagram from calibration metrics.
    
    Args:
        metrics: CalibrationMetrics object
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Plot actual calibration
    plt.bar(
        metrics.bin_confidences,
        metrics.bin_accuracies,
        width=0.1,
        alpha=0.5,
        edgecolor='black',
        label=f'ECE = {metrics.ece:.3f}'
    )
    
    plt.xlabel('Mean Predicted Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


class UncertaintyCalibrator:
    """
    Calibrator for uncertainty features (entropy, logit gap).
    Ensures these features are well-calibrated for gate decisions.
    """
    
    def __init__(self):
        self.entropy_scaler = TemperatureScaling()
        self.logit_gap_scaler = TemperatureScaling()
        self.calibrated = False
    
    def fit(
        self,
        entropy_values: torch.Tensor,
        logit_gaps: torch.Tensor,
        should_trigger: torch.Tensor
    ):
        """
        Fit calibration for uncertainty features.
        
        Args:
            entropy_values: Entropy values [N]
            logit_gaps: Logit gap values [N]
            should_trigger: Binary labels for whether gate should trigger [N]
        """
        # Calibrate entropy as a predictor of trigger
        entropy_logits = entropy_values.unsqueeze(1)
        self.entropy_temp = self.entropy_scaler.fit(
            entropy_logits,
            should_trigger,
            lr=0.01,
            max_iter=50
        )
        
        # Calibrate logit gap
        gap_logits = -logit_gaps.unsqueeze(1)  # Negative because low gap = high uncertainty
        self.gap_temp = self.logit_gap_scaler.fit(
            gap_logits,
            should_trigger,
            lr=0.01,
            max_iter=50
        )
        
        self.calibrated = True
    
    def calibrate_entropy(self, entropy: torch.Tensor) -> torch.Tensor:
        """Apply calibration to entropy values."""
        if not self.calibrated:
            return entropy
        return entropy / self.entropy_scaler.temperature
    
    def calibrate_logit_gap(self, gap: torch.Tensor) -> torch.Tensor:
        """Apply calibration to logit gap values."""
        if not self.calibrated:
            return gap
        return gap / self.logit_gap_scaler.temperature


def calibrate_model(
    model,
    val_loader,
    device: str = 'cuda'
) -> Dict[str, CalibrationMetrics]:
    """
    Full model calibration pipeline.
    
    Args:
        model: Model to calibrate
        val_loader: Validation data loader
        device: Device to use
        
    Returns:
        Dictionary of calibration metrics for different components
    """
    model.eval()
    
    # Collect predictions
    all_logits = []
    all_labels = []
    all_entropies = []
    all_gaps = []
    all_skill_logits = []
    all_skill_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            obs, labels = batch
            
            # Get model predictions
            output = model(obs)
            
            all_logits.append(output['logits'])
            all_labels.append(labels)
            
            # Collect uncertainty features
            if 'entropy' in output:
                all_entropies.append(output['entropy'])
            if 'logit_gap' in output:
                all_gaps.append(output['logit_gap'])
            if 'skill_logits' in output:
                all_skill_logits.append(output['skill_logits'])
                all_skill_labels.append(obs.get('skill_labels', torch.zeros_like(output['skill_logits'])))
    
    # Concatenate all
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Calibrate main predictions
    temp_scaler = TemperatureScaling()
    optimal_temp = temp_scaler.fit(all_logits, all_labels)
    
    # Compute ECE before and after calibration
    probs_before = F.softmax(all_logits, dim=-1)
    metrics_before = compute_ece(probs_before, all_labels)
    
    probs_after = F.softmax(all_logits / optimal_temp, dim=-1)
    metrics_after = compute_ece(probs_after, all_labels)
    metrics_after.temperature = optimal_temp
    
    results = {
        'main_before': metrics_before,
        'main_after': metrics_after
    }
    
    # Calibrate skill logits if available
    if all_skill_logits:
        all_skill_logits = torch.cat(all_skill_logits)
        all_skill_labels = torch.cat(all_skill_labels)
        
        skill_calibrator = MultiLabelCalibrator(num_labels=all_skill_logits.shape[1])
        skill_calibrator.fit(all_skill_logits, all_skill_labels)
        
        # Compute ECE for each skill
        for i in range(all_skill_logits.shape[1]):
            skill_probs = torch.sigmoid(all_skill_logits[:, i])
            skill_metrics = compute_ece(skill_probs, all_skill_labels[:, i])
            results[f'skill_{i}'] = skill_metrics
    
    # Calibrate uncertainty features if available
    if all_entropies and all_gaps:
        all_entropies = torch.cat(all_entropies)
        all_gaps = torch.cat(all_gaps)
        
        # Create pseudo-labels for when gate should trigger
        # (high entropy or low gap)
        should_trigger = (all_entropies > all_entropies.median()) | (all_gaps < all_gaps.median())
        
        uncertainty_calibrator = UncertaintyCalibrator()
        uncertainty_calibrator.fit(all_entropies, all_gaps, should_trigger.float())
        
        results['uncertainty_calibrator'] = uncertainty_calibrator
    
    return results
