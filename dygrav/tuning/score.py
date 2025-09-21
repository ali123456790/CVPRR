"""Scoring functions for hyperparameter optimization."""

from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np


def bi_objective(
    ga: float,
    tr: float,
    tr_star: float = 0.10,
    alpha: float = 0.5
) -> float:
    """
    Bi-objective scoring function for DyGRAV threshold calibration.
    
    This function balances grounding accuracy (GA) against trigger rate (TR),
    with a target trigger rate of tr_star. The alpha parameter controls the
    trade-off between maximizing GA and staying close to the target TR.
    
    Args:
        ga: Grounding accuracy (0.0 to 1.0)
        tr: Trigger rate (0.0 to 1.0) 
        tr_star: Target trigger rate (default: 0.10 = 10%)
        alpha: Trade-off parameter (0.0 = only GA, 1.0 = only TR penalty)
        
    Returns:
        Bi-objective score (higher is better)
        
    Examples:
        >>> bi_objective(0.8, 0.10)  # Perfect trigger rate
        0.8
        >>> bi_objective(0.8, 0.15)  # 5% above target
        0.775
        >>> bi_objective(0.8, 0.05)  # 5% below target  
        0.775
    """
    if not (0.0 <= ga <= 1.0):
        raise ValueError(f"Grounding accuracy must be in [0, 1], got {ga}")
    if not (0.0 <= tr <= 1.0):
        raise ValueError(f"Trigger rate must be in [0, 1], got {tr}")
    if not (0.0 <= tr_star <= 1.0):
        raise ValueError(f"Target trigger rate must be in [0, 1], got {tr_star}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
    
    # Calculate trigger rate penalty
    tr_penalty = abs(tr - tr_star)
    
    # Combine objectives
    score = ga - alpha * tr_penalty
    
    return score


def multi_objective_score(
    metrics: Dict[str, float],
    weights: Dict[str, float],
    targets: Optional[Dict[str, float]] = None,
    penalties: Optional[Dict[str, float]] = None,
) -> float:
    """
    Multi-objective scoring function for complex optimization.
    
    Args:
        metrics: Dictionary of metric values
        weights: Dictionary of metric weights (must sum to 1.0)
        targets: Optional target values for penalty-based metrics
        penalties: Optional penalty weights for target deviations
        
    Returns:
        Weighted multi-objective score
        
    Examples:
        >>> metrics = {"ga": 0.8, "tr": 0.12, "spl": 0.65}
        >>> weights = {"ga": 0.5, "tr": 0.0, "spl": 0.5}
        >>> targets = {"tr": 0.10}
        >>> penalties = {"tr": 0.2}
        >>> multi_objective_score(metrics, weights, targets, penalties)
        0.721
    """
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-6):
        raise ValueError(f"Weights must sum to 1.0, got {sum(weights.values())}")
    
    score = 0.0
    
    # Add weighted metrics
    for metric, value in metrics.items():
        if metric in weights:
            score += weights[metric] * value
    
    # Apply target penalties
    if targets and penalties:
        for metric, target in targets.items():
            if metric in metrics and metric in penalties:
                deviation = abs(metrics[metric] - target)
                penalty = penalties[metric] * deviation
                score -= penalty
    
    return score


def pareto_dominance_score(
    metrics: Dict[str, float],
    reference_metrics: List[Dict[str, float]],
    maximize: List[str],
    minimize: List[str],
) -> Dict[str, Any]:
    """
    Calculate Pareto dominance score against reference points.
    
    Args:
        metrics: Current metrics
        reference_metrics: List of reference metric dictionaries
        maximize: List of metrics to maximize
        minimize: List of metrics to minimize
        
    Returns:
        Dictionary with dominance statistics
    """
    dominates = 0
    dominated_by = 0
    non_dominated = 0
    
    for ref_metrics in reference_metrics:
        dominance = compare_pareto_dominance(
            metrics, ref_metrics, maximize, minimize
        )
        
        if dominance == 1:
            dominates += 1
        elif dominance == -1:
            dominated_by += 1
        else:
            non_dominated += 1
    
    total = len(reference_metrics)
    
    return {
        "dominates": dominates,
        "dominated_by": dominated_by,
        "non_dominated": non_dominated,
        "dominance_ratio": dominates / total if total > 0 else 0.0,
        "dominated_ratio": dominated_by / total if total > 0 else 0.0,
        "pareto_score": (dominates - dominated_by) / total if total > 0 else 0.0,
    }


def compare_pareto_dominance(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float], 
    maximize: List[str],
    minimize: List[str],
) -> int:
    """
    Compare two metric sets for Pareto dominance.
    
    Args:
        metrics_a: First metric set
        metrics_b: Second metric set
        maximize: Metrics to maximize
        minimize: Metrics to minimize
        
    Returns:
        1 if A dominates B, -1 if B dominates A, 0 if non-dominated
    """
    a_better = False
    b_better = False
    
    # Check maximize metrics
    for metric in maximize:
        if metric in metrics_a and metric in metrics_b:
            if metrics_a[metric] > metrics_b[metric]:
                a_better = True
            elif metrics_a[metric] < metrics_b[metric]:
                b_better = True
    
    # Check minimize metrics
    for metric in minimize:
        if metric in metrics_a and metric in metrics_b:
            if metrics_a[metric] < metrics_b[metric]:
                a_better = True
            elif metrics_a[metric] > metrics_b[metric]:
                b_better = True
    
    if a_better and not b_better:
        return 1   # A dominates B
    elif b_better and not a_better:
        return -1  # B dominates A
    else:
        return 0   # Non-dominated


def calculate_hypervolume(
    points: List[Dict[str, float]],
    reference_point: Dict[str, float],
    maximize: List[str],
    minimize: List[str],
) -> float:
    """
    Calculate hypervolume indicator for multi-objective optimization.
    
    Args:
        points: List of metric dictionaries
        reference_point: Reference point for hypervolume calculation
        maximize: Metrics to maximize
        minimize: Metrics to minimize
        
    Returns:
        Hypervolume indicator value
        
    Note:
        This is a simplified 2D hypervolume calculation.
        For higher dimensions, consider using specialized libraries.
    """
    if len(maximize) + len(minimize) != 2:
        raise NotImplementedError("Hypervolume calculation only supports 2D problems")
    
    if not points:
        return 0.0
    
    # Transform points to maximization problem
    transformed_points = []
    
    for point in points:
        transformed = {}
        
        for metric in maximize:
            if metric in point:
                transformed[metric] = point[metric]
        
        for metric in minimize:
            if metric in point:
                transformed[metric] = -point[metric]  # Flip for maximization
        
        transformed_points.append(transformed)
    
    # Transform reference point
    ref_transformed = {}
    for metric in maximize:
        if metric in reference_point:
            ref_transformed[metric] = reference_point[metric]
    
    for metric in minimize:
        if metric in reference_point:
            ref_transformed[metric] = -reference_point[metric]
    
    # Calculate 2D hypervolume using sweep line algorithm
    metrics = list(ref_transformed.keys())
    if len(metrics) != 2:
        raise ValueError("Exactly 2 metrics required for hypervolume calculation")
    
    # Sort points by first metric
    sorted_points = sorted(
        transformed_points,
        key=lambda p: p[metrics[0]],
        reverse=True
    )
    
    hypervolume = 0.0
    prev_y = ref_transformed[metrics[1]]
    
    for point in sorted_points:
        x = point[metrics[0]]
        y = point[metrics[1]]
        
        if x > ref_transformed[metrics[0]] and y > prev_y:
            width = x - ref_transformed[metrics[0]]
            height = y - prev_y
            hypervolume += width * height
            prev_y = y
    
    return hypervolume


def sensitivity_analysis(
    parameter_values: List[float],
    metric_values: List[float],
    parameter_name: str = "parameter",
    metric_name: str = "metric",
) -> Dict[str, Any]:
    """
    Perform sensitivity analysis for parameter tuning.
    
    Args:
        parameter_values: List of parameter values
        metric_values: Corresponding metric values
        parameter_name: Name of the parameter
        metric_name: Name of the metric
        
    Returns:
        Dictionary with sensitivity statistics
    """
    if len(parameter_values) != len(metric_values):
        raise ValueError("Parameter and metric arrays must have same length")
    
    if len(parameter_values) < 2:
        raise ValueError("At least 2 data points required for sensitivity analysis")
    
    # Convert to numpy arrays
    params = np.array(parameter_values)
    metrics = np.array(metric_values)
    
    # Calculate derivatives (finite differences)
    param_diffs = np.diff(params)
    metric_diffs = np.diff(metrics)
    
    # Avoid division by zero
    nonzero_mask = param_diffs != 0
    if not np.any(nonzero_mask):
        derivatives = np.zeros_like(param_diffs)
    else:
        derivatives = np.zeros_like(param_diffs)
        derivatives[nonzero_mask] = metric_diffs[nonzero_mask] / param_diffs[nonzero_mask]
    
    # Calculate sensitivity metrics
    sensitivity_stats = {
        "parameter_name": parameter_name,
        "metric_name": metric_name,
        "num_points": len(parameter_values),
        "parameter_range": {
            "min": float(np.min(params)),
            "max": float(np.max(params)),
            "mean": float(np.mean(params)),
            "std": float(np.std(params)),
        },
        "metric_range": {
            "min": float(np.min(metrics)),
            "max": float(np.max(metrics)),
            "mean": float(np.mean(metrics)),
            "std": float(np.std(metrics)),
        },
        "sensitivity": {
            "mean_derivative": float(np.mean(derivatives)),
            "std_derivative": float(np.std(derivatives)),
            "max_derivative": float(np.max(np.abs(derivatives))),
            "correlation": float(np.corrcoef(params, metrics)[0, 1]) if len(params) > 1 else 0.0,
        },
        "optimal_point": {
            "parameter": float(params[np.argmax(metrics)]),
            "metric": float(np.max(metrics)),
        },
    }
    
    # Determine sensitivity level
    max_derivative = sensitivity_stats["sensitivity"]["max_derivative"]
    if max_derivative < 0.1:
        sensitivity_level = "low"
    elif max_derivative < 1.0:
        sensitivity_level = "medium"
    else:
        sensitivity_level = "high"
    
    sensitivity_stats["sensitivity"]["level"] = sensitivity_level
    
    return sensitivity_stats
