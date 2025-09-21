"""Threshold calibration with grid search and TPE optimization."""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from .score import bi_objective, sensitivity_analysis


class TauSweeper:
    """
    Threshold (τ) calibration using grid search warm-start and TPE optimization.
    
    This class implements the protocol for calibrating DyGRAV threshold parameters:
    1. Grid search to warm-start TPE with promising regions
    2. Tree-structured Parzen Estimator (TPE) optimization for 50 trials
    3. Sensitivity analysis and visualization
    4. Optimal τ values frozen to artifacts/threshold.json
    """
    
    def __init__(
        self,
        evaluation_function: Callable[[float], Dict[str, float]],
        tau_range: Tuple[float, float] = (0.0, 1.0),
        target_trigger_rate: float = 0.10,
        alpha: float = 0.5,
        grid_size: int = 21,
        tpe_trials: int = 50,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        """
        Initialize threshold sweeper.
        
        Args:
            evaluation_function: Function that takes τ and returns metrics dict
                                Must return at least {"ga": float, "tr": float}
            tau_range: Range of τ values to explore
            target_trigger_rate: Target trigger rate (tr_star)
            alpha: Trade-off parameter for bi-objective function
            grid_size: Number of grid points for warm-start
            tpe_trials: Number of TPE optimization trials
            random_seed: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.evaluation_function = evaluation_function
        self.tau_range = tau_range
        self.target_trigger_rate = target_trigger_rate
        self.alpha = alpha
        self.grid_size = grid_size
        self.tpe_trials = tpe_trials
        self.random_seed = random_seed
        self.verbose = verbose
        
        # Results storage
        self.grid_results: List[Dict[str, Any]] = []
        self.tpe_results: List[Dict[str, Any]] = []
        self.best_tau: Optional[float] = None
        self.best_score: Optional[float] = None
        self.best_metrics: Optional[Dict[str, float]] = None
        
        # Validation
        if not OPTUNA_AVAILABLE:
            print("Warning: optuna not available. TPE optimization will be skipped.")
            print("Install with: pip install optuna")
        
        if self.verbose:
            print(f"✓ TauSweeper initialized")
            print(f"  τ range: {tau_range}")
            print(f"  Target TR: {target_trigger_rate:.1%}")
            print(f"  Grid size: {grid_size}")
            print(f"  TPE trials: {tpe_trials}")
    
    def run_grid_search(self) -> List[Dict[str, Any]]:
        """
        Run grid search to warm-start TPE optimization.
        
        Returns:
            List of grid search results
        """
        if self.verbose:
            print("\n" + "="*60)
            print("GRID SEARCH WARM-START")
            print("="*60)
        
        # Generate grid points
        tau_values = np.linspace(
            self.tau_range[0], 
            self.tau_range[1], 
            self.grid_size
        )
        
        grid_results = []
        
        for i, tau in enumerate(tau_values):
            if self.verbose and (i + 1) % 5 == 0:
                print(f"  Grid point {i+1}/{self.grid_size}: τ={tau:.3f}")
            
            try:
                # Evaluate τ
                start_time = time.time()
                metrics = self.evaluation_function(tau)
                eval_time = time.time() - start_time
                
                # Calculate bi-objective score
                ga = metrics.get("ga", 0.0)
                tr = metrics.get("tr", 0.0)
                score = bi_objective(ga, tr, self.target_trigger_rate, self.alpha)
                
                result = {
                    "tau": tau,
                    "score": score,
                    "metrics": metrics,
                    "evaluation_time": eval_time,
                    "method": "grid_search",
                }
                
                grid_results.append(result)
                
                if self.verbose and i < 5:  # Show first few results
                    print(f"    GA: {ga:.3f}, TR: {tr:.3f}, Score: {score:.3f}")
                
            except Exception as e:
                if self.verbose:
                    print(f"    Error evaluating τ={tau:.3f}: {e}")
                continue
        
        self.grid_results = grid_results
        
        if self.verbose:
            print(f"✓ Grid search complete: {len(grid_results)} points")
            if grid_results:
                best_grid = max(grid_results, key=lambda x: x["score"])
                print(f"  Best grid point: τ={best_grid['tau']:.3f}, score={best_grid['score']:.3f}")
        
        return grid_results
    
    def run_tpe_optimization(self, warm_start_results: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Run TPE optimization, optionally warm-started with grid results.
        
        Args:
            warm_start_results: Optional warm-start results from grid search
            
        Returns:
            List of TPE optimization results
        """
        if not OPTUNA_AVAILABLE:
            if self.verbose:
                print("⚠️  Skipping TPE optimization (optuna not available)")
            return []
        
        if self.verbose:
            print("\n" + "="*60)
            print("TPE OPTIMIZATION")
            print("="*60)
        
        # Create Optuna study
        sampler = TPESampler(seed=self.random_seed, n_startup_trials=10)
        study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"tau_calibration_{int(time.time())}",
        )
        
        # Warm-start with grid results if available
        if warm_start_results:
            for result in warm_start_results:
                study.enqueue_trial({
                    "tau": result["tau"]
                })
            
            if self.verbose:
                print(f"  Warm-started with {len(warm_start_results)} grid points")
        
        # Define objective function for Optuna
        def objective(trial):
            tau = trial.suggest_float("tau", self.tau_range[0], self.tau_range[1])
            
            try:
                metrics = self.evaluation_function(tau)
                ga = metrics.get("ga", 0.0)
                tr = metrics.get("tr", 0.0)
                score = bi_objective(ga, tr, self.target_trigger_rate, self.alpha)
                
                # Store additional info in trial user attributes
                trial.set_user_attr("ga", ga)
                trial.set_user_attr("tr", tr)
                trial.set_user_attr("metrics", metrics)
                
                return score
                
            except Exception as e:
                if self.verbose:
                    print(f"    Trial failed for τ={tau:.3f}: {e}")
                return float('-inf')  # Return very low score for failed trials
        
        # Run optimization
        if self.verbose:
            print(f"  Running {self.tpe_trials} TPE trials...")
        
        study.optimize(objective, n_trials=self.tpe_trials, show_progress_bar=False)
        
        # Extract results
        tpe_results = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                result = {
                    "tau": trial.params["tau"],
                    "score": trial.value,
                    "metrics": trial.user_attrs.get("metrics", {}),
                    "evaluation_time": trial.duration.total_seconds() if trial.duration else 0.0,
                    "method": "tpe",
                    "trial_number": trial.number,
                }
                tpe_results.append(result)
        
        self.tpe_results = tpe_results
        
        if self.verbose:
            print(f"✓ TPE optimization complete: {len(tpe_results)} trials")
            if tpe_results:
                best_tpe = max(tpe_results, key=lambda x: x["score"])
                print(f"  Best TPE result: τ={best_tpe['tau']:.3f}, score={best_tpe['score']:.3f}")
        
        return tpe_results
    
    def find_optimal_tau(self) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal τ from all results.
        
        Returns:
            Tuple of (optimal_tau, metrics_dict)
        """
        all_results = self.grid_results + self.tpe_results
        
        if not all_results:
            raise ValueError("No results available. Run optimization first.")
        
        best_result = max(all_results, key=lambda x: x["score"])
        
        self.best_tau = best_result["tau"]
        self.best_score = best_result["score"]
        self.best_metrics = best_result["metrics"]
        
        if self.verbose:
            print(f"\n✓ Optimal τ found: {self.best_tau:.4f}")
            print(f"  Score: {self.best_score:.4f}")
            print(f"  Metrics: {self.best_metrics}")
        
        return self.best_tau, self.best_metrics
    
    def generate_sensitivity_plots(
        self,
        output_dir: str = "sensitivity_plots",
        show_plots: bool = False,
    ) -> Dict[str, str]:
        """
        Generate sensitivity plots showing GA/TR vs τ.
        
        Args:
            output_dir: Directory to save plots
            show_plots: Whether to display plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_results = self.grid_results + self.tpe_results
        if not all_results:
            if self.verbose:
                print("⚠️  No results available for plotting")
            return {}
        
        # Extract data
        tau_values = [r["tau"] for r in all_results]
        ga_values = [r["metrics"].get("ga", 0.0) for r in all_results]
        tr_values = [r["metrics"].get("tr", 0.0) for r in all_results]
        scores = [r["score"] for r in all_results]
        
        # Sort by τ for plotting
        sorted_indices = np.argsort(tau_values)
        tau_sorted = [tau_values[i] for i in sorted_indices]
        ga_sorted = [ga_values[i] for i in sorted_indices]
        tr_sorted = [tr_values[i] for i in sorted_indices]
        scores_sorted = [scores[i] for i in sorted_indices]
        
        plot_files = {}
        
        # Plot 1: GA and TR vs τ
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(tau_sorted, ga_sorted, 'b-o', label='Grounding Accuracy', markersize=4)
        plt.axhline(y=np.mean(ga_sorted), color='b', linestyle='--', alpha=0.5, label='Mean GA')
        plt.xlabel('Threshold (τ)')
        plt.ylabel('Grounding Accuracy')
        plt.title('Grounding Accuracy vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.plot(tau_sorted, tr_sorted, 'r-o', label='Trigger Rate', markersize=4)
        plt.axhline(y=self.target_trigger_rate, color='g', linestyle='--', 
                   label=f'Target TR ({self.target_trigger_rate:.1%})')
        plt.axhline(y=np.mean(tr_sorted), color='r', linestyle='--', alpha=0.5, label='Mean TR')
        plt.xlabel('Threshold (τ)')
        plt.ylabel('Trigger Rate')
        plt.title('Trigger Rate vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 3)
        plt.plot(tau_sorted, scores_sorted, 'g-o', label='Bi-objective Score', markersize=4)
        if self.best_tau is not None:
            plt.axvline(x=self.best_tau, color='g', linestyle='--', 
                       label=f'Optimal τ ({self.best_tau:.3f})')
        plt.xlabel('Threshold (τ)')
        plt.ylabel('Bi-objective Score')
        plt.title('Bi-objective Score vs Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 2, 4)
        plt.scatter(tr_sorted, ga_sorted, c=tau_sorted, cmap='viridis', s=50, alpha=0.7)
        plt.colorbar(label='Threshold (τ)')
        plt.axvline(x=self.target_trigger_rate, color='g', linestyle='--', 
                   label=f'Target TR ({self.target_trigger_rate:.1%})')
        plt.xlabel('Trigger Rate')
        plt.ylabel('Grounding Accuracy')
        plt.title('GA vs TR Trade-off')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        
        sensitivity_plot = output_path / "tau_sensitivity.png"
        plt.savefig(sensitivity_plot, dpi=150, bbox_inches='tight')
        plot_files["sensitivity"] = str(sensitivity_plot)
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # Plot 2: Optimization progress
        if self.tpe_results:
            plt.figure(figsize=(10, 6))
            
            tpe_trials = [r["trial_number"] for r in self.tpe_results]
            tpe_scores = [r["score"] for r in self.tpe_results]
            
            plt.plot(tpe_trials, tpe_scores, 'b-o', markersize=4, label='TPE Trials')
            
            # Running best
            running_best = []
            best_so_far = float('-inf')
            for score in tpe_scores:
                best_so_far = max(best_so_far, score)
                running_best.append(best_so_far)
            
            plt.plot(tpe_trials, running_best, 'r-', linewidth=2, label='Running Best')
            
            plt.xlabel('Trial Number')
            plt.ylabel('Bi-objective Score')
            plt.title('TPE Optimization Progress')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            progress_plot = output_path / "tpe_progress.png"
            plt.savefig(progress_plot, dpi=150, bbox_inches='tight')
            plot_files["progress"] = str(progress_plot)
            
            if show_plots:
                plt.show()
            else:
                plt.close()
        
        if self.verbose:
            print(f"✓ Sensitivity plots saved to {output_path}")
            for name, path in plot_files.items():
                print(f"  {name}: {path}")
        
        return plot_files
    
    def save_results(
        self,
        artifacts_dir: str = "artifacts",
        threshold_filename: str = "threshold.json",
        results_filename: str = "tau_sweep_results.json",
    ) -> Dict[str, str]:
        """
        Save optimal τ and full results to artifacts.
        
        Args:
            artifacts_dir: Directory to save artifacts
            threshold_filename: Name of threshold JSON file
            results_filename: Name of full results JSON file
            
        Returns:
            Dictionary mapping artifact names to file paths
        """
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(exist_ok=True)
        
        if self.best_tau is None:
            raise ValueError("No optimal τ found. Run optimization first.")
        
        # Save threshold.json
        threshold_data = {
            "optimal_tau": self.best_tau,
            "best_score": self.best_score,
            "best_metrics": self.best_metrics,
            "target_trigger_rate": self.target_trigger_rate,
            "alpha": self.alpha,
            "calibration_timestamp": time.time(),
            "calibration_method": "grid_search_tpe",
        }
        
        threshold_file = artifacts_path / threshold_filename
        with open(threshold_file, 'w') as f:
            json.dump(threshold_data, f, indent=2)
        
        # Save full results
        results_data = {
            "configuration": {
                "tau_range": self.tau_range,
                "target_trigger_rate": self.target_trigger_rate,
                "alpha": self.alpha,
                "grid_size": self.grid_size,
                "tpe_trials": self.tpe_trials,
                "random_seed": self.random_seed,
            },
            "grid_results": self.grid_results,
            "tpe_results": self.tpe_results,
            "optimal_result": {
                "tau": self.best_tau,
                "score": self.best_score,
                "metrics": self.best_metrics,
            },
            "sensitivity_analysis": self._compute_sensitivity_analysis(),
            "timestamp": time.time(),
        }
        
        results_file = artifacts_path / results_filename
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        file_paths = {
            "threshold": str(threshold_file),
            "results": str(results_file),
        }
        
        if self.verbose:
            print(f"✓ Results saved to {artifacts_path}")
            print(f"  Threshold: {threshold_file}")
            print(f"  Full results: {results_file}")
        
        return file_paths
    
    def _compute_sensitivity_analysis(self) -> Dict[str, Any]:
        """Compute sensitivity analysis for all results."""
        all_results = self.grid_results + self.tpe_results
        if not all_results:
            return {}
        
        tau_values = [r["tau"] for r in all_results]
        ga_values = [r["metrics"].get("ga", 0.0) for r in all_results]
        tr_values = [r["metrics"].get("tr", 0.0) for r in all_results]
        scores = [r["score"] for r in all_results]
        
        return {
            "ga_sensitivity": sensitivity_analysis(tau_values, ga_values, "tau", "ga"),
            "tr_sensitivity": sensitivity_analysis(tau_values, tr_values, "tau", "tr"),
            "score_sensitivity": sensitivity_analysis(tau_values, scores, "tau", "score"),
        }
    
    def run_full_calibration(
        self,
        artifacts_dir: str = "artifacts",
        plots_dir: str = "sensitivity_plots",
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete τ calibration pipeline.
        
        Args:
            artifacts_dir: Directory to save artifacts
            plots_dir: Directory to save plots
            save_plots: Whether to generate and save plots
            
        Returns:
            Complete calibration results
        """
        if self.verbose:
            print("TAU CALIBRATION PIPELINE")
            print("=" * 80)
            print(f"Target trigger rate: {self.target_trigger_rate:.1%}")
            print(f"Alpha (trade-off): {self.alpha}")
            print("=" * 80)
        
        start_time = time.time()
        
        # 1. Grid search warm-start
        grid_results = self.run_grid_search()
        
        # 2. TPE optimization
        tpe_results = self.run_tpe_optimization(grid_results)
        
        # 3. Find optimal τ
        optimal_tau, optimal_metrics = self.find_optimal_tau()
        
        # 4. Generate plots
        plot_files = {}
        if save_plots:
            plot_files = self.generate_sensitivity_plots(plots_dir)
        
        # 5. Save artifacts
        artifact_files = self.save_results(artifacts_dir)
        
        total_time = time.time() - start_time
        
        # Summary
        summary = {
            "optimal_tau": optimal_tau,
            "optimal_metrics": optimal_metrics,
            "optimal_score": self.best_score,
            "grid_points": len(grid_results),
            "tpe_trials": len(tpe_results),
            "total_evaluations": len(grid_results) + len(tpe_results),
            "calibration_time": total_time,
            "artifact_files": artifact_files,
            "plot_files": plot_files,
        }
        
        if self.verbose:
            print("\n" + "="*80)
            print("CALIBRATION SUMMARY")
            print("="*80)
            print(f"Optimal τ: {optimal_tau:.4f}")
            print(f"Best score: {self.best_score:.4f}")
            print(f"Optimal metrics: {optimal_metrics}")
            print(f"Total evaluations: {summary['total_evaluations']}")
            print(f"Calibration time: {total_time:.1f}s")
            print(f"Artifacts saved: {len(artifact_files)} files")
            if plot_files:
                print(f"Plots saved: {len(plot_files)} files")
            print("="*80)
        
        return summary


def run_tau_sweep(
    evaluation_function: Callable[[float], Dict[str, float]],
    config: Optional[Dict[str, Any]] = None,
    output_dir: str = "calibration_output",
) -> Dict[str, Any]:
    """
    Convenience function to run τ calibration with default settings.
    
    Args:
        evaluation_function: Function to evaluate τ values
        config: Optional configuration overrides
        output_dir: Output directory for all artifacts
        
    Returns:
        Calibration summary results
    """
    # Default configuration
    default_config = {
        "tau_range": (0.0, 1.0),
        "target_trigger_rate": 0.10,
        "alpha": 0.5,
        "grid_size": 21,
        "tpe_trials": 50,
        "random_seed": 42,
        "verbose": True,
    }
    
    if config:
        default_config.update(config)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize sweeper
    sweeper = TauSweeper(
        evaluation_function=evaluation_function,
        **default_config
    )
    
    # Run calibration
    results = sweeper.run_full_calibration(
        artifacts_dir=str(output_path / "artifacts"),
        plots_dir=str(output_path / "plots"),
        save_plots=True,
    )
    
    return results
