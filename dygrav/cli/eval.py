"""Synthetic evaluator CLI for DyGRAV system testing."""
from ..eval.evaluator import run_eval

def main():
    """Run synthetic evaluation."""
    print("[eval] Starting synthetic evaluation...")
    
    # Run evaluation with 30 episodes
    results = run_eval(n=30)
    
    print(f"[eval] trigger_rate={results['trigger_rate']:.2f}  GA={results['GA']:.2f}")
    print(f"[eval] Episodes: {results['n_episodes']}, Triggers: {results['n_triggers']}")
    print("[eval] Evaluation complete!")

if __name__ == "__main__":
    main()
