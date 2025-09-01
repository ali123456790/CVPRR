import numpy as np
from dygrav.eval.evaluator import run_eval

if __name__ == "__main__":
    trials = 5
    rates, gas = [], []
    for _ in range(trials):
        out = run_eval(n=30)
        rates.append(out["trigger_rate"]); gas.append(out["GA"])
    print(f"trigger_rate mean={np.mean(rates):.2f}±{np.std(rates):.2f}  GA={np.mean(gas):.2f}")
