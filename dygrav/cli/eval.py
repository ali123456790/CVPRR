from ..eval.evaluator import run_eval

def main():
    out = run_eval(n=30)
    print(f"[eval] trigger_rate={out['trigger_rate']:.2f}  GA={out['GA']:.2f}")

if __name__ == "__main__":
    main()
