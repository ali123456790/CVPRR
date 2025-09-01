# DyGRAV — Production-Ready Scaffold

DyGRAV (**Dy**namic **G**rounding & **R**elational **A**gent for **V**LN) is a safe, reproducible
starting point for inference-time dynamic grounding in Vision-Language Navigation. The repo ships with:
- **Runnable stubs** + **passing unit tests** (no heavy ML deps required).
- A **synthetic training loop** (PyTorch Lightning) and **evaluation harness** that exercise the
  DyGRAV hooks end-to-end before wiring real datasets.
- Clear **non-negotiable rules** to protect scientific validity (latency budgets, ablations, seeds).

> Purpose: move fast **without breaking the science**—every change must be reproducible, measurable,
> and attributable (via ablations).

---

## Quickstart (lightweight, runs tests only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"     # pytest, ruff, black, mypy
pytest -q                   # all tests should pass
Full stack (with trainer/eval on synthetic)
bash
Copy code
pip install -e ".[full,dev]"      # adds torch, lightning, open-clip, etc.

# Train (synthetic data; proves end-to-end plumbing)
python -m dygrav.cli.train

# Evaluate (synthetic; prints trigger rate & GA)
python -m dygrav.cli.eval
Expected example:

ini
Copy code
[eval] trigger_rate=1.00  GA=1.00
(Synthetic path is intentionally “perfect” to validate wiring.)

Repository Layout
dygrav/modules/ – ambiguity detector, VLM query wrapper (OpenCLIP-ready), scene-graph builder, fusion

dygrav/core/ – policy integration points + shared types

dygrav/metrics/ – navigation (SR, SPL, NE) + grounding (GA) metrics

dygrav/eval/ – evaluation harness (synthetic), ablation scaffolding

dygrav/lightning/ – minimal Lightning NavLightningModule (synthetic)

dygrav/data/ – SyntheticDataModule (real RxR loader to be added later)

dygrav/detectors/ – SimpleDetector stub (swap for YOLO/DETR later)

tests/ – unit tests validating the DyGRAV interfaces

scripts/ – env setup, RxR-FG criteria stub, threshold sweep scaffold

docs/ – metrics, ADRs, pipeline diagram

Hard Rules (non-negotiable)
Determinism: fix seeds; record them in logs for every run.

On-Demand Only: DyGRAV triggers only under ambiguity. Keep trigger rate conservative on real val splits (target ≤ 30%).

Latency Budget: added latency ≤ 20 ms only when DyGRAV triggers. Measure per stage (VLM, SG) and report.

Ablation Gates: every claim must include −VLM, −SG, and −Ambiguity ablations—gains must drop when modules are removed.

One Detector in Mainline: avoid detector churn until DyGRAV path is stable.

Repro Logs: write JSONL per step: policy_conf, attn_entropy, trigger_reason, vlm_top, n_rel.

No Silent API Breaks: keep interfaces for Region, VLMResult, RelationEdge, DygravSignal stable.

Metrics & Success Bars
We report standard VLN metrics and a DyGRAV-specific measure:

SR (Success Rate) — primary.

SPL (Success weighted by Path Length).

NE (Navigation Error, meters).

GA (Grounding Accuracy) — % steps where chosen referent IoU ≥ τ w.r.t. annotated referent.
Report GA by category: attributes vs relations.

Pilot Success Targets (vs. strong transformer baseline on RxR-FG):

+5–8 SR on RxR-FG.

≤ +20 ms average added latency when triggered.

No worse SPL/NE; clean ablation attribution (−VLM/−SG/−Ambiguity).

What’s Implemented vs. What’s Next
✅ Now

Unit tests (no heavy deps).

Synthetic trainer (python -m dygrav.cli.train).

Synthetic evaluator (python -m dygrav.cli.eval) with GA & trigger-rate display.

OpenCLIP backend stubbed and ready (switch when you install GPUs/weights).

➡️ Next

Threshold Calibration: tune tau_conf / tau_entropy (see scripts/sweep_tau.py).

Detector Swap: replace SimpleDetector with YOLO/DETR returning List[Region].

Crop Caching & Batching: enable before any real timing.

Real Data: add dygrav/data/rxr.py and complete scripts/build_rxr_fg.py.

Ablations + Report Table: run baseline vs. DyGRAV vs. ablations; publish SR/SPL/NE/GA (+ latency).

CI Gates: once thresholds are tuned on real val splits, add a CI check that fails on trigger-rate > target.

Git & Branching
bash
Copy code
git checkout -b feat/trainer-eval-ci
# make atomic commits; keep tests green
git push -u origin feat/trainer-eval-ci
Open PRs against main. Use conventional commits:

feat: ..., fix: ..., chore: ..., docs: ..., test: ...

License
MIT

yaml
Copy code

---

## 3) (Optional) Add CI so PRs don’t regress

Create `.github/workflows/ci.yml`:

```yaml
name: dygrav-ci

on:
  push:
    branches: [ main, feat/** ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install dev deps (lightweight)
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Run unit tests
        run: |
          source .venv/bin/activate
          pytest -q

  synthetic-eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - name: Install full deps (CPU)
        run: |
          python -m venv .venv
          source .venv/bin/activate
          pip install --upgrade pip
          pip install -e ".[full,dev]"
      - name: Run synthetic trainer and evaluator
        run: |
          source .venv/bin/activate
          python -m dygrav.cli.train
          python -m dygrav.cli.eval
Commit & push:

bash
Copy code
git add README.md .github/workflows/ci.yml
git commit -m "docs: update README with hard rules; ci: add tests + synthetic eval"
git push
4) Cursor-safe workflow recap
Work on branch feat/trainer-eval-ci.

Make one edit at a time, then pytest -q.

If Cursor glitches: git restore --staged . && git checkout -- . to rollback uncommitted changes.

Keep interfaces (Region, DygravSignal, etc.) unchanged to avoid cascading edits.