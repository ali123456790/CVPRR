# DyGRAV — Complete Dynamic Grounding System

DyGRAV (**Dy**namic **G**rounding & **R**elational **A**gent for **V**LN) is a complete implementation of a cost-aware, multi-expert navigation system with learned gate control. The system includes:

- **Learned Gate Model** with Gumbel-Softmax for expert selection
- **Three Expert Modules**: Micro-Graph, VER-slice, and LLM-tools
- **Cost-Aware Training** with dual ascent for budget constraints
- **Complete Feature Pipeline** with novelty tracking and skill classification
- **Production-Ready Code** with comprehensive tests and documentation

> **Status**: Full implementation complete according to mathematical specification. Ready for training and deployment.

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

## Repository Structure

```
dygrav/
├── core/                    # Core policy and types
│   ├── enhanced_policy.py   # Complete inference with learned gate
│   ├── policy.py           # Original policy (legacy)
│   └── types.py            # Shared data structures
├── modules/                # Expert modules and gate
│   ├── gate_model.py       # Learned gate with Gumbel-Softmax
│   ├── gate_features.py    # Feature extraction pipeline
│   ├── ver_slice.py        # VER-slice expert (3D reasoning)
│   ├── llm_tools.py        # LLM-tools expert (symbolic)
│   ├── scene_graph.py      # Enhanced Micro-Graph expert
│   ├── vlm_query.py        # Vision-Language Model queries
│   ├── ambiguity.py        # Rule-based ambiguity detector
│   └── fusion.py           # Expert output fusion
├── training/               # Training pipeline
│   ├── cost_aware_trainer.py  # Cost-aware training with dual ascent
│   └── calibration.py      # Model calibration methods
├── data/                   # Data loaders and datasets
├── detectors/              # Object detection
├── metrics/                # Evaluation metrics
├── eval/                   # Evaluation harness
└── utils/                  # Utilities and caching

tests/                      # Comprehensive test suite
├── test_full_integration.py  # End-to-end system tests
└── test_*.py              # Unit tests for each component

docs/                       # Documentation
├── IMPLEMENTATION_COMPLETE.md  # Full implementation guide
└── metrics.md             # Metrics documentation
```

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

## What's Implemented

✅ **Complete System Implementation**
- Learned gate model with Gumbel-Softmax selection
- All three expert modules (Micro-Graph, VER-slice, LLM-tools)
- Cost-aware training with dual ascent optimization
- Complete feature extraction pipeline
- Calibration methods for uncertainty and skill logits
- Enhanced inference loop with cost tracking
- Comprehensive test suite

✅ **Mathematical Specification Compliance**
- All equations from the specification implemented
- Budget constraints with dual ascent
- Gumbel-Softmax for differentiable selection
- Novelty tracking with visual memory bank
- CLIP margin and logit gap calculations

✅ **Production Ready**
- PyTorch Lightning training pipeline
- Comprehensive error handling
- Cost tracking and metrics logging
- Modular, extensible architecture

## Usage Example

```python
from dygrav.core.enhanced_policy import EnhancedPolicyWithGate
from dygrav.modules.gate_model import GateModel
from dygrav.detectors.yolo import YoloV10Detector

# Initialize components
backbone = load_pretrained_backbone()  # Your navigation backbone
detector = YoloV10Detector()
gate_model = GateModel(input_dim=10, hidden_dim=64)

# Create policy
policy = EnhancedPolicyWithGate(
    backbone=backbone,
    detector=detector,
    gate_model=gate_model,
    use_learned_gate=True,
    log_metrics=True
)

# Run inference
obs = get_observation()  # RGB, depth, instruction, etc.
output = policy.step(obs)

# Access results
action = output['action']
triggered = output['dygrav_triggered']
module_used = output['dygrav_module']
cost = output['dygrav_cost']
```

## Training Example

```python
from dygrav.training.cost_aware_trainer import CostAwareNavModule, TrainingConfig
import pytorch_lightning as pl

# Configure training
config = TrainingConfig(
    backbone_type="hamt",
    target_budget=50.0,  # 50ms per episode
    num_epochs=100,
    warmup_epochs=20
)

# Create module
module = CostAwareNavModule(config)

# Train with PyTorch Lightning
trainer = pl.Trainer(max_epochs=config.num_epochs)
trainer.fit(module, train_dataloader, val_dataloader)
```

## Next Steps

1. **Data Integration**: Connect real navigation datasets (R2R, RxR)
2. **Hardware Profiling**: Measure actual latency/energy on target device
3. **Training**: Run two-stage training on real data
4. **Calibration**: Calibrate on validation set
5. **Evaluation**: Measure SR, SPL, NE, GA metrics
6. **Deployment**: Integrate with navigation environment

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