"""Profile DyGRAV experts to build latency/energy cost LUT.

Usage:
  python -m dygrav.tools.profile_experts --runs 100 --out configs/cost_lut.json
"""

from __future__ import annotations
import argparse
import json
import time
from typing import Dict, Any

import numpy as np
import torch

from ..modules.scene_graph import SceneGraphBuilder
from ..modules.ver_slice import create_ver_slice_expert
from ..modules.llm_tools import create_llm_tools_expert
from ..core.types import Region


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def profile_callable(callable_fn, *args, n_runs: int = 100, warmup: int = 10) -> float:
    # warmup
    for _ in range(warmup):
        _ = callable_fn(*args)
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = callable_fn(*args)
    _sync()
    dt = (time.perf_counter() - t0) / float(n_runs) * 1000.0
    return dt


def build_dummy_inputs(image_w: int = 640, image_h: int = 480) -> Dict[str, Any]:
    # Regions for scene graph (xyxy, score, label)
    regions = [
        Region((100, 120, 180, 220), 0.9, "chair"),
        Region((220, 140, 300, 240), 0.85, "table"),
        Region((320, 200, 420, 340), 0.80, "sofa"),
        Region((50,  260, 120, 340), 0.75, "door"),
    ]
    depth = torch.rand(image_h, image_w) * 5.0
    instruction = "Go up the stairs and turn left at the door"
    rgb = np.zeros((image_h, image_w, 3), dtype=np.uint8)
    return {"regions": regions, "depth": depth, "instruction": instruction, "rgb": rgb}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--out", type=str, default="configs/cost_lut.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on device: {device}")

    dummy = build_dummy_inputs()

    # Scene Graph builder: wrap call
    sg = SceneGraphBuilder(k_cap=6)
    def sg_call():
        return sg.build(dummy["regions"], clip_scores=None, instruction_spans=dummy["instruction"].split())

    # VER-slice expert
    ver = create_ver_slice_expert(1.0)
    def ver_call():
        return ver(dummy["depth"], dummy["instruction"], dummy["rgb"], radius=1.0)

    # LLM-tools expert
    llm = create_llm_tools_expert(16)
    tool_calls = llm.parse_instruction(dummy["instruction"], {"regions": dummy["regions"]})
    def llm_call():
        results = []
        for call in tool_calls[:1]:
            results.append(llm.execute_tool(call, {"instruction": dummy["instruction"]}, dummy["regions"]))
        if results:
            _ = llm.generate_action_bias(results)
        return results

    lut: Dict[str, Dict[str, float]] = {}

    for name, fn in [
        ("micro_graph", sg_call),
        ("ver_slice", ver_call),
        ("llm_tools", llm_call),
    ]:
        ms = profile_callable(fn, n_runs=args.runs)
        lut[name] = {"latency_ms": float(ms), "energy_j": float(ms) * 0.001}
        print(f"{name:12s}: {ms:.3f} ms")

    with open(args.out, "w") as f:
        json.dump(lut, f, indent=2)
    print(f"\nSaved cost LUT to {args.out}")


if __name__ == "__main__":
    main()


