from typing import List, Dict

def success_rate(successes: List[bool]) -> float:
    return sum(1 for s in successes if s) / max(1, len(successes))

def spl(successes: List[bool], path_lens: List[float], shortest_paths: List[float]) -> float:
    num = 0.0
    denom = 0.0
    for s, l, sp in zip(successes, path_lens, shortest_paths):
        num += (1.0 if s else 0.0) * (sp / max(l, sp, 1e-6))
        denom += 1.0
    return num / max(1.0, denom)

def navigation_error(errors: List[float]) -> float:
    return sum(errors) / max(1, len(errors))
