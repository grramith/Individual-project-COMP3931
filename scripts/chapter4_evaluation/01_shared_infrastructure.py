import os
from pathlib import Path

def _find_project_root():
    sentinel_dirs = {'data', 'scripts', 'models'}
    candidate = Path.cwd().resolve()
    while True:
        children = {p.name for p in candidate.iterdir() if p.is_dir()}
        if sentinel_dirs <= children:
            return candidate
        if candidate.parent == candidate:
            raise RuntimeError("Could not locate project root (need data/, scripts/, models/)")
        candidate = candidate.parent

PROJECT_ROOT = _find_project_root()
os.chdir(PROJECT_ROOT)
print(f"Project root: {PROJECT_ROOT}")

EVAL_DIR = "data/results/evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)

TRADING_DAYS = 252
TX_COST_DEFAULT = 0.0005
INITIAL_CAPITAL = 1000.0