from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .task import Task
from .eval import evaluate


def _load_task(path: str) -> Task:
    p = Path(path).resolve()
    if not p.exists():
        raise SystemExit(f"task file not found: {p}")
    spec = importlib.util.spec_from_file_location(f"_mb_task_{p.stem}", p)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load task module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "task"):
        raise SystemExit(f"{p}: must define a top-level `task` of type Task")
    if not isinstance(mod.task, Task):
        raise SystemExit(f"{p}: `task` is not a metalbench.Task instance")
    return mod.task


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="metalbench")
    ap.add_argument("task", help="path to a task .py file (must export `task: Task`)")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters",  type=int, default=50)
    args = ap.parse_args(argv)

    task = _load_task(args.task)
    result = evaluate(task, seed=args.seed, warmup=args.warmup, iters=args.iters)
    print(json.dumps(result, indent=2))
    return 0 if result["correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
