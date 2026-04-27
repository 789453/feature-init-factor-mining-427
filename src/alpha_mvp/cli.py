from __future__ import annotations
import argparse
from .config import RunConfig, EvalConfig
from .pipeline import run_pipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duckdb", default=None)
    p.add_argument("--pool-json", default=None)
    p.add_argument("--start", default="20240101")
    p.add_argument("--end", default="20260430")
    p.add_argument("--out", default="outputs/run")
    p.add_argument("--max-exprs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-simulated", action="store_true")
    args = p.parse_args()
    cfg = RunConfig(
        start=args.start, end=args.end, out_dir=args.out,
        max_exprs=args.max_exprs, batch_size=args.batch_size, seed=args.seed,
        duckdb_path=args.duckdb, pool_json=args.pool_json, use_simulated=args.use_simulated,
        eval=EvalConfig(),
    )
    summary = run_pipeline(cfg)
    print(summary)

if __name__ == "__main__":
    main()
