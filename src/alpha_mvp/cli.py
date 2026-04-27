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
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--write-every", type=int, default=200)
    p.add_argument("--checkpoint-pct", type=float, default=0.05)
    p.add_argument("--first-checkpoint-pct", type=float, default=0.02)
    p.add_argument("--topk-checkpoint", type=int, default=50)
    p.add_argument("--sqlite-path", default=None)
    p.add_argument("--train-end", default="20250831")
    p.add_argument("--test-start", default="20250901")
    p.add_argument("--progress-min-interval-sec", type=float, default=5.0)
    p.add_argument("--expr-file", default=None, help="Path to expression file (with index)")
    p.add_argument("--start-expr", type=int, default=1, help="Start expression index (1-based)")
    p.add_argument("--end-expr", type=int, default=None, help="End expression index (inclusive)")
    args = p.parse_args()
    cfg = RunConfig(
        start=args.start, end=args.end, out_dir=args.out,
        max_exprs=args.max_exprs, batch_size=args.batch_size, seed=args.seed,
        duckdb_path=args.duckdb, pool_json=args.pool_json, use_simulated=args.use_simulated,
        eval=EvalConfig(),
        resume=args.resume,
        checkpoint_pct=args.checkpoint_pct,
        first_checkpoint_pct=args.first_checkpoint_pct,
        topk_checkpoint=args.topk_checkpoint,
        write_every=args.write_every,
        sqlite_path=args.sqlite_path,
        train_end=args.train_end,
        test_start=args.test_start,
        progress_min_interval_sec=args.progress_min_interval_sec,
        expr_file=args.expr_file,
        start_expr=args.start_expr,
        end_expr=args.end_expr,
    )
    summary = run_pipeline(cfg)
    print(summary)

if __name__ == "__main__":
    main()