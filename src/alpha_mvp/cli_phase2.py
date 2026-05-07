from __future__ import annotations
import argparse
from .config import RunConfig, EvalConfig
from .phase2_pipeline import run_phase2

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duckdb", default=None)
    p.add_argument("--pool-json", default=None)
    p.add_argument("--start", default="20240101")
    p.add_argument("--end", default="20260430")
    p.add_argument("--out", default="outputs/phase2")
    p.add_argument("--max-exprs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-simulated", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--write-every", type=int, default=200)
    p.add_argument("--sqlite-path", default=None, help="Path to results duckdb")
    p.add_argument("--train-end", default="20250831")
    p.add_argument("--test-start", default="20250901")
    p.add_argument("--progress-min-interval-sec", type=float, default=5.0)
    p.add_argument("--force-rerun", action="store_true", help="Force recalculation even if job exists")
    
    # Phase 2 Field Selection
    p.add_argument("--fields", default=None, help="Comma separated list of fields to include")
    p.add_argument("--exclude-fields", default=None, help="Comma separated list of fields to exclude")
    p.add_argument("--field-file", default=None, help="Path to JSON file containing fields")
    p.add_argument("--field-set", default=None, help="Name of a predefined field set")

    p.add_argument("--expr-file", default=None, help="Path to expression file")
    p.add_argument("--start-expr", type=int, default=1)
    p.add_argument("--end-expr", type=int, default=None)

    args = p.parse_args()
    
    fields = args.fields.split(",") if args.fields else None
    exclude_fields = args.exclude_fields.split(",") if args.exclude_fields else None

    cfg = RunConfig(
        start=args.start, end=args.end, out_dir=args.out,
        max_exprs=args.max_exprs, batch_size=args.batch_size, seed=args.seed,
        duckdb_path=args.duckdb, pool_json=args.pool_json, use_simulated=args.use_simulated,
        eval=EvalConfig(),
        resume=args.resume,
        write_every=args.write_every,
        sqlite_path=args.sqlite_path,
        train_end=args.train_end,
        test_start=args.test_start,
        progress_min_interval_sec=args.progress_min_interval_sec,
        force_rerun=args.force_rerun,
        fields=fields,
        exclude_fields=exclude_fields,
        field_file=args.field_file,
        field_set=args.field_set,
        expr_file=args.expr_file,
        start_expr=args.start_expr,
        end_expr=args.end_expr
    )
    
    summary = run_phase2(cfg)
    print("Phase 2 Summary:", summary)

if __name__ == "__main__":
    main()
