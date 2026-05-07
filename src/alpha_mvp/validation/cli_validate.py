import argparse
from .config import ValidationConfig
from .runner import run_validation

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duckdb", required=True)
    p.add_argument("--top100", default="outputs/real_all/top100.csv")
    p.add_argument("--out", default="outputs/validation_real_all")
    p.add_argument("--start", default="20180101")
    p.add_argument("--end", default="20260430")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--rebalance-n", type=int, default=5)
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--alphalens-top-n", type=int, default=10)
    p.add_argument("--cache-features", action="store_true", default=True)
    p.add_argument("--cache-factors", action="store_true", default=True)
    p.add_argument("--overwrite-cache", action="store_true", default=False)
    p.add_argument("--duckdb-threads", type=int, default=24)
    p.add_argument("--duckdb-memory-limit", default="24GB")
    p.add_argument("--skip-alphalens", action="store_true", default=False)
    p.add_argument("--skip-vectorbot", action="store_true", default=False)
    p.add_argument("--skip-reports", action="store_true", default=False)
    p.add_argument("--train-end", default="20250831")
    p.add_argument("--test-start", default="20250901")
    p.add_argument("--from-step", type=int, default=1, help="Start from step N (1-8)")
    args = p.parse_args()

    cfg = ValidationConfig(
        duckdb_path=args.duckdb,
        top100_path=args.top100,
        out_dir=args.out,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        rebalance_n=args.rebalance_n,
        top_quantiles=(0.05, 0.10),
        alphalens_top_n=0 if args.skip_alphalens else args.alphalens_top_n,
        duckdb_threads=args.duckdb_threads,
        duckdb_memory_limit=args.duckdb_memory_limit,
        train_end=args.train_end,
        test_start=args.test_start,
        top_n=args.top_n,
        overwrite_cache=args.overwrite_cache,
        skip_vectorbot=args.skip_vectorbot,
        skip_alphalens=args.skip_alphalens,
        skip_reports=args.skip_reports,
        from_step=args.from_step,
    )

    run_validation(cfg)

if __name__ == "__main__":
    main()