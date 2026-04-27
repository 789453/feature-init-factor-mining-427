import argparse
from .config import ValidationConfig
from .runner import run_validation

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duckdb", required=True)
    p.add_argument("--top100", default="outputs/real_all/top100.csv")
    p.add_argument("--out", default="outputs/validation_real_all")
    p.add_argument("--start", default="20190101")
    p.add_argument("--end", default="20260430")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--rebalance-n", type=int, default=5)
    p.add_argument("--alphalens-top-n", type=int, default=10)
    p.add_argument("--duckdb-threads", type=int, default=24)
    p.add_argument("--duckdb-memory-limit", default="24GB")
    args = p.parse_args()

    cfg = ValidationConfig(
        duckdb_path=args.duckdb,
        top100_path=args.top100,
        out_dir=args.out,
        start=args.start,
        end=args.end,
        horizon=args.horizon,
        rebalance_n=args.rebalance_n,
        alphalens_top_n=args.alphalens_top_n,
        duckdb_threads=args.duckdb_threads,
        duckdb_memory_limit=args.duckdb_memory_limit,
    )
    run_validation(cfg)

if __name__ == "__main__":
    main()