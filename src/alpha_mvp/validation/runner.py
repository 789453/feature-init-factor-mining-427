from pathlib import Path

def run_validation(cfg):
    from .top_factors import load_top_factors
    from .market_data import load_market_data
    from .factor_compute import build_feature_panels, compute_factor_panels
    from .analytics import run_factor_analytics
    from .vectorbt_runner import run_vectorbt_like_backtest
    from .alphalens_runner import run_alphalens_batch

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading top factors from {cfg.top100_path}")
    top_factors = load_top_factors(cfg.top100_path)
    print(f"  Loaded {len(top_factors)} factors")

    print(f"[2/6] Loading market data from DuckDB")
    raw = load_market_data(cfg)
    print(f"  Loaded {len(raw)} rows, {raw['ts_code'].nunique()} stocks, {raw['trade_date'].nunique()} dates")

    print(f"[3/6] Building feature panels")
    feature_df, panels, dates, codes, meta = build_feature_panels(raw)
    print(f"  Panels: {list(panels.keys())}, shape: {panels['close'].shape}")

    print(f"[4/6] Computing factor panels")
    factor_panels, status = compute_factor_panels(
        top_factors, panels, dates, codes, cfg
    )
    print(f"  Computed {len(factor_panels)} factors")
    status.to_csv(out / "factor_compute_status.csv", index=False, encoding="utf-8-sig")

    print(f"[5/6] Running factor analytics (IC/RankIC/group metrics)")
    metrics = run_factor_analytics(
        factor_panels=factor_panels,
        close=panels["close"],
        dates=dates,
        codes=codes,
        meta=meta,
        cfg=cfg,
    )
    print(f"  Analytics complete")

    print(f"[6/6] Running vectorbt-like backtest")
    bt = run_vectorbt_like_backtest(
        factor_panels=factor_panels,
        close=panels["close"],
        dates=dates,
        codes=codes,
        factor_metrics=metrics["summary"],
        cfg=cfg,
    )
    print(f"  Backtest complete")

    selected = metrics["summary"].head(cfg.alphalens_top_n)["factor_id"].tolist()
    if selected:
        print(f"Running Alphalens for top {len(selected)} factors")
        run_alphalens_batch(selected, factor_panels, panels["close"], dates, codes, meta, cfg)

    print(f"\nValidation complete. Results in {out}")
    return metrics, bt