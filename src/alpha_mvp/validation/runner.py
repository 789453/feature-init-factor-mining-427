from pathlib import Path
import pandas as pd

def run_validation(cfg):
    from .top_factors import load_top_factors
    from .market_data import load_market_data
    from ..cache import build_market_feature_cache, load_market_feature_cache
    from ..cache import compute_and_cache_factors, load_factor_panel
    from .analytics_fast import run_batch_analytics
    from .vectorbot import run_vectorbot
    from .group_metrics import run_group_metrics
    from .alphalens_runner import run_alphalens_batch
    from .report_builder import build_validation_report
    from .plots import generate_factor_plots

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    feature_cache_dir = out / "cache"
    feature_cache_dir.mkdir(parents=True, exist_ok=True)

    panels = None
    dates = None
    codes = None
    factor_panels = {}

    if cfg.from_step <= 1:
        print(f"[1/8] Loading top factors from {cfg.top100_path}")
        top_factors = load_top_factors(cfg.top100_path, top_n=cfg.top_n)
        print(f"  Loaded {len(top_factors)} factors")
    else:
        top_factors = load_top_factors(cfg.top100_path, top_n=cfg.top_n)
        print(f"[1/8] Skipping (--from-step={cfg.from_step})")

    if cfg.from_step <= 2:
        print(f"[2/8] Loading/Building market feature cache")
        cache_dir = feature_cache_dir / "market_features"
        if cache_dir.exists() and (cache_dir / "manifest.json").exists():
            print(f"  Loading from cache: {cache_dir}")
            panels, dates, codes, feature_manifest = load_market_feature_cache(str(feature_cache_dir))
            print(f"  Loaded {len(panels)} feature panels, {len(dates)} dates, {len(codes)} codes")
        else:
            print(f"  Loading market data from DuckDB")
            raw = load_market_data(cfg)
            print(f"  Loaded {len(raw)} rows")
            print(f"  Building feature cache")
            manifest = build_market_feature_cache(raw, str(feature_cache_dir))
            panels, dates, codes, feature_manifest = load_market_feature_cache(str(feature_cache_dir))
            print(f"  Built {len(panels)} feature panels")
    else:
        cache_dir = feature_cache_dir / "market_features"
        if cache_dir.exists() and (cache_dir / "manifest.json").exists():
            print(f"[2/8] Loading market feature cache (--from-step={cfg.from_step})")
            panels, dates, codes, feature_manifest = load_market_feature_cache(str(feature_cache_dir))
            print(f"  Loaded {len(panels)} feature panels, {len(dates)} dates, {len(codes)} codes")
        else:
            raise RuntimeError(f"Cannot skip to step 3+: market features cache not found at {cache_dir}")
        print(f"[2/8] Skipping (--from-step={cfg.from_step})")

    if cfg.from_step <= 3:
        print(f"[3/8] Computing/Loading factor panels")
        factor_cache_dir = feature_cache_dir / "factor_panels"
        if factor_cache_dir.exists() and (factor_cache_dir / "manifest.json").exists() and not cfg.overwrite_cache:
            print(f"  Loading from cache: {factor_cache_dir}")
            manifest = load_factor_manifest(str(feature_cache_dir))
            done_fids = [f["factor_id"] for f in manifest.get("factors", []) if f.get("status") == "OK"]
            for fid in done_fids:
                arr = load_factor_panel(fid, str(feature_cache_dir), dates, codes)
                if arr is not None:
                    factor_panels[fid] = arr
            print(f"  Loaded {len(factor_panels)} factor panels from cache")
        else:
            print(f"  Computing factor panels (overwrite={cfg.overwrite_cache})")
            factor_results = compute_and_cache_factors(
                top_factors, panels, dates, codes, str(feature_cache_dir), overwrite=cfg.overwrite_cache
            )
            manifest = load_factor_manifest(str(feature_cache_dir))
            done_fids = [f["factor_id"] for f in manifest.get("factors", []) if f.get("status") == "OK"]
            for fid in done_fids:
                arr = load_factor_panel(fid, str(feature_cache_dir), dates, codes)
                if arr is not None:
                    factor_panels[fid] = arr
            print(f"  Computed {len(factor_panels)} factor panels")
    else:
        factor_cache_dir = feature_cache_dir / "factor_panels"
        print(f"[3/8] Loading factor panels from cache (--from-step={cfg.from_step})")
        if factor_cache_dir.exists() and (factor_cache_dir / "manifest.json").exists():
            manifest = load_factor_manifest(str(feature_cache_dir))
            done_fids = [f["factor_id"] for f in manifest.get("factors", []) if f.get("status") == "OK"]
            for fid in done_fids:
                arr = load_factor_panel(fid, str(feature_cache_dir), dates, codes)
                if arr is not None:
                    factor_panels[fid] = arr
            print(f"  Loaded {len(factor_panels)} factor panels from cache")
        else:
            raise RuntimeError(f"Cannot skip to step 4+: factor cache not found at {factor_cache_dir}")

    analytics_results = {"Summary": pd.DataFrame(), "rolling_ic": pd.DataFrame()}
    if cfg.from_step <= 4:
        print(f"[4/8] Running batch analytics (IC/RankIC)")
        if len(factor_panels) > 0:
            analytics_results = run_batch_analytics(
                factor_panels, panels["close"], dates, codes, str(out), cfg
            )
            print(f"  Analytics complete: {len(analytics_results.get('Summary', []))} factors")
        else:
            print(f"  Skipping: no factor panels loaded")
    else:
        summary_path = Path(str(out)) / "metrics" / "summary.csv"
        rolling_path = Path(str(out)) / "metrics" / "rolling_ic.parquet"
        if summary_path.exists() and rolling_path.exists():
            print(f"[4/8] Loading analytics from cache (--from-step={cfg.from_step})")
            analytics_results["Summary"] = pd.read_csv(summary_path)
            analytics_results["rolling_ic"] = pd.read_parquet(rolling_path)
        else:
            print(f"[4/8] Skipping (--from-step={cfg.from_step})")

    group_results = {"Size": pd.DataFrame(), "Industry": pd.DataFrame()}
    if cfg.from_step <= 5:
        print(f"[5/8] Running group metrics (size/industry)")
        circ_mv = panels.get("circ_mv")
        industry = panels.get("industry")
        if circ_mv is not None and industry is not None and len(factor_panels) > 0:
            from .analytics_fast import forward_returns
            fwd = forward_returns(panels["close"], horizon=cfg.horizon)
            group_results = run_group_metrics(
                factor_panels, fwd, circ_mv, industry, dates, codes, str(out)
            )
            print(f"  Group metrics complete")
        else:
            print(f"  Skipping: missing data or no factor panels")
    else:
        size_path = Path(str(out)) / "metrics" / "group_size.csv"
        ind_path = Path(str(out)) / "metrics" / "group_industry.csv"
        industry = panels.get("industry") if panels else None
        if size_path.exists() and ind_path.exists():
            print(f"[5/8] Loading group metrics from cache (--from-step={cfg.from_step})")
            group_results["Size"] = pd.read_csv(size_path)
            group_results["Industry"] = pd.read_csv(ind_path)
        else:
            print(f"[5/8] Skipping (--from-step={cfg.from_step})")

    bt_results = {"Summary": pd.DataFrame()}
    if not cfg.skip_vectorbot and cfg.from_step <= 6:
        print(f"[6/8] Running VectorBot backtest")
        summary_df = analytics_results.get("Summary")
        if summary_df is not None and not summary_df.empty and len(factor_panels) > 0:
            bt_results = run_vectorbot(
                factor_panels, panels["close"], dates, codes,
                summary_df, str(out), cfg
            )
            print(f"  VectorBot complete")
        else:
            print(f"  Skipping VectorBot: no factor metrics or panels")
    else:
        print(f"[6/8] Skipping VectorBot (--skip-vectorbot or --from-step)")

    if not cfg.skip_alphalens and cfg.from_step <= 7:
        print(f"[7/8] Running Alphalens for top {cfg.alphalens_top_n} factors")
        summary_df = analytics_results.get("Summary")
        if summary_df is not None and not summary_df.empty and cfg.alphalens_top_n > 0 and industry is not None:
            selected = summary_df.head(cfg.alphalens_top_n)["factor_id"].tolist()
            if selected:
                run_alphalens_batch(
                    selected, factor_panels, panels["close"], dates, codes,
                    industry, cfg, str(out)
                )
                print(f"  Alphalens complete for {len(selected)} factors")
            else:
                print(f"  Skipping Alphalens: no selected factors")
        else:
            print(f"  Skipping Alphalens: no factor metrics")
    else:
        print(f"[7/8] Skipping Alphalens (--skip-alphalens or --from-step)")

    if not cfg.skip_reports and cfg.from_step <= 8:
        print(f"[8/8] Generating plots and reports")
        eq_df = pd.DataFrame()
        vb_equity_path = Path(str(out)) / "vectorbot" / "equity_curves.parquet"
        if vb_equity_path.exists():
            eq_df = pd.read_parquet(vb_equity_path)
            print(f"  Loaded equity curves")

        rolling_ic_df = analytics_results.get("rolling_ic", pd.DataFrame())
        summary_df = analytics_results.get("Summary", pd.DataFrame())
        if not rolling_ic_df.empty:
            generate_factor_plots(
                summary_df,
                rolling_ic_df,
                eq_df,
                group_results.get("Size", pd.DataFrame()),
                group_results.get("Industry", pd.DataFrame()),
                str(out)
            )
            print(f"  Generated plots")

        if not summary_df.empty:
            build_validation_report(
                str(out), cfg,
                summary_df,
                group_results.get("Size", pd.DataFrame()),
                group_results.get("Industry", pd.DataFrame())
            )
            print(f"  Generated HTML report")
    else:
        print(f"[8/8] Skipping reports (--skip-reports or --from-step)")

    print(f"\nValidation complete. Results in {out}")
    return analytics_results, bt_results

def load_factor_manifest(out_dir: str) -> dict:
    import json
    manifest_path = Path(out_dir) / "factor_panels" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return {"factors": []}