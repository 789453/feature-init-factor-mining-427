from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

from .config import RunConfig
from .data import load_from_duckdb, make_simulated_data
from .fields import add_basic_features, DEFAULT_FEATURES
from .grammar import generate_templates
from .evaluator import BatchEvaluator, make_panels
from .metrics import forward_returns, summarize_factor
from .parser import parse_expr, canonical

def run_pipeline(cfg: RunConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.use_simulated:
        raw = make_simulated_data(seed=cfg.seed)
    else:
        if not cfg.duckdb_path:
            raise ValueError("duckdb_path is required unless use_simulated=True")
        raw = load_from_duckdb(cfg.duckdb_path, cfg.pool_json, cfg.start, cfg.end)

    try:
        raw.to_parquet(out / "raw_loaded_sample.parquet", index=False)
    except Exception:
        raw.head(1000).to_csv(out / "raw_loaded_sample.csv", index=False, encoding="utf-8-sig")

    df = add_basic_features(raw)
    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    panels, dates, codes = make_panels(df, feature_cols, value_col="close")
    fwd = forward_returns(panels["close"], horizon=cfg.eval.forward_days)

    exprs = generate_templates(feature_cols, cfg.eval.windows, max_exprs=cfg.max_exprs)
    (out / "generated_expressions.txt").write_text("\\n".join(exprs), encoding="utf-8")

    ev = BatchEvaluator(
        panels={k: v for k, v in panels.items() if k != "close"},
        dates=dates,
        codes=codes,
        windows=cfg.eval.windows,
        max_depth=cfg.eval.max_depth,
        max_nodes=cfg.eval.max_nodes,
    )

    records = []
    for expr in exprs:
        arr, status = ev.eval_expr(expr)
        if arr is None:
            records.append({"expr": expr, "status": status})
            continue
        m = summarize_factor(arr, fwd, dates, min_daily_valid=cfg.eval.min_daily_valid_names)
        records.append({"expr": expr, "canonical": canonical(parse_expr(expr)), "status": status, **m})

    res = pd.DataFrame(records)
    if "mean_rank_ic" in res:
        res["score"] = res["mean_rank_ic"].abs() * res["rank_icir"].abs().replace([np.inf, -np.inf], np.nan)
        res = res.sort_values("score", ascending=False, na_position="last")
    res.to_csv(out / "factor_results.csv", index=False, encoding="utf-8-sig")
    res.head(100).to_csv(out / "top100.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_raw_rows": int(len(raw)),
        "n_dates": int(len(dates)),
        "n_codes": int(len(codes)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "n_exprs": int(len(exprs)),
        "n_results": int(len(res)),
        "out_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
