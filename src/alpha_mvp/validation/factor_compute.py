import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from alpha_mvp.fields import add_basic_features, DEFAULT_FEATURES
from alpha_mvp.evaluator import BatchEvaluator
from alpha_mvp.pipeline import make_panels

def panel_to_long(arr, dates, codes, value_name="value"):
    df = pd.DataFrame(arr, index=pd.Index(dates, name="trade_date"), columns=codes)
    s = df.stack(dropna=False).rename(value_name).reset_index()
    s.columns = ["trade_date", "ts_code", value_name]
    return s

def build_feature_panels(raw_df):
    df = add_basic_features(raw_df)
    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    panels, dates, codes = make_panels(df, feature_cols, value_col="close")
    meta = (
        df[["trade_date", "ts_code", "industry", "circ_mv"]]
        .drop_duplicates(["trade_date", "ts_code"])
    )
    return df, panels, dates, codes, meta

def compute_factor_panels(top_factors, panels, dates, codes, cfg):
    evaluator = BatchEvaluator(
        panels={k: v for k, v in panels.items() if k != "close"},
        dates=dates,
        codes=codes,
        windows=(10, 20, 30, 40, 50),
        max_depth=6,
        max_nodes=16,
    )

    out_dir = Path(cfg.out_dir) / "factor_panels"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    status_rows = []

    for _, row in top_factors.iterrows():
        fid = row["factor_id"]
        expr = row["factor_expr"]
        arr, status = evaluator.eval_expr(expr)

        status_rows.append({
            "factor_id": fid,
            "expr": expr,
            "status": status,
            "coverage": float(np.isfinite(arr).mean()) if arr is not None else np.nan,
        })

        if arr is None:
            continue

        results[fid] = arr

        if cfg.write_factor_panels:
            long_df = panel_to_long(arr, dates, codes, value_name="factor")
            long_df["factor_id"] = fid
            long_df.to_parquet(out_dir / f"{fid}.parquet", index=False)

    return results, pd.DataFrame(status_rows)