from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from ..fields import add_basic_features, DEFAULT_FEATURES
from ..pipeline import make_panels
from .panel_io import panel_to_parquet, save_codes_codes, save_dates

FEATURE_VERSION = "v1"

def build_market_feature_cache(raw_df: pd.DataFrame, out_dir: str, dtype: str = "float32") -> dict:
    out_path = Path(out_dir)
    feature_dir = out_path / "market_features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    df = add_basic_features(raw_df)
    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]

    panels, dates, codes = make_panels(df, feature_cols, value_col="close")

    meta_df = df[["trade_date", "ts_code", "industry", "circ_mv"]].drop_duplicates(["trade_date", "ts_code"])

    circ_pivot = meta_df.pivot(index="trade_date", columns="ts_code", values="circ_mv")
    industry_pivot = meta_df.pivot(index="trade_date", columns="ts_code", values="industry")

    panels["circ_mv"] = circ_pivot.reindex(index=dates, columns=codes).to_numpy(dtype=np.float32)

    industry_str = np.empty_like(industry_pivot.to_numpy(), dtype=object)
    raw_industry = industry_pivot.to_numpy()
    for i in range(raw_industry.shape[0]):
        for j in range(raw_industry.shape[1]):
            val = raw_industry[i, j]
            industry_str[i, j] = str(val) if pd.notna(val) else ""
    panels["industry"] = industry_str

    manifest = {
        "start": dates[0] if dates else "",
        "end": dates[-1] if dates else "",
        "n_dates": len(dates),
        "n_codes": len(codes),
        "features": list(panels.keys()),
        "dtype": dtype,
        "field_version": FEATURE_VERSION,
        "created_at": datetime.now().isoformat(),
    }

    save_codes_codes(codes, feature_dir / "codes.parquet")
    save_dates(dates, feature_dir / "dates.parquet")

    for name, arr in panels.items():
        if arr.dtype == object:
            str_df = pd.DataFrame(arr, index=dates, columns=codes)
            str_df.index.name = "trade_date"
            str_df.to_parquet(feature_dir / f"{name}.parquet", index=True)
        else:
            arr = arr.astype(dtype)
            panel_to_parquet(arr, dates, codes, feature_dir / f"{name}.parquet")

    with open(feature_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest

def load_market_feature_cache(out_dir: str) -> tuple[dict[str, np.ndarray], list[str], list[str], dict]:
    feature_dir = Path(out_dir) / "market_features"

    with open(feature_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    codes = pd.read_parquet(feature_dir / "codes.parquet")["ts_code"].tolist()
    dates = pd.read_parquet(feature_dir / "dates.parquet")["trade_date"].astype(str).tolist()

    panels = {}
    for feat in manifest["features"]:
        p = feature_dir / f"{feat}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.strftime("%Y%m%d")
            df = df.reindex(index=dates, columns=codes)
            if feat == "industry":
                arr = df.to_numpy(object)
            else:
                arr = df.to_numpy(dtype=np.float32)
            panels[feat] = arr

    return panels, dates, codes, manifest