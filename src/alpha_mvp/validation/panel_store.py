import numpy as np
import pandas as pd
from pathlib import Path

def panel_to_long(arr, dates, codes, value_name="value"):
    df = pd.DataFrame(arr, index=pd.Index(dates, name="trade_date"), columns=codes)
    s = df.stack(dropna=False).rename(value_name).reset_index()
    s.columns = ["trade_date", "ts_code", value_name]
    return s

def long_to_panel(long_df, dates, codes, value_name="value"):
    df = long_df.pivot(index="trade_date", columns="ts_code", values=value_name)
    df = df.reindex(index=dates, columns=codes)
    return df.to_numpy()

def load_factor_panel(fid, out_dir, dates, codes):
    path = Path(out_dir) / "factor_panels" / f"{fid}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return long_to_panel(df, dates, codes, "factor")

def save_factor_panel(fid, arr, dates, codes, out_dir):
    long_df = panel_to_long(arr, dates, codes, value_name="factor")
    long_df["factor_id"] = fid
    out_path = Path(out_dir) / "factor_panels"
    out_path.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(out_path / f"{fid}.parquet", index=False)