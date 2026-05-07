from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def panel_to_parquet(arr: np.ndarray, dates: list[str], codes: list[str], path: Path) -> None:
    str_dates = [str(d) for d in dates]
    df = pd.DataFrame(arr, index=pd.Index(str_dates, name="trade_date"), columns=codes)
    df.to_parquet(path, index=True)

def parquet_to_panel(path: Path, dates: list[str] | None = None, codes: list[str] | None = None) -> tuple[np.ndarray, list[str], list[str]]:
    df = pd.read_parquet(path)
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime("%Y%m%d")
    df = df.sort_index()
    if dates is not None:
        df = df.reindex(index=[str(d) for d in dates])
    dates_out = df.index.tolist()
    if codes is not None:
        df = df.reindex(columns=codes)
    codes_out = df.columns.tolist()
    arr = df.to_numpy(dtype=np.float32)
    return arr, dates_out, codes_out

def save_codes_codes(codes: list[str], path: Path) -> None:
    pd.DataFrame({"ts_code": codes}).to_parquet(path, index=False)

def load_codes_codes(path: Path) -> list[str]:
    return pd.read_parquet(path)["ts_code"].tolist()

def save_dates(dates: list[str], path: Path) -> None:
    pd.DataFrame({"trade_date": [str(d) for d in dates]}).to_parquet(path, index=False)

def load_dates(path: Path) -> list[str]:
    df = pd.read_parquet(path)
    if "trade_date" in df.columns:
        return df["trade_date"].astype(str).tolist()
    return df.index.astype(str).tolist()