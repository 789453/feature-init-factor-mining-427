from __future__ import annotations
import json
import hashlib
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from ..evaluator import BatchEvaluator
from .panel_io import panel_to_parquet

def make_factor_id(expr: str, rank: int) -> str:
    h = hashlib.md5(expr.encode("utf-8")).hexdigest()[:8]
    return f"F{rank:04d}_{h}"

def load_factor_manifest(out_dir: str) -> dict:
    manifest_path = Path(out_dir) / "factor_panels" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            return json.load(f)
    return {"factors": []}

def save_factor_manifest(out_dir: str, manifest: dict) -> None:
    manifest_path = Path(out_dir) / "factor_panels" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def load_factor_panel(fid: str, out_dir: str, dates: list[str] | None = None, codes: list[str] | None = None) -> np.ndarray | None:
    p = Path(out_dir) / "factor_panels" / f"{fid}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)

    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime("%Y%m%d")

    df.index = df.index.astype(str)

    if dates is not None:
        str_dates = [str(d) for d in dates]
        df = df.reindex(index=str_dates)
    if codes is not None:
        df = df.reindex(columns=codes)

    return df.to_numpy(dtype=np.float32)

def compute_and_cache_factors(
    top_factors: pd.DataFrame,
    panels: dict[str, np.ndarray],
    dates: list[str],
    codes: list[str],
    out_dir: str,
    overwrite: bool = False,
    max_depth: int = 6,
    max_nodes: int = 16,
) -> dict:
    factor_dir = Path(out_dir) / "factor_panels"
    factor_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_factor_manifest(out_dir)
    done_ids = {f["factor_id"] for f in manifest.get("factors", [])}

    evaluator = BatchEvaluator(
        panels={k: v for k, v in panels.items() if k not in ("close", "industry")},
        dates=dates,
        codes=codes,
        windows=(10, 20, 30, 40, 50),
        max_depth=max_depth,
        max_nodes=max_nodes,
    )

    results = []
    for _, row in top_factors.iterrows():
        fid = row["factor_id"]
        expr = row["factor_expr"]

        if fid in done_ids and not overwrite:
            results.append({"factor_id": fid, "expr": expr, "status": "SKIPPED", "coverage": np.nan})
            continue

        arr, status = evaluator.eval_expr(expr)
        if arr is None:
            results.append({"factor_id": fid, "expr": expr, "status": status, "coverage": np.nan})
        else:
            coverage = float(np.isfinite(arr).mean())
            arr = arr.astype(np.float32)
            panel_to_parquet(arr, dates, codes, factor_dir / f"{fid}.parquet")
            results.append({"factor_id": fid, "expr": expr, "status": "OK", "coverage": coverage})

    all_factors = []
    for r in results:
        existing = next((f for f in manifest.get("factors", []) if f["factor_id"] == r["factor_id"]), None)
        if existing:
            existing.update(r)
            all_factors.append(existing)
        else:
            all_factors.append(r)

    new_manifest = {
        "factors": all_factors,
        "updated_at": datetime.now().isoformat(),
    }
    save_factor_manifest(out_dir, new_manifest)

    return {r["factor_id"]: r for r in results}