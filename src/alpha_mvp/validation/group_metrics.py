from __future__ import annotations
import numpy as np
import pandas as pd

def make_size_bucket(circ_mv: np.ndarray, n_buckets: int = 5) -> np.ndarray:
    T, N = circ_mv.shape
    log_mv = np.log1p(circ_mv)
    bucket = np.full((T, N), -1, dtype=np.int8)

    for t in range(T):
        row = log_mv[t]
        m = np.isfinite(row)
        if m.sum() < 50:
            continue
        ranks = pd.Series(row[m]).rank(pct=True).to_numpy()
        b = np.floor(ranks * n_buckets).astype(np.int8)
        b[b == n_buckets] = n_buckets - 1
        bucket[t, np.where(m)[0]] = b

    return bucket

def group_by_size(factor: np.ndarray, fwd: np.ndarray, circ_mv: np.ndarray, dates: list[str]) -> pd.DataFrame:
    bucket = make_size_bucket(circ_mv, n_buckets=5)
    bucket_names = ["micro", "small", "mid", "large", "mega"]

    ic_dict = {name: [] for name in bucket_names}

    for t in range(factor.shape[0]):
        fa = factor[t]
        fr = fwd[t]
        b = bucket[t]

        for bi, name in enumerate(bucket_names):
            mask = (b == bi) & np.isfinite(fa) & np.isfinite(fr)
            if mask.sum() >= 10:
                ic_dict[name].append(float(np.corrcoef(fa[mask], fr[mask])[0, 1]))

    rows = []
    for name in bucket_names:
        vals = ic_dict[name]
        rows.append({
            "bucket": name,
            "mean_rank_ic": float(np.nanmean(vals)) if vals else np.nan,
            "std_rank_ic": float(np.nanstd(vals)) if vals else np.nan,
            "positive_ratio": float(np.nanmean([v > 0 for v in vals])) if vals else np.nan,
            "n_samples": len(vals),
        })
    return pd.DataFrame(rows)

def group_by_industry(factor: np.ndarray, fwd: np.ndarray, industry: np.ndarray, dates: list[str], codes: list[str]) -> pd.DataFrame:
    T, N = factor.shape
    code_to_industry = {}
    for j in range(N):
        val = industry[0, j] if industry.shape[0] > 0 else None
        if val and str(val).strip():
            code_to_industry[codes[j]] = str(val).strip()
        else:
            code_to_industry[codes[j]] = "UNKNOWN"

    industry_list = sorted(set(code_to_industry.values()))

    ic_dict = {ind: [] for ind in industry_list}

    for t in range(T):
        fa = factor[t]
        fr = fwd[t]
        for j in range(N):
            code = codes[j]
            ind = code_to_industry.get(code, "UNKNOWN")
            fa_j = fa[j]
            fr_j = fr[j]
            if np.isfinite(fa_j) and np.isfinite(fr_j):
                ic_dict[ind].append(float(fa_j * fr_j))

    rows = []
    for ind in industry_list:
        vals = ic_dict.get(ind, [])
        valid_vals = [v for v in vals if v != 0]
        if len(valid_vals) >= 10:
            rows.append({
                "industry": ind,
                "mean_rank_ic": float(np.nanmean(valid_vals)),
                "n_samples": len(valid_vals),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mean_rank_ic", ascending=False)
    return result

def run_group_metrics(
    factor_panels: dict[str, np.ndarray],
    fwd: np.ndarray,
    circ_mv: np.ndarray,
    industry: np.ndarray,
    dates: list[str],
    codes: list[str],
    out_dir: str,
) -> dict:
    from pathlib import Path
    out_path = Path(out_dir)
    metrics_dir = out_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    first_factor = next(iter(factor_panels.values()))

    size_df = group_by_size(first_factor, fwd, circ_mv, dates)
    size_df.to_csv(metrics_dir / "group_size.csv", index=False, encoding="utf-8-sig")

    industry_df = group_by_industry(first_factor, fwd, industry, dates, codes)
    industry_df.to_csv(metrics_dir / "group_industry.csv", index=False, encoding="utf-8-sig")

    return {"Size": size_df, "Industry": industry_df}