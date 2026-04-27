from __future__ import annotations
import numpy as np
import pandas as pd
from . import fastops

def forward_returns(close: np.ndarray, horizon=5) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=float)
    out[:-horizon] = close[horizon:] / close[:-horizon] - 1
    return out

def _daily_corr(x, y, rank=False):
    return fastops.daily_corr(x, y, rank=rank)

def turnover_proxy(factor: np.ndarray) -> float:
    ranks = fastops.rank_cs(factor)
    diff = np.nanmean(np.abs(ranks[1:] - ranks[:-1]))
    return float(diff) if np.isfinite(diff) else np.nan

def quantile_spread(factor: np.ndarray, fwd: np.ndarray, q=5) -> float:
    spreads = []
    for i in range(factor.shape[0]):
        a, r = factor[i], fwd[i]
        m = np.isfinite(a) & np.isfinite(r)
        if m.sum() < q * 10:
            continue
        ranks = fastops.rank_cs(a[m].reshape(1, -1)).flatten()
        top = r[m][ranks >= 1 - 1/q]
        bot = r[m][ranks <= 1/q]
        if len(top) > 0 and len(bot) > 0:
            spreads.append(np.nanmean(top) - np.nanmean(bot))
    return float(np.nanmean(spreads)) if spreads else np.nan

def summarize_factor(factor: np.ndarray, fwd: np.ndarray, dates: list[str], min_daily_valid=30) -> dict:
    valid = np.isfinite(factor)
    coverage = float(np.nanmean(valid))
    usable = valid.sum(axis=1) >= min_daily_valid
    factor2 = factor[usable]
    fwd2 = fwd[usable]
    dates2 = np.array(dates)[usable]
    ic = _daily_corr(factor2, fwd2, rank=False)
    ric = _daily_corr(factor2, fwd2, rank=True)
    mean_ic = float(np.nanmean(ic)) if len(ic) else np.nan
    std_ic = float(np.nanstd(ic)) if len(ic) else np.nan
    mean_ric = float(np.nanmean(ric)) if len(ric) else np.nan
    std_ric = float(np.nanstd(ric)) if len(ric) else np.nan
    years = pd.to_datetime(dates2).year if len(dates2) else []
    by_year = {}
    for y in sorted(set(years)):
        mask = np.array(years) == y
        by_year[str(y)] = float(np.nanmean(ric[mask])) if mask.any() else np.nan
    return {
        "coverage": coverage,
        "usable_days": int(np.sum(usable)),
        "mean_ic": mean_ic,
        "icir": mean_ic / std_ic if std_ic and np.isfinite(std_ic) else np.nan,
        "mean_rank_ic": mean_ric,
        "rank_icir": mean_ric / std_ric if std_ric and np.isfinite(std_ric) else np.nan,
        "positive_rank_ic_ratio": float(np.nanmean(ric > 0)) if len(ric) else np.nan,
        "turnover_proxy": turnover_proxy(factor2) if len(factor2) else np.nan,
        "quantile_spread": quantile_spread(factor2, fwd2) if len(factor2) else np.nan,
        "year_rank_ic": by_year,
    }

def summarize_factor_split(
    factor: np.ndarray,
    fwd: np.ndarray,
    dates: list[str],
    train_end: str = "20250831",
    test_start: str = "20250901",
    min_daily_valid: int = 30,
) -> dict:
    base = summarize_factor(factor, fwd, dates, min_daily_valid=min_daily_valid)
    d = np.array(dates).astype(str)
    train_mask = d <= train_end
    test_mask = d >= test_start

    train = summarize_factor(
        factor[train_mask],
        fwd[train_mask],
        d[train_mask].tolist(),
        min_daily_valid=min_daily_valid,
    )
    test = summarize_factor(
        factor[test_mask],
        fwd[test_mask],
        d[test_mask].tolist(),
        min_daily_valid=min_daily_valid,
    )

    base.update({
        "train_mean_rank_ic": train.get("mean_rank_ic"),
        "train_rank_icir": train.get("rank_icir"),
        "test_mean_rank_ic": test.get("mean_rank_ic"),
        "test_rank_icir": test.get("rank_icir"),
    })

    tr = base.get("train_mean_rank_ic")
    te = base.get("test_mean_rank_ic")
    if tr is not None and te is not None and np.isfinite(tr) and np.isfinite(te):
        same_sign = 1.0 if tr * te > 0 else 0.25
        base["score"] = abs(te) * abs(base.get("test_rank_icir", 0) or 0) * same_sign
    else:
        base["score"] = np.nan
    return base