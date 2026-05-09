from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def forward_returns(close: np.ndarray, horizon: int = 5) -> np.ndarray:
    out = np.full_like(close, np.nan, dtype=np.float32)
    out[:-horizon] = close[horizon:] / close[:-horizon] - 1
    return out

def daily_pearson_ic(factor: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    T, N = factor.shape
    ic = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        fa = factor[t]
        fr = fwd[t]
        m = np.isfinite(fa) & np.isfinite(fr)
        if m.sum() < 10:
            continue
        ic[t] = np.corrcoef(fa[m], fr[m])[0, 1]
    return ic

def daily_rank_ic(factor: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    T, N = factor.shape
    ric = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        fa = factor[t]
        fr = fwd[t]
        m = np.isfinite(fa) & np.isfinite(fr)
        if m.sum() < 10:
            continue
        ranks_fa = pd.Series(fa[m]).rank()
        ranks_fr = pd.Series(fr[m]).rank()
        ric[t] = np.corrcoef(ranks_fa, ranks_fr)[0, 1]
    return ric

def rolling_ic_series(ic: np.ndarray, window: int) -> np.ndarray:
    T = len(ic)
    out = np.full(T, np.nan, dtype=np.float32)
    for t in range(window, T):
        window_vals = ic[t-window:t]
        valid = np.isfinite(window_vals)
        if valid.sum() >= max(5, window // 2):
            out[t] = np.nanmean(window_vals[valid])
    return out

def calc_factor_metrics(
    fid: str,
    factor: np.ndarray,
    fwd: np.ndarray,
    dates: list[str],
    train_end: str,
    test_start: str,
) -> dict:
    ic = daily_pearson_ic(factor, fwd)
    ric = daily_rank_ic(factor, fwd)

    d = np.array(dates).astype(str)
    train_mask = d <= train_end
    test_mask = d >= test_start

    def period_stats(x, mask):
        vals = x[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return {"mean": np.nan, "std": np.nan}
        return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    train_ic = period_stats(ic, train_mask)
    test_ic = period_stats(ic, test_mask)
    train_ric = period_stats(ric, train_mask)
    test_ric = period_stats(ric, test_mask)

    rolling_20 = rolling_ic_series(ric, 20)
    rolling_60 = rolling_ic_series(ric, 60)
    rolling_120 = rolling_ic_series(ric, 120)

    return {
        "factor_id": fid,
        "coverage": float(np.nanmean(np.isfinite(factor))),
        "usable_days": int(np.sum(np.isfinite(factor).sum(axis=1) >= 30)),
        "mean_ic": train_ic["mean"],
        "icir": train_ic["mean"] / train_ic["std"] if train_ic["std"] > 0 else np.nan,
        "mean_rank_ic": train_ric["mean"],
        "rank_icir": train_ric["mean"] / train_ric["std"] if train_ric["std"] > 0 else np.nan,
        "positive_rank_ic_ratio": float(np.nanmean(ric > 0)),
        "train_mean_rank_ic": train_ric["mean"],
        "train_rank_icir": train_ric["mean"] / train_ric["std"] if train_ric["std"] > 0 else np.nan,
        "test_mean_rank_ic": test_ric["mean"],
        "test_rank_icir": test_ric["mean"] / test_ric["std"] if test_ric["std"] > 0 else np.nan,
        "rolling_20_rank_ic": float(np.nanmean(rolling_20[np.isfinite(rolling_20)])) if np.isfinite(rolling_20).sum() > 0 else np.nan,
        "rolling_60_rank_ic": float(np.nanmean(rolling_60[np.isfinite(rolling_60)])) if np.isfinite(rolling_60).sum() > 0 else np.nan,
        "rolling_120_rank_ic": float(np.nanmean(rolling_120[np.isfinite(rolling_120)])) if np.isfinite(rolling_120).sum() > 0 else np.nan,
    }

def run_batch_analytics(
    factor_panels: dict[str, np.ndarray],
    close: np.ndarray,
    dates: list[str],
    codes: list[str],
    out_dir: str,
    cfg,
) -> dict:
    out_path = Path(out_dir)
    metrics_dir = out_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    fwd = forward_returns(close, horizon=cfg.horizon)

    summary_rows = []
    rolling_rows = []

    for fid, factor in factor_panels.items():
        m = calc_factor_metrics(fid, factor, fwd, dates, cfg.train_end, cfg.test_start)

        tr = m.get("train_mean_rank_ic")
        te = m.get("test_mean_rank_ic")
        if tr is not None and te is not None and np.isfinite(tr) and np.isfinite(te):
            same_sign = 1.0 if tr * te > 0 else 0.25
            m["score"] = abs(te) * abs(m.get("test_rank_icir") or 0) * same_sign
        else:
            m["score"] = np.nan

        summary_rows.append(m)

        ric = daily_rank_ic(factor, fwd)
        r20 = rolling_ic_series(ric, 20)
        r60 = rolling_ic_series(ric, 60)
        r120 = rolling_ic_series(ric, 120)

        for i, d in enumerate(dates):
            rolling_rows.append({
                "factor_id": fid,
                "date": d,
                "rank_ic": ric[i] if i < len(ric) else np.nan,
                "rolling_20": r20[i] if i < len(r20) else np.nan,
                "rolling_60": r60[i] if i < len(r60) else np.nan,
                "rolling_120": r120[i] if i < len(r120) else np.nan,
            })

    summary_df = pd.DataFrame(summary_rows)
    rolling_df = pd.DataFrame(rolling_rows)

    summary_df.to_csv(metrics_dir / "summary.csv", index=False, encoding="utf-8-sig")
    rolling_df.to_parquet(metrics_dir / "rolling_ic.parquet", index=False)

    return {"summary": summary_df, "rolling_ic": rolling_df}