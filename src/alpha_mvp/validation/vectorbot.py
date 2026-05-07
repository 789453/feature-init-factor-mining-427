from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def make_rebalance_mask(n_dates: int, rebalance_n: int) -> np.ndarray:
    mask = np.zeros(n_dates, dtype=bool)
    mask[::rebalance_n] = True
    mask[0] = True
    return mask

def make_top_weights(
    factor: np.ndarray,
    dates: list[str],
    top_pct: float = 0.05,
    rebalance_n: int = 5,
    direction: int = 1,
) -> np.ndarray:
    T, N = factor.shape
    x = factor * direction
    weights = np.zeros((T, N), dtype=np.float32)
    rebalance = make_rebalance_mask(T, rebalance_n)
    last_w = np.zeros(N, dtype=np.float32)

    for t in range(T):
        if rebalance[t]:
            row = x[t]
            valid = np.isfinite(row)
            n_valid = valid.sum()
            if n_valid > 0:
                k = max(1, int(n_valid * top_pct))
                valid_idx = np.where(valid)[0]
                selected = valid_idx[np.argsort(row[valid])[-k:]]
                w = np.zeros(N, dtype=np.float32)
                w[selected] = 1.0 / k
                last_w = w
        weights[t] = last_w
    return weights

def portfolio_returns(
    weights: np.ndarray,
    close: np.ndarray,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    ret = close[1:] / close[:-1] - 1
    w = weights[:-1]
    gross = np.nansum(w * ret, axis=1)

    turnover = np.nansum(np.abs(weights[1:] - weights[:-1]), axis=1)
    cost = turnover * (fee_bps + slippage_bps) / 10000.0

    net = gross - cost
    return net, turnover

def equity_curve(returns: np.ndarray) -> np.ndarray:
    eq = np.nancumprod(1 + returns)
    return eq

def calc_period_stats(returns: np.ndarray, return_dates: list[str], start: str, end: str) -> dict:
    d = np.array(return_dates).astype(str)
    mask = (d >= start) & (d <= end)
    r = returns[mask]
    if len(r) == 0 or np.isnan(r).all():
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    ann_ret = np.nanmean(r) * 252
    ann_vol = np.nanstd(r) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}

def max_drawdown(equity: np.ndarray) -> float:
    peak = 0.0
    max_dd = 0.0
    for e in equity:
        if np.isfinite(e):
            if e > peak:
                peak = e
            dd = (e - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
    return max_dd

def run_vectorbot(
    factor_panels: dict[str, np.ndarray],
    close: np.ndarray,
    dates: list[str],
    codes: list[str],
    factor_metrics: pd.DataFrame,
    out_dir: str,
    cfg,
) -> dict:
    out_path = Path(out_dir)
    vb_dir = out_path / "vectorbot"
    vb_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = factor_metrics.set_index("factor_id")

    return_dates = dates[1:]

    summary_rows = []
    equity_rows = []

    for fid, factor in factor_panels.items():
        if fid not in metrics_df.index:
            continue

        mric = metrics_df.loc[fid, "mean_rank_ic"] if "mean_rank_ic" in metrics_df.columns else 0
        direction = 1 if mric >= 0 else -1

        for tq in cfg.top_quantiles:
            weights = make_top_weights(factor, dates, top_pct=tq, rebalance_n=cfg.rebalance_n, direction=direction)
            returns, turnover = portfolio_returns(weights, close, fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps)
            eq = equity_curve(returns)

            train_stats = calc_period_stats(returns, return_dates, dates[0], cfg.train_end)
            test_stats = calc_period_stats(returns, return_dates, cfg.test_start, dates[-1])

            summary_rows.append({
                "factor_id": fid,
                "top_pct": tq,
                "direction": direction,
                "ann_return": train_stats["ann_return"],
                "ann_vol": train_stats["ann_vol"],
                "sharpe": train_stats["sharpe"],
                "max_drawdown": max_drawdown(eq),
                "win_rate": float(np.nanmean(returns > 0)),
                "avg_turnover": float(np.nanmean(turnover)),
                "total_return": float(eq[-1]) if len(eq) > 0 and np.isfinite(eq[-1]) else np.nan,
                "train_ann_return": train_stats["ann_return"],
                "test_ann_return": test_stats["ann_return"],
                "test_sharpe": test_stats["sharpe"],
            })

            for i, e in enumerate(eq):
                if i < len(return_dates):
                    equity_rows.append({
                        "factor_id": fid,
                        "top_pct": tq,
                        "date": return_dates[i],
                        "equity": e,
                    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(vb_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")

    if equity_rows:
        equity_df = pd.DataFrame(equity_rows)
        equity_df.to_parquet(vb_dir / "equity_curves.parquet", index=False)

    return {"Summary": summary_df}