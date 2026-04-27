import numpy as np
import pandas as pd
from pathlib import Path

def make_rebalance_mask(dates, n=5):
    mask = np.zeros(len(dates), dtype=bool)
    mask[::n] = True
    return mask

def make_top_weights(factor, dates, top_pct=0.05, rebalance_n=5, direction=1):
    x = factor * direction
    T, N = x.shape
    weights = np.zeros((T, N), dtype=float)
    rebalance = make_rebalance_mask(dates, rebalance_n)
    last_w = np.zeros(N, dtype=float)

    for t in range(T):
        if rebalance[t]:
            row = x[t]
            valid = np.isfinite(row)
            n_valid = valid.sum()
            if n_valid > 0:
                k = max(1, int(n_valid * top_pct))
                valid_idx = np.where(valid)[0]
                selected = valid_idx[np.argsort(row[valid])[-k:]]
                w = np.zeros(N, dtype=float)
                w[selected] = 1.0 / k
                last_w = w
        weights[t] = last_w
    return weights

def portfolio_returns(weights, close, fee_bps=10.0, slippage_bps=5.0):
    ret = close[1:] / close[:-1] - 1
    w = weights[:-1]
    gross = np.nansum(w * ret, axis=1)

    turnover = np.nansum(np.abs(weights[1:] - weights[:-1]), axis=1)
    cost = turnover * (fee_bps + slippage_bps) / 10000.0

    net = gross - cost
    return net, turnover

def calc_stats(returns, dates, train_end, test_start):
    d = np.array(dates).astype(str)
    train_mask = d <= train_end
    test_mask = d >= test_start

    def period_stats(r, mask):
        r = r[mask[1:]]
        if len(r) == 0 or np.isnan(r).all():
            return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
        ann_ret = np.nanmean(r) * 252
        ann_vol = np.nanstd(r) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        return {"ann_return": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe}

    train_stats = period_stats(returns, train_mask)
    test_stats = period_stats(returns, test_mask)

    cumret = np.nancumprod(1 + returns) - 1
    max_dd = 0.0
    peak = 0.0
    for r in cumret:
        if np.isfinite(r):
            if r > peak:
                peak = r
            dd = (r - peak) / (1 + peak)
            if dd < max_dd:
                max_dd = dd

    total_ret = float(cumret[-1]) if len(cumret) > 0 and np.isfinite(cumret[-1]) else np.nan

    win_rate = float(np.nanmean(returns > 0)) if len(returns) > 0 else np.nan

    return {
        "total_return": total_ret,
        "ann_return": train_stats["ann_return"],
        "ann_vol": train_stats["ann_vol"],
        "sharpe": train_stats["sharpe"],
        "max_drawdown": max_dd,
        "calmar": abs(train_stats["ann_return"] / max_dd) if max_dd != 0 else np.nan,
        "win_rate": win_rate,
        "train_ann_return": train_stats["ann_return"],
        "test_ann_return": test_stats["ann_return"],
        "test_sharpe": test_stats["sharpe"],
    }

def run_vectorbt_like_backtest(factor_panels, close, dates, codes, factor_metrics, cfg):
    out_dir = Path(cfg.out_dir) / "vectorbt"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = factor_metrics.set_index("factor_id")
    top_quantiles = cfg.top_quantiles

    all_equity_rows = []

    for fid, factor in factor_panels.items():
        if fid not in metrics_df.index:
            continue
        mric = metrics_df.loc[fid, "mean_rank_ic"]
        direction = 1 if mric >= 0 else -1

        for tq in top_quantiles:
            weights = make_top_weights(factor, dates, top_pct=tq, rebalance_n=cfg.rebalance_n, direction=direction)
            returns, turnover = portfolio_returns(weights, close, fee_bps=cfg.fee_bps, slippage_bps=cfg.slippage_bps)

            stats = calc_stats(returns, dates, cfg.train_end, cfg.test_start)
            stats["factor_id"] = fid
            stats["top_pct"] = tq
            stats["direction"] = direction
            stats["avg_turnover"] = float(np.nanmean(turnover)) if len(turnover) > 0 else np.nan

            all_equity_rows.append(stats)

            equity = np.nancumprod(1 + returns)
            for i, eq in enumerate(equity):
                all_equity_rows.append({
                    "factor_id": fid,
                    "top_pct": tq,
                    "date": dates[i + 1] if i + 1 < len(dates) else dates[-1],
                    "equity": eq,
                })

    summary_df = pd.DataFrame([r for r in all_equity_rows if "equity" not in r])
    summary_df.to_csv(out_dir / "portfolio_summary.csv", index=False, encoding="utf-8-sig")

    equity_df = pd.DataFrame([r for r in all_equity_rows if "equity" in r])
    if not equity_df.empty:
        equity_df.to_parquet(out_dir / "equity_curves.parquet", index=False)

    return {"summary": summary_df}