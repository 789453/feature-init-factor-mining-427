import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from alpha_mvp.validation.size_industry import make_size_bucket, get_industry_map
from alpha_mvp.validation.panel_store import load_factor_panel

def forward_returns(close, horizon=5):
    out = np.full_like(close, np.nan, dtype=float)
    out[:-horizon] = close[horizon:] / close[:-horizon] - 1
    return out

def daily_rank_ic(factor, fwd):
    T, N = factor.shape
    ic = np.full(T, np.nan)
    ric = np.full(T, np.nan)
    for t in range(T):
        fa = factor[t]
        fr = fwd[t]
        m = np.isfinite(fa) & np.isfinite(fr)
        if m.sum() < 10:
            continue
        fa_m = fa[m]
        fr_m = fr[m]
        ic[t] = np.corrcoef(fa_m, fr_m)[0, 1]
        ranks_fa = pd.Series(fa_m).rank()
        ranks_fr = pd.Series(fr_m).rank()
        ric[t] = np.corrcoef(ranks_fa, ranks_fr)[0, 1]
    return ic, ric

def rolling_ic(ric, windows=(20, 60, 120)):
    result = {}
    for w in windows:
        out = np.full(len(ric), np.nan)
        for i in range(w, len(ric)):
            window = ric[i-w:i]
            m = np.isfinite(window)
            if m.sum() >= w // 2:
                out[i] = np.nanmean(window[m])
        result[f"rolling_{w}_rank_ic"] = out
    return result

def factor_summary(factor, fwd, dates, min_daily_valid=30):
    valid = np.isfinite(factor)
    coverage = float(np.nanmean(valid))
    usable = valid.sum(axis=1) >= min_daily_valid
    factor2 = factor[usable]
    fwd2 = fwd[usable]
    dates2 = np.array(dates)[usable]

    ic, ric = daily_rank_ic(factor2, fwd2)

    mean_ic = float(np.nanmean(ic)) if len(ic) else np.nan
    std_ic = float(np.nanstd(ic)) if len(ic) else np.nan
    mean_ric = float(np.nanmean(ric)) if len(ric) else np.nan
    std_ric = float(np.nanstd(ric)) if len(ric) else np.nan

    r = rolling_ic(ric)
    return {
        "coverage": coverage,
        "usable_days": int(np.sum(usable)),
        "mean_ic": mean_ic,
        "icir": mean_ic / std_ic if std_ic and np.isfinite(std_ic) else np.nan,
        "mean_rank_ic": mean_ric,
        "rank_icir": mean_ric / std_ric if std_ric and np.isfinite(std_ric) else np.nan,
        "positive_rank_ic_ratio": float(np.nanmean(ric > 0)) if len(ric) else np.nan,
        **r,
    }

def factor_summary_split(factor, fwd, dates, train_end, test_start, min_daily_valid=30):
    base = factor_summary(factor, fwd, dates, min_daily_valid)
    d = np.array(dates).astype(str)
    train_mask = d <= train_end
    test_mask = d >= test_start

    train = factor_summary(factor[train_mask], fwd[train_mask], d[train_mask].tolist(), min_daily_valid)
    test = factor_summary(factor[test_mask], fwd[test_mask], d[test_mask].tolist(), min_daily_valid)

    base["train_mean_rank_ic"] = train.get("mean_rank_ic")
    base["train_rank_icir"] = train.get("rank_icir")
    base["test_mean_rank_ic"] = test.get("mean_rank_ic")
    base["test_rank_icir"] = test.get("rank_icir")

    tr = base.get("train_mean_rank_ic")
    te = base.get("test_mean_rank_ic")
    if tr is not None and te is not None and np.isfinite(tr) and np.isfinite(te):
        same_sign = 1.0 if tr * te > 0 else 0.25
        base["score"] = abs(te) * abs(base.get("test_rank_icir", 0) or 0) * same_sign
    else:
        base["score"] = np.nan
    return base

def group_metrics_by_size(factor, fwd, meta, dates, codes, cfg):
    buckets = make_size_bucket(meta, dates, codes, n_buckets=5)
    bucket_names = ["micro", "small", "mid", "large", "mega"]
    size_labels = {i: bucket_names[i] if i >= 0 else "unknown" for i in range(5)}

    ic_dict = {}
    for b in range(5):
        mask = buckets == b
        for t in range(len(dates)):
            fa = factor[t] * mask[t]
            fr = fwd[t]
            m = mask[t] & np.isfinite(fa) & np.isfinite(fr)
            if m.sum() < 10:
                continue
            key = f"bucket_{bucket_names[b]}"
            if key not in ic_dict:
                ic_dict[key] = []
            fa_m = fa[m]
            fr_m = fr[m]
            ic_dict[key].append(np.corrcoef(fa_m, fr_m)[0, 1])

    rows = []
    for b in range(5):
        name = bucket_names[b]
        vals = ic_dict.get(f"bucket_{name}", [np.nan])
        rows.append({
            "bucket": name,
            "mean_rank_ic": float(np.nanmean(vals)) if vals else np.nan,
            "std_rank_ic": float(np.nanstd(vals)) if vals else np.nan,
            "positive_ratio": float(np.nanmean([v > 0 for v in vals])) if vals else np.nan,
        })
    return pd.DataFrame(rows)

def group_metrics_by_industry(factor, fwd, meta, dates, codes):
    industry_map = get_industry_map(meta, codes)
    unique_industries = sorted(set(industry_map.values()))

    ic_dict = {}
    for code, ind in industry_map.items():
        try:
            col_idx = codes.index(code)
        except ValueError:
            continue
        for t in range(len(dates)):
            fa = factor[t, col_idx]
            fr = fwd[t, col_idx]
            if not (np.isfinite(fa) and np.isfinite(fr)):
                continue
            if ind not in ic_dict:
                ic_dict[ind] = []
            ic_dict[ind].append(np.corrcoef([fa], [fr])[0, 1])

    rows = []
    for ind, vals in ic_dict.items():
        rows.append({
            "industry": ind,
            "mean_rank_ic": float(np.nanmean(vals)) if vals else np.nan,
            "std_rank_ic": float(np.nanstd(vals)) if vals else np.nan,
            "positive_ratio": float(np.nanmean([v > 0 for v in vals])) if vals else np.nan,
            "n_samples": len(vals),
        })
    return pd.DataFrame(rows).sort_values("mean_rank_ic", ascending=False)

def run_factor_analytics(factor_panels, close, dates, codes, meta, cfg):
    fwd = forward_returns(close, horizon=cfg.horizon)
    out_dir = Path(cfg.out_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rolling_ic_rows = []
    summary_rows = []

    for fid, factor in factor_panels.items():
        m = factor_summary_split(factor, fwd, dates, cfg.train_end, cfg.test_start)
        m["factor_id"] = fid
        summary_rows.append(m)

        ric_df = pd.DataFrame({"date": dates})
        ic, ric = daily_rank_ic(factor, fwd)
        ric_df["rank_ic"] = ric
        r = rolling_ic(ric)
        for k, v in r.items():
            ric_df[k] = v
        ric_df["factor_id"] = fid
        rolling_ic_rows.append(ric_df)

    summary_df = pd.DataFrame(summary_rows)
    rolling_ic_df = pd.concat(rolling_ic_rows, ignore_index=True)

    summary_df.to_csv(out_dir / "factor_metrics_summary.csv", index=False, encoding="utf-8-sig")
    rolling_ic_df.to_csv(out_dir / "rolling_ic.csv", index=False, encoding="utf-8-sig")

    size_metrics = group_metrics_by_size(
        next(iter(factor_panels.values())),
        fwd, meta, dates, codes, cfg
    )
    size_metrics.to_csv(out_dir / "group_metrics_by_size.csv", index=False, encoding="utf-8-sig")

    industry_metrics = group_metrics_by_industry(
        next(iter(factor_panels.values())),
        fwd, meta, dates, codes
    )
    industry_metrics.to_csv(out_dir / "group_metrics_by_industry.csv", index=False, encoding="utf-8-sig")

    return {
        "summary": summary_df,
        "rolling_ic": rolling_ic_df,
        "size_metrics": size_metrics,
        "industry_metrics": industry_metrics,
    }