import numpy as np
import pandas as pd

def make_size_bucket(meta, dates, codes, n_buckets=5):
    circ = meta.pivot(index="trade_date", columns="ts_code", values="circ_mv").reindex(index=dates, columns=codes)
    log_mv = np.log1p(circ.to_numpy(dtype=float))
    bucket = np.full(log_mv.shape, -1, dtype=int)
    for t in range(log_mv.shape[0]):
        row = log_mv[t]
        m = np.isfinite(row)
        if m.sum() < 50:
            continue
        ranks = pd.Series(row[m]).rank(pct=True).to_numpy()
        b = np.floor(ranks * n_buckets).astype(int)
        b[b == n_buckets] = n_buckets - 1
        bucket[t, np.where(m)[0]] = b
    return bucket

def get_industry_map(meta, codes):
    latest = meta.sort_values("trade_date").groupby("ts_code").last()
    industry = latest.get("industry", pd.Series(dtype=str))
    result = {}
    for code in codes:
        if code in industry.index:
            ind = industry.get(code, None)
            if pd.notna(ind):
                result[code] = str(ind)
            else:
                result[code] = "UNKNOWN"
        else:
            result[code] = "UNKNOWN"
    return result