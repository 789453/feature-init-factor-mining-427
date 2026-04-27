from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

TABLE_CANDIDATES = {
    "daily": ["silver.fact_stock_daily", "stock_daily", "silver.stock_daily"],
    "moneyflow": ["silver.fact_stock_moneyflow", "stock_moneyflow", "silver.stock_moneyflow"],
    "cyq": ["silver.fact_stock_cyq_perf", "stock_cyq_perf", "silver.stock_cyq_perf"],
    "basic": ["silver.fact_stock_daily_basic", "stock_daily_basic", "silver.stock_daily_basic"],
    "snapshot": ["silver.fact_stock_basic_snapshot", "stock_basic_snapshot", "silver.stock_basic_snapshot"],
}

def load_pool(path: str | None) -> list[str] | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _first_existing_table(conn, candidates):
    existing = set()
    try:
        rows = conn.execute("select table_schema || '.' || table_name as name from information_schema.tables").fetchall()
        existing |= {r[0] for r in rows}
        rows2 = conn.execute("select table_name as name from information_schema.tables").fetchall()
        existing |= {r[0] for r in rows2}
    except Exception:
        pass
    for t in candidates:
        if t in existing:
            return t
    return candidates[0]

def load_from_duckdb(duckdb_path: str, pool_json: str | None, start: str, end: str) -> pd.DataFrame:
    import duckdb
    pool = load_pool(pool_json)
    pool_clause = ""
    if pool:
        codes = "','".join(pool)
        pool_clause = f" and d.ts_code in ('{codes}')"

    conn = duckdb.connect(duckdb_path, read_only=True)
    daily = _first_existing_table(conn, TABLE_CANDIDATES["daily"])
    money = _first_existing_table(conn, TABLE_CANDIDATES["moneyflow"])
    cyq = _first_existing_table(conn, TABLE_CANDIDATES["cyq"])
    basic = _first_existing_table(conn, TABLE_CANDIDATES["basic"])
    snap = _first_existing_table(conn, TABLE_CANDIDATES["snapshot"])

    q = f"""
    select
      d.ts_code, d.trade_date,
      d.open, d.high, d.low, d.close, d.pre_close, d.pct_chg, d.vol, d.amount,
      m.buy_sm_amount, m.sell_sm_amount, m.buy_md_amount, m.sell_md_amount,
      m.buy_lg_amount, m.sell_lg_amount, m.net_mf_amount,
      c.his_low, c.his_high, c.cost_5pct, c.cost_15pct, c.cost_50pct,
      c.cost_85pct, c.cost_95pct, c.weight_avg, c.winner_rate,
      b.turnover_rate, b.turnover_rate_f, b.volume_ratio, b.total_mv, b.circ_mv,
      s.industry
    from {daily} d
    left join {money} m on d.ts_code=m.ts_code and d.trade_date=m.trade_date
    left join {cyq} c on d.ts_code=c.ts_code and d.trade_date=c.trade_date
    left join {basic} b on d.ts_code=b.ts_code and d.trade_date=b.trade_date
    left join {snap} s on d.ts_code=s.ts_code
    where d.trade_date >= '{start}' and d.trade_date <= '{end}'
    {pool_clause}
    order by d.trade_date, d.ts_code
    """
    df = conn.execute(q).fetchdf()
    conn.close()
    return df

def make_simulated_data(n_days=90, n_stocks=60, start="2024-01-01", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    codes = [f"{i:06d}.SZ" for i in range(n_stocks)]
    idx = pd.MultiIndex.from_product([dates.strftime("%Y%m%d"), codes], names=["trade_date", "ts_code"])
    df = pd.DataFrame(index=idx).reset_index()
    n = len(df)

    base = rng.lognormal(mean=3.3, sigma=0.4, size=n_stocks)
    ret = rng.normal(0, 0.018, size=(n_days, n_stocks))
    flow_signal = rng.normal(0, 1, size=(n_days, n_stocks))
    ret[1:] += 0.0025 * np.tanh(flow_signal[:-1])
    close = base[None, :] * np.exp(np.cumsum(ret, axis=0))
    open_ = close * (1 + rng.normal(0, 0.006, size=close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, size=close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, size=close.shape)))
    pre_close = np.vstack([close[0], close[:-1]])
    vol = rng.lognormal(12, 0.5, size=close.shape)
    amount = vol * close * 0.01
    flat = lambda x: x.reshape(-1)
    df["open"], df["high"], df["low"], df["close"] = flat(open_), flat(high), flat(low), flat(close)
    df["pre_close"], df["pct_chg"], df["vol"], df["amount"] = flat(pre_close), flat((close / pre_close - 1) * 100), flat(vol), flat(amount)

    for side in ["sm", "md", "lg"]:
        scale = {"sm": 0.7, "md": 0.5, "lg": 0.35}[side]
        buy = amount * np.maximum(0.01, 0.5 + scale * flow_signal + rng.normal(0, 0.2, close.shape))
        sell = amount * np.maximum(0.01, 0.5 - scale * flow_signal + rng.normal(0, 0.2, close.shape))
        df[f"buy_{side}_amount"] = flat(buy)
        df[f"sell_{side}_amount"] = flat(sell)
    df["net_mf_amount"] = df["buy_lg_amount"] + df["buy_md_amount"] - df["sell_lg_amount"] - df["sell_md_amount"]

    df["cost_50pct"] = df["close"] * (1 + rng.normal(0, 0.04, n))
    df["cost_5pct"] = df["cost_50pct"] * (1 - rng.uniform(0.05, 0.15, n))
    df["cost_15pct"] = df["cost_50pct"] * (1 - rng.uniform(0.03, 0.10, n))
    df["cost_85pct"] = df["cost_50pct"] * (1 + rng.uniform(0.03, 0.10, n))
    df["cost_95pct"] = df["cost_50pct"] * (1 + rng.uniform(0.05, 0.15, n))
    df["weight_avg"] = df["cost_50pct"] * (1 + rng.normal(0, 0.02, n))
    df["winner_rate"] = np.clip(50 + 300 * (df["close"] / df["weight_avg"] - 1) + rng.normal(0, 10, n), 0, 100)
    df["his_low"] = df["close"] * rng.uniform(0.55, 0.9, n)
    df["his_high"] = df["close"] * rng.uniform(1.1, 1.8, n)

    df["turnover_rate"] = rng.uniform(0.5, 8, n)
    df["turnover_rate_f"] = df["turnover_rate"] * rng.uniform(1.0, 2.0, n)
    df["volume_ratio"] = rng.lognormal(0, 0.5, n)
    df["circ_mv"] = rng.lognormal(6.5, 0.8, n)
    df["total_mv"] = df["circ_mv"] * rng.uniform(1.0, 1.8, n)
    industries = np.array(["电子", "医药", "机械", "化工", "计算机", "银行", "汽车", "电力"])
    code_ind = {c: industries[i % len(industries)] for i, c in enumerate(codes)}
    df["industry"] = df["ts_code"].map(code_ind)
    return df
