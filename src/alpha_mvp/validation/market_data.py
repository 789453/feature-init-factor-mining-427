import duckdb
import pandas as pd

TABLE_CANDIDATES = {
    "daily": ["silver.fact_stock_daily", "stock_daily", "silver.stock_daily"],
    "moneyflow": ["silver.fact_stock_moneyflow", "stock_moneyflow", "silver.stock_moneyflow"],
    "cyq": ["silver.fact_stock_cyq_perf", "stock_cyq_perf", "silver.stock_cyq_perf"],
    "basic": ["silver.fact_stock_daily_basic", "stock_daily_basic", "silver.stock_daily_basic"],
    "snapshot": ["silver.fact_stock_basic_snapshot", "stock_basic_snapshot", "silver.stock_basic_snapshot"],
}

def first_existing_table(conn, candidates):
    rows = conn.execute(
        "select table_schema || '.' || table_name as name from information_schema.tables"
    ).fetchall()
    names = {r[0] for r in rows}
    rows2 = conn.execute("select table_name from information_schema.tables").fetchall()
    names |= {r[0] for r in rows2}
    for t in candidates:
        if t in names:
            return t
    return candidates[0]

def load_market_data(cfg):
    conn = duckdb.connect(cfg.duckdb_path, read_only=True)
    conn.execute(f"set threads={cfg.duckdb_threads}")
    conn.execute(f"set memory_limit='{cfg.duckdb_memory_limit}'")

    daily = first_existing_table(conn, TABLE_CANDIDATES["daily"])
    money = first_existing_table(conn, TABLE_CANDIDATES["moneyflow"])
    cyq = first_existing_table(conn, TABLE_CANDIDATES["cyq"])
    basic = first_existing_table(conn, TABLE_CANDIDATES["basic"])
    snap = first_existing_table(conn, TABLE_CANDIDATES["snapshot"])

    q = f'''
    select
      d.ts_code, d.trade_date,
      d.open, d.high, d.low, d.close, d.pre_close, d.pct_chg, d.vol, d.amount,

      m.buy_sm_amount, m.sell_sm_amount,
      m.buy_md_amount, m.sell_md_amount,
      m.buy_lg_amount, m.sell_lg_amount,
      m.net_mf_amount,

      c.his_low, c.his_high,
      c.cost_5pct, c.cost_15pct, c.cost_50pct,
      c.cost_85pct, c.cost_95pct,
      c.weight_avg, c.winner_rate,

      b.turnover_rate, b.turnover_rate_f,
      b.volume_ratio, b.total_mv, b.circ_mv,

      s.industry, s.name, s.market, s.list_date
    from {daily} d
    left join {money} m
      on d.ts_code=m.ts_code and d.trade_date=m.trade_date
    left join {cyq} c
      on d.ts_code=c.ts_code and d.trade_date=c.trade_date
    left join {basic} b
      on d.ts_code=b.ts_code and d.trade_date=b.trade_date
    left join {snap} s
      on d.ts_code=s.ts_code
    where d.trade_date >= '{cfg.start}'
      and d.trade_date <= '{cfg.end}'
      and d.close is not null
      and d.close > {cfg.min_price}
    order by d.trade_date, d.ts_code
    '''
    df = conn.execute(q).fetchdf()
    conn.close()
    return df