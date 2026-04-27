from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

SCHEMA = """
create table if not exists factor_results (
    run_id text not null,
    expr text not null,
    canonical text,
    status text,
    coverage real,
    usable_days integer,
    mean_ic real,
    icir real,
    mean_rank_ic real,
    rank_icir real,
    positive_rank_ic_ratio real,
    turnover_proxy real,
    quantile_spread real,
    train_mean_rank_ic real,
    train_rank_icir real,
    test_mean_rank_ic real,
    test_rank_icir real,
    score real,
    error text,
    created_at text default current_timestamp,
    primary key (run_id, canonical)
);

create index if not exists idx_factor_results_run_score
on factor_results(run_id, score desc);
"""

class ResultStore:
    def __init__(self, sqlite_path: str):
        self.path = Path(sqlite_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=60)
        self.conn.execute("pragma journal_mode=WAL")
        self.conn.execute("pragma synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def completed_keys(self, run_id: str) -> set[str]:
        rows = self.conn.execute(
            "select canonical from factor_results where run_id=?",
            (run_id,)
        ).fetchall()
        return {r[0] for r in rows}

    def upsert_many(self, run_id: str, rows: list[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        df["run_id"] = run_id
        cols = [
            "run_id", "expr", "canonical", "status", "coverage", "usable_days",
            "mean_ic", "icir", "mean_rank_ic", "rank_icir",
            "positive_rank_ic_ratio", "turnover_proxy", "quantile_spread",
            "train_mean_rank_ic", "train_rank_icir",
            "test_mean_rank_ic", "test_rank_icir",
            "score", "error",
        ]
        for c in cols:
            if c not in df:
                df[c] = None
        sql = f"""
        insert or replace into factor_results ({",".join(cols)})
        values ({",".join(["?"] * len(cols))})
        """
        self.conn.executemany(sql, df[cols].itertuples(index=False, name=None))
        self.conn.commit()

    def load_all(self, run_id: str) -> pd.DataFrame:
        return pd.read_sql_query(
            "select * from factor_results where run_id=? order by score desc",
            self.conn,
            params=(run_id,),
        )

    def close(self):
        self.conn.close()