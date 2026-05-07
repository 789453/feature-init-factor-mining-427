from __future__ import annotations
import duckdb
import json
import os
from datetime import datetime
from .template_builder import ExpressionRecord
from .expr_meta import ExprMeta

class DuckDBJobStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        self._init_tables()

    def _init_tables(self):
        # 1. 运行记录表
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_signature VARCHAR PRIMARY KEY,
            job_id VARCHAR,
            start_date VARCHAR,
            end_date VARCHAR,
            field_set_hash VARCHAR,
            grammar_hash VARCHAR,
            data_hash VARCHAR,
            universe_hash VARCHAR,
            eval_hash VARCHAR,
            code_hash VARCHAR,
            manifest_json JSON,
            status VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP
        )
        """)

        # 2. 表达式目录表 (全局唯一)
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS expression_catalog (
            expr_hash VARCHAR PRIMARY KEY,
            canonical VARCHAR,
            expr VARCHAR,
            template_name VARCHAR,
            template_family VARCHAR,
            template_order INTEGER,
            complexity_tier INTEGER,
            fields JSON,
            operators JSON,
            windows JSON,
            n_fields INTEGER,
            n_unique_fields INTEGER,
            n_ops INTEGER,
            depth INTEGER,
            nodes INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # 3. 表达式任务表 (特定运行下的任务)
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS expression_jobs (
            run_signature VARCHAR,
            expr_hash VARCHAR,
            canonical VARCHAR,
            status VARCHAR,
            priority DOUBLE,
            shard_id INTEGER,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            error VARCHAR,
            PRIMARY KEY(run_signature, expr_hash)
        )
        """)

        # 4. 因子结果表
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS factor_results (
            run_signature VARCHAR,
            expr_hash VARCHAR,
            canonical VARCHAR,
            expr VARCHAR,
            dimension INTEGER,
            status VARCHAR,
            coverage DOUBLE,
            usable_days INTEGER,
            raw_train_mean_rank_ic DOUBLE,
            raw_test_mean_rank_ic DOUBLE,
            oriented_train_mean_rank_ic DOUBLE,
            oriented_test_mean_rank_ic DOUBLE,
            oriented_test_rank_icir DOUBLE,
            positive_oriented_rank_ic_ratio DOUBLE,
            turnover_proxy DOUBLE,
            quantile_spread DOUBLE,
            yearly_positive_ratio DOUBLE,
            complexity_score DOUBLE,
            score_raw DOUBLE,
            score_ranked DOUBLE,
            error VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(run_signature, expr_hash)
        )
        """)

        # 5. 关联表 (用于 attribution)
        self.con.execute("CREATE TABLE IF NOT EXISTS expression_field_link (expr_hash VARCHAR, field VARCHAR, position INTEGER, PRIMARY KEY(expr_hash, field, position))")
        self.con.execute("CREATE TABLE IF NOT EXISTS expression_operator_link (expr_hash VARCHAR, operator VARCHAR, position INTEGER, PRIMARY KEY(expr_hash, operator, position))")
        self.con.execute("CREATE TABLE IF NOT EXISTS expression_window_link (expr_hash VARCHAR, \"window\" INTEGER, position INTEGER, PRIMARY KEY(expr_hash, \"window\", position))")

    def init_run(self, manifest: dict, run_signature: str):
        self.con.execute("""
            INSERT OR IGNORE INTO runs 
            (run_signature, job_id, start_date, end_date, field_set_hash, grammar_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_signature, 
            manifest.get("job_id"), 
            manifest.get("start"), 
            manifest.get("end"),
            manifest.get("field_set_hash"),
            manifest.get("grammar_hash"),
            "PENDING"
        ))

    def upsert_expressions(self, records: list[ExpressionRecord], metas: list[ExprMeta]):
        # 批量插入表达式目录
        for r, m in zip(records, metas):
            self.con.execute("""
                INSERT OR IGNORE INTO expression_catalog 
                (expr_hash, canonical, expr, template_name, template_family, template_order, complexity_tier, 
                 fields, operators, windows, n_fields, n_unique_fields, n_ops, depth, nodes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                r.expr_hash, r.canonical, r.expr, r.template_name, r.template_family, r.template_order, r.complexity_tier,
                json.dumps(m.fields), json.dumps(m.operators), json.dumps(m.windows),
                m.n_fields, m.n_unique_fields, m.n_ops, m.depth, m.nodes
            ))
            
            # 插入关联表
            for i, f in enumerate(m.fields):
                self.con.execute("INSERT OR IGNORE INTO expression_field_link VALUES (?, ?, ?)", (r.expr_hash, f, i))
            for i, o in enumerate(m.operators):
                self.con.execute("INSERT OR IGNORE INTO expression_operator_link VALUES (?, ?, ?)", (r.expr_hash, o, i))
            for i, w in enumerate(m.windows):
                self.con.execute('INSERT OR IGNORE INTO expression_window_link (expr_hash, "window", position) VALUES (?, ?, ?)', (r.expr_hash, w, i))

    def enqueue_jobs(self, run_signature: str, records: list[ExpressionRecord]):
        for r in records:
            self.con.execute("""
                INSERT OR IGNORE INTO expression_jobs (run_signature, expr_hash, canonical, status)
                VALUES (?, ?, ?, ?)
            """, (run_signature, r.expr_hash, r.canonical, "PENDING"))

    def completed_expr_hashes(self, run_signature: str) -> set[str]:
        res = self.con.execute("SELECT expr_hash FROM expression_jobs WHERE run_signature = ? AND status = 'COMPLETED'", (run_signature,)).fetchall()
        return {r[0] for r in res}

    def fetch_todo(self, run_signature: str, limit: int = 1000) -> list[dict]:
        res = self.con.execute("""
            SELECT j.expr_hash, j.canonical, c.expr, c.template_name, c.template_family, c.template_order, c.complexity_tier
            FROM expression_jobs j
            JOIN expression_catalog c ON j.expr_hash = c.expr_hash
            WHERE j.run_signature = ? AND j.status = 'PENDING'
            LIMIT ?
        """, (run_signature, limit)).fetchall()
        
        return [
            {
                "expr_hash": r[0],
                "canonical": r[1],
                "expr": r[2],
                "template_name": r[3],
                "template_family": r[4],
                "template_order": r[5],
                "complexity_tier": r[6]
            } for r in res
        ]

    def write_results(self, run_signature: str, results: list[dict]):
        for res in results:
            expr_hash = res["expr_hash"]
            # 更新任务状态
            self.con.execute("""
                UPDATE expression_jobs SET status = 'COMPLETED', finished_at = ?
                WHERE run_signature = ? AND expr_hash = ?
            """, (datetime.now(), run_signature, expr_hash))
            
            # 插入因子结果
            cols = [
                "run_signature", "expr_hash", "canonical", "expr", "dimension", "status",
                "coverage", "usable_days", "raw_train_mean_rank_ic", "raw_test_mean_rank_ic",
                "oriented_train_mean_rank_ic", "oriented_test_mean_rank_ic", "oriented_test_rank_icir",
                "positive_oriented_rank_ic_ratio", "turnover_proxy", "quantile_spread",
                "yearly_positive_ratio", "complexity_score", "score_raw"
            ]
            placeholders = ",".join(["?" for _ in cols])
            vals = [res.get(c) for c in cols]
            
            self.con.execute(f"INSERT OR REPLACE INTO factor_results ({','.join(cols)}) VALUES ({placeholders})", vals)

    def mark_failed(self, run_signature: str, expr_hash: str, error: str):
        self.con.execute("""
            UPDATE expression_jobs SET status = 'FAILED', error = ?, finished_at = ?
            WHERE run_signature = ? AND expr_hash = ?
        """, (error, datetime.now(), run_signature, expr_hash))

    def close(self):
        self.con.close()
