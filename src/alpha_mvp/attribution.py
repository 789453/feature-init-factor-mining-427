from __future__ import annotations
import pandas as pd
import duckdb

def compute_field_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    query = f"""
    SELECT 
        l.field,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover
    FROM expression_field_link l
    JOIN factor_results r ON l.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY l.field
    ORDER BY median_score_ranked DESC
    """
    return con.execute(query).df()

def compute_operator_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    query = f"""
    SELECT 
        l.operator,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir
    FROM expression_operator_link l
    JOIN factor_results r ON l.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY l.operator
    ORDER BY median_score_ranked DESC
    """
    return con.execute(query).df()

def compute_window_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    query = f"""
    SELECT 
        l."window",
        COUNT(DISTINCT r.expr_hash) as n_expr,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic
    FROM expression_window_link l
    JOIN factor_results r ON l.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY l."window"
    ORDER BY l."window" ASC
    """
    return con.execute(query).df()

def compute_template_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    query = f"""
    SELECT 
        c.template_family,
        c.template_name,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic
    FROM expression_catalog c
    JOIN factor_results r ON c.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY c.template_family, c.template_name
    ORDER BY median_score_ranked DESC
    """
    return con.execute(query).df()

def run_all_attribution(con: duckdb.DuckDBPyConnection, run_signature: str, out_dir: str):
    import os
    export_dir = os.path.join(out_dir, "exports")
    os.makedirs(export_dir, exist_ok=True)
    
    compute_field_stats(con, run_signature).to_csv(os.path.join(export_dir, "attribution_field.csv"), index=False)
    compute_operator_stats(con, run_signature).to_csv(os.path.join(export_dir, "attribution_operator.csv"), index=False)
    compute_window_stats(con, run_signature).to_csv(os.path.join(export_dir, "attribution_window.csv"), index=False)
    compute_template_stats(con, run_signature).to_csv(os.path.join(export_dir, "attribution_template.csv"), index=False)
