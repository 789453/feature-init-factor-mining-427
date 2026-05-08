"""
扩展的归因分析模块
实现组合有效性统计和更详细的归因分析
"""
from __future__ import annotations
import pandas as pd
import duckdb
from pathlib import Path
from typing import Dict, List, Any

def compute_field_operator_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    """字段×算子组合统计"""
    query = f"""
    SELECT 
        f.field,
        o.operator,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) as n_ok,
        CASE 
            WHEN COUNT(DISTINCT r.expr_hash) > 0 
            THEN COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) * 100.0 / COUNT(DISTINCT r.expr_hash)
            ELSE 0 
        END as valid_rate,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.score_ranked) as p75_score_ranked,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY r.score_ranked) as p90_score_ranked,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.99) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top1pct_rate,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.95) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top5pct_rate,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        MEDIAN(r.oriented_test_mean_rank_ic) as median_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover,
        AVG(r.complexity_score) as mean_complexity
    FROM expression_field_link f
    JOIN expression_operator_link o ON f.expr_hash = o.expr_hash
    JOIN factor_results r ON f.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY f.field, o.operator
    HAVING COUNT(DISTINCT r.expr_hash) >= 20
    ORDER BY median_score_ranked DESC
    """
    return con.execute(query).df()

def compute_field_window_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    """字段×窗口组合统计"""
    query = f"""
    SELECT 
        f.field,
        w.window,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) as n_ok,
        CASE 
            WHEN COUNT(DISTINCT r.expr_hash) > 0 
            THEN COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) * 100.0 / COUNT(DISTINCT r.expr_hash)
            ELSE 0 
        END as valid_rate,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.score_ranked) as p75_score_ranked,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY r.score_ranked) as p90_score_ranked,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.99) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top1pct_rate,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.95) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top5pct_rate,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        MEDIAN(r.oriented_test_mean_rank_ic) as median_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover,
        AVG(r.complexity_score) as mean_complexity
    FROM expression_field_link f
    JOIN expression_window_link w ON f.expr_hash = w.expr_hash
    JOIN factor_results r ON f.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY f.field, w.window
    HAVING COUNT(DISTINCT r.expr_hash) >= 20
    ORDER BY f.field, w.window
    """
    return con.execute(query).df()

def compute_operator_window_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    """算子×窗口组合统计"""
    query = f"""
    SELECT 
        o.operator,
        w.window,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) as n_ok,
        CASE 
            WHEN COUNT(DISTINCT r.expr_hash) > 0 
            THEN COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) * 100.0 / COUNT(DISTINCT r.expr_hash)
            ELSE 0 
        END as valid_rate,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.score_ranked) as p75_score_ranked,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY r.score_ranked) as p90_score_ranked,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.99) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top1pct_rate,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.95) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top5pct_rate,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        MEDIAN(r.oriented_test_mean_rank_ic) as median_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover,
        AVG(r.complexity_score) as mean_complexity
    FROM expression_operator_link o
    JOIN expression_window_link w ON o.expr_hash = w.expr_hash
    JOIN factor_results r ON o.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY o.operator, w.window
    HAVING COUNT(DISTINCT r.expr_hash) >= 20
    ORDER BY o.operator, w.window
    """
    return con.execute(query).df()

def compute_template_operator_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    """模板×算子组合统计"""
    query = f"""
    SELECT 
        c.template_family,
        c.template_name,
        o.operator,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) as n_ok,
        CASE 
            WHEN COUNT(DISTINCT r.expr_hash) > 0 
            THEN COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) * 100.0 / COUNT(DISTINCT r.expr_hash)
            ELSE 0 
        END as valid_rate,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.score_ranked) as p75_score_ranked,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY r.score_ranked) as p90_score_ranked,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.99) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top1pct_rate,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.95) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top5pct_rate,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        MEDIAN(r.oriented_test_mean_rank_ic) as median_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover,
        AVG(r.complexity_score) as mean_complexity
    FROM expression_catalog c
    JOIN expression_operator_link o ON c.expr_hash = o.expr_hash
    JOIN factor_results r ON c.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY c.template_family, c.template_name, o.operator
    HAVING COUNT(DISTINCT r.expr_hash) >= 20
    ORDER BY c.template_family, median_score_ranked DESC
    """
    return con.execute(query).df()

def compute_template_window_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    """模板×窗口组合统计"""
    query = f"""
    SELECT 
        c.template_family,
        c.template_name,
        w.window,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) as n_ok,
        CASE 
            WHEN COUNT(DISTINCT r.expr_hash) > 0 
            THEN COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) * 100.0 / COUNT(DISTINCT r.expr_hash)
            ELSE 0 
        END as valid_rate,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.score_ranked) as p75_score_ranked,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY r.score_ranked) as p90_score_ranked,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.99) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top1pct_rate,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.95) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top5pct_rate,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        MEDIAN(r.oriented_test_mean_rank_ic) as median_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover,
        AVG(r.complexity_score) as mean_complexity
    FROM expression_catalog c
    JOIN expression_window_link w ON c.expr_hash = w.expr_hash
    JOIN factor_results r ON c.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY c.template_family, c.template_name, w.window
    HAVING COUNT(DISTINCT r.expr_hash) >= 20
    ORDER BY c.template_family, w.window
    """
    return con.execute(query).df()

def compute_template_combo_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    """模板组合统计"""
    query = f"""
    SELECT 
        c.template_family,
        c.template_name,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) as n_ok,
        CASE 
            WHEN COUNT(DISTINCT r.expr_hash) > 0 
            THEN COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) * 100.0 / COUNT(DISTINCT r.expr_hash)
            ELSE 0 
        END as valid_rate,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.score_ranked) as p75_score_ranked,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY r.score_ranked) as p90_score_ranked,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.99) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top1pct_rate,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.95) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top5pct_rate,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        MEDIAN(r.oriented_test_mean_rank_ic) as median_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover,
        AVG(r.complexity_score) as mean_complexity
    FROM expression_catalog c
    JOIN factor_results r ON c.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY c.template_family, c.template_name
    HAVING COUNT(DISTINCT r.expr_hash) >= 20
    ORDER BY median_score_ranked DESC
    """
    return con.execute(query).df()

def compute_order_complexity_stats(con: duckdb.DuckDBPyConnection, run_signature: str) -> pd.DataFrame:
    """阶数复杂度统计"""
    query = f"""
    SELECT 
        c.template_order,
        c.complexity_tier,
        COUNT(DISTINCT r.expr_hash) as n_expr,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) as n_ok,
        CASE 
            WHEN COUNT(DISTINCT r.expr_hash) > 0 
            THEN COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked IS NOT NULL) * 100.0 / COUNT(DISTINCT r.expr_hash)
            ELSE 0 
        END as valid_rate,
        AVG(r.score_ranked) as mean_score_ranked,
        MEDIAN(r.score_ranked) as median_score_ranked,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY r.score_ranked) as p75_score_ranked,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY r.score_ranked) as p90_score_ranked,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.99) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top1pct_rate,
        COUNT(DISTINCT r.expr_hash) FILTER (WHERE r.score_ranked >= (SELECT PERCENTILE_CONT(0.95) FROM factor_results WHERE run_signature = '{run_signature}' AND score_ranked IS NOT NULL)) * 100.0 / COUNT(DISTINCT r.expr_hash) as top5pct_rate,
        AVG(r.oriented_test_mean_rank_ic) as mean_oriented_test_rank_ic,
        MEDIAN(r.oriented_test_mean_rank_ic) as median_oriented_test_rank_ic,
        AVG(r.oriented_test_rank_icir) as mean_oriented_test_rank_icir,
        AVG(r.coverage) as mean_coverage,
        AVG(r.turnover_proxy) as mean_turnover,
        AVG(r.complexity_score) as mean_complexity
    FROM expression_catalog c
    JOIN factor_results r ON c.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}'
    GROUP BY c.template_order, c.complexity_tier
    ORDER BY c.template_order, c.complexity_tier
    """
    return con.execute(query).df()

def compute_coarse_to_fine_decay(con: duckdb.DuckDBPyConnection, coarse_signature: str, fine_signature: str) -> pd.DataFrame:
    """粗筛到细筛的衰减统计"""
    query = f"""
    WITH coarse_scores AS (
        SELECT expr_hash, score_ranked as coarse_score
        FROM factor_results
        WHERE run_signature = '{coarse_signature}'
    ),
    fine_scores AS (
        SELECT expr_hash, score_ranked as fine_score
        FROM factor_results
        WHERE run_signature = '{fine_signature}'
    )
    SELECT 
        c.expr_hash,
        c.coarse_score,
        f.fine_score,
        f.fine_score - c.coarse_score as score_decay,
        CASE 
            WHEN c.coarse_score > 0 THEN (f.fine_score - c.coarse_score) / c.coarse_score * 100.0
            ELSE NULL 
        END as decay_ratio_percent
    FROM coarse_scores c
    JOIN fine_scores f ON c.expr_hash = f.expr_hash
    WHERE c.coarse_score IS NOT NULL AND f.fine_score IS NOT NULL
    ORDER BY score_decay DESC
    """
    return con.execute(query).df()

def run_extended_attribution(con: duckdb.DuckDBPyConnection, run_signature: str, out_dir: str, 
                             coarse_signature: str = None, fine_signature: str = None):
    """运行扩展的归因分析"""
    export_dir = Path(out_dir) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础归因分析（保持向后兼容）
    compute_field_stats(con, run_signature).to_csv(export_dir / "attribution_field.csv", index=False)
    compute_operator_stats(con, run_signature).to_csv(export_dir / "attribution_operator.csv", index=False)
    compute_window_stats(con, run_signature).to_csv(export_dir / "attribution_window.csv", index=False)
    compute_template_stats(con, run_signature).to_csv(export_dir / "attribution_template.csv", index=False)
    
    # 扩展归因分析
    compute_field_operator_stats(con, run_signature).to_csv(export_dir / "attribution_field_operator.csv", index=False)
    compute_field_window_stats(con, run_signature).to_csv(export_dir / "attribution_field_window.csv", index=False)
    compute_operator_window_stats(con, run_signature).to_csv(export_dir / "attribution_operator_window.csv", index=False)
    compute_template_operator_stats(con, run_signature).to_csv(export_dir / "attribution_template_operator.csv", index=False)
    compute_template_window_stats(con, run_signature).to_csv(export_dir / "attribution_template_window.csv", index=False)
    compute_template_combo_stats(con, run_signature).to_csv(export_dir / "attribution_template_combo.csv", index=False)
    compute_order_complexity_stats(con, run_signature).to_csv(export_dir / "attribution_order_complexity.csv", index=False)
    
    # 粗筛到细筛的衰减分析
    if coarse_signature and fine_signature:
        decay_stats = compute_coarse_to_fine_decay(con, coarse_signature, fine_signature)
        decay_stats.to_csv(export_dir / "attribution_coarse_to_fine_decay.csv", index=False)
        
        # 生成衰减统计摘要
        decay_summary = {
            "total_expressions": len(decay_stats),
            "mean_decay": decay_stats["score_decay"].mean(),
            "median_decay": decay_stats["score_decay"].median(),
            "std_decay": decay_stats["score_decay"].std(),
            "positive_decay_ratio": (decay_stats["score_decay"] > 0).mean() * 100,
            "large_decay_count": (decay_stats["score_decay"] < -0.1).sum(),
            "improved_count": (decay_stats["score_decay"] > 0.1).sum()
        }
        
        # 保存衰减摘要
        import json
        with open(export_dir / "decay_summary.json", "w") as f:
            json.dump(decay_summary, f, indent=2, default=str)

def generate_attribution_summary(con: duckdb.DuckDBPyConnection, run_signature: str) -> Dict[str, Any]:
    """生成归因分析摘要"""
    # 基础统计
    basic_stats = con.execute(f"""
    SELECT 
        COUNT(DISTINCT expr_hash) as total_expressions,
        COUNT(DISTINCT expr_hash) FILTER (WHERE score_ranked IS NOT NULL) as valid_expressions,
        AVG(score_ranked) as mean_score,
        MEDIAN(score_ranked) as median_score,
        PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY score_ranked) as p90_score,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY score_ranked) as p99_score
    FROM factor_results
    WHERE run_signature = '{run_signature}'
    """).fetchone()
    
    # 模板族统计
    template_stats = con.execute(f"""
    SELECT 
        template_family,
        COUNT(DISTINCT expr_hash) as count,
        AVG(score_ranked) as mean_score
    FROM expression_catalog c
    JOIN factor_results r ON c.expr_hash = r.expr_hash
    WHERE r.run_signature = '{run_signature}' AND r.score_ranked IS NOT NULL
    GROUP BY template_family
    ORDER BY mean_score DESC
    """).df().to_dict('records')
    
    return {
        "basic_stats": {
            "total_expressions": basic_stats[0],
            "valid_expressions": basic_stats[1],
            "valid_rate": basic_stats[1] / basic_stats[0] * 100 if basic_stats[0] > 0 else 0,
            "mean_score": basic_stats[2],
            "median_score": basic_stats[3],
            "p90_score": basic_stats[4],
            "p99_score": basic_stats[5]
        },
        "template_performance": template_stats
    }