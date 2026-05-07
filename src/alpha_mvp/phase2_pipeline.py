from __future__ import annotations
import json
import time
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

from .config import RunConfig
from .data import load_from_duckdb, make_simulated_data
from .fields import add_basic_features
from .field_registry import resolve_fields, get_field_set_hash
from .template_spec import TEMPLATE_SPECS, COMPLEXITY_BUDGETS
from .template_builder import generate_expressions_from_specs, ExpressionRecord
from .expr_meta import extract_meta
from .job_store import DuckDBJobStore
from .scoring import summarize_factor_oriented, apply_ranked_score
from .evaluator import BatchEvaluator, make_panels
from .metrics import forward_returns
from .parser import parse_expr, canonical
from .attribution import run_all_attribution

def compute_run_signature(manifest: dict) -> str:
    content = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

def run_phase2(cfg: RunConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 加载数据
    if cfg.use_simulated:
        raw = make_simulated_data(seed=cfg.seed)
    else:
        if not cfg.duckdb_path:
            raise ValueError("duckdb_path is required unless use_simulated=True")
        raw = load_from_duckdb(cfg.duckdb_path, cfg.pool_json, cfg.start, cfg.end)

    # 2. 准备基础特征
    df = add_basic_features(raw)
    
    # 3. 解析字段
    available_cols = list(df.columns)
    selected_fields = resolve_fields(available_cols, cfg.fields, cfg.exclude_fields, cfg.field_file)
    
    # 4. 准备面板数据
    panels, dates, codes = make_panels(df, selected_fields, value_col="close")
    fwd = forward_returns(panels["close"], horizon=cfg.eval.forward_days)
    
    # 5. 计算训练集/测试集 mask
    dates_arr = np.array(dates)
    train_mask = dates_arr <= cfg.train_end
    test_mask = dates_arr >= cfg.test_start

    # 6. 生成表达式
    if cfg.expr_file:
        from .grammar import load_expression_range
        expr_strings = load_expression_range(cfg.expr_file, cfg.start_expr, cfg.end_expr)
        expr_records = []
        for e in expr_strings:
            node = parse_expr(e)
            can = canonical(node)
            expr_records.append(ExpressionRecord(
                expr=e, canonical=can, expr_hash=hashlib.sha256(can.encode()).hexdigest(),
                template_name="manual", template_family="manual", template_order=0, complexity_tier=1
            ))
    else:
        expr_records = generate_expressions_from_specs(
            fields=selected_fields,
            windows=cfg.eval.windows,
            specs=TEMPLATE_SPECS,
            max_exprs=cfg.max_exprs,
            seed=cfg.seed
        )

    # 7. 提取元数据并初始化 JobStore
    metas = [extract_meta(r.expr, r) for r in expr_records]
    
    manifest = {
        "job_id": f"phase2_{time.strftime('%Y%m%d_%H%M%S')}",
        "start": cfg.start,
        "end": cfg.end,
        "field_set_hash": get_field_set_hash(selected_fields),
        "grammar_hash": hashlib.sha256(str(TEMPLATE_SPECS).encode()).hexdigest(), # 简化版
        "eval_hash": hashlib.sha256(str(cfg.eval).encode()).hexdigest(),
        "selected_fields": selected_fields,
        "windows": list(cfg.eval.windows)
    }
    run_signature = compute_run_signature(manifest)
    manifest["run_signature"] = run_signature
    
    db_path = cfg.sqlite_path or str(out / "phase2_results.duckdb")
    store = DuckDBJobStore(db_path)
    
    if cfg.force_rerun:
        print(f"[Phase2] Force rerun enabled. Clearing previous results for signature {run_signature[:8]}")
        store.con.execute("DELETE FROM expression_jobs WHERE run_signature = ?", (run_signature,))
        store.con.execute("DELETE FROM factor_results WHERE run_signature = ?", (run_signature,))

    store.init_run(manifest, run_signature)
    store.upsert_expressions(expr_records, metas)
    store.enqueue_jobs(run_signature, expr_records)

    # 8. 运行评估
    # 在 Phase 2 中，生成器已经根据 ComplexityTier 进行了校验，评估器可以放宽限制
    ev = BatchEvaluator(
        panels={k: v for k, v in panels.items() if k != "close"},
        dates=dates,
        codes=codes,
        windows=cfg.eval.windows,
        max_depth=10,  # 允许更深，因为生成器已校验
        max_nodes=50,
    )

    # 检查是否已存在结果，实现增量过滤
    completed_hashes = store.completed_expr_hashes(run_signature)
    todo_all = [r for r in expr_records if r.expr_hash not in completed_hashes]
    total = len(todo_all)
    
    if total == 0:
        print(f"[Phase2] All {len(expr_records)} expressions already completed for signature {run_signature[:8]}.")
    else:
        print(f"[Phase2] Starting evaluation: {total} new expressions (Total {len(expr_records)}), run_signature={run_signature[:8]}")

    results_buffer = []
    last_progress = time.time()
    
    # 分批处理以提高效率
    batch_size = cfg.batch_size
    for i in range(0, total, batch_size):
        batch_tasks = todo_all[i:i+batch_size]
        batch_exprs = [t.expr for t in batch_tasks]
        
        # 并行计算这一批表达式
        batch_results = ev.eval_batch(batch_exprs, max_workers=8)
        
        for task, (arr, status) in zip(batch_tasks, batch_results):
            expr_hash = task.expr_hash
            can = task.canonical
            expr = task.expr
            
            try:
                if arr is None:
                    res = {"run_signature": run_signature, "expr_hash": expr_hash, "canonical": can, "expr": expr, "status": status, "error": status}
                else:
                    m = summarize_factor_oriented(arr, fwd, train_mask, test_mask, dates=dates)
                    from .metrics import turnover_proxy
                    cov = float(np.nanmean(np.isfinite(arr)))
                    to = turnover_proxy(arr)
                    
                    # 获取元数据
                    meta = next((m for m in metas if m.expr_hash == expr_hash), None)
                    nodes = meta.nodes if meta else 10
                    complexity_score = 1.0 - (nodes / 50.0)
                    
                    res = {
                        "run_signature": run_signature, "expr_hash": expr_hash, "canonical": can, "expr": expr, "status": "OK",
                        "coverage": cov, "usable_days": int(np.sum(np.any(~np.isnan(arr), axis=1))),
                        "turnover_proxy": to, "complexity_score": complexity_score, **m
                    }
            except Exception as e:
                res = {"run_signature": run_signature, "expr_hash": expr_hash, "canonical": can, "expr": expr, "status": "ERROR", "error": str(e)}
            
            results_buffer.append(res)

        if len(results_buffer) >= cfg.write_every:
            store.write_results(run_signature, results_buffer)
            results_buffer.clear()
            
        if time.time() - last_progress > cfg.progress_min_interval_sec:
            print(f"[Phase2 Progress] {min(i+batch_size, total)}/{total} ({min(i+batch_size, total)/total:.1%})")
            last_progress = time.time()

    if results_buffer:
        store.write_results(run_signature, results_buffer)
    
    # 9. 计算 Rank Score 并导出结果
    query = """
    SELECT r.*, c.template_family, c.template_name, c.nodes, c.complexity_tier
    FROM factor_results r
    JOIN expression_catalog c ON r.expr_hash = c.expr_hash
    WHERE r.run_signature = ?
    """
    all_res = store.con.execute(query, (run_signature,)).df()
    
    all_res = apply_ranked_score(all_res)
    
    # 增加分层交替显示逻辑 (Interleaved Display)
    def interleave_top_results(df, top_n=100):
        if df.empty: return df
        if "template_family" not in df.columns: return df.head(top_n)
        
        families = df["template_family"].unique()
        groups = [df[df["template_family"] == f] for f in families]
        
        interleaved = []
        for i in range(top_n):
            for g in groups:
                if i < len(g):
                    interleaved.append(g.iloc[i])
                if len(interleaved) >= top_n:
                    break
            if len(interleaved) >= top_n:
                break
        return pd.DataFrame(interleaved)

    # 写回数据库
    for _, row in all_res.iterrows():
        store.con.execute("""
            UPDATE factor_results 
            SET score_ranked = ?, score_raw = ?, yearly_positive_ratio = ?, complexity_score = ?
            WHERE run_signature = ? AND expr_hash = ?
        """, (row["score_ranked"], row["score_raw"], row.get("yearly_positive_ratio", 0), row.get("complexity_score", 0), run_signature, row["expr_hash"]))
    
    # 导出 CSV
    export_dir = out / "exports"
    export_dir.mkdir(exist_ok=True)
    
    full_sorted = all_res.sort_values("score_ranked", ascending=False)
    full_sorted.to_csv(out / "factor_results_phase2.csv", index=False)
    
    top_interleaved = interleave_top_results(full_sorted, top_n=100)
    top_interleaved.to_csv(out / "top100_phase2.csv", index=False)
    
    # 10. 运行 Attribution
    run_all_attribution(store.con, run_signature, str(out))
    
    print(f"[Phase2] Finished. Results in {out}")
    store.close()
    
    return manifest
