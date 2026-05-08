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
from .template_config import load_template_config
from .expr_meta import extract_meta
from .job_store import DuckDBJobStore
from .scoring import summarize_factor_oriented, apply_ranked_score
from .evaluator import BatchEvaluator, make_panels
from .metrics import forward_returns
from .parser import parse_expr, canonical
from .attribution import run_all_attribution
from .attribution_extended import run_extended_attribution
from .ranking import interleave_by_group, stratified_ranking
from .signature import build_manifest_from_config, compute_run_signature
from .candidate_sampler_extended import select_for_fine_screen, export_candidate_expr_file

def compute_run_signature(manifest: dict) -> str:
    content = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

def _write_checkpoint(out: Path, df: pd.DataFrame, pct: float, topk: int):
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    if df.empty:
        return
    fn = ckpt_dir / f"top{topk}_pct_{pct:.0%}.csv"
    df.sort_values("score_ranked", ascending=False, na_position="last").head(topk).to_csv(
        fn, index=False, encoding="utf-8-sig"
    )
    print(f"[Checkpoint] Saved {topk} results at {pct:.0%} to {fn}")

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
        # 使用YAML模板配置或默认配置
        specs = TEMPLATE_SPECS
        budgets = COMPLEXITY_BUDGETS
        
        template_config_path = getattr(cfg, 'template_config_path', None)
        if template_config_path and Path(template_config_path).exists():
            try:
                loaded_specs, loaded_budgets, _ = load_template_config(template_config_path)
                specs = loaded_specs
                budgets = loaded_budgets
                print(f"[Phase2] Loaded template config from {template_config_path}")
            except Exception as e:
                print(f"[Phase2] Warning: Failed to load template config: {e}")
                print("[Phase2] Using default template specs")
        
        expr_records = generate_expressions_from_specs(
            fields=selected_fields,
            windows=cfg.eval.windows,
            specs=specs,
            budgets=budgets,
            max_exprs=cfg.max_exprs,
            seed=cfg.seed,
            template_config_path=template_config_path
        )

    # 7. 提取元数据并构建完整的manifest
    metas = [extract_meta(r.expr, r) for r in expr_records]
    
    # 使用新的签名系统
    template_config_path = getattr(cfg, 'template_config_path', None)
    manifest = build_manifest_from_config(cfg, template_config_path or "default", selected_fields)
    run_signature = compute_run_signature(manifest)
    manifest["run_signature"] = run_signature
    
    print(f"[Phase2] Run signature: {run_signature[:16]}")
    print(f"[Phase2] Pool: {manifest['pool_signature']['pool_json']} ({manifest['pool_signature']['n_codes']} codes)")
    print(f"[Phase2] Fields: {len(selected_fields)} fields")
    print(f"[Phase2] Expressions: {len(expr_records)} generated")
    
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
        max_depth=10,
        max_nodes=50,
    )

    # 检查是否已存在结果，实现增量过滤 (断点恢复)
    completed_hashes = store.completed_expr_hashes(run_signature)
    todo_all = [r for r in expr_records if r.expr_hash not in completed_hashes]
    total = len(todo_all)
    total_all = len(expr_records)
    
    if total == 0:
        print(f"[Phase2] All {total_all} expressions already completed for signature {run_signature[:8]}.")
    else:
        print(f"[Phase2] Starting: {total} new / {total_all} total expressions (resume={cfg.resume})")

    results_buffer = []
    last_progress = time.time()
    next_checkpoint_ratio = cfg.first_checkpoint_pct
    checkpoint_ratios_done = set()
    done_count = len(completed_hashes)
    
    # 分批处理以提高效率
    batch_size = cfg.batch_size
    for i in range(0, total, batch_size):
        batch_start_idx = i
        batch_end_idx = min(i + batch_size, total)
        batch_tasks = todo_all[batch_start_idx:batch_end_idx]
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
            
        # 进度报告和checkpoint
        done_count = len(completed_hashes) + batch_end_idx
        ratio = done_count / total_all
        now = time.time()
        
        if ratio >= next_checkpoint_ratio and next_checkpoint_ratio not in checkpoint_ratios_done:
            # 写入当前所有结果并保存checkpoint
            store.write_results(run_signature, results_buffer)
            results_buffer.clear()
            
            # 查询当前所有结果
            query = """
            SELECT r.*, c.template_family, c.template_name, c.nodes, c.complexity_tier
            FROM factor_results r
            JOIN expression_catalog c ON r.expr_hash = c.expr_hash
            WHERE r.run_signature = ?
            """
            current_df = store.con.execute(query, (run_signature,)).df()
            if not current_df.empty:
                current_df = apply_ranked_score(current_df)
                _write_checkpoint(out, current_df, next_checkpoint_ratio, cfg.topk_checkpoint)
            checkpoint_ratios_done.add(next_checkpoint_ratio)
            next_checkpoint_ratio = max(
                cfg.checkpoint_pct,
                (int(ratio / cfg.checkpoint_pct) + 1) * cfg.checkpoint_pct,
            )
            
        if now - last_progress > cfg.progress_min_interval_sec:
            eval_speed = batch_end_idx / (now - last_progress) if now > last_progress else 0
            print(f"[Phase2 Progress] {done_count}/{total_all} ({ratio:.1%}) | Speed: {eval_speed:.0f} batch/s | Buffer: {len(results_buffer)}")
            last_progress = now

    if results_buffer:
        store.write_results(run_signature, results_buffer)
        results_buffer.clear()
    
    # 9. 计算 Rank Score 并导出结果
    query = """
    SELECT r.*, c.template_family, c.template_name, c.nodes, c.complexity_tier
    FROM factor_results r
    JOIN expression_catalog c ON r.expr_hash = c.expr_hash
    WHERE r.run_signature = ?
    """
    all_res = store.con.execute(query, (run_signature,)).df()
    
    all_res = apply_ranked_score(all_res)
    
    # 增强的分层交替显示逻辑
    def interleave_top_results(df, top_n=100, strategy="interleave"):
        if df.empty: 
            return df
        if "template_family" not in df.columns: 
            return df.head(top_n)
        
        if strategy == "interleave":
            # 使用新的interleave_by_group函数
            return interleave_by_group(df, group_cols=["template_family"], top_n=top_n)
        elif strategy == "stratified":
            # 使用分层排序
            return stratified_ranking(
                df,
                score_col="score_ranked",
                group_cols=["template_family"],
                top_n=top_n,
                min_per_group=5,
                max_per_group=30
            )
        else:
            # 默认的简单排序
            return df.head(top_n)

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
    
    # 使用增强的分层显示
    ranking_strategy = getattr(cfg, 'ranking_strategy', 'interleave')
    top_interleaved = interleave_top_results(full_sorted, top_n=100, strategy=ranking_strategy)
    top_interleaved.to_csv(out / "top100_phase2.csv", index=False)

    # 10. 运行 Attribution (使用扩展版本)
    extended_attribution = getattr(cfg, 'extended_attribution', True)
    if extended_attribution:
        coarse_sig = getattr(cfg, 'coarse_signature', None)
        run_extended_attribution(store.con, run_signature, str(out), coarse_sig)
    else:
        run_all_attribution(store.con, run_signature, str(out))

    # 11. 导出细筛候选表达式文件
    if getattr(cfg, 'phase_type', None) == 'coarse':
        # 粗筛阶段结束后导出候选
        fine_candidates = select_for_fine_screen(
            all_res,
            top_k=getattr(cfg, 'candidate_top_k', 1000),
            sample_n=getattr(cfg, 'candidate_sample_n', 1000),
            min_per_template_family=getattr(cfg, 'candidate_min_per_family', 20),
            alpha=getattr(cfg, 'candidate_alpha', 0.85),
            seed=cfg.seed
        )
        if not fine_candidates.empty:
            candidate_file = export_dir / "fine_candidates.expr"
            export_candidate_expr_file(fine_candidates, str(candidate_file))
            print(f"[Phase2] Exported {len(fine_candidates)} candidates for fine screening")

    print(f"[Phase2] Finished. Results in {out}")
    store.close()

    return manifest
