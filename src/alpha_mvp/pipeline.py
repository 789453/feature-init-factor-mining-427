from __future__ import annotations
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

from .config import RunConfig
from .data import load_from_duckdb, make_simulated_data
from .fields import add_basic_features, DEFAULT_FEATURES
from .grammar import generate_templates, load_expression_range
from .evaluator import BatchEvaluator, make_panels
from .metrics import forward_returns, summarize_factor_split
from .parser import parse_expr, canonical
from .store import ResultStore

def _write_checkpoint(out: Path, all_results: pd.DataFrame, pct: float, topk: int):
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    if all_results.empty:
        return
    fn = ckpt_dir / f"top{topk}_pct_{pct:.0%}.csv"
    all_results.sort_values("score", ascending=False, na_position="last").head(topk).to_csv(
        fn, index=False, encoding="utf-8-sig"
    )

def run_pipeline(cfg: RunConfig) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if cfg.use_simulated:
        raw = make_simulated_data(seed=cfg.seed)
    else:
        if not cfg.duckdb_path:
            raise ValueError("duckdb_path is required unless use_simulated=True")
        raw = load_from_duckdb(cfg.duckdb_path, cfg.pool_json, cfg.start, cfg.end)

    try:
        raw.to_parquet(out / "raw_loaded_sample.parquet", index=False)
    except Exception:
        raw.head(1000).to_csv(out / "raw_loaded_sample.csv", index=False, encoding="utf-8-sig")

    df = add_basic_features(raw)
    feature_cols = [c for c in DEFAULT_FEATURES if c in df.columns]
    panels, dates, codes = make_panels(df, feature_cols, value_col="close")
    fwd = forward_returns(panels["close"], horizon=cfg.eval.forward_days)

    if cfg.expr_file:
        exprs = load_expression_range(cfg.expr_file, cfg.start_expr, cfg.end_expr)
        total = len(exprs)
    else:
        exprs = generate_templates(feature_cols, cfg.eval.windows, max_exprs=cfg.max_exprs)
        total = len(exprs)
    (out / "generated_expressions.txt").write_text("\n".join(exprs), encoding="utf-8")

    ev = BatchEvaluator(
        panels={k: v for k, v in panels.items() if k != "close"},
        dates=dates,
        codes=codes,
        windows=cfg.eval.windows,
        max_depth=cfg.eval.max_depth,
        max_nodes=cfg.eval.max_nodes,
    )

    run_id = f"{cfg.start}_{cfg.end}_{len(codes)}_{cfg.max_exprs}"
    sqlite_path = cfg.sqlite_path or str(out / "factor_results.sqlite3")
    store = ResultStore(sqlite_path)

    done = store.completed_keys(run_id) if cfg.resume else set()
    exprs_to_run = []
    for e in exprs:
        key = canonical(parse_expr(e))
        if key not in done:
            exprs_to_run.append(e)

    total = len(exprs)
    buffer = []
    last_progress = time.time()
    next_checkpoint_ratio = cfg.first_checkpoint_pct
    checkpoint_ratios_done = set()

    print(f"[start] total={total}, resuming={len(done)}, to_run={len(exprs_to_run)}")

    for idx, expr in enumerate(exprs_to_run, start=len(done) + 1):
        try:
            key = canonical(parse_expr(expr))
            arr, status = ev.eval_expr(expr)
            if arr is None:
                rec = {"expr": expr, "canonical": key, "status": status, "error": status}
            else:
                m = summarize_factor_split(
                    arr, fwd, dates,
                    train_end=cfg.train_end,
                    test_start=cfg.test_start,
                    min_daily_valid=cfg.eval.min_daily_valid_names,
                )
                rec = {"expr": expr, "canonical": key, "status": status, **m, "error": None}
        except Exception as e:
            rec = {"expr": expr, "canonical": expr, "status": "ERROR", "error": repr(e)}

        buffer.append(rec)

        if len(buffer) >= cfg.write_every:
            store.upsert_many(run_id, buffer)
            buffer.clear()

        ratio = idx / total
        now = time.time()

        if ratio >= next_checkpoint_ratio and next_checkpoint_ratio not in checkpoint_ratios_done:
            store.upsert_many(run_id, buffer)
            buffer.clear()
            all_df = store.load_all(run_id)
            _write_checkpoint(out, all_df, next_checkpoint_ratio, cfg.topk_checkpoint)
            checkpoint_ratios_done.add(next_checkpoint_ratio)
            next_checkpoint_ratio = max(
                cfg.checkpoint_pct,
                (int(ratio / cfg.checkpoint_pct) + 1) * cfg.checkpoint_pct,
            )

        if now - last_progress >= cfg.progress_min_interval_sec:
            print(f"[progress] {idx}/{total} ({ratio:.1%}), cache={len(ev._cache)}")
            last_progress = now

    store.upsert_many(run_id, buffer)
    res = store.load_all(run_id)
    res.to_csv(out / "factor_results.csv", index=False, encoding="utf-8-sig")
    res.head(100).to_csv(out / "top100.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_raw_rows": int(len(raw)),
        "n_dates": int(len(dates)),
        "n_codes": int(len(codes)),
        "n_features": int(len(feature_cols)),
        "features": feature_cols,
        "n_exprs": int(total),
        "n_results": int(len(res)),
        "n_resumed": int(len(done)),
        "expr_file": cfg.expr_file,
        "start_expr": cfg.start_expr,
        "end_expr": cfg.end_expr,
        "out_dir": str(out),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    store.close()
    return summary