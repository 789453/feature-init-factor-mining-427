from __future__ import annotations
from itertools import combinations
from pathlib import Path
from .parser import canonical, parse_expr
from .validator import Validator

CORE_TS = ["TsMean", "TsStd", "TsIr", "TsMinMaxDiff", "TsRank", "TsDelta", "TsDiv", "TsPctChange", "TsWMA", "TsEMA"]

def interleave_groups(groups: list[list[str]], max_exprs: int | None = None):
    out = []
    max_len = max(len(g) for g in groups)
    for i in range(max_len):
        for g in groups:
            if i < len(g):
                out.append(g[i])
                if max_exprs and len(out) >= max_exprs:
                    return out
    return out

def generate_all_templates(fields: list[str], windows=(10,20,30,40,50),
                           allow_heavy_ops=False) -> tuple[list[str], dict]:
    validator = Validator(set(fields), set(windows))
    seen = set()
    stats = {
        "single_field": 0,
        "binary": 0,
        "pair_corr": 0,
        "directional": 0,
        "total": 0,
    }

    def add_expr(e: str) -> bool:
        try:
            node = parse_expr(e)
            key = canonical(node)
            if key in seen:
                return False
            vr = validator.validate(node)
            if not vr.ok:
                return False
            seen.add(key)
            return True
        except Exception:
            return False

    single_field_group = []
    for f in fields:
        for w in windows:
            for ts in CORE_TS:
                e = f"Rank({ts}(${f},{w}))"
                if add_expr(e):
                    single_field_group.append(e)
                    stats["single_field"] += 1
                if ts in {"TsDelta", "TsPctChange", "TsIr", "TsRank"}:
                    e = f"SLog1p({ts}(${f},{w}))"
                    if add_expr(e):
                        single_field_group.append(e)
                        stats["single_field"] += 1

    binary_group = []
    for f1, f2 in combinations(fields, 2):
        for w in windows:
            e = f"Sub(Rank(TsMean(${f1},{w})),Rank(TsMean(${f2},{w})))"
            if add_expr(e):
                binary_group.append(e)
                stats["binary"] += 1
            e = f"Sub(Rank(TsDelta(${f1},{w})),Rank(TsDelta(${f2},{w})))"
            if add_expr(e):
                binary_group.append(e)
                stats["binary"] += 1
            e = f"Div(Rank(TsMean(${f1},{w})),Rank(TsMean(${f2},{w})))"
            if add_expr(e):
                binary_group.append(e)
                stats["binary"] += 1

    pair_corr_group = []
    pair_fields = fields[: min(len(fields), 18)]
    for f1, f2 in combinations(pair_fields, 2):
        for w in windows:
            e = f"Rank(TsCorr(${f1},${f2},{w}))"
            if add_expr(e):
                pair_corr_group.append(e)
                stats["pair_corr"] += 1
            if allow_heavy_ops:
                e = f"Rank(TsCov(${f1},${f2},{w}))"
                if add_expr(e):
                    pair_corr_group.append(e)
                    stats["pair_corr"] += 1

    directional_group = []
    named = set(fields)
    directional_pairs = [
        ("big_vs_small_flow", "ret_1d"),
        ("main_net_ratio", "turnover_log"),
        ("flow_imbalance", "ret_1d"),
        ("lg_net_ratio", "sm_net_ratio"),
        ("chip_cost_bias", "winner_rate_norm"),
        ("chip_width_90", "turnover_log"),
        ("hist_price_position", "ret_1d"),
        ("active_big_buy_pressure", "ret_1d"),
    ]
    for a, b in directional_pairs:
        if a in named and b in named:
            for w in windows:
                e = f"Rank(TsCorr(${a},${b},{w}))"
                if add_expr(e):
                    directional_group.append(e)
                    stats["directional"] += 1
                e = f"Sub(Rank(TsMean(${a},{w})),Rank(TsMean(${b},{w})))"
                if add_expr(e):
                    directional_group.append(e)
                    stats["directional"] += 1
                e = f"Rank(TsDelta(${a},{w}))"
                if add_expr(e):
                    directional_group.append(e)
                    stats["directional"] += 1

    stats["total"] = stats["single_field"] + stats["binary"] + stats["pair_corr"] + stats["directional"]

    all_exprs = interleave_groups(
        [single_field_group, binary_group, pair_corr_group, directional_group]
    )
    return all_exprs, stats

def generate_templates(fields: list[str], windows=(10,20,30,40,50), max_exprs=5000,
                       allow_heavy_ops=False) -> list[str]:
    all_exprs, _ = generate_all_templates(fields, windows, allow_heavy_ops)
    return all_exprs[:max_exprs]

def save_all_expressions(fields: list[str], windows=(10,20,30,40,50),
                         allow_heavy_ops=False, out_dir: str = "outputs/expressions") -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_exprs, stats = generate_all_templates(fields, windows, allow_heavy_ops)

    expr_file = out_path / "all_expressions.txt"
    with open(expr_file, "w", encoding="utf-8") as f:
        for i, e in enumerate(all_exprs, 1):
            f.write(f"{i}\t{e}\n")

    meta_file = out_path / "expression_stats.json"
    import json
    meta = {
        "total": stats["total"],
        "single_field": stats["single_field"],
        "binary": stats["binary"],
        "pair_corr": stats["pair_corr"],
        "directional": stats["directional"],
        "windows": list(windows),
        "fields": fields,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "total": stats["total"],
        "expr_file": str(expr_file),
        "meta_file": str(meta_file),
    }

def load_expression_range(expr_file: str, start: int = 1, end: int | None = None) -> list[str]:
    exprs = []
    file_path = Path(expr_file)
    
    if file_path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(expr_file)
        # 尝试寻找 expr 列
        col = "expr" if "expr" in df.columns else (df.columns[0] if len(df.columns) > 0 else None)
        if col:
            raw_list = df[col].tolist()
            # start 是 1-based index
            s_idx = start - 1
            e_idx = end if end is not None else len(raw_list)
            exprs = raw_list[s_idx:e_idx]
    else:
        with open(expr_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    try:
                        idx = int(parts[0])
                        if start <= idx <= (end if end is not None else float("inf")):
                            exprs.append(parts[1])
                    except ValueError:
                        # 如果不是 tab 分隔的带索引格式，则尝试作为纯文本行处理
                        exprs.append(line)
                else:
                    exprs.append(line)
    return exprs