from __future__ import annotations
from itertools import combinations
from .parser import canonical, parse_expr
from .validator import Validator

CORE_TS = ["TsMean", "TsStd", "TsIr", "TsMinMaxDiff", "TsRank", "TsDelta", "TsDiv", "TsPctChange", "TsWMA", "TsEMA"]

def generate_templates(fields: list[str], windows=(10,20,30,40,50), max_exprs=5000,
                       allow_heavy_ops=False) -> list[str]:
    exprs = []
    for f in fields:
        for w in windows:
            for ts in CORE_TS:
                exprs.append(f"Rank({ts}(${f},{w}))")
                if ts in {"TsDelta", "TsPctChange", "TsIr", "TsRank"}:
                    exprs.append(f"SLog1p({ts}(${f},{w}))")

    for f1, f2 in combinations(fields, 2):
        for w in windows:
            exprs.append(f"Sub(Rank(TsMean(${f1},{w})),Rank(TsMean(${f2},{w})))")
            exprs.append(f"Sub(Rank(TsDelta(${f1},{w})),Rank(TsDelta(${f2},{w})))")
            exprs.append(f"Div(Rank(TsMean(${f1},{w})),Rank(TsMean(${f2},{w})))")

    pair_fields = fields[: min(len(fields), 18)]
    for f1, f2 in combinations(pair_fields, 2):
        for w in windows:
            exprs.append(f"Rank(TsCorr(${f1},${f2},{w}))")
            if allow_heavy_ops:
                exprs.append(f"Rank(TsCov(${f1},${f2},{w}))")

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
                exprs.append(f"Rank(TsCorr(${a},${b},{w}))")
                exprs.append(f"Sub(Rank(TsMean(${a},{w})),Rank(TsMean(${b},{w})))")
                exprs.append(f"Rank(TsDelta(${a},{w}))")

    validator = Validator(set(fields), set(windows))
    out, seen = [], set()
    for e in exprs:
        try:
            node = parse_expr(e)
            key = canonical(node)
            if key in seen:
                continue
            vr = validator.validate(node)
            if not vr.ok:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= max_exprs:
                break
        except Exception:
            continue
    return out
