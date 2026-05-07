from __future__ import annotations
import random
import hashlib
from dataclasses import dataclass
from typing import Sequence
from .template_spec import TemplateSpec, ComplexityBudget, TEMPLATE_SPECS, COMPLEXITY_BUDGETS
from .parser import parse_expr, canonical
from .validator import Validator

@dataclass
class ExpressionRecord:
    expr: str
    canonical: str
    expr_hash: str
    template_name: str
    template_family: str
    template_order: int
    complexity_tier: int

def get_expr_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()

def generate_expressions_from_specs(
    fields: list[str],
    windows: list[int],
    specs: list[TemplateSpec] = TEMPLATE_SPECS,
    budgets: dict[int, ComplexityBudget] = COMPLEXITY_BUDGETS,
    max_exprs: int | None = None,
    seed: int = 42,
) -> list[ExpressionRecord]:
    random.seed(seed)
    all_records = []
    seen_canonical = set()
    
    # 获取所有窗口的集合，用于验证
    all_windows = set(windows)
    for spec in specs:
        if spec.short_windows:
            all_windows.update(spec.short_windows)
        if spec.long_windows:
            all_windows.update(spec.long_windows)

    # 遍历每个模板规格
    for spec in specs:
        if not spec.enabled:
            continue
            
        spec_records = []
        budget = budgets.get(spec.complexity_tier)
        validator = Validator(
            fields=set(fields), 
            windows=all_windows,
            max_depth=budget.max_depth,
            max_nodes=budget.max_nodes,
            max_ts_ops=budget.max_ts_ops,
            max_pair_ops=budget.max_pair_ops,
            max_binary_ops=budget.max_binary_ops
        )

        if spec.family == "single":
            for f in fields:
                for w in windows:
                    for ts_op in spec.ts_ops:
                        for unary in spec.unary_pre:
                            # 构造基础一阶变换
                            f_expr = f"${f}"
                            if unary != "Id":
                                f_expr = f"{unary}({f_expr})"
                            
                            ts_expr = f"{ts_op}({f_expr},{w})"
                            
                            for outer in spec.outer_transforms:
                                if outer == "Rank":
                                    expr = f"Rank({ts_expr})"
                                elif outer == "SLog1p":
                                    expr = f"SLog1p({ts_expr})"
                                elif outer == "RankSLog1p":
                                    expr = f"Rank(SLog1p({ts_expr}))"
                                else:
                                    expr = ts_expr
                                
                                _add_if_valid(expr, spec, spec_records, seen_canonical, validator)

        elif spec.family == "binary":
            # 简化：只对相同窗口的二元组合
            for i, f1 in enumerate(fields):
                for f2 in fields[i+1:]:
                    for w in windows:
                        for ts_op in spec.ts_ops:
                            for b_op in spec.binary_ops:
                                for outer in spec.outer_transforms:
                                    # 结构：Outer(B_Op(TsOp(f1,w), TsOp(f2,w)))
                                    ts1 = f"{ts_op}(${f1},{w})"
                                    ts2 = f"{ts_op}(${f2},{w})"
                                    b_expr = f"{b_op}({ts1},{ts2})"
                                    
                                    if outer == "Rank":
                                        expr = f"Rank({b_expr})"
                                    elif outer == "SLog1p":
                                        expr = f"SLog1p({b_expr})"
                                    elif outer == "RankSLog1p":
                                        expr = f"Rank(SLog1p({b_expr}))"
                                    else:
                                        expr = b_expr
                                    
                                    _add_if_valid(expr, spec, spec_records, seen_canonical, validator)

        elif spec.family == "multi_window":
            # 结构：Rank(B_Op(TsOp(f, sw), TsOp(f, lw)))
            sws = spec.short_windows if spec.short_windows else [5, 10, 20]
            lws = spec.long_windows if spec.long_windows else [40, 60]
            
            for f in fields:
                for sw in sws:
                    for lw in lws:
                        if sw >= lw: continue
                        for ts_op in spec.ts_ops:
                            for b_op in spec.binary_ops:
                                ts_sw = f"{ts_op}(${f},{sw})"
                                ts_lw = f"{ts_op}(${f},{lw})"
                                expr = f"Rank({b_op}({ts_sw},{ts_lw}))"
                                _add_if_valid(expr, spec, spec_records, seen_canonical, validator)

        # 其他高阶模板可以按需实现，或者使用随机采样逻辑
        # ... 这里先实现核心的几个

        # 打乱并加入总列表
        random.shuffle(spec_records)
        if spec.max_count:
            spec_records = spec_records[:spec.max_count]
        all_records.extend(spec_records)

    # 全局打乱
    random.shuffle(all_records)
    if max_exprs:
        all_records = all_records[:max_exprs]
        
    return all_records

def _add_if_valid(expr: str, spec: TemplateSpec, records: list[ExpressionRecord], seen: set[str], validator: Validator):
    try:
        node = parse_expr(expr)
        v = validator.validate(node)
        if not v.ok:
            return
        
        can = canonical(node)
        if can in seen:
            return
        
        seen.add(can)
        records.append(ExpressionRecord(
            expr=expr,
            canonical=can,
            expr_hash=get_expr_hash(can),
            template_name=spec.name,
            template_family=spec.family,
            template_order=spec.order,
            complexity_tier=spec.complexity_tier
        ))
    except Exception:
        pass
