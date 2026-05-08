from __future__ import annotations
import random
import hashlib
from dataclasses import dataclass
from typing import Sequence
from .template_spec import TemplateSpec, ComplexityBudget, TEMPLATE_SPECS, COMPLEXITY_BUDGETS
from .template_config import load_template_config
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

def stable_keep(key: str, rate: float, seed: int) -> bool:
    """确定性采样函数，确保同一套字段、模板、seed下生成结果稳定"""
    h = hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()
    v = int(h[:12], 16) / float(16**12)
    return v < rate

def generate_expressions_from_specs(
    fields: list[str],
    windows: list[int],
    specs: list[TemplateSpec] = TEMPLATE_SPECS,
    budgets: dict[int, ComplexityBudget] = COMPLEXITY_BUDGETS,
    max_exprs: int | None = None,
    seed: int = 42,
    template_config_path: str = None,
) -> list[ExpressionRecord]:
    random.seed(seed)
    all_records = []
    seen_canonical = set()
    
    # 如果提供了模板配置文件路径，优先使用配置文件
    if template_config_path:
        try:
            loaded_specs, loaded_budgets, _ = load_template_config(template_config_path)
            specs = loaded_specs
            budgets = loaded_budgets
        except Exception as e:
            print(f"Warning: Failed to load template config from {template_config_path}: {e}")
            print("Using default template specs")
    
    # 获取所有窗口的集合，用于验证
    all_windows = set(windows)
    for spec in specs:
        if spec.short_windows:
            all_windows.update(spec.short_windows)
        if spec.long_windows:
            all_windows.update(spec.long_windows)

    # 构建器映射表
    BUILDERS = {
        "single": build_single,
        "binary_same_ts": build_binary_same_ts,
        "binary": build_binary_same_ts,          # 兼容旧family
        "binary_mixed_ts": build_binary_mixed_ts,
        "binary_mixed": build_binary_mixed_ts,   # 修复：匹配template_spec.py中的family值
        "multi_window": build_multi_window,
        "triple": build_triple_modulation,
        "quad": build_quad_balanced,
    }

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
        
        # 使用对应的构建器
        builder_func = BUILDERS.get(spec.family)
        if builder_func:
            builder_func(spec, fields, windows, spec_records, seen_canonical, validator, seed)
        else:
            print(f"Warning: No builder found for family '{spec.family}'")



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

def build_single(spec: TemplateSpec, fields: list[str], windows: list[int], 
                 records: list[ExpressionRecord], seen: set[str], validator: Validator, seed: int):
    """构建单变量模板"""
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
                        
                        _add_if_valid(expr, spec, records, seen, validator)

def build_binary_same_ts(spec: TemplateSpec, fields: list[str], windows: list[int],
                        records: list[ExpressionRecord], seen: set[str], validator: Validator, seed: int):
    """构建相同时间窗口的二元模板"""
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
                            
                            _add_if_valid(expr, spec, records, seen, validator)

def build_binary_mixed_ts(spec: TemplateSpec, fields: list[str], windows: list[int],
                         records: list[ExpressionRecord], seen: set[str], validator: Validator, seed: int):
    """构建混合时间窗口的二元模板"""
    left_ts_ops = getattr(spec, 'left_ts_ops', spec.ts_ops)
    right_ts_ops = getattr(spec, 'right_ts_ops', spec.ts_ops)
    
    # 为了避免组合爆炸，使用确定性采样
    total_combinations = len(fields) * len(fields) * len(windows) * max(len(left_ts_ops), 1) * max(len(right_ts_ops), 1)
    sample_rate = min(0.3, 10000.0 / max(total_combinations, 1))
    
    for i, f1 in enumerate(fields):
        for j, f2 in enumerate(fields):
            if i == j:  # 避免相同字段
                continue
            for w in windows:
                for left_op in left_ts_ops:
                    for right_op in right_ts_ops:
                        for b_op in spec.binary_ops:
                            for outer in spec.outer_transforms:
                                # 确定性采样
                                combo_key = f"{f1}_{f2}_{w}_{left_op}_{right_op}_{b_op}"
                                if not stable_keep(combo_key, sample_rate, seed):
                                    continue
                                
                                # 结构：Outer(B_Op(LeftTsOp(f1,w), RightTsOp(f2,w)))
                                left_ts = f"{left_op}(${f1},{w})"
                                right_ts = f"{right_op}(${f2},{w})"
                                b_expr = f"{b_op}({left_ts},{right_ts})"
                                
                                if outer == "Rank":
                                    expr = f"Rank({b_expr})"
                                else:
                                    expr = b_expr
                                
                                _add_if_valid(expr, spec, records, seen, validator)

def build_multi_window(spec: TemplateSpec, fields: list[str], windows: list[int],
                      records: list[ExpressionRecord], seen: set[str], validator: Validator, seed: int):
    """构建多窗口模板"""
    sws = spec.short_windows if spec.short_windows else [5, 10, 20]
    lws = spec.long_windows if spec.long_windows else [40, 60]
    
    for f in fields:
        for sw in sws:
            for lw in lws:
                if sw >= lw: 
                    continue
                for ts_op in spec.ts_ops:
                    for b_op in spec.binary_ops:
                        ts_sw = f"{ts_op}(${f},{sw})"
                        ts_lw = f"{ts_op}(${f},{lw})"
                        expr = f"Rank({b_op}({ts_sw},{ts_lw}))"
                        _add_if_valid(expr, spec, records, seen, validator)

def build_triple_modulation(spec: TemplateSpec, fields: list[str], windows: list[int],
                         records: list[ExpressionRecord], seen: set[str], validator: Validator, seed: int):
    """构建三阶调制模板"""
    forms = spec.forms if spec.forms else [
        "Rank(Mul(Sub(A,B),C))",
        "Rank(Div(Sub(A,B),Abs(C)))",
        "Rank(Sub(Mul(A,B),C))",
        "Rank(Add(Mul(A,B),C))",
        "Rank(Mul(Div(A,B),C))"
    ]
    
    # 为了避免组合爆炸，使用确定性采样
    total_combinations = len(fields) * len(fields) * len(fields) * len(windows) * len(forms)
    sample_rate = min(0.1, 30000.0 / total_combinations)
    
    for form in forms:
        for w in windows:
            # 随机选择三个不同的字段
            if len(fields) < 3:
                continue
                
            # 使用确定性采样选择字段组合
            for i in range(min(100, len(fields))):  # 限制每种子形式的尝试次数
                combo_key = f"triple_{form}_{w}_{i}"
                if not stable_keep(combo_key, sample_rate, seed):
                    continue
                
                # 随机选择三个字段
                selected_fields = random.sample(fields, 3)
                f1, f2, f3 = selected_fields
                
                # 替换形式中的占位符
                expr = form.replace("A", f"TsMean(${f1},{w})" if "TsMean" in form else f"${f1}")
                expr = expr.replace("B", f"TsMean(${f2},{w})" if "TsMean" in form else f"${f2}")
                expr = expr.replace("C", f"TsMean(${f3},{w})" if "TsMean" in form else f"${f3}")
                
                # 如果形式中包含TsMean等操作符，需要构建完整的表达式
                if "TsMean" not in form:
                    # 简单的字段替换
                    expr = form.replace("A", f"${f1}").replace("B", f"${f2}").replace("C", f"${f3}")
                    # 添加时间序列操作
                    if "TsMean" in spec.ts_ops:
                        expr = expr.replace("A", f"TsMean(${f1},{w})")
                        expr = expr.replace("B", f"TsMean(${f2},{w})")
                        expr = expr.replace("C", f"TsMean(${f3},{w})")
                
                _add_if_valid(expr, spec, records, seen, validator)

def build_quad_balanced(spec: TemplateSpec, fields: list[str], windows: list[int],
                       records: list[ExpressionRecord], seen: set[str], validator: Validator, seed: int):
    """构建四阶平衡模板"""
    forms = spec.forms if spec.forms else [
        "Rank(Sub(Mul(A,B),Mul(C,D)))",
        "Rank(Sub(Div(A,B),Div(C,D)))",
        "Rank(Mul(Sub(A,B),Sub(C,D)))"
    ]
    
    # 为了避免组合爆炸，使用确定性采样
    total_combinations = len(fields) * len(fields) * len(fields) * len(fields) * len(windows) * len(forms)
    sample_rate = min(0.05, 10000.0 / total_combinations)
    
    for form in forms:
        for w in windows:
            # 随机选择四个不同的字段
            if len(fields) < 4:
                continue
                
            # 使用确定性采样选择字段组合
            for i in range(min(50, len(fields))):  # 限制每种子形式的尝试次数
                combo_key = f"quad_{form}_{w}_{i}"
                if not stable_keep(combo_key, sample_rate, seed):
                    continue
                
                # 随机选择四个字段
                selected_fields = random.sample(fields, 4)
                f1, f2, f3, f4 = selected_fields
                
                # 替换形式中的占位符，添加时间序列操作
                expr = form
                if "TsMean" in spec.ts_ops:
                    expr = expr.replace("A", f"TsMean(${f1},{w})")
                    expr = expr.replace("B", f"TsMean(${f2},{w})")
                    expr = expr.replace("C", f"TsMean(${f3},{w})")
                    expr = expr.replace("D", f"TsMean(${f4},{w})")
                else:
                    expr = expr.replace("A", f"${f1}").replace("B", f"${f2}")
                    expr = expr.replace("C", f"${f3}").replace("D", f"${f4}")
                
                _add_if_valid(expr, spec, records, seen, validator)

def _add_if_valid(expr: str, spec: TemplateSpec, records: list[ExpressionRecord], seen: set[str], validator: Validator):
    """验证并添加表达式到记录列表"""
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
