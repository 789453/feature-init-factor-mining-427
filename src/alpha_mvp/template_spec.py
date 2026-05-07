from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

@dataclass(frozen=True)
class TemplateSpec:
    name: str
    family: str
    order: int
    enabled: bool = True
    max_count: int | None = None
    outer_transforms: Sequence[str] = ("Rank",)
    unary_pre: Sequence[str] = ("Id",)
    ts_ops: Sequence[str] = ()
    binary_ops: Sequence[str] = ()
    pair_ops: Sequence[str] = ()
    windows: Sequence[int] = ()
    short_windows: Sequence[int] = ()
    long_windows: Sequence[int] = ()
    complexity_tier: int = 1

@dataclass(frozen=True)
class ComplexityBudget:
    tier: int
    max_depth: int
    max_nodes: int
    max_ts_ops: int
    max_pair_ops: int
    max_binary_ops: int
    max_unary_ops: int

# 默认复杂度预算
COMPLEXITY_BUDGETS = {
    1: ComplexityBudget(tier=1, max_depth=4, max_nodes=10, max_ts_ops=2, max_pair_ops=1, max_binary_ops=1, max_unary_ops=3),
    2: ComplexityBudget(tier=2, max_depth=5, max_nodes=14, max_ts_ops=3, max_pair_ops=1, max_binary_ops=2, max_unary_ops=4),
    3: ComplexityBudget(tier=3, max_depth=6, max_nodes=18, max_ts_ops=4, max_pair_ops=1, max_binary_ops=3, max_unary_ops=5),
    4: ComplexityBudget(tier=4, max_depth=7, max_nodes=24, max_ts_ops=4, max_pair_ops=1, max_binary_ops=4, max_unary_ops=6),
}

# 核心模板规格定义
TEMPLATE_SPECS = [
    TemplateSpec(
        name="single_ts_outer_rank",
        family="single",
        order=1,
        unary_pre=("Id", "Abs", "SLog1p"),
        ts_ops=("TsMean", "TsStd", "TsIr", "TsRank", "TsDelta", "TsDiv", "TsPctChange", "TsWMA", "TsEMA"),
        outer_transforms=("Rank", "SLog1p", "RankSLog1p"),
        complexity_tier=1,
    ),
    TemplateSpec(
        name="binary_same_ts",
        family="binary",
        order=2,
        ts_ops=("TsMean", "TsDelta", "TsStd", "TsEMA", "TsWMA"),
        binary_ops=("Add", "Sub", "Mul", "Div", "Greater", "Less"),
        outer_transforms=("Rank", "SLog1p", "RankSLog1p"),
        complexity_tier=2,
    ),
    TemplateSpec(
        name="binary_mixed_ts",
        family="binary_mixed",
        order=2,
        ts_ops=("TsMean", "TsStd", "TsIr", "TsRank", "TsDelta", "TsEMA"),
        binary_ops=("Sub", "Div", "Mul"),
        outer_transforms=("Rank",),
        complexity_tier=2,
    ),
    TemplateSpec(
        name="triple_modulation",
        family="triple",
        order=3,
        ts_ops=("TsMean", "TsDelta", "TsEMA"),
        binary_ops=("Add", "Sub", "Mul", "Div"),
        outer_transforms=("Rank",),
        complexity_tier=3,
    ),
    TemplateSpec(
        name="quad_balanced",
        family="quad",
        order=4,
        ts_ops=("TsMean", "TsEMA"),
        binary_ops=("Sub", "Mul", "Div"),
        outer_transforms=("Rank",),
        complexity_tier=4,
    ),
    TemplateSpec(
        name="multi_window_gap",
        family="multi_window",
        order=1,
        ts_ops=("TsMean", "TsEMA", "TsStd", "TsRank"),
        binary_ops=("Sub", "Div"),
        outer_transforms=("Rank",),
        complexity_tier=2,
    ),
]
