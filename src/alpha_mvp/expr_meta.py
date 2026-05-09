from __future__ import annotations
from dataclasses import dataclass, field
from .parser import Node, parse_expr
from .template_builder import ExpressionRecord

@dataclass
class ExprMeta:
    canonical: str
    expr_hash: str
    template_name: str | None
    template_family: str | None
    template_order: int | None
    fields: list[str] = field(default_factory=list)
    operators: list[str] = field(default_factory=list)
    unary_ops: list[str] = field(default_factory=list)
    binary_ops: list[str] = field(default_factory=list)
    ts_ops: list[str] = field(default_factory=list)
    pair_ops: list[str] = field(default_factory=list)
    windows: list[int] = field(default_factory=list)
    n_fields: int = 0
    n_unique_fields: int = 0
    n_ops: int = 0
    depth: int = 0
    nodes: int = 0
    has_rank: bool = False
    has_slog1p: bool = False
    has_log: bool = False
    has_mul: bool = False
    has_div: bool = False
    has_corr: bool = False

def extract_meta(expr: str, template_record: ExpressionRecord | None = None) -> ExprMeta:
    """
    从表达式中提取元数据，用于后续统计分析
    """
    node = parse_expr(expr)
    
    fields = []
    ops = []
    windows = []
    
    # 算子分类定义（需与 ops.py 保持一致）
    from .ops import UNARY, BINARY, ROLLING, PAIR_ROLLING
    
    def walk(n):
        if n.kind == "field":
            fields.append(n.value)
            return 1, 1 # depth, nodes
        elif n.kind == "const":
            return 1, 1
        elif n.kind == "op":
            ops.append(n.value)
            child_depths = []
            child_nodes = 0
            for a in n.args:
                if a.kind == "const" and n.value in (set(ROLLING) | set(PAIR_ROLLING)):
                    windows.append(int(a.value))
                else:
                    d, m = walk(a)
                    child_depths.append(d)
                    child_nodes += m
            return 1 + max(child_depths, default=0), 1 + child_nodes
        return 0, 0

    depth, nodes = walk(node)
    
    unique_fields = sorted(list(set(fields)))
    
    meta = ExprMeta(
        canonical=template_record.canonical if template_record else expr, # 假设传入了 record 则使用其 canonical
        expr_hash=template_record.expr_hash if template_record else "",
        template_name=template_record.template_name if template_record else None,
        template_family=template_record.template_family if template_record else None,
        template_order=template_record.template_order if template_record else None,
        fields=fields,
        operators=ops,
        windows=windows,
        n_fields=len(fields),
        n_unique_fields=len(unique_fields),
        n_ops=len(ops),
        depth=depth,
        nodes=nodes,
        unary_ops=[o for o in ops if o in UNARY],
        binary_ops=[o for o in ops if o in BINARY],
        ts_ops=[o for o in ops if o in ROLLING],
        pair_ops=[o for o in ops if o in PAIR_ROLLING],
        has_rank="Rank" in ops,
        has_slog1p="SLog1p" in ops,
        has_log="Log" in ops,
        has_mul="Mul" in ops,
        has_div="Div" in ops,
        has_corr="TsCorr" in ops
    )
    
    return meta
