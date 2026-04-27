from __future__ import annotations
import numpy as np
from .parser import Node, parse_expr, canonical
from . import ops
from .validator import Validator

class EvalError(RuntimeError):
    pass

class BatchEvaluator:
    def __init__(self, panels: dict[str, np.ndarray], dates: list[str], codes: list[str], windows=(10,20,30,40,50),
                 max_depth=4, max_nodes=10):
        self.panels = panels
        self.dates = dates
        self.codes = codes
        self.windows = tuple(windows)
        self.validator = Validator(set(panels.keys()), set(windows), max_depth=max_depth, max_nodes=max_nodes)
        self._cache: dict[str, np.ndarray] = {}

    def eval_expr(self, expr: str) -> tuple[np.ndarray | None, str]:
        try:
            node = parse_expr(expr)
            vr = self.validator.validate(node)
            if not vr.ok:
                return None, vr.reason
            return self._eval_node(node), "OK"
        except Exception as e:
            return None, f"EVAL_ERROR:{type(e).__name__}:{e}"

    def _eval_node(self, node: Node) -> np.ndarray:
        key = canonical(node)
        if key in self._cache:
            return self._cache[key]
        if node.kind == "field":
            if node.value not in self.panels:
                raise EvalError(f"unknown field {node.value}")
            out = self.panels[node.value]
        elif node.kind == "const":
            out = np.full_like(next(iter(self.panels.values())), float(node.value), dtype=float)
        elif node.kind == "op":
            op = node.value
            if op in ops.UNARY:
                out = ops.UNARY[op](self._eval_node(node.args[0]))
            elif op in ops.BINARY:
                out = ops.BINARY[op](self._eval_node(node.args[0]), self._eval_node(node.args[1]))
            elif op in ops.ROLLING:
                out = ops.ROLLING[op](self._eval_node(node.args[0]), int(node.args[1].value))
            elif op in ops.PAIR_ROLLING:
                out = ops.PAIR_ROLLING[op](self._eval_node(node.args[0]), self._eval_node(node.args[1]), int(node.args[2].value))
            else:
                raise EvalError(f"unknown op {op}")
        else:
            raise EvalError(f"unknown node {node.kind}")
        out = np.asarray(out, dtype=float)
        out[~np.isfinite(out)] = np.nan
        self._cache[key] = out
        return out

def make_panels(df, feature_cols: list[str], value_col="close"):
    pivots = {}
    dates = sorted(df["trade_date"].astype(str).unique().tolist())
    codes = sorted(df["ts_code"].unique().tolist())
    for c in feature_cols + [value_col]:
        if c not in df.columns:
            continue
        p = df.pivot(index="trade_date", columns="ts_code", values=c).reindex(index=dates, columns=codes)
        pivots[c] = p.to_numpy(dtype=float)
    return pivots, dates, codes
