import hashlib
import pandas as pd

def make_factor_id(expr: str, i: int) -> str:
    h = hashlib.md5(expr.encode("utf-8")).hexdigest()[:8]
    return f"F{i:04d}_{h}"

def load_top_factors(path: str, top_n: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    expr_col = "canonical" if "canonical" in df.columns else "expr"
    df = df.dropna(subset=[expr_col]).copy()
    df = df.drop_duplicates(subset=[expr_col])
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    if top_n is not None:
        df = df.head(top_n)
    df["factor_expr"] = df[expr_col].astype(str)
    df["factor_id"] = [
        make_factor_id(expr, i + 1)
        for i, expr in enumerate(df["factor_expr"])
    ]
    return df