from pathlib import Path
from src.alpha_mvp.config import RunConfig, EvalConfig
from src.alpha_mvp.pipeline import run_pipeline
from src.alpha_mvp.parser import parse_expr, canonical
from src.alpha_mvp.validator import Validator
from src.alpha_mvp.fields import DEFAULT_FEATURES

def test_parser_validator():
    expr = "Rank(TsMean($ret_1d,10))"
    node = parse_expr(expr)
    assert canonical(node) == expr
    v = Validator(set(DEFAULT_FEATURES), {10,20,30,40,50})
    assert v.validate(node).ok

def test_pipeline_simulated(tmp_path):
    cfg = RunConfig(use_simulated=True, max_exprs=12, out_dir=str(tmp_path), eval=EvalConfig(min_daily_valid_names=20))
    summary = run_pipeline(cfg)
    assert summary["n_exprs"] > 0
    assert Path(tmp_path / "factor_results.csv").exists()
    assert Path(tmp_path / "top100.csv").exists()
