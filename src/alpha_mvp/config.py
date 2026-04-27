from dataclasses import dataclass, field
from typing import Sequence

DEFAULT_WINDOWS = (10, 20, 30, 40, 50)
DEFAULT_FORWARD_DAYS = 1

@dataclass(frozen=True)
class EvalConfig:
    windows: Sequence[int] = DEFAULT_WINDOWS
    forward_days: int = DEFAULT_FORWARD_DAYS
    max_depth: int = 4
    max_nodes: int = 10
    max_ts_ops: int = 2
    max_pair_ops: int = 1
    max_binary_ops: int = 1
    min_coverage: float = 0.55
    min_daily_valid_names: int = 30
    eps: float = 1e-9
    use_industry_neutral: bool = False

@dataclass(frozen=True)
class RunConfig:
    start: str = "20240101"
    end: str = "20260430"
    max_exprs: int = 5000
    batch_size: int = 64
    out_dir: str = "outputs/run"
    seed: int = 42
    duckdb_path: str | None = None
    pool_json: str | None = None
    use_simulated: bool = False
    eval: EvalConfig = field(default_factory=EvalConfig)
