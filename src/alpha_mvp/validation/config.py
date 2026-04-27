from dataclasses import dataclass

@dataclass(frozen=True)
class ValidationConfig:
    duckdb_path: str
    top100_path: str = "outputs/real_all/top100.csv"
    out_dir: str = "outputs/validation_real_all"
    start: str = "20180101"
    end: str = "20260430"

    horizon: int = 5
    rebalance_n: int = 5

    min_price: float = 1.0
    min_amount: float = 0.0
    exclude_st: bool = False

    top_quantiles: tuple[float, ...] = (0.05, 0.10)
    layer_quantiles: int = 5
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    train_end: str = "20250831"
    test_start: str = "20250901"

    alphalens_top_n: int = 10
    alphalens_periods: tuple[int, ...] = (1, 5, 10)
    alphalens_quantiles: int = 5
    alphalens_max_loss: float = 0.45

    factor_batch_size: int = 10
    write_factor_panels: bool = True
    use_gpu: bool = False
    duckdb_threads: int = 24
    duckdb_memory_limit: str = "24GB"