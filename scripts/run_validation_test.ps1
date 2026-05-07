Set-StrictMode -Version 2
$env:PATH = "D:\Total_Tools\miniforge3\Scripts;" + $env:PATH
$env:CONDA_DEFAULT_ENV = "universal"
$env:CONDA_PREFIX = "D:\Total_Tools\miniforge3\envs\universal"

python -m src.alpha_mvp.validation.cli_validate `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --top100 "outputs\rreal_all\top100.csv" `
  --out "outputs\validation_real_200_v3" `
  --start 20180101 `
  --end 20260430 `
  --horizon 5 `
  --rebalance-n 5 `
  --alphalens-top-n 1