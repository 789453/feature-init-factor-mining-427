$env:PATH = "D:\Total_Tools\miniforge3\Scripts;" + $env:PATH
$env:CONDA_DEFAULT_ENV = "universal"
$env:CONDA_PREFIX = "D:\Total_Tools\miniforge3\envs\universal"

python -m src.alpha_mvp.cli `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --pool-json "D:\Trading\My_factor_mining_427\static_pool_200.json" `
  --start 20240101 `
  --end 20260430 `
  --out outputs\real_200_test `
  --max-exprs 100