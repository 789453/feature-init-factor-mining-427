#!/bin/bash
# Full validation run (Steps 1-8)

export PATH="D:/Total_Tools/miniforge3/Scripts:$PATH"
export CONDA_DEFAULT_ENV="universal"
export CONDA_PREFIX="D:/Total_Tools/miniforge3/envs/universal"

python -m src.alpha_mvp.validation.cli_validate \
  --duckdb "D:/Trading/data_ever_26_3_14/data/meta/warehouse.duckdb" \
  --top100 "d:/Trading/My_factor_mining_427/outputs/real_all/top100.csv" \
  --out "d:/Trading/My_factor_mining_427/outputs/validation_full" \
  --top-n 100 \
  --alphalens-top-n 5
