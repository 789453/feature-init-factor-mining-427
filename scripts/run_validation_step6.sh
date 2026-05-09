#!/bin/bash
# Resume validation from step 6 (using cached factor panels, analytics, group metrics)

export PATH="D:/Total_Tools/miniforge3/Scripts:$PATH"
export CONDA_DEFAULT_ENV="universal"
export CONDA_PREFIX="D:/Total_Tools/miniforge3/envs/universal"

echo "Running validation from step 6 (using cached data)..."

python -m src.alpha_mvp.validation.cli_validate \
  --duckdb "D:/Trading/data_ever_26_3_14/data/meta/warehouse.duckdb" \
  --top100 "d:/Trading/My_factor_mining_427/outputs/real_all/top100.csv" \
  --out "d:/Trading/My_factor_mining_427/outputs/validation_full" \
  --top-n 100 \
  --from-step 6 \
  --alphalens-top-n 5
