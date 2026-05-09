#!/bin/bash
# Phase 2 Fine Validation - 从第4步开始 (使用缓存的因子面板)

export PATH="D:/Total_Tools/miniforge3/Scripts:$PATH"
export CONDA_DEFAULT_ENV="universal"
export CONDA_PREFIX="D:/Total_Tools/miniforge3/envs/universal"

python -m src.alpha_mvp.validation.cli_validate \
  --duckdb "D:/Trading/data_ever_26_3_14/data/meta/warehouse.duckdb" \
  --top100 "d:/Trading/My_factor_mining_427/outputs/phase2/fine_200_2018/top100_phase2.csv" \
  --out "d:/Trading/My_factor_mining_427/outputs/validation_phase2_fine" \
  --top-n 100 \
  --alphalens-top-n 10 \
  --from-step 4
