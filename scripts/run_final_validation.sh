#!/bin/bash
# 最终验证运行脚本

echo "Starting final validation..."
echo "Using top factors from fine phase"

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 运行验证流程（假设使用现有的验证脚本）
python -m src.alpha_mvp.validation.cli_validate \
  --duckdb data/market.duckdb \
  --factors outputs/phase2/fine_800_2018/top100_phase2.csv \
  --start 20180101 \
  --end 20260430 \
  --out outputs/final_validation

echo "Final validation completed!"
echo "Check outputs/final_validation/ for validation results"