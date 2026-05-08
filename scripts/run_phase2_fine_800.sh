#!/bin/bash
# Phase2细筛阶段运行脚本

echo "Starting Phase2 Fine Mining..."
echo "Pool: static_pool_800.json"
echo "Date Range: 20180101 - 20260430"
echo "Using candidates from coarse phase"

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 运行细筛阶段
python -m src.alpha_mvp.phase_runner configs/phase2/phase2_coarse_fine.yaml fine

echo "Fine phase completed!"
echo "Check outputs/phase2/fine_800_2018/ for results"
echo "Top 100 factors will be in outputs/phase2/fine_800_2018/top100_phase2.csv"