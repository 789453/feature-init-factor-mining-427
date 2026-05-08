#!/bin/bash
# Phase2粗筛阶段运行脚本

echo "Starting Phase2 Coarse Mining..."
echo "Pool: static_pool_200.json"
echo "Date Range: 20240101 - 20260430"
echo "Template Config: configs/phase2/templates_v1.yaml"

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 运行粗筛阶段
python -m src.alpha_mvp.phase_runner configs/phase2/phase2_coarse_fine.yaml coarse

echo "Coarse phase completed!"
echo "Check outputs/phase2/coarse_200_2024/ for results"
echo "Candidate expressions will be in outputs/phase2/coarse_200_2024/exports/fine_candidates.expr"