#!/bin/bash
# Phase2完整流程运行脚本（粗筛+细筛）

echo "Starting Phase2 Full Pipeline..."
echo "This will run both coarse and fine phases"

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 运行完整流程
python -m src.alpha_mvp.phase_runner configs/phase2/phase2_coarse_fine.yaml full

echo "Full pipeline completed!"
echo "Check outputs/phase2/ for complete results"
echo "Final report: outputs/phase2/fine_800_2018/final_pipeline_report.json"