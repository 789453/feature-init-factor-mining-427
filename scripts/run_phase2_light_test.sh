#!/bin/bash
# 轻量级Phase2测试运行脚本

echo "Starting Phase2 Light Test..."
echo "Using minimal configuration for quick testing"

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 运行轻量级配置
python -m src.alpha_mvp.phase_runner configs/phase2/phase2_light.yaml full

echo "Light test completed!"
echo "Check outputs/phase2_light/ for results"