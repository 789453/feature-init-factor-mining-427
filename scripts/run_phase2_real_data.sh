#!/bin/bash

# Phase 2 真实数据运行脚本
# 使用现有的DuckDB文件运行完整的粗筛+细筛流程

echo "Starting Phase 2 Real Data Execution..."
echo "======================================="

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 设置配置路径
CONFIG_PATH="configs/phase2/real_data_config.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

# 检查DuckDB文件是否存在
DUCKDB_PATH="outputs/full_run_phase2/phase2_results.duckdb"
if [ ! -f "$DUCKDB_PATH" ]; then
    echo "Error: DuckDB file not found: $DUCKDB_PATH"
    echo "Available DuckDB files:"
    find outputs/ -name "*.duckdb" | head -5
    exit 1
fi

echo "Using configuration: $CONFIG_PATH"
echo "Using DuckDB file: $DUCKDB_PATH"
echo "======================================="

# 运行完整的Phase 2流程
echo "Running Phase 2 with real data..."
python -m src.alpha_mvp.phase_runner --config "$CONFIG_PATH" --phase full

echo "Phase 2 Real Data Execution Completed!"
echo "Check outputs/phase2_real/ for results"