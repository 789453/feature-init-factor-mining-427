#!/bin/bash

# Phase 2 真实因子数据逐步运行脚本
# 使用现有的因子评估结果数据，逐步验证各个阶段

echo "======================================="
echo "Phase 2 Real Factor Data Step-by-Step Execution"
echo "======================================="

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 设置配置路径
CONFIG_PATH="configs/phase2/real_factor_data_config.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

# 检查DuckDB文件是否存在
DUCKDB_PATH="outputs/full_run_phase2/phase2_results.duckdb"
if [ ! -f "$DUCKDB_PATH" ]; then
    echo "❌ Error: DuckDB file not found: $DUCKDB_PATH"
    exit 1
fi

echo "✅ Configuration: $CONFIG_PATH"
echo "✅ DuckDB file: $DUCKDB_PATH"
echo "✅ Python environment: activated"
echo "======================================="

# 步骤0: 验证现有真实因子数据
echo "🔍 Step 0: Validating existing real factor data..."

python -c "
import duckdb
import pandas as pd

print('=== Existing Real Factor Data Analysis ===')
conn = duckdb.connect('outputs/full_run_phase2/phase2_results.duckdb')

# 检查因子结果表
try:
    results_df = conn.execute('SELECT COUNT(*) as total_factors, MIN(created_at) as earliest, MAX(created_at) as latest FROM factor_results').fetchdf()
    print(f'📊 Total factor results: {results_df.iloc[0][\"total_factors\"]}')
    print(f'📅 Date range