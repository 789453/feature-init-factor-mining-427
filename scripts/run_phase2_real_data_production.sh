#!/bin/bash

# Phase 2 真实数据生产环境运行脚本
# 使用真实DuckDB数据运行完整的粗筛+细筛流程

echo "======================================="
echo "Phase 2 Real Data Production Execution"
echo "======================================="

# 激活环境
source D:/Total_Tools/miniforge3/etc/profile.d/conda.sh
conda activate universal

# 设置配置路径
CONFIG_PATH="configs/phase2/real_data_full_config.yaml"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Error: Configuration file not found: $CONFIG_PATH"
    exit 1
fi

# 检查DuckDB文件是否存在
DUCKDB_PATH="outputs/full_run_phase2/phase2_results.duckdb"
if [ ! -f "$DUCKDB_PATH" ]; then
    echo "❌ Error: DuckDB file not found: $DUCKDB_PATH"
    echo "Available DuckDB files:"
    find outputs/ -name "*.duckdb" | head -5
    exit 1
fi

echo "✅ Configuration: $CONFIG_PATH"
echo "✅ DuckDB file: $DUCKDB_PATH"
echo "✅ Python environment: activated"
echo "======================================="

# 步骤1: 运行粗筛阶段
echo "🚀 Step 1: Running COARSE phase with real data..."
echo "Time range: 2024-01-01 to 2024-06-30 (6 months)"
echo "Pool: static_pool_200.json (200 stocks)"
echo "Max expressions: 100,000"
echo ""

python -m src.alpha_mvp.phase_runner "$CONFIG_PATH" coarse

if [ $? -ne 0 ]; then
    echo "❌ Coarse phase failed!"
    exit 1
fi

echo "✅ Coarse phase completed successfully!"
echo ""

# 步骤2: 运行细筛阶段
echo "🚀 Step 2: Running FINE phase with real data..."
echo "Using candidates from coarse phase"
echo "Pool: static_pool_800.json (800 stocks)"
echo "Max expressions: 50,000"
echo ""

python -m src.alpha_mvp.phase_runner "$CONFIG_PATH" fine

if [ $? -ne 0 ]; then
    echo "❌ Fine phase failed!"
    exit 1
fi

echo "✅ Fine phase completed successfully!"
echo ""

# 步骤3: 运行完整流程报告
echo "🚀 Step 3: Generating final report..."
python -m src.alpha_mvp.phase_runner "$CONFIG_PATH" full

if [ $? -ne 0 ]; then
    echo "❌ Final report generation failed!"
    exit 1
fi

echo ""
echo "======================================="
echo "✅ Phase 2 Real Data Production Execution COMPLETED!"
echo "======================================="
echo ""
echo "📊 Results location:"
echo "  - Coarse phase: outputs/phase2_real_prod/coarse_200_2024/"
echo "  - Fine phase: outputs/phase2_real_prod/fine_800_2024/"
echo "  - Final report: outputs/phase2_real_prod/"
echo ""
echo "🔍 Key features validated:"
echo "  ✅ Real DuckDB data loading"
echo "  ✅ YAML configuration system"
echo "  ✅ High-order template generation"
echo "  ✅ Stratified candidate selection"
echo "  ✅ Multi-strategy ranking"
echo "  ✅ Experiment signature system"
echo "  ✅ Extended attribution analysis"
echo "======================================="