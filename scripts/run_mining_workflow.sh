#!/bin/bash
# alpha-factor-mvp 挖掘流程：一阶段 (粗筛) -> 二阶段 (精炼)

# 设置环境 (根据实际情况调整)
# export PATH="D:/Total_Tools/miniforge3/Scripts:$PATH"
# source activate universal

echo "=== 开始一阶段挖掘 (2024-至今, 200只票) ==="
python -m src.alpha_mvp.cli_phase2 \
    --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" \
    --pool-json "D:\Trading\My_factor_mining_427\static_pool_200.json" \
    --start 20240101 \
    --end 20260430 \
    --max-exprs 2000 \
    --out outputs/phase1_coarse \
    --write-every 500 \
    --batch-size 100

echo "=== 一阶段完成，提取 Top 候选因子 ==="
# 假设我们选择 Top 500 进入二阶段
# 这里可以调用 candidate_sampler.py 的逻辑，或者简单读取 top100_phase2.csv
# 为演示，我们直接从一阶段输出目录读取

echo "=== 开始二阶段挖掘 (2018-至今, 800只票) ==="
# 使用一阶段生成的表达式文件进行精炼验证
python -m src.alpha_mvp.cli_phase2 \
    --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" \
    --pool-json "D:\Trading\My_factor_mining_427\static_pool_800.json" \
    --start 20180101 \
    --end 20260430 \
    --expr-file "outputs/phase1_coarse/factor_results_phase2.csv" \
    --start-expr 1 \
    --end-expr 500 \
    --out outputs/phase2_refined \
    --write-every 100 \
    --batch-size 20
