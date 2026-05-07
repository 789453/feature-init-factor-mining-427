# alpha-factor-mvp

面向 A 股日频短期资金/订单/量价特征的公式化 Alpha 挖掘原型。

## 核心能力

1. 从 DuckDB 或模拟数据加载 `daily / moneyflow / cyq_perp / daily_basic`
2. 构建约 30 个短期基础特征字段，不直接使用最原始字段
3. 使用 typed grammar + 模板枚举生成低阶表达式
4. 解析、合法性校验、复杂度控制、表达式去重
5. 在线计算表达式，不做重缓存
6. 批量计算 IC、RankIC、ICIR、覆盖率、换手 proxy、年度/分段稳定性
7. 记录所有表达式测试结果
8. 预留二阶段优化接口

## 项目结构

```
d:/Trading/My_factor_mining_427/
├── src/alpha_mvp/
│   ├── cli.py                  命令行入口
│   ├── config.py                全局配置
│   ├── fields.py               30+ 基础特征字段构建
│   ├── ops.py                  算子实现（rank_cs, daily_corr, rolling_corr等）
│   ├── parser.py               表达式解析
│   ├── validator.py            合法性校验
│   ├── grammar.py              typed grammar + 模板枚举
│   ├── data.py                 DuckDB / 模拟数据加载
│   ├── metrics.py              IC/RankIC/ICIR 等指标计算
│   ├── evaluator.py            在线批量 evaluator
│   ├── pipeline.py             主流程
│   ├── cache/                  缓存模块
│   │   ├── panel_io.py         Panel 数据读写（Parquet格式）
│   │   ├── feature_cache.py    特征面板缓存
│   │   └── factor_cache.py     因子面板缓存
│   └── validation/             验证模块（8步流水线）
│       ├── cli_validate.py     验证CLI入口
│       ├── config.py            验证配置
│       ├── runner.py            8步验证流程编排
│       ├── analytics.py         IC/RankIC批量分析
│       ├── group_metrics.py     分组度量（Size/Industry）
│       ├── vectorbot.py         VectorBot风格回测
│       ├── alphalens_runner.py  Alphalens分析
│       ├── report_builder.py    HTML报告生成
│       ├── plots.py             Plotly可视化基础
│       └── plots_advanced.py    高级可视化
├── scripts/                     运行脚本
│   ├── run_validation_full.sh  完整验证流程（步骤1-8）
│   ├── run_validation_step6.sh 从步骤6恢复验证
│   └── regenerate_plots.sh      重绘图表（不重新计算）
├── outputs/                     输出目录
│   ├── real_all/               一阶段挖掘结果
│   │   └── top100.csv          Top100因子列表
│   └── validation_full/        完整验证结果
│       ├── cache/              缓存数据
│       │   ├── feature_panels/ 特征面板缓存
│       │   └── factor_panels/   因子面板缓存
│       ├── metrics/            指标结果
│       │   ├── summary.csv     汇总统计
│       │   ├── rolling_ic.parquet 滚动IC
│       │   ├── group_size.csv  市值分组结果
│       │   └── group_industry.csv 行业分组结果
│       ├── vectorbot/          VectorBot回测结果
│       │   ├── portfolio_summary.csv
│       │   └── equity_curves.parquet
│       ├── alphalens/          Alphalens分析结果
│       │   └── F*/             各因子分析目录
│       ├── plots/              可视化图表
│       │   ├── *_rolling_ic.html  滚动IC图
│       │   ├── *_equity.html       权益曲线图
│       │   ├── top10_equity_comparison.html Top10对比
│       │   ├── vectorbot_summary.html VectorBot汇总
│       │   ├── ic_heatmap_top20.html  IC热力图
│       │   └── *.html              其他分析图
│       └── reports/
│           └── validation_report.html 验证报告
└── architectures/              设计文档
    ├── validation_stage_engineering_design.md
    ├── full_market_validation_performance_fix_plan.md
    └── fastops修正.md
```

## 快速运行

### 模拟数据测试

```bash
cd d:/Trading/My_factor_mining_427
python -m src.alpha_mvp.cli --use-simulated --out outputs/demo --max-exprs 300
```

### 真实数据验证

```bash
# 完整验证流程（8步）
bash scripts/run_validation_full.sh

# 从第6步恢复（使用缓存）
bash scripts/run_validation_step6.sh

# 重绘所有图表（不重新计算）
bash scripts/regenerate_plots.sh
```

## 验证流水线（8步）

| 步骤 | 名称 | 说明 |
|------|------|------|
| 1 | 加载Top因子 | 从top100.csv加载待验证因子 |
| 2 | 加载市场特征 | 从DuckDB加载特征面板并缓存 |
| 3 | 计算因子面板 | 批量计算100个因子的面板数据 |
| 4 | IC/RankIC分析 | 批量计算IC、RankIC、滚动IC |
| 5 | 分组度量 | 按Size、Industry分组统计 |
| 6 | VectorBot回测 | 组合回测、权益曲线、Turnover |
| 7 | Alphalens分析 | 分层收益、信息边界、换手分析 |
| 8 | 报告生成 | HTML报告、Plotly交互图表 |

## 参数说明

### CLI验证参数

```
--duckdb              DuckDB数据库路径
--top100              Top100因子CSV路径
--out                 输出目录
--start               开始日期（默认20180101）
--end                 结束日期（默认20260430）
--horizon             预测周期（默认5日）
--rebalance-n         调仓周期（默认5日）
--top-n               验证因子数量（默认100）
--alphalens-top-n     Alphalens分析因子数（默认10）
--from-step           从第N步开始（支持断点恢复）
--train-end           训练集截止日期
--test-start          测试集开始日期
--overwrite-cache     覆盖缓存重新计算
--skip-vectorbot      跳过VectorBot
--skip-alphalens      跳过Alphalens
--skip-reports        跳过报告生成
```

### 关键配置

- **股票池**: 200只A股，由static_pool_200.json定义
- **训练集**: 2018-2025年
- **测试集**: 2025年至今
- **调仓周期**: 5日
- **Top quantile**: 5%, 10%

## 输出指标

### IC/RankIC分析

| 指标 | 说明 |
|------|------|
| IC Mean | 信息系数均值 |
| IC Std | 信息系数标准差 |
| RankIC Mean | 秩相关系数均值 |
| RankIC IR | ICIR（均值/标准差） |
| Positive % | IC为正比例 |

### VectorBot回测

| 指标 | 说明 |
|------|------|
| Annual Return | 年化收益率 |
| Annual Vol | 年化波动率 |
| Sharpe | 夏普比率 |
| Max Drawdown | 最大回撤 |
| Win Rate | 胜率 |
| Avg Turnover | 平均换手率 |
| Total Return | 总收益 |

## 可视化

使用Plotly生成交互式HTML图表：

- **滚动IC图**: `*_rolling_ic.html` - 每日IC + 滚动20/60日IC
- **权益曲线**: `*_equity.html` - 各Top quantile组合收益
- **Top10对比**: `top10_equity_comparison.html` - 前10因子对比
- **VectorBot汇总**: `vectorbot_summary.html` - 4面板综合分析
- **IC热力图**: `ic_heatmap_top20.html` - 前20因子IC时间序列
- **分组图表**: `size_bucket.html`, `industry_top20.html`
- **IC分布**: `ic_distribution.html` - IC统计分布

## 缓存机制

验证流程支持完整的缓存恢复：

1. **特征面板缓存**: `cache/feature_panels/*.parquet`
2. **因子面板缓存**: `cache/factor_panels/*.parquet`
3. **指标缓存**: `metrics/summary.csv`, `metrics/rolling_ic.parquet`

使用 `--from-step N` 可从任意步骤恢复，避免重复计算。

## 依赖

- Python 3.10+
- DuckDB（数据存储）
- Numba（加速计算）
- Plotly（交互图表）
- Alphalens（因子分析）
- Pandas/NumPy（数据处理）
