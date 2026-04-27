# alpha-factor-mvp

第一阶段 MVP：面向 A 股日频短期资金/订单/量价特征的公式化 Alpha 挖掘原型。

核心能力：

1. 从 DuckDB 或模拟数据加载 `daily / moneyflow / cyq_perf / daily_basic`。
2. 构建约 30 个短期基础特征字段，不直接使用最原始字段。
3. 使用 typed grammar + 模板枚举生成低阶表达式。
4. 解析、合法性校验、复杂度控制、表达式去重。
5. 在线计算表达式，不做重缓存。
6. 批量计算 IC、RankIC、ICIR、覆盖率、换手 proxy、年度/分段稳定性。
7. 记录所有表达式测试结果。
8. 预留二阶段优化接口。

## 快速运行：模拟数据

```bash
cd alpha_factor_mvp
python -m src.alpha_mvp.cli --use-simulated --out outputs/demo --max-exprs 300
```

## 真实数据运行示例

Windows PowerShell:

```powershell
python -m src.alpha_mvp.cli `
  --duckdb "D:\Trading\data_ever_26_3_14\data\meta\warehouse.duckdb" `
  --pool-json "D:\Trading\data_ever_26_3_14\static_pool_200.json" `
  --start 20240101 `
  --end 20260430 `
  --out outputs/real_200 `
  --max-exprs 5000
```

如果你的 DuckDB 表名不是代码默认的 `silver.fact_stock_daily` 等，可以在 `src/alpha_mvp/data.py` 的 `TABLE_CANDIDATES` 中修改。

## 项目结构

```text
src/alpha_mvp/
  config.py          全局配置
  fields.py          30 个基础字段构建
  ops.py             算子实现
  parser.py          表达式解析
  validator.py       合法性校验
  grammar.py         typed grammar + 模板枚举
  data.py            DuckDB / 模拟数据加载
  metrics.py         IC/RankIC/ICIR 等
  evaluator.py       在线批量 evaluator
  pipeline.py        主流程
  cli.py             命令行
tests/
  test_smoke.py
```

## 当前设计边界

- 第一阶段不做 GP / PySR / Optuna。
- 暂不硬编码 AlphaPROBE 的 feature enum；字段由配置和数据列动态决定。
- 表达式合法性借鉴 AlphaPROBE 的表达式类别思想，但采用自研 typed DSL。
- 初期主要在线计算；只保存表达式、指标和失败原因。
