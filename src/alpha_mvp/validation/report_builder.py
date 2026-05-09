from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime

def build_validation_report(out_dir: str, cfg, summary_metrics, group_size, group_industry) -> None:
    out_path = Path(out_dir)
    reports_dir = out_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_info = {
        "duckdb_path": cfg.duckdb_path,
        "top100_path": cfg.top100_path,
        "start": cfg.start,
        "end": cfg.end,
        "horizon": cfg.horizon,
        "rebalance_n": cfg.rebalance_n,
        "run_time": datetime.now().isoformat(),
        "n_factors": len(summary_metrics),
    }

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Factor Validation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; border-left: 4px solid #007bff; padding-left: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #007bff; color: white; }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        .metric {{ font-weight: bold; color: #007bff; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
        .summary-box {{ background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 15px 0; }}
        .warning {{ background: #fff3cd; padding: 10px; border-radius: 5px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>因子验证报告</h1>
    <div class="summary-box">
        <h3>运行信息</h3>
        <p><strong>数据源:</strong> {cfg.duckdb_path}</p>
        <p><strong>因子池:</strong> {cfg.top100_path}</p>
        <p><strong>时间范围:</strong> {cfg.start} - {cfg.end}</p>
        <p><strong>预测周期:</strong> {cfg.horizon} 日</p>
        <p><strong>调仓周期:</strong> {cfg.rebalance_n} 日</p>
        <p><strong>因子数量:</strong> {run_info['n_factors']}</p>
        <p><strong>生成时间:</strong> {run_info['run_time']}</p>
    </div>

    <h2>1. IC/RankIC 统计</h2>
    <table>
        <tr>
            <th>因子ID</th>
            <th>Train RankIC</th>
            <th>Test RankIC</th>
            <th>RankICIR</th>
            <th>Score</th>
        </tr>
"""

    for _, row in summary_metrics.iterrows():
        fid = row.get("factor_id", "N/A")
        train_ric = row.get("train_mean_rank_ic", 0)
        test_ric = row.get("test_mean_rank_ic", 0)
        rank_icir = row.get("rank_icir", 0)
        score = row.get("score", 0)

        train_class = "positive" if train_ric > 0 else "negative"
        test_class = "positive" if test_ric > 0 else "negative"

        html += f"""        <tr>
            <td>{fid}</td>
            <td class="{train_class}">{train_ric:.4f}</td>
            <td class="{test_class}">{test_ric:.4f}</td>
            <td>{rank_icir:.4f}</td>
            <td class="metric">{score:.6f}</td>
        </tr>
"""

    html += """    </table>

    <h2>2. 市值分组表现</h2>
    <table>
        <tr>
            <th>市值桶</th>
            <th>Mean RankIC</th>
            <th>Std</th>
            <th>正IC比例</th>
        </tr>
"""

    for _, row in group_size.iterrows():
        html += f"""        <tr>
            <td>{row.get('bucket', 'N/A')}</td>
            <td>{row.get('mean_rank_ic', 0):.4f}</td>
            <td>{row.get('std_rank_ic', 0):.4f}</td>
            <td>{row.get('positive_ratio', 0):.2%}</td>
        </tr>
"""

    html += """    </table>

    <h2>3. 行业分组表现 (Top 20)</h2>
    <table>
        <tr>
            <th>行业</th>
            <th>Mean RankIC</th>
            <th>样本数</th>
        </tr>
"""

    top_ind = group_industry.head(20)
    for _, row in top_ind.iterrows():
        html += f"""        <tr>
            <td>{row.get('industry', 'N/A')}</td>
            <td>{row.get('mean_rank_ic', 0):.4f}</td>
            <td>{row.get('n_samples', 0)}</td>
        </tr>
"""

    html += """    </table>

    <h2>4. 可视化链接</h2>
    <div class="summary-box">
        <p>交互式图表位于 <code>plots/</code> 目录:</p>
        <ul>
            <li>Equity Curves: <code>plots/*_equity.html</code></li>
            <li>Rolling IC: <code>plots/*_rolling_ic.html</code></li>
            <li>Size Bucket: <code>plots/size_bucket.html</code></li>
            <li>Industry: <code>plots/industry_top20.html</code></li>
        </ul>
    </div>

    <h2>5. Alphalens 诊断</h2>
    <div class="summary-box">
        <p>Alphalens tear sheets 位于 <code>alphalens/</code> 目录，每个因子有:</p>
        <ul>
            <li><code>summary_tear_sheet.png</code></li>
            <li><code>returns_tear_sheet.png</code></li>
            <li><code>information_tear_sheet.png</code></li>
        </ul>
    </div>

    <h2>6. 评估标准检查</h2>
    <div class="summary-box">
        <p>推荐筛选条件:</p>
        <ul>
            <li>abs(test_rank_ic) > 0.015</li>
            <li>abs(test_rank_icir) > 0.15</li>
            <li>train/test same sign</li>
            <li>行业 positive ratio > 50%</li>
        </ul>
    </div>

</div>
</body>
</html>"""

    report_path = reports_dir / "validation_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Validation report generated: {report_path}")