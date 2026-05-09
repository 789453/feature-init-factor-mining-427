"""
重绘验证结果图表脚本 - 不重新计算，只读取缓存数据并生成可视化
使用 src.alpha_mvp.validation.plots 模块
输出到 reports 目录
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

OUT_DIR = Path("d:/Trading/My_factor_mining_427/outputs/validation_phase2_fine")
REPORTS_DIR = OUT_DIR / "reports"


def load_metrics():
    summary = pd.read_csv(OUT_DIR / "metrics" / "summary.csv")
    rolling_ic = pd.read_parquet(OUT_DIR / "metrics" / "rolling_ic.parquet")
    group_size = pd.read_csv(OUT_DIR / "metrics" / "group_size.csv")
    group_ind = pd.read_csv(OUT_DIR / "metrics" / "group_industry.csv")
    return summary, rolling_ic, group_size, group_ind


def load_vectorbot():
    vb_dir = OUT_DIR / "vectorbot"
    if not vb_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    summary = pd.read_csv(vb_dir / "portfolio_summary.csv")
    equity = pd.read_parquet(vb_dir / "equity_curves.parquet")
    return summary, equity


def main():
    from src.alpha_mvp.validation.plots import (
        plot_rolling_ic_all,
        plot_equity_all,
        plot_top10_equity_comparison,
        plot_size_bucket,
        plot_industry,
        plot_ic_heatmap,
        plot_factor_ic_distribution,
        plot_alphalens_from_cache,
        plot_vectorbot_summary,
    )

    print("Loading cached data...")
    summary, rolling_ic, group_size, group_ind = load_metrics()
    vb_summary, equity = load_vectorbot()

    print(f"  Loaded {len(summary)} factors, {len(rolling_ic)} IC records")
    print(f"  VectorBot: {len(vb_summary)} portfolios, {len(equity)} equity records")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")

    print("  - Rolling IC plots (factor-specific)...")
    plot_rolling_ic_all(rolling_ic, REPORTS_DIR)

    print("  - Equity curves (factor-specific)...")
    plot_equity_all(equity, REPORTS_DIR)

    print("  - Equity comparison Top 10 (global)...")
    plot_top10_equity_comparison(summary, equity, REPORTS_DIR)

    print("  - Size bucket plot (global)...")
    plot_size_bucket(group_size, REPORTS_DIR)

    print("  - Industry plot (global)...")
    plot_industry(group_ind, REPORTS_DIR)

    print("  - IC heatmap (global)...")
    plot_ic_heatmap(rolling_ic, REPORTS_DIR)

    print("  - IC distribution (global)...")
    plot_factor_ic_distribution(rolling_ic, REPORTS_DIR)

    print("  - VectorBot Portfolio Summary (global)...")
    plot_vectorbot_summary(vb_summary, REPORTS_DIR)

    print("  - Alphalens from cache (factor-specific)...")
    plot_alphalens_from_cache(REPORTS_DIR)

    print(f"\nAll plots saved to {REPORTS_DIR}")
    print(f"Global plots: {len(list(REPORTS_DIR.glob('*.html')))}")
    print(f"Factor plots: {len(list((REPORTS_DIR / 'plots').glob('*.html')))}")


if __name__ == "__main__":
    main()
