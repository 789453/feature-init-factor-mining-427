"""
重绘验证结果图表脚本 - 不重新计算，只读取缓存数据并生成可视化
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json


OUT_DIR = Path("d:/Trading/My_factor_mining_427/outputs/validation_full")
PLOT_DIR = OUT_DIR / "plots"


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


def load_factor_cache():
    cache_dir = OUT_DIR / "cache" / "factor_panels"
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    factors = {}
    for item in manifest.get("factors", []):
        fid = item["factor_id"]
        p = cache_dir / f"{fid}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            factors[fid] = df
    return factors


def plot_rolling_ic_all(rolling_ic: pd.DataFrame, out_dir: Path):
    if rolling_ic.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for fid in rolling_ic["factor_id"].unique():
        df = rolling_ic[rolling_ic["factor_id"] == fid].sort_values("date")
        if df.empty:
            continue

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.08,
                           row_heights=[0.6, 0.4],
                           subplot_titles=(f"{fid} Rolling RankIC", "Daily RankIC"))

        fig.add_trace(go.Scatter(x=df["date"], y=df["rank_ic"],
                                 name="Daily RankIC", opacity=0.4,
                                 line=dict(color="lightblue")),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_20"],
                                 name="Rolling 20", line=dict(color="blue")),
                    row=1, col=1)
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_60"],
                                 name="Rolling 60", line=dict(color="red")),
                    row=1, col=1)

        fig.add_trace(go.Bar(x=df["date"], y=df["rank_ic"].clip(-0.3, 0.3),
                             name="Daily IC", marker_color=np.where(df["rank_ic"] > 0, "green", "red"),
                             opacity=0.5),
                    row=2, col=1)

        fig.update_layout(height=500, showlegend=True,
                         title=dict(text=fid, x=0.5))
        fig.write_html(out_dir / f"{fid}_rolling_ic.html")


def plot_equity_all(equity: pd.DataFrame, out_dir: Path):
    if equity.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    for fid in equity["factor_id"].unique():
        df = equity[equity["factor_id"] == fid].sort_values("date")
        if df.empty:
            continue

        fig = go.Figure()
        colors = {0.05: "blue", 0.1: "red", 0.2: "green"}
        for tq in sorted(df["top_pct"].unique()):
            dff = df[df["top_pct"] == tq]
            fig.add_trace(go.Scatter(
                x=dff["date"], y=dff["equity"],
                name=f"Top {tq*100:.0f}%",
                line=dict(color=colors.get(tq, "gray"))
            ))

        fig.update_layout(
            height=400,
            title=dict(text=f"{fid} Equity Curves", x=0.5),
            xaxis_title="Date",
            yaxis_title="Equity (1=initial)",
            showlegend=True
        )
        fig.write_html(out_dir / f"{fid}_equity.html")


def plot_equity_comparison(equity: pd.DataFrame, summary: pd.DataFrame, out_dir: Path):
    if equity.empty or summary.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    top_fids = summary.nlargest(10, "test_sharpe")["factor_id"].tolist()

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    for i, fid in enumerate(top_fids):
        df = equity[(equity["factor_id"] == fid) & (equity["top_pct"] == 0.1)].sort_values("date")
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["equity"],
                name=fid.split("_")[0], mode="lines",
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        height=500,
        title=dict(text="Top 10 Factors - Equity Curve Comparison (Top 10%)", x=0.5),
        xaxis_title="Date",
        yaxis_title="Equity",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html(out_dir / "top10_equity_comparison.html")


def plot_size_bucket(size_df: pd.DataFrame, out_dir: Path):
    if size_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = px.bar(size_df, x="bucket", y="mean_rank_ic", error_y=size_df["std_rank_ic"],
                 color=size_df["mean_rank_ic"], color_continuous_scale="RdYlGn",
                 title="Size Bucket RankIC")
    fig.update_layout(height=350)
    fig.write_html(out_dir / "size_bucket.html")


def plot_industry_top(industry_df: pd.DataFrame, out_dir: Path):
    if industry_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    top = industry_df.nlargest(20, "mean_rank_ic")
    fig = px.bar(top, x="industry", y="mean_rank_ic",
                 color="mean_rank_ic", color_continuous_scale="RdYlGn",
                 title="Top 20 Industries by Mean RankIC")
    fig.update_layout(height=400, xaxis_tickangle=-45)
    fig.write_html(out_dir / "industry_top20.html")


def plot_ic_heatmap(rolling_ic: pd.DataFrame, out_dir: Path):
    if rolling_ic.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    pivot = rolling_ic.pivot_table(values="rank_ic", index="factor_id", columns="date")

    top_fids = rolling_ic.groupby("factor_id")["rank_ic"].mean().nlargest(20).index.tolist()
    pivot_top = pivot.loc[pivot.index.isin(top_fids)]

    fig = go.Figure(data=go.Heatmap(
        z=pivot_top.values,
        x=pivot_top.columns,
        y=pivot_top.index,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="RankIC")
    ))
    fig.update_layout(height=600, title="Top 20 Factors - IC Heatmap (Date x Factor)")
    fig.write_html(out_dir / "ic_heatmap_top20.html")


def plot_vectorbot_summary(summary: pd.DataFrame, out_dir: Path):
    if summary.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(rows=2, cols=2,
                       subplot_titles=("Annual Return vs Max Drawdown", "Sharpe Distribution",
                                      "Turnover vs Win Rate", "Train vs Test Return"),
                       specs=[[{"type": "scatter"}, {"type": "histogram"}],
                              [{"type": "scatter"}, {"type": "scatter"}]])

    df = summary[summary["top_pct"] == 0.1].copy()

    fig.add_trace(go.Scatter(x=df["ann_return"], y=df["max_drawdown"],
                             text=df["factor_id"], mode="markers",
                             marker=dict(color=df["sharpe"], colorscale="Viridis", size=8)),
                 row=1, col=1)

    fig.add_trace(go.Histogram(x=df["sharpe"], name="Sharpe", marker_color="lightblue"),
                 row=1, col=2)

    fig.add_trace(go.Scatter(x=df["avg_turnover"], y=df["win_rate"],
                             text=df["factor_id"], mode="markers"),
                 row=2, col=1)

    fig.add_trace(go.Scatter(x=df["train_ann_return"], y=df["test_ann_return"],
                             text=df["factor_id"], mode="markers",
                             marker=dict(color=df["sharpe"], colorscale="Viridis")),
                 row=2, col=2)

    fig.update_layout(height=700, showlegend=False,
                     title=dict(text="VectorBot Portfolio Summary (Top 10%)", x=0.5))
    fig.write_html(out_dir / "vectorbot_summary.html")


def plot_factor_ic_distribution(rolling_ic: pd.DataFrame, out_dir: Path):
    if rolling_ic.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    ic_stats = rolling_ic.groupby("factor_id").agg({
        "rank_ic": ["mean", "std", lambda x: (x > 0).mean()]
    }).reset_index()
    ic_stats.columns = ["factor_id", "ic_mean", "ic_std", "positive_ratio"]

    fig = make_subplots(rows=1, cols=3,
                       subplot_titles=("IC Mean Distribution", "IC Std Distribution", "Positive IC Ratio"))

    fig.add_trace(go.Histogram(x=ic_stats["ic_mean"], marker_color="steelblue", name="IC Mean"),
                 row=1, col=1)
    fig.add_trace(go.Histogram(x=ic_stats["ic_std"], marker_color="orange", name="IC Std"),
                 row=1, col=2)
    fig.add_trace(go.Histogram(x=ic_stats["positive_ratio"], marker_color="green", name="Positive Ratio"),
                 row=1, col=3)

    fig.update_layout(height=350, showlegend=False,
                     title=dict(text="Factor IC Statistics Distribution", x=0.5))
    fig.write_html(out_dir / "ic_distribution.html")


def plot_alphalens_from_cache(out_dir: Path):
    al_dir = out_dir / "alphalens"
    if not al_dir.exists():
        return

    for fid_dir in sorted(al_dir.iterdir()):
        if not fid_dir.is_dir():
            continue
        fid = fid_dir.name
        clean_path = fid_dir / "clean_factor_data.parquet"
        if not clean_path.exists():
            continue

        try:
            clean = pd.read_parquet(clean_path)
            if clean.empty:
                continue

            plot_alphalens_factor(clean, fid, out_dir / "plots")
            plot_alphalens_quantile_returns(clean, fid, out_dir / "plots")
        except Exception as e:
            print(f"Failed to process {fid}: {e}")


def plot_alphalens_factor(clean: pd.DataFrame, fid: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = clean.index.get_level_values("date").unique()
    mean_ic_by_date = clean.groupby("date")["factor"].corr(clean["1D"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       vertical_spacing=0.08,
                       row_heights=[0.6, 0.4],
                       subplot_titles=(f"{fid} Factor Values", "Forward Return IC"))

    factor_vals = clean["factor"].unstack()
    if not factor_vals.empty:
        factor_mean = factor_vals.mean(axis=1)
        fig.add_trace(go.Scatter(x=factor_mean.index, y=factor_mean.values,
                                 name="Mean Factor", line=dict(color="blue")),
                     row=1, col=1)

    fig.add_trace(go.Scatter(x=mean_ic_by_date.index, y=mean_ic_by_date.values,
                             name="IC (1D)", opacity=0.5, line=dict(color="green")),
                 row=2, col=1)

    fig.update_layout(height=500, showlegend=True,
                     title=dict(text=f"{fid} - Alphalens Analysis", x=0.5))
    fig.write_html(out_dir / f"{fid}_alphalens.html")


def plot_alphalens_quantile_returns(clean: pd.DataFrame, fid: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    qt_returns = clean.groupby(["date", "factor_quantile"])["1D"].mean().reset_index()
    qt_pivot = qt_returns.pivot(index="date", columns="factor_quantile", values="1D")

    if qt_pivot.empty or len(qt_pivot.columns) < 2:
        return

    fig = go.Figure()
    colors = ["red", "orange", "gray", "lightgreen", "green"]
    for i, q in enumerate(sorted(qt_pivot.columns)):
        cum_ret = (1 + qt_pivot[q]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum_ret.index, y=cum_ret.values,
            name=f"Q{int(q)}",
            line=dict(color=colors[i] if i < len(colors) else "gray")
        ))

    fig.update_layout(
        height=400,
        title=dict(text=f"{fid} - Quantile Cumulative Returns", x=0.5),
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        showlegend=True
    )
    fig.write_html(out_dir / f"{fid}_quantile_returns.html")


def main():
    print("Loading cached data...")
    summary, rolling_ic, group_size, group_ind = load_metrics()
    vb_summary, equity = load_vectorbot()

    print(f"  Loaded {len(summary)} factors, {len(rolling_ic)} IC records")
    print(f"  VectorBot: {len(vb_summary)} portfolios, {len(equity)} equity records")

    print("Generating plots...")

    print("  - Rolling IC plots...")
    plot_rolling_ic_all(rolling_ic, PLOT_DIR)

    print("  - Equity curves...")
    plot_equity_all(equity, PLOT_DIR)

    print("  - Equity comparison...")
    plot_equity_comparison(equity, vb_summary, PLOT_DIR)

    print("  - Size bucket plot...")
    plot_size_bucket(group_size, PLOT_DIR)

    print("  - Industry plot...")
    plot_industry_top(group_ind, PLOT_DIR)

    print("  - IC heatmap...")
    plot_ic_heatmap(rolling_ic, PLOT_DIR)

    print("  - VectorBot summary...")
    plot_vectorbot_summary(vb_summary, PLOT_DIR)

    print("  - IC distribution...")
    plot_factor_ic_distribution(rolling_ic, PLOT_DIR)

    print("  - Alphalens from cache...")
    plot_alphalens_from_cache(OUT_DIR)

    print(f"\nAll plots saved to {PLOT_DIR}")
    print(f"Total HTML files: {len(list(PLOT_DIR.glob('*.html')))}")


if __name__ == "__main__":
    main()
