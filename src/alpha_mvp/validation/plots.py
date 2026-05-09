from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def plot_equity_curve(equity_df: pd.DataFrame, out_dir: Path) -> None:
    if equity_df.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for fid in equity_df["factor_id"].unique():
        df = equity_df[equity_df["factor_id"] == fid]
        for tq in df["top_pct"].unique():
            dff = df[df["top_pct"] == tq]
            fig = px.line(dff, x="date", y="equity", title=f"{fid} Top{tq*100:.0f}% Equity Curve")
            fig.write_html(plot_dir / f"{fid}_top{tq}_equity.html")

def plot_rolling_ic(rolling_ic_df: pd.DataFrame, out_dir: Path) -> None:
    if rolling_ic_df.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for fid in rolling_ic_df["factor_id"].unique():
        df = rolling_ic_df[rolling_ic_df["factor_id"] == fid].sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["date"], y=df["rank_ic"], name="Daily RankIC", opacity=0.5))
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_20"], name="Rolling 20"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["rolling_60"], name="Rolling 60"))
        fig.update_layout(title=f"{fid} Rolling RankIC")
        fig.write_html(plot_dir / f"{fid}_rolling_ic.html")

def plot_rolling_ic_all(rolling_ic: pd.DataFrame, out_dir: Path) -> None:
    if rolling_ic.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

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
        fig.write_html(plot_dir / f"{fid}_rolling_ic.html")

def plot_equity_all(equity: pd.DataFrame, out_dir: Path) -> None:
    if equity.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

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
        fig.write_html(plot_dir / f"{fid}_equity.html")

def plot_top10_equity_comparison(factor_metrics: pd.DataFrame, equity_df: pd.DataFrame, out_dir: Path) -> None:
    if equity_df.empty or factor_metrics.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    score_col = "test_sharpe" if "test_sharpe" in factor_metrics.columns else "test_mean_rank_ic"
    top_fids = factor_metrics.nlargest(10, score_col)["factor_id"].tolist()

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, fid in enumerate(top_fids):
        df = equity_df[(equity_df["factor_id"] == fid) & (equity_df["top_pct"] == 0.1)].sort_values("date")
        if not df.empty:
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["equity"],
                name=fid.split("_")[0],
                mode="lines",
                line=dict(color=colors[i % len(colors)])
            ))

    fig.update_layout(
        height=600,
        width=1200,
        title=dict(text="Top 10 Factors - Equity Curve Comparison (Top 10%)", x=0.5),
        xaxis_title="Date",
        yaxis_title="Equity",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.write_html(out_dir / "top10_equity_comparison.html")

def plot_size_bucket(size_df: pd.DataFrame, out_dir: Path) -> None:
    if size_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = px.bar(size_df, x="bucket", y="mean_rank_ic", error_y=size_df["std_rank_ic"],
                 color=size_df["mean_rank_ic"], color_continuous_scale="RdYlGn",
                 title="Size Bucket RankIC")
    fig.update_layout(height=350)
    fig.write_html(out_dir / "size_bucket.html")

def plot_industry(industry_df: pd.DataFrame, out_dir: Path) -> None:
    if industry_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    top_industries = industry_df.nlargest(20, "mean_rank_ic")
    fig = px.bar(top_industries, x="industry", y="mean_rank_ic",
                 color="mean_rank_ic", color_continuous_scale="RdYlGn",
                 title="Top 20 Industries by Mean RankIC")
    fig.update_layout(height=400, xaxis_tickangle=-45)
    fig.write_html(out_dir / "industry_top20.html")

def plot_ic_heatmap(rolling_ic: pd.DataFrame, out_dir: Path) -> None:
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

def plot_vectorbot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
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

def plot_factor_ic_distribution(rolling_ic: pd.DataFrame, out_dir: Path) -> None:
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

def plot_alphalens_from_cache(out_dir: Path) -> None:
    al_dir = out_dir / "alphalens"
    if not al_dir.exists():
        al_dir = out_dir.parent / "alphalens"
    if not al_dir.exists():
        return

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

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

            plot_alphalens_factor(clean, fid, plot_dir)
            plot_alphalens_quantile_returns(clean, fid, plot_dir)
        except Exception as e:
            print(f"Failed to process {fid}: {e}")

def plot_alphalens_factor(clean: pd.DataFrame, fid: str, out_dir: Path) -> None:
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

def plot_alphalens_quantile_returns(clean: pd.DataFrame, fid: str, out_dir: Path) -> None:
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

def generate_factor_plots(factor_metrics: pd.DataFrame, rolling_ic: pd.DataFrame,
                         equity: pd.DataFrame, size_df: pd.DataFrame,
                         industry_df: pd.DataFrame, out_dir: str) -> None:
    out_path = Path(out_dir)
    plot_dir = out_path / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path.mkdir(parents=True, exist_ok=True)

    plot_rolling_ic_all(rolling_ic, out_path)
    plot_equity_all(equity, out_path)
    plot_alphalens_from_cache(out_path)

    plot_top10_equity_comparison(factor_metrics, equity, out_path)
    plot_size_bucket(size_df, out_path)
    plot_industry(industry_df, out_path)
    plot_ic_heatmap(rolling_ic, out_path)
    plot_factor_ic_distribution(rolling_ic, out_path)