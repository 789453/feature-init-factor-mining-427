from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

def plot_size_bucket(size_df: pd.DataFrame, out_dir: Path) -> None:
    if size_df.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig = px.bar(size_df, x="bucket", y="mean_rank_ic", error_y=size_df["std_rank_ic"],
                  title="Size Bucket RankIC")
    fig.write_html(plot_dir / "size_bucket.html")

def plot_industry(industry_df: pd.DataFrame, out_dir: Path) -> None:
    if industry_df.empty:
        return
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    top_industries = industry_df.nlargest(20, "mean_rank_ic")
    fig = px.bar(top_industries, x="industry", y="mean_rank_ic", title="Top 20 Industries by RankIC")
    fig.write_html(plot_dir / "industry_top20.html")

def generate_factor_plots(factor_metrics: pd.DataFrame, rolling_ic: pd.DataFrame,
                         equity: pd.DataFrame, size_df: pd.DataFrame,
                         industry_df: pd.DataFrame, out_dir: str) -> None:
    out_path = Path(out_dir)
    plot_equity_curve(equity, out_path)
    plot_rolling_ic(rolling_ic, out_path)
    plot_size_bucket(size_df, out_path)
    plot_industry(industry_df, out_path)