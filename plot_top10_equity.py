"""
临时脚本：重新绘制 Top10 Equity Comparison 图
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

OUT_DIR = Path("d:/Trading/My_factor_mining_427/outputs/validation_full")
PLOT_DIR = OUT_DIR / "plots"

def main():
    print("Loading data...")
    vb_dir = OUT_DIR / "vectorbot"
    summary = pd.read_csv(vb_dir / "portfolio_summary.csv")
    equity = pd.read_parquet(vb_dir / "equity_curves.parquet")

    print(f"Summary shape: {summary.shape}")
    print(f"Equity shape: {equity.shape}")
    print(f"Columns in summary: {summary.columns.tolist()}")
    print(f"Columns in equity: {equity.columns.tolist()}")
    print(f"Unique top_pct values: {equity['top_pct'].unique()}")

    top_fids = summary.nlargest(10, "test_sharpe")["factor_id"].tolist()
    print(f"Top 10 factor IDs: {top_fids}")

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, fid in enumerate(top_fids):
        df = equity[(equity["factor_id"] == fid) & (equity["top_pct"] == 0.1)].sort_values("date")
        print(f"  {fid}: {len(df)} records")
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

    output_path = PLOT_DIR / "top10_equity_comparison.html"
    fig.write_html(output_path)
    print(f"\nSaved to {output_path}")
    print(f"Number of traces in figure: {len(fig.data)}")

if __name__ == "__main__":
    main()
