from __future__ import annotations
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def panel_to_alphalens_series(arr, dates, codes):
    df = pd.DataFrame(
        arr,
        index=pd.to_datetime(dates, format="%Y%m%d"),
        columns=codes,
    )
    s = df.stack(dropna=False)
    s.index.names = ["date", "asset"]
    return s.sort_index()

def run_alphalens_for_factor(fid, factor_arr, dates, codes, close, industry_map, cfg, out_dir):
    al_out = Path(out_dir) / "alphalens" / fid
    al_out.mkdir(parents=True, exist_ok=True)

    factor = panel_to_alphalens_series(factor_arr, dates, codes)
    prices = pd.DataFrame(
        close,
        index=pd.to_datetime(dates, format="%Y%m%d"),
        columns=codes,
    )

    try:
        import alphalens as al

        clean = al.utils.get_clean_factor_and_forward_returns(
            factor=factor,
            prices=prices,
            groupby=industry_map,
            quantiles=cfg.alphalens_quantiles,
            periods=cfg.alphalens_periods,
            max_loss=cfg.alphalens_max_loss,
        )

        clean.to_parquet(al_out / "clean_factor_data.parquet")

        al.tears.create_summary_tear_sheet(clean)
        plt.savefig(al_out / "summary_tear_sheet.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        al.tears.create_returns_tear_sheet(clean)
        plt.savefig(al_out / "returns_tear_sheet.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        al.tears.create_information_tear_sheet(clean)
        plt.savefig(al_out / "information_tear_sheet.png", dpi=150, bbox_inches="tight")
        plt.close("all")

        return clean
    except Exception as e:
        print(f"Alphalens failed for {fid}: {e}")
        return None

def run_alphalens_batch(selected_fids, factor_panels, close, dates, codes, industry, cfg, out_dir):
    from .group_metrics import make_size_bucket

    code_to_industry = {}
    latest_industry = {}
    for t in range(industry.shape[0]):
        for j, code in enumerate(codes):
            if code not in latest_industry:
                val = industry[t, j]
                if val and str(val) != "":
                    latest_industry[code] = str(val)
    for code in codes:
        code_to_industry[code] = latest_industry.get(code, "UNKNOWN")

    results = []
    for fid in selected_fids:
        if fid not in factor_panels:
            continue
        factor = factor_panels[fid]
        clean = run_alphalens_for_factor(fid, factor, dates, codes, close, code_to_industry, cfg, out_dir)
        results.append({"factor_id": fid, "status": "OK" if clean is not None else "FAILED"})
    return pd.DataFrame(results)