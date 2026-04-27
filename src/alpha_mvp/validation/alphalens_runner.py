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

def run_alphalens_batch(selected_fids, factor_panels, close, dates, codes, meta, cfg):
    from .size_industry import get_industry_map
    industry_map = get_industry_map(meta, codes)

    results = []
    for fid in selected_fids:
        if fid not in factor_panels:
            continue
        factor = factor_panels[fid]
        clean = run_alphalens_for_factor(fid, factor, dates, codes, close, industry_map, cfg, cfg.out_dir)
        results.append({"factor_id": fid, "status": "OK" if clean is not None else "FAILED"})
    return pd.DataFrame(results)