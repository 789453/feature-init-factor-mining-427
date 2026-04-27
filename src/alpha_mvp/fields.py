from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-9

def safe_div(a, b):
    return a / np.where(np.abs(b) < EPS, np.nan, b)

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ts_code", "trade_date"]).copy()
    g = df.groupby("ts_code", sort=False)

    df["ret_1d"] = g["close"].pct_change()
    df["hl_range"] = safe_div(df["high"] - df["low"], df["pre_close"])
    df["oc_ret"] = safe_div(df["close"] - df["open"], df["open"])
    df["upper_shadow"] = safe_div(df["high"] - np.maximum(df["open"], df["close"]), df["pre_close"])
    df["lower_shadow"] = safe_div(np.minimum(df["open"], df["close"]) - df["low"], df["pre_close"])
    df["close_pos"] = safe_div(df["close"] - df["low"], df["high"] - df["low"])
    df["gap_ret"] = safe_div(df["open"] - df["pre_close"], df["pre_close"])
    df["intraday_reversal"] = -safe_div(df["close"] - df["open"], df["high"] - df["low"])

    df["amount_log"] = np.log1p(df.get("amount", np.nan))
    df["vol_log"] = np.log1p(df.get("vol", np.nan))
    df["turnover_log"] = np.log1p(df["turnover_rate"]) if "turnover_rate" in df else np.nan
    df["amount_per_vol"] = safe_div(df.get("amount", np.nan), df.get("vol", np.nan))
    df["price_volume_pressure"] = df["ret_1d"] * df["vol_log"]
    df["amplitude_turnover"] = df["hl_range"] * df["turnover_log"]
    df["volume_ratio_x_ret"] = df.get("volume_ratio", np.nan) * df["ret_1d"]

    sm_buy = df.get("buy_sm_amount", np.nan)
    sm_sell = df.get("sell_sm_amount", np.nan)
    md_buy = df.get("buy_md_amount", np.nan)
    md_sell = df.get("sell_md_amount", np.nan)
    lg_buy = df.get("buy_lg_amount", np.nan)
    lg_sell = df.get("sell_lg_amount", np.nan)

    df["sm_net_ratio"] = safe_div(sm_buy - sm_sell, sm_buy + sm_sell)
    df["md_net_ratio"] = safe_div(md_buy - md_sell, md_buy + md_sell)
    df["lg_net_ratio"] = safe_div(lg_buy - lg_sell, lg_buy + lg_sell)
    df["main_net_ratio"] = safe_div((md_buy + lg_buy) - (md_sell + lg_sell), md_buy + lg_buy + md_sell + lg_sell)
    df["retail_pressure"] = -df["sm_net_ratio"]
    df["big_vs_small_flow"] = df["lg_net_ratio"] - df["sm_net_ratio"]
    df["flow_imbalance"] = safe_div(df.get("net_mf_amount", np.nan), df.get("amount", np.nan))
    df["large_order_intensity"] = safe_div(lg_buy + lg_sell, sm_buy + sm_sell + md_buy + md_sell + lg_buy + lg_sell)
    df["active_big_buy_pressure"] = safe_div(lg_buy, lg_buy + lg_sell) - 0.5

    df["chip_width_90"] = safe_div(df.get("cost_95pct", np.nan) - df.get("cost_5pct", np.nan), df.get("cost_50pct", np.nan))
    df["chip_width_70"] = safe_div(df.get("cost_85pct", np.nan) - df.get("cost_15pct", np.nan), df.get("cost_50pct", np.nan))
    df["chip_cost_bias"] = safe_div(df["close"] - df.get("weight_avg", np.nan), df.get("weight_avg", np.nan))
    df["chip_median_bias"] = safe_div(df["close"] - df.get("cost_50pct", np.nan), df.get("cost_50pct", np.nan))
    df["chip_upper_pressure"] = safe_div(df.get("cost_95pct", np.nan) - df["close"], df["close"])
    df["chip_lower_support"] = safe_div(df["close"] - df.get("cost_5pct", np.nan), df["close"])
    df["winner_rate_norm"] = df.get("winner_rate", np.nan) / 100.0
    df["hist_price_position"] = safe_div(df["close"] - df.get("his_low", np.nan), df.get("his_high", np.nan) - df.get("his_low", np.nan))
    df["winner_cost_divergence"] = df["winner_rate_norm"] - df["chip_cost_bias"]

    df["size_log"] = np.log1p(df.get("circ_mv", np.nan))
    df["free_turnover_gap"] = df.get("turnover_rate_f", np.nan) - df.get("turnover_rate", np.nan)
    df["liquidity_crowding"] = df["turnover_log"] * df.get("volume_ratio", np.nan)

    for c in DEFAULT_FEATURES:
        if c in df:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
    return df

DEFAULT_FEATURES = [
    "ret_1d", "hl_range", "oc_ret", "close_pos", "gap_ret", "intraday_reversal",
    "amount_log", "vol_log", "turnover_log", "price_volume_pressure", "amplitude_turnover",
    "volume_ratio_x_ret", "sm_net_ratio", "md_net_ratio", "lg_net_ratio", "main_net_ratio",
    "retail_pressure", "big_vs_small_flow", "flow_imbalance", "large_order_intensity",
    "active_big_buy_pressure", "chip_width_90", "chip_width_70", "chip_cost_bias",
    "chip_median_bias", "chip_upper_pressure", "chip_lower_support", "winner_rate_norm",
    "hist_price_position", "winner_cost_divergence"
]
