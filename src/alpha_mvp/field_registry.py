from __future__ import annotations
import pandas as pd
import hashlib
import json

# 定义字段注册表版本
FIELD_FORMULA_VERSION = "2026-05-07-v1"

# 基础特征注册表
FIELD_REGISTRY = {
    "ret_1d": {"version": "v1", "deps": ["close"], "group": "price"},
    "hl_range": {"version": "v1", "deps": ["high", "low", "pre_close"], "group": "price"},
    "oc_ret": {"version": "v1", "deps": ["open", "close"], "group": "price"},
    "upper_shadow": {"version": "v1", "deps": ["high", "open", "close", "pre_close"], "group": "price"},
    "lower_shadow": {"version": "v1", "deps": ["low", "open", "close", "pre_close"], "group": "price"},
    "close_pos": {"version": "v1", "deps": ["high", "low", "close"], "group": "price"},
    "gap_ret": {"version": "v1", "deps": ["open", "pre_close"], "group": "price"},
    "intraday_reversal": {"version": "v1", "deps": ["open", "close", "high", "low"], "group": "price"},
    "amount_log": {"version": "v1", "deps": ["amount"], "group": "volume"},
    "vol_log": {"version": "v1", "deps": ["vol"], "group": "volume"},
    "turnover_log": {"version": "v1", "deps": ["turnover_rate"], "group": "volume"},
    "amount_per_vol": {"version": "v1", "deps": ["amount", "vol"], "group": "volume"},
    "price_volume_pressure": {"version": "v1", "deps": ["ret_1d", "vol_log"], "group": "mixed"},
    "amplitude_turnover": {"version": "v1", "deps": ["hl_range", "turnover_log"], "group": "mixed"},
    "volume_ratio_x_ret": {"version": "v1", "deps": ["volume_ratio", "ret_1d"], "group": "mixed"},
    "sm_net_ratio": {"version": "v1", "deps": ["buy_sm_amount", "sell_sm_amount"], "group": "flow"},
    "md_net_ratio": {"version": "v1", "deps": ["buy_md_amount", "sell_md_amount"], "group": "flow"},
    "lg_net_ratio": {"version": "v1", "deps": ["buy_lg_amount", "sell_lg_amount"], "group": "flow"},
    "main_net_ratio": {"version": "v1", "deps": ["buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount"], "group": "flow"},
    "retail_pressure": {"version": "v1", "deps": ["sm_net_ratio"], "group": "flow"},
    "big_vs_small_flow": {"version": "v1", "deps": ["lg_net_ratio", "sm_net_ratio"], "group": "flow"},
    "flow_imbalance": {"version": "v1", "deps": ["net_mf_amount", "amount"], "group": "flow"},
    "large_order_intensity": {"version": "v1", "deps": ["buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount", "buy_lg_amount", "sell_lg_amount"], "group": "flow"},
    "active_big_buy_pressure": {"version": "v1", "deps": ["buy_lg_amount", "sell_lg_amount"], "group": "flow"},
    "chip_width_90": {"version": "v1", "deps": ["cost_95pct", "cost_5pct", "cost_50pct"], "group": "chip"},
    "chip_width_70": {"version": "v1", "deps": ["cost_85pct", "cost_15pct", "cost_50pct"], "group": "chip"},
    "chip_cost_bias": {"version": "v1", "deps": ["close", "weight_avg"], "group": "chip"},
    "chip_median_bias": {"version": "v1", "deps": ["close", "cost_50pct"], "group": "chip"},
    "chip_upper_pressure": {"version": "v1", "deps": ["cost_95pct", "close"], "group": "chip"},
    "chip_lower_support": {"version": "v1", "deps": ["cost_5pct", "close"], "group": "chip"},
    "winner_rate_norm": {"version": "v1", "deps": ["winner_rate"], "group": "chip"},
    "hist_price_position": {"version": "v1", "deps": ["close", "his_low", "his_high"], "group": "chip"},
    "winner_cost_divergence": {"version": "v1", "deps": ["winner_rate_norm", "chip_cost_bias"], "group": "chip"},
    "size_log": {"version": "v1", "deps": ["circ_mv"], "group": "other"},
    "free_turnover_gap": {"version": "v1", "deps": ["turnover_rate_f", "turnover_rate"], "group": "other"},
    "liquidity_crowding": {"version": "v1", "deps": ["turnover_log", "volume_ratio"], "group": "other"},
}

def resolve_fields(available_columns: list[str], 
                   include: list[str] | None = None, 
                   exclude: list[str] | None = None, 
                   field_file: str | None = None) -> list[str]:
    """
    根据 include, exclude 和 field_file 解析最终要使用的字段列表
    """
    all_known_fields = set(FIELD_REGISTRY.keys())
    
    # 初始字段集：如果是 None，默认使用全部已知字段
    if include:
        selected = set(include)
    elif field_file:
        with open(field_file, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                selected = set(data)
            elif isinstance(data, dict) and "fields" in data:
                selected = set(data["fields"])
            else:
                selected = all_known_fields
    else:
        selected = all_known_fields
        
    # 排除逻辑
    if exclude:
        selected = selected - set(exclude)
        
    # 最终检查：必须是已知字段且在 available_columns 中（或者能通过 fields.py 构建）
    # 这里我们只检查是否是已知字段，具体构建在 pipeline 中由 add_basic_features 处理
    final_fields = [f for f in FIELD_REGISTRY.keys() if f in selected]
    
    return final_fields

def get_field_set_hash(fields: list[str]) -> str:
    """
    计算字段集的 hash，用于 run_signature
    """
    field_info = {f: FIELD_REGISTRY.get(f, {"version": "unknown"}) for f in sorted(fields)}
    content = json.dumps({
        "formula_version": FIELD_FORMULA_VERSION,
        "fields": field_info
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()
