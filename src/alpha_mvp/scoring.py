from __future__ import annotations
import numpy as np
import pandas as pd

def infer_dimension(train_rank_ic_series: np.ndarray) -> int:
    """
    根据训练集 RankIC 均值判断因子方向
    """
    med = np.nanmedian(train_rank_ic_series)
    if not np.isfinite(med) or abs(med) < 1e-6:
        return 0
    return 1 if med > 0 else -1

def apply_ranked_score(df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """
    对结果进行 Rank 之后再加权计算最终 Score
    """
    if df.empty:
        return df
        
    if weights is None:
        # 默认权重
        weights = {
            "r_edge": 0.30,        # oriented_test_mean_rank_ic
            "r_ir": 0.20,          # oriented_test_rank_icir
            "r_hit": 0.15,         # positive_oriented_rank_ic_ratio
            "r_stability": 0.15,   # yearly_positive_ratio
            "r_coverage": 0.10,    # coverage
            "r_turnover": 0.05,    # -turnover_proxy
            "r_complexity": 0.05,  # complexity_score
        }
    
    df = df.copy()
    
    # 确保 complexity_score 存在
    if "complexity_score" not in df and "nodes" in df:
        # 节点数越少，得分越高。假设最大节点数为 50
        df["complexity_score"] = 1.0 - (df["nodes"] / 50.0).clip(0, 1)
    
    # 计算 score_raw (原始指标的简单加权)
    df["score_raw"] = (
        df.get("oriented_test_mean_rank_ic", 0) * 10 + 
        df.get("oriented_test_rank_icir", 0) * 2 +
        df.get("yearly_positive_ratio", 0) * 5 +
        df.get("complexity_score", 0) * 2
    )

    # 计算各组件的 percentile rank
    if "oriented_test_mean_rank_ic" in df:
        df["r_edge"] = df["oriented_test_mean_rank_ic"].rank(pct=True)
    if "oriented_test_rank_icir" in df:
        df["r_ir"] = df["oriented_test_rank_icir"].rank(pct=True)
    if "positive_oriented_rank_ic_ratio" in df:
        df["r_hit"] = df["positive_oriented_rank_ic_ratio"].rank(pct=True)
    if "yearly_positive_ratio" in df:
        df["r_stability"] = df["yearly_positive_ratio"].rank(pct=True)
    if "coverage" in df:
        df["r_coverage"] = df["coverage"].rank(pct=True)
    if "turnover_proxy" in df:
        df["r_turnover"] = (-df["turnover_proxy"]).rank(pct=True)
    if "complexity_score" in df:
        df["r_complexity"] = df["complexity_score"].rank(pct=True)
    elif "nodes" in df:
        df["r_complexity"] = (-df["nodes"]).rank(pct=True)

    # 初始 score 为 0
    df["score_ranked"] = 0.0
    for key, weight in weights.items():
        if key in df:
            df["score_ranked"] += df[key] * weight
            
    return df

def summarize_factor_oriented(factor_values: np.ndarray, fwd_returns: np.ndarray, 
                              train_mask: np.ndarray, test_mask: np.ndarray,
                              dates: np.ndarray | None = None) -> dict:
    """
    计算考虑方向后的因子指标
    """
    from .fastops import daily_corr
    
    # 1. 计算每日 RankIC
    daily_ic = daily_corr(factor_values, fwd_returns, rank=True)
    
    # 2. 判断方向 (仅使用训练集)
    train_ic = daily_ic[train_mask]
    dimension = infer_dimension(train_ic)
    
    if dimension == 0:
        return {"status": "INVALID_DIRECTION", "dimension": 0}
        
    # 3. 统一方向后的 IC
    oriented_ic = daily_ic * dimension
    oriented_train_ic = oriented_ic[train_mask]
    oriented_test_ic = oriented_ic[test_mask]
    
    # 4. 计算指标
    def calc_metrics(ic_series):
        if len(ic_series) == 0 or np.all(np.isnan(ic_series)):
            return 0.0, 0.0, 0.0
        mean = np.nanmean(ic_series)
        std = np.nanstd(ic_series)
        ir = mean / std if std > 1e-9 else 0.0
        hit = np.nanmean(ic_series > 0)
        return mean, ir, hit

    tr_mean, tr_ir, tr_hit = calc_metrics(oriented_train_ic)
    te_mean, te_ir, te_hit = calc_metrics(oriented_test_ic)
    
    # 5. 计算年度稳定性 (Yearly Positive Ratio)
    yearly_stability = 0.0
    if dates is not None:
        years = np.array([str(d)[:4] for d in dates])
        unique_years = np.unique(years)
        yearly_hits = []
        for y in unique_years:
            mask = (years == y) & test_mask
            if np.any(mask):
                y_ic = oriented_ic[mask]
                if np.any(np.isfinite(y_ic)):
                    yearly_hits.append(np.nanmean(y_ic > 0))
        if yearly_hits:
            # 这里的稳定性定义为：年度胜率大于 50% 的年份比例
            yearly_stability = np.mean(np.array(yearly_hits) > 0.5)

    return {
        "dimension": dimension,
        "oriented_train_mean_rank_ic": tr_mean,
        "oriented_test_mean_rank_ic": te_mean,
        "oriented_test_rank_icir": te_ir,
        "positive_oriented_rank_ic_ratio": te_hit,
        "yearly_positive_ratio": yearly_stability,
        "raw_train_mean_rank_ic": np.nanmean(train_ic),
        "raw_test_mean_rank_ic": np.nanmean(daily_ic[test_mask]),
        "status": "OK"
    }
