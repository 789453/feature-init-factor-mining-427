from __future__ import annotations
import numpy as np
import pandas as pd

def select_for_fine_screen(
    results: pd.DataFrame,
    top_k: int = 1000,
    sample_n: int = 1000,
    min_per_template_family: int = 20,
    alpha: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    """
    从粗筛结果中选择候选进入细筛。
    逻辑：
    1. 选择 Top K 个评分最高的表达式。
    2. 对剩余部分进行概率抽样（评分越高概率越大）。
    3. 确保每个模板族至少有最小数量的代表。
    """
    if results.empty:
        return results
        
    # 确保有 score_ranked 字段
    score_col = "score_ranked" if "score_ranked" in results.columns else "score_raw"
    df = results.sort_values(score_col, ascending=False).reset_index(drop=True)
    
    # 1. Top K
    top = df.head(top_k)
    
    # 2. 概率抽样
    tail = df.iloc[top_k:].copy()
    if not tail.empty:
        rank = np.arange(1, len(tail) + 1)
        p = 1 / np.power(rank, alpha)
        p = p / p.sum()
        
        sampled_n = min(sample_n, len(tail))
        sampled = tail.sample(n=sampled_n, weights=p, random_state=seed)
    else:
        sampled = pd.DataFrame()
        
    # 3. 模板族保底
    family_keep = pd.DataFrame()
    if "template_family" in df.columns:
        family_keep = (
            df.groupby("template_family", group_keys=False)
              .apply(lambda x: x.nlargest(min_per_template_family, score_col))
        )
        
    # 合并去重
    combined = pd.concat([top, sampled, family_keep]).drop_duplicates(subset=["expr_hash"])
    
    return combined.sort_values(score_col, ascending=False)
