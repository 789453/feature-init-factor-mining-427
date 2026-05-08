"""
分层排序和打散模块
实现多种排序策略，确保top结果具有良好的多样性
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json

def sort_by_score(df: pd.DataFrame, score_col: str = "score_ranked", ascending: bool = False) -> pd.DataFrame:
    """
    按评分排序
    
    Args:
        df: 数据框
        score_col: 评分列名
        ascending: 是否升序
        
    Returns:
        排序后的数据框
    """
    if df.empty or score_col not in df.columns:
        return df
    return df.sort_values(score_col, ascending=ascending).reset_index(drop=True)

def interleave_by_group(
    df: pd.DataFrame,
    group_cols: List[str] = ("template_family",),
    top_n: int = 100,
    score_col: str = "score_ranked",
    random_state: int = 42
) -> pd.DataFrame:
    """
    按组交替抽取，避免top100全部被单一模板族占据
    
    Args:
        df: 排序后的因子结果数据框
        group_cols: 分组列名列表
        top_n: 最终选择的数量
        score_col: 评分列名
        
    Returns:
        交替抽取后的数据框
    """
    if df.empty or len(group_cols) == 0:
        return df.head(top_n)
    
    # 确保按评分降序排列
    df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    
    # 过滤掉不存在的分组列
    existing_group_cols = [col for col in group_cols if col in df_sorted.columns]
    if not existing_group_cols:
        return df_sorted.head(top_n)
    
    # 分组
    groups = df_sorted.groupby(existing_group_cols)
    
    # 为每个组分配权重（基于组内排名）
    result_rows = []
    group_data = {}
    
    # 收集每个组的数据
    for name, group in groups:
        group_data[name] = group.reset_index(drop=True)
    
    if not group_data:
        return df.head(top_n)
    
    # 轮询选择，确保每个组都有代表
    np.random.seed(random_state)
    selected_indices = set()
    
    # 第一轮：从每个组中选择最好的
    for name, group_df in group_data.items():
        if len(group_df) > 0:
            best_idx = group_df.index[0]
            if best_idx not in selected_indices:
                result_rows.append(group_df.iloc[0])
                selected_indices.add(best_idx)
    
    # 后续轮次：继续轮询选择
    round_num = 1
    while len(result_rows) < top_n and len(selected_indices) < len(df_sorted):
        for name, group_df in group_data.items():
            if len(result_rows) >= top_n:
                break
                
            # 选择当前轮次的元素
            if round_num < len(group_df):
                candidate_idx = group_df.index[round_num]
                if candidate_idx not in selected_indices:
                    result_rows.append(group_df.iloc[round_num])
                    selected_indices.add(candidate_idx)
        
        round_num += 1
        
        # 如果所有组都没有更多元素，停止
        if all(round_num >= len(group_df) for group_df in group_data.values()):
            break
    
    # 如果还不够，从剩余的最好元素中补充
    if len(result_rows) < top_n:
        remaining = df_sorted[~df_sorted.index.isin(selected_indices)].head(top_n - len(result_rows))
        result_rows.extend([row for _, row in remaining.iterrows()])
    
    if not result_rows:
        return df.head(top_n)
    
    result_df = pd.DataFrame(result_rows)
    
    # 确保返回的数据框包含原始列
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = None
    
    return result_df[df.columns].reset_index(drop=True)

def stratified_ranking(
    df: pd.DataFrame,
    score_col: str = "score_ranked",
    group_cols: List[str] = ("template_family",),
    top_n: int = 100,
    min_per_group: int = 5,
    max_per_group: int = 50,
    random_state: int = 42
) -> pd.DataFrame:
    """
    分层排序，确保每个组都有足够的代表性
    
    Args:
        df: 数据框
        score_col: 评分列名
        group_cols: 分组列名列表
        top_n: 最终选择的数量
        min_per_group: 每组最少选择数量
        max_per_group: 每组最多选择数量
        random_state: 随机种子
        
    Returns:
        分层选择后的数据框
    """
    if df.empty:
        return df
    
    # 按评分排序
    df_sorted = sort_by_score(df, score_col)
    
    # 分组
    if not group_cols or len(group_cols) == 0:
        return df_sorted.head(top_n)
    
    # 过滤掉不存在的分组列
    existing_group_cols = [col for col in group_cols if col in df_sorted.columns]
    if not existing_group_cols:
        return df_sorted.head(top_n)
    
    groups = df_sorted.groupby(existing_group_cols)
    n_groups = len(groups)
    
    # 计算每组的配额
    if n_groups == 0:
        return df_sorted.head(top_n)
    
    # 基础配额：确保每组至少有min_per_group个
    base_quota = min_per_group
    remaining_slots = top_n - (n_groups * min_per_group)
    
    if remaining_slots < 0:
        # 如果基础配额就超过了top_n，按比例分配
        base_quota = max(1, top_n // n_groups)
        remaining_slots = top_n - (n_groups * base_quota)
    
    selected_rows = []
    
    # 从每个组中选择
    for name, group in groups:
        group_size = len(group)
        
        # 确定该组的配额
        if remaining_slots > 0:
            # 按比例分配剩余名额
            group_quota = base_quota + int((len(group) / len(df_sorted)) * remaining_slots)
            group_quota = min(group_quota, max_per_group)
        else:
            group_quota = min(base_quota, max_per_group)
        
        # 从组中选择最好的group_quota个
        selected_from_group = group.head(min(group_quota, group_size))
        selected_rows.extend([row for _, row in selected_from_group.iterrows()])
    
    # 创建结果数据框
    if selected_rows:
        result_df = pd.DataFrame(selected_rows)
        # 按评分重新排序
        result_df = sort_by_score(result_df, score_col)
        # 限制最终数量
        return result_df.head(top_n)
    else:
        return df_sorted.head(top_n)

def diversity_aware_ranking(
    df: pd.DataFrame,
    score_col: str = "score_ranked",
    diversity_cols: List[str] = ("template_family", "field"),
    top_n: int = 100,
    diversity_weight: float = 0.3,
    random_state: int = 42
) -> pd.DataFrame:
    """
    多样性感知的排序，平衡评分和多样性
    
    Args:
        df: 数据框
        score_col: 评分列名
        diversity_cols: 多样性列名列表
        top_n: 最终选择的数量
        diversity_weight: 多样性权重（0-1）
        random_state: 随机种子
        
    Returns:
        多样性排序后的数据框
    """
    if df.empty:
        return df
    
    # 过滤掉不存在的多样性列
    existing_diversity_cols = [col for col in diversity_cols if col in df.columns]
    if not existing_diversity_cols:
        # 如果没有多样性列，直接按评分排序
        return df.sort_values(score_col, ascending=False).head(top_n)
    
    np.random.seed(random_state)
    
    # 标准化评分
    df_sorted = sort_by_score(df, score_col)
    df_sorted["score_normalized"] = (df_sorted[score_col] - df_sorted[score_col].min()) / (df_sorted[score_col].max() - df_sorted[score_col].min())
    
    selected_indices = []
    selected_combinations = set()
    
    # 逐步选择，考虑多样性
    for i in range(min(top_n, len(df_sorted))):
        best_score = -np.inf
        best_idx = -1
        
        # 在剩余的数据中选择
        remaining_indices = [idx for idx in df_sorted.index if idx not in selected_indices]
        
        if not remaining_indices:
            break
        
        for idx in remaining_indices:
            row = df_sorted.loc[idx]
            
            # 计算多样性得分
            diversity_score = 0
            if diversity_cols:
                current_combination = tuple(row[col] for col in diversity_cols if col in row)
                if current_combination not in selected_combinations:
                    diversity_score = 1.0
            
            # 综合得分
            combined_score = (1 - diversity_weight) * row["score_normalized"] + diversity_weight * diversity_score
            
            if combined_score > best_score:
                best_score = combined_score
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            # 更新已选择的组合
            if diversity_cols:
                best_row = df_sorted.loc[best_idx]
                combination = tuple(best_row[col] for col in diversity_cols if col in best_row)
                selected_combinations.add(combination)
    
    # 返回选中的行
    result_df = df_sorted.loc[selected_indices].drop(columns=["score_normalized"])
    return result_df.reset_index(drop=True)

def create_ranking_report(
    original_df: pd.DataFrame,
    ranked_df: pd.DataFrame,
    ranking_method: str,
    score_col: str = "score_ranked",
    group_cols: List[str] = ("template_family",),
    output_path: str = None
) -> Dict[str, Any]:
    """
    创建排序报告，对比原始结果和排序结果
    
    Args:
        original_df: 原始数据框
        ranked_df: 排序后的数据框
        ranking_method: 排序方法名称
        score_col: 评分列名
        group_cols: 分组列名
        output_path: 输出文件路径（可选）
        
    Returns:
        报告字典
    """
    report = {
        "ranking_method": ranking_method,
        "original_count": len(original_df),
        "selected_count": len(ranked_df),
        "score_comparison": {
            "original_mean": original_df[score_col].mean(),
            "selected_mean": ranked_df[score_col].mean(),
            "original_median": original_df[score_col].median(),
            "selected_median": ranked_df[score_col].median(),
            "score_loss": (original_df[score_col].head(len(ranked_df)).mean() - ranked_df[score_col].mean()) / original_df[score_col].head(len(ranked_df)).mean() * 100
        }
    }
    
    # 分组分布分析
    if group_cols and all(col in original_df.columns for col in group_cols):
        orig_dist = original_df.head(len(ranked_df)).groupby(list(group_cols)).size()
        ranked_dist = ranked_df.groupby(list(group_cols)).size()
        
        report["group_distribution"] = {
            "original": orig_dist.to_dict(),
            "selected": ranked_dist.to_dict(),
            "diversity_improvement": _calculate_diversity_improvement(orig_dist, ranked_dist)
        }
    
    # 保存报告
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
    
    return report

def _calculate_diversity_improvement(original_dist: pd.Series, selected_dist: pd.Series) -> float:
    """计算多样性改进程度"""
    # 使用基尼系数或熵来衡量多样性
    def entropy(dist):
        probs = dist / dist.sum()
        return -np.sum(probs * np.log(probs + 1e-10))
    
    orig_entropy = entropy(original_dist)
    selected_entropy = entropy(selected_dist)
    
    if orig_entropy == 0:
        return 0
    
    return (selected_entropy - orig_entropy) / orig_entropy * 100

def apply_ranking_strategy(
    df: pd.DataFrame,
    strategy: str = "interleave",
    score_col: str = "score_ranked",
    group_cols: List[str] = ("template_family",),
    top_n: int = 100,
    **kwargs
) -> pd.DataFrame:
    """
    应用指定的排序策略
    
    Args:
        df: 数据框
        strategy: 排序策略 ('simple', 'interleave', 'stratified', 'diversity')
        score_col: 评分列名
        group_cols: 分组列名
        top_n: 最终选择的数量
        **kwargs: 其他策略特定参数
        
    Returns:
        排序后的数据框
    """
    if strategy == "simple":
        return sort_by_score(df, score_col).head(top_n)
    elif strategy == "interleave":
        return interleave_by_group(df, group_cols, top_n, score_col, kwargs.get("random_state", 42))
    elif strategy == "stratified":
        return stratified_ranking(
            df, score_col, group_cols, top_n,
            min_per_group=kwargs.get("min_per_group", 5),
            max_per_group=kwargs.get("max_per_group", 50),
            random_state=kwargs.get("random_state", 42)
        )
    elif strategy == "diversity":
        return diversity_aware_ranking(
            df, score_col, group_cols, top_n,
            diversity_weight=kwargs.get("diversity_weight", 0.3),
            random_state=kwargs.get("random_state", 42)
        )
    else:
        raise ValueError(f"Unknown ranking strategy: {strategy}")

def batch_ranking(
    df: pd.DataFrame,
    strategies: List[str],
    score_col: str = "score_ranked",
    group_cols: List[str] = ("template_family",),
    top_n: int = 100,
    output_dir: str = None
) -> Dict[str, pd.DataFrame]:
    """
    批量应用多种排序策略，并生成对比报告
    
    Args:
        df: 数据框
        strategies: 排序策略列表
        score_col: 评分列名
        group_cols: 分组列名
        top_n: 最终选择的数量
        output_dir: 输出目录（可选）
        
    Returns:
        策略名称到结果数据框的映射
    """
    results = {}
    reports = {}
    
    for strategy in strategies:
        print(f"Applying ranking strategy: {strategy}")
        ranked_df = apply_ranking_strategy(df, strategy, score_col, group_cols, top_n)
        results[strategy] = ranked_df
        
        # 生成报告
        if output_dir:
            report_path = Path(output_dir) / f"ranking_report_{strategy}.json"
            report = create_ranking_report(df, ranked_df, strategy, score_col, group_cols, str(report_path))
            reports[strategy] = report
    
    # 生成对比报告
    if output_dir and reports:
        comparison_path = Path(output_dir) / "ranking_comparison.json"
        with open(comparison_path, "w") as f:
            json.dump(reports, f, indent=2, default=str)
    
    return results