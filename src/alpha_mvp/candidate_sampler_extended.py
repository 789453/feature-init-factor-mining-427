"""
扩展的候选采样器模块
实现分层打散排序和候选导出功能
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

def select_for_fine_screen(
    results: pd.DataFrame,
    top_k: int = 1000,
    sample_n: int = 1000,
    min_per_template_family: int = 20,
    min_per_field: int = 20,
    min_per_operator: int = 20,
    alpha: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    """
    从粗筛结果中选择候选进入细筛。
    
    Args:
        results: 包含因子结果的数据框
        top_k: 选择评分最高的K个表达式
        sample_n: 从尾部概率采样的数量
        min_per_template_family: 每个模板族最少保留的数量
        min_per_field: 每个字段最少保留的数量
        min_per_operator: 每个算子最少保留的数量
        alpha: 概率采样衰减系数
        seed: 随机种子
        
    Returns:
        选中的候选表达式数据框
    """
    if results.empty:
        return results
        
    # 确保有 score_ranked 字段
    score_col = "score_ranked" if "score_ranked" in results.columns else "score_raw"
    df = results.sort_values(score_col, ascending=False).reset_index(drop=True)
    
    # 1. Top K
    top = df.head(top_k).copy()
    
    # 2. 概率抽样
    tail = df.iloc[top_k:].copy()
    sampled = pd.DataFrame()
    if not tail.empty:
        rank = np.arange(1, len(tail) + 1)
        p = 1 / np.power(rank, alpha)
        p = p / p.sum()
        
        sampled_n = min(sample_n, len(tail))
        # 使用replace=True避免权重采样问题
        sampled = tail.sample(n=sampled_n, weights=p, random_state=seed, replace=True)
    
    # 3. 保底机制
    guaranteed = pd.DataFrame()
    
    # 模板族保底
    if "template_family" in df.columns:
        family_keep = (
            df.groupby("template_family", group_keys=False)
              .apply(lambda x: x.nlargest(min_per_template_family, score_col))
        )
        guaranteed = pd.concat([guaranteed, family_keep])
    
    # 字段保底
    if "fields" in df.columns or "field" in df.columns:
        field_col = "fields" if "fields" in df.columns else "field"
        # 简化处理：假设字段信息在单独的链接表中，这里需要根据实际情况调整
        # 这里使用表达式哈希作为代理
        if "expr_hash" in df.columns:
            field_keep = df.groupby("expr_hash").head(min_per_field)
            guaranteed = pd.concat([guaranteed, field_keep])
    
    # 算子保底
    if "operators" in df.columns or "operator" in df.columns:
        op_col = "operators" if "operators" in df.columns else "operator"
        op_keep = (
            df.groupby(op_col, group_keys=False)
              .apply(lambda x: x.nlargest(min_per_operator, score_col))
        )
        guaranteed = pd.concat([guaranteed, op_keep])
    
    # 合并去重
    all_parts = [part for part in [top, sampled, guaranteed] if not part.empty]
    if not all_parts:
        return pd.DataFrame()
    
    combined = pd.concat(all_parts).drop_duplicates(subset=["expr_hash"])
    
    # 最终限制返回数量不超过top_k
    return combined.sort_values(score_col, ascending=False).head(top_k).reset_index(drop=True)

def stratified_candidate_select(
    df: pd.DataFrame,
    score_col: str = "score_ranked",
    top_k: int = 2000,
    sample_n: int = 3000,
    min_per_template_family: int = 100,
    min_per_field: int = 20,
    min_per_operator: int = 20,
    alpha: float = 0.85,
    seed: int = 42,
    export_fields: List[str] = None
) -> pd.DataFrame:
    """
    分层候选选择，确保各种维度都有足够的代表性
    
    Args:
        df: 因子结果数据框
        score_col: 评分列名
        top_k: 高分段选择数量
        sample_n: 尾部采样数量
        min_per_template_family: 每模板族最少数量
        min_per_field: 每字段最少数量
        min_per_operator: 每算子最少数量
        alpha: 概率采样衰减系数
        seed: 随机种子
        export_fields: 需要导出的字段列表
        
    Returns:
        分层选择后的候选数据框
    """
    if df.empty:
        return df
    
    # 默认导出字段
    if export_fields is None:
        export_fields = ["expr", "expr_hash", score_col, "template_family", "template_name"]
    
    # 确保所有必需的字段都存在
    available_fields = [col for col in export_fields if col in df.columns]
    
    # 使用增强的选择策略
    selected = select_for_fine_screen(
        df[available_fields],
        top_k=top_k,
        sample_n=sample_n,
        min_per_template_family=min_per_template_family,
        min_per_field=min_per_field,
        min_per_operator=min_per_operator,
        alpha=alpha,
        seed=seed
    )
    
    return selected

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
    
    # 按评分降序排列
    df_sorted = df.sort_values(score_col, ascending=False)
    
    # 分组
    # 检查分组列是否存在
    existing_group_cols = [col for col in group_cols if col in df_sorted.columns]
    if not existing_group_cols:
        return df.head(top_n)
    
    groups = df_sorted.groupby(existing_group_cols)
    
    # 交替抽取
    result_rows = []
    max_per_group = (top_n // len(groups)) + 1
    
    # 使用轮询方式从每个组中抽取
    group_iters = {name: group.iterrows() for name, group in groups}
    
    for i in range(top_n):
        if not group_iters:
            break
            
        # 轮询每个组
        for group_name in list(group_iters.keys()):
            try:
                _, row = next(group_iters[group_name])
                result_rows.append(row)
                if len(result_rows) >= top_n:
                    break
            except StopIteration:
                # 该组已经没有更多数据
                del group_iters[group_name]
        
        if len(result_rows) >= top_n:
            break
    
    if not result_rows:
        return df.head(top_n)
    
    result_df = pd.DataFrame(result_rows)
    
    # 确保返回的数据框包含原始列
    for col in df.columns:
        if col not in result_df.columns:
            result_df[col] = None
    
    return result_df[df.columns].reset_index(drop=True)

def export_candidate_expr_file(
    candidates: pd.DataFrame,
    out_path: str,
    expr_col: str = "expr",
    add_header: bool = False
) -> None:
    """
    导出候选表达式到文件
    
    Args:
        candidates: 候选数据框
        out_path: 输出文件路径
        expr_col: 表达式列名
        add_header: 是否添加表头
    """
    if candidates.empty:
        print("Warning: No candidates to export")
        return
    
    # 确保输出目录存在
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 导出表达式
    candidates[[expr_col]].to_csv(
        out_path, 
        index=False, 
        header=add_header,
        encoding='utf-8'
    )
    
    print(f"Exported {len(candidates)} candidate expressions to {out_path}")

def export_candidate_analysis(
    candidates: pd.DataFrame,
    output_dir: str,
    score_col: str = "score_ranked"
) -> None:
    """
    导出候选分析结果，包括按模板族、字段、算子的分布
    
    Args:
        candidates: 候选数据框
        output_dir: 输出目录
        score_col: 评分列名
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 按模板族分析
    if "template_family" in candidates.columns:
        by_family = candidates.groupby("template_family").agg({
            score_col: ["count", "mean", "std", "min", "max"],
            "expr_hash": "nunique"
        }).round(4)
        by_family.columns = ["count", "mean_score", "std_score", "min_score", "max_score", "unique_exprs"]
        by_family.to_csv(output_path / "fine_candidates_by_template.csv")
    
    # 按字段分析（如果可用）
    # 这里需要根据实际的数据结构来实现
    
    # 按算子分析（如果可用）
    # 这里需要根据实际的数据结构来实现
    
    # 总体统计
    summary = {
        "total_candidates": len(candidates),
        "unique_expressions": candidates["expr_hash"].nunique() if "expr_hash" in candidates.columns else len(candidates),
        "mean_score": candidates[score_col].mean(),
        "median_score": candidates[score_col].median(),
        "score_std": candidates[score_col].std(),
        "min_score": candidates[score_col].min(),
        "max_score": candidates[score_col].max()
    }
    
    # 保存摘要
    import json
    with open(output_path / "candidate_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

def create_candidate_report(
    original_results: pd.DataFrame,
    selected_candidates: pd.DataFrame,
    output_path: str,
    score_col: str = "score_ranked"
) -> None:
    """
    创建候选选择报告，对比原始结果和选择结果
    
    Args:
        original_results: 原始因子结果
        selected_candidates: 选中的候选
        output_path: 输出文件路径
        score_col: 评分列名
    """
    report = {
        "selection_summary": {
            "original_count": len(original_results),
            "selected_count": len(selected_candidates),
            "selection_ratio": len(selected_candidates) / len(original_results) * 100 if len(original_results) > 0 else 0
        },
        "score_comparison": {
            "original_mean": original_results[score_col].mean(),
            "selected_mean": selected_candidates[score_col].mean(),
            "original_median": original_results[score_col].median(),
            "selected_median": selected_candidates[score_col].median(),
            "original_p90": original_results[score_col].quantile(0.9),
            "selected_p90": selected_candidates[score_col].quantile(0.9)
        }
    }
    
    # 模板族分布对比
    if "template_family" in original_results.columns and "template_family" in selected_candidates.columns:
        orig_family_dist = original_results["template_family"].value_counts(normalize=True)
        sel_family_dist = selected_candidates["template_family"].value_counts(normalize=True)
        
        report["template_family_distribution"] = {
            "original": orig_family_dist.to_dict(),
            "selected": sel_family_dist.to_dict()
        }
    
    # 保存报告
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"Candidate selection report saved to {output_path}")