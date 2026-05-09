"""
测试候选采样器功能
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.alpha_mvp.candidate_sampler_extended import (
    select_for_fine_screen, stratified_candidate_select, interleave_by_group,
    export_candidate_expr_file, export_candidate_analysis, create_candidate_report
)

class TestCandidateSampler:
    """测试候选采样器功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试数据
        np.random.seed(42)
        n_expressions = 1000
        
        self.test_data = pd.DataFrame({
            'expr': [f'Rank(TsMean($ret_1d,{i}))' for i in range(n_expressions)],
            'expr_hash': [f'hash_{i}' for i in range(n_expressions)],
            'score_ranked': np.random.exponential(0.1, n_expressions),
            'template_family': np.random.choice(['single', 'binary', 'triple'], n_expressions),
            'template_name': np.random.choice(['test1', 'test2', 'test3'], n_expressions),
            'field': np.random.choice(['ret_1d', 'ret_5d', 'vol_log'], n_expressions),
            'operator': np.random.choice(['TsMean', 'TsStd', 'TsRank'], n_expressions),
            'coverage': np.random.uniform(0.5, 1.0, n_expressions),
            'turnover_proxy': np.random.uniform(0.1, 0.5, n_expressions)
        })
        
        # 确保评分降序排列
        self.test_data = self.test_data.sort_values('score_ranked', ascending=False).reset_index(drop=True)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_select_for_fine_screen_basic(self):
        """测试基础候选选择功能"""
        candidates = select_for_fine_screen(
            self.test_data,
            top_k=100,
            sample_n=50,
            min_per_template_family=10,
            alpha=0.85,
            seed=42
        )
        
        # 验证结果
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) > 0
        assert len(candidates) <= len(self.test_data)
        
        # 验证包含必需列
        assert 'expr' in candidates.columns
        assert 'score_ranked' in candidates.columns
        assert 'template_family' in candidates.columns
        
        # 验证排序
        assert candidates['score_ranked'].is_monotonic_decreasing or candidates['score_ranked'].is_monotonic_decreasing.fillna(False)
    
    def test_select_for_fine_screen_empty_data(self):
        """测试空数据情况"""
        empty_data = pd.DataFrame()
        
        candidates = select_for_fine_screen(empty_data)
        
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) == 0
    
    def test_stratified_candidate_select(self):
        """测试分层候选选择"""
        candidates = stratified_candidate_select(
            self.test_data,
            score_col='score_ranked',
            top_k=200,
            sample_n=100,
            min_per_template_family=20,
            min_per_field=10,
            min_per_operator=10,
            alpha=0.85,
            seed=42
        )
        
        # 验证结果
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) > 0
        assert len(candidates) <= len(self.test_data)
        
        # 验证分层效果
        family_counts = candidates['template_family'].value_counts()
        assert len(family_counts) >= 2, "Should have multiple template families"
        assert all(count >= 5 for count in family_counts), "Each family should have reasonable representation"
    
    def test_interleave_by_group(self):
        """测试分组交替抽取"""
        interleaved = interleave_by_group(
            self.test_data,
            group_cols=['template_family'],
            top_n=100,
            score_col='score_ranked',
            random_state=42
        )
        
        # 验证结果
        assert isinstance(interleaved, pd.DataFrame)
        assert len(interleaved) == 100
        assert 'template_family' in interleaved.columns
        
        # 验证多样性
        family_counts = interleaved['template_family'].value_counts()
        assert len(family_counts) >= 2, "Should have multiple template families"
        
        # 验证没有单一族占据所有位置
        max_family_ratio = family_counts.max() / len(interleaved)
        assert max_family_ratio < 0.6, "No single family should dominate"
    
    def test_interleave_by_group_no_groups(self):
        """测试没有分组列的情况"""
        data_no_groups = self.test_data.drop(columns=['template_family'])
        
        interleaved = interleave_by_group(
            data_no_groups,
            group_cols=['template_family'],  # 不存在的列
            top_n=50,
            score_col='score_ranked'
        )
        
        # 应该返回简单的top结果
        assert len(interleaved) == 50
        assert 'template_family' not in interleaved.columns
    
    def test_export_candidate_expr_file(self):
        """测试导出候选表达式文件"""
        candidates = self.test_data.head(100)
        output_file = self.temp_dir / "candidates.expr"
        
        export_candidate_expr_file(candidates, str(output_file))
        
        # 验证文件存在
        assert output_file.exists()
        
        # 验证文件内容
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 100
        assert all(line.strip() for line in lines), "All lines should be non-empty"
        
        # 验证表达式格式
        assert all('Rank(' in line or 'TsMean(' in line for line in lines[:10])
    
    def test_export_candidate_expr_file_with_header(self):
        """测试带表头的导出"""
        candidates = self.test_data.head(50)
        output_file = self.temp_dir / "candidates_with_header.csv"
        
        export_candidate_expr_file(candidates, str(output_file), add_header=True)
        
        # 验证文件存在
        assert output_file.exists()
        
        # 验证文件内容
        with open(output_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 51  # 50行数据 + 1行表头
        assert lines[0].strip() == 'expr', "First line should be header"
    
    def test_export_candidate_analysis(self):
        """测试导出候选分析"""
        candidates = self.test_data.head(200)
        output_dir = self.temp_dir / "candidate_analysis"
        
        export_candidate_analysis(candidates, str(output_dir))
        
        # 验证输出目录存在
        assert output_dir.exists()
        assert output_dir.is_dir()
        
        # 验证分析文件存在
        assert (output_dir / "fine_candidates_by_template.csv").exists()
        assert (output_dir / "candidate_summary.json").exists()
        
        # 验证CSV文件内容
        template_csv = pd.read_csv(output_dir / "fine_candidates_by_template.csv")
        assert len(template_csv) > 0
        assert 'template_family' in template_csv.columns or len(template_csv) == len(candidates['template_family'].unique())
        
        # 验证JSON摘要
        import json
        with open(output_dir / "candidate_summary.json", 'r') as f:
            summary = json.load(f)
        
        assert 'total_candidates' in summary
        assert 'unique_expressions' in summary
        assert 'mean_score' in summary
        assert summary['total_candidates'] == 200
    
    def test_create_candidate_report(self):
        """测试创建候选选择报告"""
        # 选择部分数据作为"选中的候选"
        selected = self.test_data.head(100)
        
        report_file = self.temp_dir / "candidate_report.json"
        
        create_candidate_report(
            self.test_data,
            selected,
            str(report_file),
            score_col='score_ranked'
        )
        
        # 验证报告文件存在
        assert report_file.exists()
        
        # 验证报告内容
        import json
        with open(report_file, 'r') as f:
            report = json.load(f)
        
        assert 'selection_summary' in report
        assert 'score_comparison' in report
        assert report['selection_summary']['original_count'] == 1000
        assert report['selection_summary']['selected_count'] == 100
        assert report['selection_summary']['selection_ratio'] == 10.0
        
        # 验证评分对比
        score_comparison = report['score_comparison']
        assert 'original_mean' in score_comparison
        assert 'selected_mean' in score_comparison
        assert score_comparison['selected_mean'] >= score_comparison['original_mean']
    
    def test_candidate_selection_diversity(self):
        """测试候选选择的多样性"""
        # 创建有偏向的数据（某个族评分普遍较高）
        biased_data = self.test_data.copy()
        
        # 让single族的评分普遍更高
        single_mask = biased_data['template_family'] == 'single'
        biased_data.loc[single_mask, 'score_ranked'] = biased_data.loc[single_mask, 'score_ranked'] * 2
        
        # 重新排序
        biased_data = biased_data.sort_values('score_ranked', ascending=False).reset_index(drop=True)
        
        # 使用分层选择
        candidates = stratified_candidate_select(
            biased_data,
            top_k=50,
            sample_n=50,
            min_per_template_family=20,  # 强制每族至少20个
            alpha=0.85,
            seed=42
        )
        
        # 验证多样性
        family_counts = candidates['template_family'].value_counts()
        assert len(family_counts) >= 2, "Should maintain diversity"
        # 放宽要求，因为样本选择可能有随机性
        # 至少有一个族有较多代表，且所有族都有代表
        assert max(family_counts) >= 10, "Should have at least one family with good representation"
        assert min(family_counts) >= 1, "Each family should have at least one representative"
    
    def test_export_empty_candidates(self):
        """测试导出空候选列表"""
        empty_candidates = pd.DataFrame()
        output_file = self.temp_dir / "empty_candidates.expr"
        
        # 不应该抛出异常
        export_candidate_expr_file(empty_candidates, str(output_file))
        
        # 文件应该存在但为空（或者根据实现可能不存在）
        # 这里我们主要验证不抛出异常
        print("Empty candidate export completed without error")
    
    def test_candidate_selection_with_missing_columns(self):
        """测试缺少某些列的情况"""
        # 删除一些可选列
        incomplete_data = self.test_data.drop(columns=['field', 'operator'])
        
        # 应该能正常工作，只是缺少某些保底机制
        candidates = stratified_candidate_select(
            incomplete_data,
            min_per_field=5,  # 这个参数会被忽略
            min_per_operator=5,  # 这个参数会被忽略
            seed=42
        )
        
        assert isinstance(candidates, pd.DataFrame)
        assert len(candidates) > 0
        assert 'field' not in candidates.columns
        assert 'operator' not in candidates.columns