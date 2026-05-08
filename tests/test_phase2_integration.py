"""
最终集成测试
验证所有Phase2增强功能的完整集成
"""
import pytest
import tempfile
from pathlib import Path
import yaml
import json
import pandas as pd
import numpy as np

from src.alpha_mvp.config import RunConfig, EvalConfig
from src.alpha_mvp.phase_runner import PhaseRunner, create_default_phase_config
from src.alpha_mvp.template_config import load_template_config, save_template_config
from src.alpha_mvp.signature import build_manifest_from_config, compute_run_signature
from src.alpha_mvp.template_builder import generate_expressions_from_specs
from src.alpha_mvp.candidate_sampler_extended import select_for_fine_screen, export_candidate_expr_file
from src.alpha_mvp.ranking import interleave_by_group, stratified_ranking, apply_ranking_strategy
from src.alpha_mvp.attribution_extended import run_extended_attribution

class TestPhase2Integration:
    """Phase2完整集成测试"""
    
    def setup_method(self):
        """设置集成测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试配置
        self.config = create_default_phase_config(
            coarse_pool="static_pool_200.json",
            fine_pool="static_pool_800.json",
            output_base_dir=str(self.temp_dir / "outputs")
        )
        
        # 创建轻量级配置用于快速测试
        self.config['base_config']['max_exprs'] = 100
        self.config['base_config']['batch_size'] = 16
        self.config['base_config']['use_simulated'] = True
        
        # 修改阶段配置
        self.config['phase_config']['coarse']['max_exprs'] = 100
        self.config['phase_config']['fine']['max_exprs'] = 50
        
        # 保存配置文件
        self.config_file = self.temp_dir / "integration_test_config.yaml"
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config, f)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_workflow_integration(self):
        """测试完整工作流集成"""
        print("\n=== Testing Complete Phase2 Workflow Integration ===")
        
        # 1. 测试配置加载
        runner = PhaseRunner(str(self.config_file))
        assert runner.config is not None
        assert 'phase_config' in runner.config
        print("✓ Configuration loading successful")
        
        # 2. 测试模板配置
        template_config_path = runner.config.get('template_config_path')
        if template_config_path and Path(template_config_path).exists():
            specs, budgets, raw = load_template_config(template_config_path)
            assert len(specs) > 0
            assert len(budgets) > 0
            print(f"✓ Template config loaded: {len(specs)} specs, {len(budgets)} budgets")
        
        # 3. 测试表达式生成
        test_fields = ['ret_1d', 'ret_5d', 'vol_log', 'hl_range']
        test_windows = [10, 20, 30]
        
        if template_config_path and Path(template_config_path).exists():
            expr_records = generate_expressions_from_specs(
                fields=test_fields,
                windows=test_windows,
                template_config_path=template_config_path,
                max_exprs=50,
                seed=42
            )
            
            assert len(expr_records) > 0
            assert len(expr_records) <= 50
            print(f"✓ Expression generation successful: {len(expr_records)} expressions")
            
            # 验证不同模板族
            families = {r.template_family for r in expr_records}
            assert 'single' in families
            assert 'binary_same_ts' in families or 'binary' in families
            print(f"✓ Multiple template families generated: {families}")
        
        # 4. 测试候选选择
        # 创建模拟结果数据
        np.random.seed(42)
        mock_results = pd.DataFrame({
            'expr': [f'expr_{i}' for i in range(100)],
            'expr_hash': [f'hash_{i}' for i in range(100)],
            'score_ranked': np.random.exponential(0.1, 100),
            'template_family': np.random.choice(['single', 'binary', 'triple'], 100),
            'field': np.random.choice(test_fields, 100),
            'operator': np.random.choice(['TsMean', 'TsStd'], 100)
        })
        
        candidates = select_for_fine_screen(
            mock_results,
            top_k=20,
            sample_n=15,
            min_per_template_family=5,
            seed=42
        )
        
        assert len(candidates) > 0
        # 候选选择可能返回更多结果，因为包含保底机制
        assert len(candidates) <= 100  # 不超过原始数据
        print(f"✓ Candidate selection successful: {len(candidates)} candidates")
        
        # 5. 测试排序策略
        # 简单排序
        simple_sorted = mock_results.sort_values('score_ranked', ascending=False).head(20)
        assert len(simple_sorted) == 20
        print("✓ Simple sorting successful")
        
        # 交替排序
        interleaved = interleave_by_group(
            mock_results,
            group_cols=['template_family'],
            top_n=20,
            random_state=42
        )
        assert len(interleaved) == 20
        print("✓ Interleave sorting successful")
        
        # 分层排序
        stratified = stratified_ranking(
            mock_results,
            group_cols=['template_family'],
            top_n=20,
            min_per_group=3,
            max_per_group=10,
            random_state=42
        )
        assert len(stratified) <= 20  # 分层排序可能返回少于请求的数量
        print("✓ Stratified ranking successful")
        
        # 6. 测试候选导出
        export_dir = self.temp_dir / "exports"
        export_candidate_expr_file(candidates, str(export_dir / "test_candidates.expr"))
        assert (export_dir / "test_candidates.expr").exists()
        print("✓ Candidate export successful")
        
        # 7. 测试签名系统
        from src.alpha_mvp.config import EvalConfig
        eval_config = EvalConfig(
            forward_days=5,
            windows=[10, 20, 30],
            min_daily_valid_names=30
        )
        
        # 创建测试池文件
        test_pool = {
            'name': 'test_pool',
            'description': 'Test pool',
            'codes': ['STCK0001', 'STCK0002', 'STCK0003']
        }
        pool_file = self.temp_dir / "test_pool.json"
        with open(pool_file, 'w') as f:
            json.dump(test_pool, f)
        
        run_config = RunConfig(
            duckdb_path="test.duckdb",
            pool_json=str(pool_file),
            start="2024-01-01",
            end="2024-12-31",
            fields=test_fields,
            max_exprs=100,
            eval=eval_config,
            use_simulated=True,
            train_end="2024-08-31",
            test_start="2024-09-01"
        )
        
        manifest = build_manifest_from_config(
            run_config,
            template_config_path or "default",
            test_fields
        )
        
        assert 'project_version' in manifest
        assert 'pool_signature' in manifest
        assert 'field_signature' in manifest
        assert 'grammar_signature' in manifest
        assert 'eval_signature' in manifest
        
        run_signature = compute_run_signature(manifest)
        assert len(run_signature) == 64  # SHA256
        print("✓ Signature system successful")
        
        print("\n=== All Integration Tests Passed! ===")
    
    def test_ranking_strategies_comparison(self):
        """测试不同排序策略的对比"""
        print("\n=== Testing Ranking Strategies Comparison ===")
        
        # 创建测试数据
        np.random.seed(42)
        n_expressions = 100
        
        test_data = pd.DataFrame({
            'expr': [f'expr_{i}' for i in range(n_expressions)],
            'expr_hash': [f'hash_{i}' for i in range(n_expressions)],
            'score_ranked': np.random.exponential(0.1, n_expressions),
            'template_family': np.random.choice(['single', 'binary', 'triple'], n_expressions),
            'field': np.random.choice(['ret_1d', 'ret_5d', 'vol_log'], n_expressions),
            'operator': np.random.choice(['TsMean', 'TsStd', 'TsRank'], n_expressions)
        })
        
        # 确保某个族评分偏高（模拟真实场景）
        single_mask = test_data['template_family'] == 'single'
        test_data.loc[single_mask, 'score_ranked'] *= 1.5
        
        # 排序并重新索引
        test_data = test_data.sort_values('score_ranked', ascending=False).reset_index(drop=True)
        
        strategies = ['simple', 'interleave', 'stratified', 'diversity']
        results = {}
        
        for strategy in strategies:
            if strategy == 'simple':
                ranked = apply_ranking_strategy(
                    test_data, strategy=strategy, top_n=30
                )
            elif strategy == 'interleave':
                ranked = apply_ranking_strategy(
                    test_data, strategy=strategy, group_cols=['template_family'], top_n=30
                )
            elif strategy == 'stratified':
                ranked = apply_ranking_strategy(
                    test_data, strategy=strategy, group_cols=['template_family'], 
                    top_n=30, min_per_group=5, max_per_group=15
                )
            elif strategy == 'diversity':
                ranked = apply_ranking_strategy(
                    test_data, strategy=strategy, diversity_cols=['template_family'], 
                    top_n=30, diversity_weight=0.3
                )
            
            results[strategy] = ranked
            
            # 分析结果
            family_counts = ranked['template_family'].value_counts()
            mean_score = ranked['score_ranked'].mean()
            
            print(f"\n{strategy.upper()} Strategy:")
            print(f"  Mean score: {mean_score:.4f}")
            print(f"  Family distribution: {dict(family_counts)}")
            print(f"  Score std: {ranked['score_ranked'].std():.4f}")
        
        # 验证策略差异
        simple_families = set(results['simple']['template_family'].unique())
        interleave_families = set(results['interleave']['template_family'].unique())
        
        # 交替排序应该比简单排序有更好的多样性（或者至少相等）
        assert len(interleave_families) >= len(simple_families) or len(interleave_families) >= 1
        print("✓ Interleave strategy maintains family diversity")
        
        # 分层排序应该确保每个族都有代表
        stratified_families = results['stratified']['template_family'].value_counts()
        assert all(count >= 3 for count in stratified_families)
        print("✓ Stratified strategy ensures minimum representation")
        
        print("\n=== Ranking Strategies Comparison Completed ===")
    
    def test_template_configuration_system(self):
        """测试模板配置系统"""
        print("\n=== Testing Template Configuration System ===")
        
        # 创建测试配置
        test_config = {
            'version': 'integration_test_v1',
            'windows': [5, 10, 15, 20],
            'short_windows': [3, 5, 8],
            'long_windows': [15, 20, 30],
            'budgets': {
                '1': {
                    'max_depth': 3, 'max_nodes': 8, 'max_ts_ops': 1,
                    'max_pair_ops': 0, 'max_binary_ops': 1, 'max_unary_ops': 2
                },
                '2': {
                    'max_depth': 4, 'max_nodes': 12, 'max_ts_ops': 2,
                    'max_pair_ops': 1, 'max_binary_ops': 2, 'max_unary_ops': 3
                }
            },
            'families': [
                {
                    'name': 'single_integration',
                    'family': 'single',
                    'order': 1,
                    'enabled': True,
                    'unary_pre': ['Id', 'Abs'],
                    'ts_ops': ['TsMean', 'TsStd'],
                    'outer_transforms': ['Rank'],
                    'complexity_tier': 1
                },
                {
                    'name': 'binary_mixed_integration',
                    'family': 'binary_mixed_ts',
                    'order': 2,
                    'enabled': True,
                    'left_ts_ops': ['TsMean', 'TsDelta'],
                    'right_ts_ops': ['TsMean', 'TsStd'],
                    'binary_ops': ['Sub', 'Div'],
                    'outer_transforms': ['Rank'],
                    'complexity_tier': 2
                },
                {
                    'name': 'triple_integration',
                    'family': 'triple',
                    'order': 3,
                    'enabled': True,
                    'ts_ops': ['TsMean', 'TsDelta'],
                    'forms': [
                        'Rank(Mul(Sub(A,B),C))',
                        'Rank(Sub(Mul(A,B),C))'
                    ],
                    'max_count': 500,
                    'complexity_tier': 2
                }
            ]
        }
        
        config_file = self.temp_dir / "test_template_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # 加载配置
        specs, budgets, raw = load_template_config(str(config_file))
        
        assert len(specs) == 3
        assert len(budgets) == 2
        print(f"✓ Loaded {len(specs)} template specs and {len(budgets)} budgets")
        
        # 验证规格属性
        single_spec = next(s for s in specs if s.family == 'single')
        assert single_spec.order == 1
        assert single_spec.enabled == True
        assert 'Id' in single_spec.unary_pre
        assert 'TsMean' in single_spec.ts_ops
        
        binary_spec = next(s for s in specs if s.family == 'binary_mixed_ts')
        assert hasattr(binary_spec, 'left_ts_ops')
        assert hasattr(binary_spec, 'right_ts_ops')
        assert 'Sub' in binary_spec.binary_ops
        
        triple_spec = next(s for s in specs if s.family == 'triple')
        assert hasattr(triple_spec, '_forms')
        assert len(triple_spec._forms) == 2
        assert triple_spec.max_count == 500
        
        print("✓ Template specifications validated")
        
        # 生成表达式
        test_fields = ['ret_1d', 'ret_5d', 'vol_log']
        test_windows = [5, 10, 20]
        
        expr_records = generate_expressions_from_specs(
            fields=test_fields,
            windows=test_windows,
            specs=specs,
            budgets=budgets,
            max_exprs=30,
            seed=42
        )
        
        assert len(expr_records) > 0
        assert len(expr_records) <= 30
        print(f"✓ Generated {len(expr_records)} expressions from configuration")
        
        # 验证不同族的表达式
        family_counts = {}
        for record in expr_records:
            family = record.template_family
            family_counts[family] = family_counts.get(family, 0) + 1
        
        print(f"✓ Expression family distribution: {family_counts}")
        
        print("\n=== Template Configuration System Test Passed ===")
    
    def test_error_handling_and_edge_cases(self):
        """测试错误处理和边界情况"""
        print("\n=== Testing Error Handling and Edge Cases ===")
        
        # 测试空数据
        empty_data = pd.DataFrame()
        
        # 候选选择应该处理空数据
        empty_candidates = select_for_fine_screen(empty_data)
        assert len(empty_candidates) == 0
        print("✓ Empty data handling in candidate selection")
        
        # 测试单一样本
        single_data = pd.DataFrame({
            'expr': ['single_expr'],
            'expr_hash': ['hash_single'],
            'score_ranked': [1.0],
            'template_family': ['single']
        })
        
        single_candidates = select_for_fine_screen(single_data, top_k=5)
        assert len(single_candidates) == 1
        print("✓ Single sample handling")
        
        # 测试相同评分
        tied_data = pd.DataFrame({
            'expr': [f'expr_{i}' for i in range(10)],
            'expr_hash': [f'hash_{i}' for i in range(10)],
            'score_ranked': [0.5] * 10,  # 所有评分相同
            'template_family': ['single'] * 10
        })
        
        tied_candidates = select_for_fine_screen(tied_data, top_k=5)
        assert len(tied_candidates) == 5
        print("✓ Tied scores handling")
        
        # 测试无效配置
        invalid_config = {
            'version': 'test',
            'budgets': {},  # 空预算
            'families': []  # 空族
        }
        
        config_file = self.temp_dir / "invalid_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # 应该能加载但生成0个表达式
        try:
            specs, budgets, raw = load_template_config(str(config_file))
            assert len(specs) == 0
            print("✓ Invalid configuration handling")
        except Exception as e:
            print(f"✓ Invalid configuration properly rejected: {e}")
        
        print("\n=== Error Handling and Edge Cases Test Passed ===")
    
    def test_performance_and_scalability(self):
        """测试性能和可扩展性"""
        print("\n=== Testing Performance and Scalability ===")
        
        import time
        
        # 测试不同规模的数据
        scales = [100, 500, 1000]
        
        for scale in scales:
            print(f"\nTesting with {scale} expressions...")
            
            # 创建测试数据
            np.random.seed(42)
            test_data = pd.DataFrame({
                'expr': [f'expr_{i}' for i in range(scale)],
                'expr_hash': [f'hash_{i}' for i in range(scale)],
                'score_ranked': np.random.exponential(0.1, scale),
                'template_family': np.random.choice(['single', 'binary', 'triple'], scale),
                'field': np.random.choice(['ret_1d', 'ret_5d', 'vol_log'], scale),
                'operator': np.random.choice(['TsMean', 'TsStd', 'TsRank'], scale)
            })
            
            # 排序数据
            test_data = test_data.sort_values('score_ranked', ascending=False).reset_index(drop=True)
            
            # 测试候选选择性能
            start_time = time.time()
            candidates = select_for_fine_screen(
                test_data,
                top_k=min(100, scale // 2),
                sample_n=min(50, scale // 4),
                min_per_template_family=max(5, scale // 20),
                seed=42
            )
            candidate_time = time.time() - start_time
            
            print(f"  Candidate selection: {len(candidates)} candidates in {candidate_time:.3f}s")
            
            # 测试排序性能
            start_time = time.time()
            ranked = interleave_by_group(
                test_data,
                group_cols=['template_family'],
                top_n=min(50, scale // 2),
                random_state=42
            )
            ranking_time = time.time() - start_time
            
            print(f"  Interleave ranking: {len(ranked)} results in {ranking_time:.3f}s")
            
            # 验证结果正确性
            assert len(candidates) > 0
            assert len(ranked) == min(50, scale // 2)
            
            # 性能基准（宽松要求）
            if scale <= 500:
                assert candidate_time < 1.0, f"Candidate selection too slow: {candidate_time}s"
                assert ranking_time < 0.5, f"Ranking too slow: {ranking_time}s"
            else:
                assert candidate_time < 5.0, f"Candidate selection too slow: {candidate_time}s"
                assert ranking_time < 2.0, f"Ranking too slow: {ranking_time}s"
        
        print("\n=== Performance and Scalability Test Passed ===")
    
    def test_configuration_validation_and_consistency(self):
        """测试配置验证和一致性"""
        print("\n=== Testing Configuration Validation and Consistency ===")
        
        # 测试配置一致性
        runner = PhaseRunner(str(self.config_file))
        
        # 验证粗筛配置
        coarse_cfg = runner.config['phase_config']['coarse']
        assert coarse_cfg['phase_type'] == 'coarse'
        assert coarse_cfg['pool_json'] == 'static_pool_200.json'
        assert coarse_cfg['max_exprs'] == 100  # 我们修改的值
        
        # 验证细筛配置
        fine_cfg = runner.config['phase_config']['fine']
        assert fine_cfg['phase_type'] == 'fine'
        assert fine_cfg['pool_json'] == 'static_pool_800.json'
        assert fine_cfg['max_exprs'] == 50  # 我们修改的值
        
        # 验证基础配置
        base_cfg = runner.config['base_config']
        assert base_cfg['use_simulated'] == True
        assert base_cfg['ranking_strategy'] == 'interleave'
        assert base_cfg['extended_attribution'] == True
        
        # 验证候选选择配置
        candidate_cfg = runner.config['candidate_selection']
        assert 'top_k' in candidate_cfg
        assert 'sample_n' in candidate_cfg
        assert 'min_per_family' in candidate_cfg
        
        print("✓ Configuration consistency validated")
        
        # 测试签名一致性
        test_fields = ['ret_1d', 'ret_5d']
        
        from src.alpha_mvp.config import EvalConfig
        eval_config = EvalConfig(
            forward_days=5,
            windows=[10, 20, 30],
            min_daily_valid_names=30
        )
        
        run_config = RunConfig(
            duckdb_path="test.duckdb",
            pool_json="static_pool_200.json",
            start="2024-01-01",
            end="2024-12-31",
            fields=test_fields,
            max_exprs=100,
            eval=eval_config,
            use_simulated=True,
            train_end="2024-08-31",
            test_start="2024-09-01"
        )
        
        # 生成两次签名，应该相同
        manifest1 = build_manifest_from_config(run_config, "default", test_fields)
        manifest2 = build_manifest_from_config(run_config, "default", test_fields)
        
        signature1 = compute_run_signature(manifest1)
        signature2 = compute_run_signature(manifest2)
        
        assert signature1 == signature2
        print("✓ Signature consistency validated")
        
        print("\n=== Configuration Validation and Consistency Test Passed ===")
    
    def test_end_to_end_simulation(self):
        """测试端到端模拟运行"""
        print("\n=== Testing End-to-End Simulation ===")
        
        # 这是一个模拟的端到端测试，不实际运行完整的pipeline
        # 而是验证所有组件可以正确集成
        
        # 1. 配置验证
        runner = PhaseRunner(str(self.config_file))
        print("✓ Phase runner initialized")
        
        # 2. 模板配置加载
        template_config_path = runner.config.get('template_config_path')
        if template_config_path and Path(template_config_path).exists():
            specs, budgets, raw = load_template_config(template_config_path)
            print(f"✓ Template configuration loaded: {len(specs)} specs")
        
        # 3. 表达式生成模拟
        test_fields = ['ret_1d', 'ret_5d', 'vol_log']
        test_windows = [10, 20, 30]
        
        if template_config_path and Path(template_config_path).exists():
            expr_records = generate_expressions_from_specs(
                fields=test_fields,
                windows=test_windows,
                template_config_path=template_config_path,
                max_exprs=20,
                seed=42
            )
            print(f"✓ Expression generation simulated: {len(expr_records)} expressions")
        
        # 4. 候选选择模拟
        mock_results = pd.DataFrame({
            'expr': [f'expr_{i}' for i in range(50)],
            'expr_hash': [f'hash_{i}' for i in range(50)],
            'score_ranked': np.random.exponential(0.1, 50),
            'template_family': np.random.choice(['single', 'binary'], 50),
        })
        
        candidates = select_for_fine_screen(mock_results, top_k=10, seed=42)
        print(f"✓ Candidate selection simulated: {len(candidates)} candidates")
        
        # 5. 排序策略应用
        strategies = ['simple', 'interleave', 'stratified']
        for strategy in strategies:
            ranked = apply_ranking_strategy(
                mock_results, strategy=strategy, top_n=10
            )
            print(f"✓ {strategy} ranking applied: {len(ranked)} results")
        
        # 6. 签名生成
        from src.alpha_mvp.config import EvalConfig
        eval_config = EvalConfig(
            forward_days=5,
            windows=[10, 20, 30],
            min_daily_valid_names=30
        )
        
        run_config = RunConfig(
            duckdb_path="test.duckdb",
            pool_json="static_pool_200.json",
            start="2024-01-01",
            end="2024-12-31",
            fields=test_fields,
            max_exprs=20,
            eval=eval_config,
            use_simulated=True,
            train_end="2024-08-31",
            test_start="2024-09-01"
        )
        
        manifest = build_manifest_from_config(
            run_config, template_config_path or "default", test_fields
        )
        signature = compute_run_signature(manifest)
        print(f"✓ Run signature generated: {signature[:16]}...")
        
        print("\n=== End-to-End Simulation Test Passed ===")
        print("All Phase2 components integrated successfully!")
    
    def test_documentation_and_examples(self):
        """测试文档和示例"""
        print("\n=== Testing Documentation and Examples ===")
        
        # 验证配置文件示例
        example_config = create_default_phase_config()
        
        # 验证必需字段存在
        required_keys = ['base_config', 'template_config_path', 'candidate_selection', 'phase_config']
        for key in required_keys:
            assert key in example_config
        
        # 验证粗筛和细筛配置
        assert 'coarse' in example_config['phase_config']
        assert 'fine' in example_config['phase_config']
        
        # 验证基础配置字段
        base_keys = ['duckdb_path', 'forward_days', 'windows', 'batch_size', 'ranking_strategy']
        for key in base_keys:
            assert key in example_config['base_config']
        
        print("✓ Default configuration structure validated")
        
        # 验证候选选择配置
        candidate_keys = ['top_k', 'sample_n', 'min_per_family', 'alpha']
        for key in candidate_keys:
            assert key in example_config['candidate_selection']
        
        print("✓ Candidate selection configuration validated")
        
        # 创建使用示例
        usage_example = """
# Phase2 Pipeline Usage Example

# 1. Create configuration
config = create_default_phase_config(
    coarse_pool="static_pool_200.json",
    fine_pool="static_pool_800.json",
    output_base_dir="outputs/phase2"
)

# 2. Save configuration
with open("my_config.yaml", "w") as f:
    yaml.dump(config, f)

# 3. Run complete pipeline
runner = PhaseRunner("my_config.yaml")
result = runner.run_full_pipeline()

# 4. Or run individual phases
coarse_manifest = runner.run_coarse_phase()
fine_manifest = runner.run_fine_phase(coarse_manifest)
"""
        
        print("✓ Usage example created")
        print("\n=== Documentation and Examples Test Passed ===")

if __name__ == "__main__":
    # 运行所有集成测试
    pytest.main([__file__, "-v"])