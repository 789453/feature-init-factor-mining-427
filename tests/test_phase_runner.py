"""
测试阶段运行器功能
"""
import pytest
import tempfile
from pathlib import Path
import yaml
import json

from src.alpha_mvp.phase_runner import (
    PhaseRunner, create_default_phase_config, run_phase_runner
)

class TestPhaseRunner:
    """测试阶段运行器功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试配置
        self.test_config = {
            'base_config': {
                'duckdb_path': 'test_market.duckdb',
                'forward_days': 5,
                'train_end': '2024-08-31',
                'test_start': '2024-09-01',
                'windows': [10, 20, 30],
                'batch_size': 32,
                'write_every': 10,
                'progress_min_interval_sec': 5,
                'seed': 42,
                'extended_attribution': True,
                'ranking_strategy': 'interleave',
                'use_simulated': True  # 使用模拟数据
            },
            'template_config_path': 'configs/phase2/templates_light.yaml',
            'candidate_selection': {
                'top_k': 100,
                'sample_n': 50,
                'min_per_family': 20,
                'min_per_field': 5,
                'min_per_operator': 5,
                'alpha': 0.85
            },
            'phase_config': {
                'coarse': {
                    'pool_json': 'static_pool_200.json',
                    'start': '2024-01-01',
                    'end': '2024-04-30',
                    'max_exprs': 500,
                    'out_dir': str(self.temp_dir / 'test_coarse'),
                    'phase_type': 'coarse',
                    'top_n_display': 50
                },
                'fine': {
                    'pool_json': 'static_pool_800.json',
                    'start': '2024-01-01',
                    'end': '2024-04-30',
                    'max_exprs': 200,
                    'out_dir': str(self.temp_dir / 'test_fine'),
                    'phase_type': 'fine',
                    'top_n_display': 50
                }
            }
        }
        
        # 创建配置文件
        self.config_file = self.temp_dir / 'test_phase_config.yaml'
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # 创建测试数据文件
        self.test_duckdb = self.temp_dir / 'test_market.duckdb'
        
        # 创建测试股票池文件
        test_pool = {
            'name': 'test_pool',
            'description': 'Test stock pool for unit tests',
            'codes': [f'STCK{i:04d}' for i in range(50)],
            'creation_date': '2024-01-01',
            'criteria': {
                'min_listing_days': 252,
                'min_market_cap': 1e9,
                'max_st_days_ratio': 0.05
            }
        }
        self.test_pool_file = self.temp_dir / 'static_pool_200.json'
        with open(self.test_pool_file, 'w') as f:
            json.dump(test_pool, f)
        
        self.test_pool_file_800 = self.temp_dir / 'static_pool_800.json'
        test_pool_800 = {
            'name': 'test_pool_800',
            'description': 'Test stock pool for unit tests',
            'codes': [f'STCK{i:04d}' for i in range(200)],
            'creation_date': '2024-01-01',
            'criteria': {
                'min_listing_days': 252,
                'min_market_cap': 1e9,
                'max_st_days_ratio': 0.05
            }
        }
        with open(self.test_pool_file_800, 'w') as f:
            json.dump(test_pool_800, f)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_phase_runner_initialization(self):
        """测试阶段运行器初始化"""
        runner = PhaseRunner(str(self.config_file))
        
        assert runner.config_path == self.config_file
        assert isinstance(runner.config, dict)
        assert 'base_config' in runner.config
        assert 'phase_config' in runner.config
        assert 'coarse' in runner.config['phase_config']
        assert 'fine' in runner.config['phase_config']
    
    def test_phase_runner_invalid_config(self):
        """测试无效配置处理"""
        invalid_config_file = self.temp_dir / 'invalid_config.yaml'
        invalid_config_file.write_text('invalid: yaml: content: [')
        
        with pytest.raises(Exception):  # YAML解析错误
            PhaseRunner(str(invalid_config_file))
    
    def test_load_config_nonexistent_file(self):
        """测试加载不存在的配置文件"""
        nonexistent_file = self.temp_dir / 'nonexistent.yaml'
        
        with pytest.raises(FileNotFoundError):
            PhaseRunner(str(nonexistent_file))
    
    def test_build_run_config_coarse(self):
        """测试构建粗筛运行配置"""
        runner = PhaseRunner(str(self.config_file))
        
        from src.alpha_mvp.config import RunConfig
        coarse_cfg = runner.config['phase_config']['coarse']
        
        # 修改配置以使用测试文件
        coarse_cfg['pool_json'] = str(self.test_pool_file)
        coarse_cfg['duckdb_path'] = str(self.test_duckdb)
        coarse_cfg['use_simulated'] = True
        
        run_config = runner._build_run_config(coarse_cfg, phase_type="coarse")
        
        assert isinstance(run_config, RunConfig)
        assert runner.phase_metadata["coarse"]["phase_type"] == "coarse"
        assert run_config.pool_json == str(self.test_pool_file)
        assert run_config.max_exprs == 500
        assert run_config.start == "2024-01-01"
        assert run_config.end == "2024-04-30"
    
    def test_build_run_config_fine(self):
        """测试构建细筛运行配置"""
        runner = PhaseRunner(str(self.config_file))
        
        from src.alpha_mvp.config import RunConfig
        fine_cfg = runner.config['phase_config']['fine']
        
        # 修改配置以使用测试文件
        fine_cfg['pool_json'] = str(self.test_pool_file_800)
        fine_cfg['duckdb_path'] = str(self.test_duckdb)
        fine_cfg['use_simulated'] = True
        
        # 创建模拟的粗筛manifest
        coarse_manifest = {
            'run_signature': 'test_coarse_signature_123',
            'output_dir': str(self.temp_dir / 'test_coarse'),
            'pool_signature': {'pool_json': str(self.test_pool_file)}
        }
        
        run_config = runner._build_run_config(fine_cfg, phase_type="fine", coarse_manifest=coarse_manifest)
        
        assert isinstance(run_config, RunConfig)
        assert runner.phase_metadata["fine"]["phase_type"] == "fine"
        assert runner.phase_metadata["fine"]["coarse_signature"] == 'test_coarse_signature_123'
        assert run_config.pool_json == str(self.test_pool_file_800)
        assert run_config.max_exprs == 200
    
    def test_create_default_phase_config(self):
        """测试创建默认阶段配置"""
        config = create_default_phase_config()
        
        assert isinstance(config, dict)
        assert 'base_config' in config
        assert 'template_config_path' in config
        assert 'candidate_selection' in config
        assert 'phase_config' in config
        
        # 验证粗筛配置
        coarse_cfg = config['phase_config']['coarse']
        assert coarse_cfg['pool_json'] == 'static_pool_200.json'
        assert coarse_cfg['start'] == '20240101'
        assert coarse_cfg['phase_type'] == 'coarse'
        
        # 验证细筛配置
        fine_cfg = config['phase_config']['fine']
        assert fine_cfg['pool_json'] == 'static_pool_800.json'
        assert fine_cfg['start'] == '20180101'
        assert fine_cfg['phase_type'] == 'fine'
    
    def test_save_phase_summary(self):
        """测试保存阶段摘要"""
        runner = PhaseRunner(str(self.config_file))
        
        test_manifest = {
            'run_signature': 'test_signature_123',
            'output_dir': str(self.temp_dir / 'test_output'),
            'pool_signature': {'pool_json': 'test_pool.json', 'n_codes': 100},
            'field_signature': {'field_set_hash': 'field_hash_123'},
            'grammar_signature': {'template_yaml': 'test_templates.yaml'},
            'total_expressions': 1000,
            'valid_expressions': 800
        }
        
        runner._save_phase_summary("test", test_manifest, duration=123.45)
        
        # 验证摘要文件已创建
        summary_file = Path(test_manifest['output_dir']) / "test_summary.json"
        assert summary_file.exists()
        
        # 验证摘要内容
        import json
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        assert summary['phase'] == "test"
        assert summary['duration_seconds'] == 123.45
        assert summary['run_signature'] == 'test_signature_123'
        assert summary['total_expressions'] == 1000
        assert summary['valid_expressions'] == 800
    
    def test_generate_decay_report(self):
        """测试生成衰减报告"""
        runner = PhaseRunner(str(self.config_file))
        
        coarse_manifest = {
            'run_signature': 'coarse_sig_123',
            'output_dir': str(self.temp_dir / 'coarse_output'),
            'pool_signature': {'pool_json': 'static_pool_200.json'},
            'data_signature': {'start': '2024-01-01', 'end': '2024-06-30'}
        }
        
        fine_manifest = {
            'run_signature': 'fine_sig_456',
            'output_dir': str(self.temp_dir / 'fine_output'),
            'pool_signature': {'pool_json': 'static_pool_800.json'},
            'data_signature': {'start': '2024-01-01', 'end': '2024-12-31'}
        }
        
        runner._generate_decay_report(coarse_manifest, fine_manifest)
        
        # 验证衰减报告文件已创建
        decay_file = Path(fine_manifest['output_dir']) / "coarse_to_fine_decay_report.json"
        assert decay_file.exists()
        
        # 验证报告内容
        with open(decay_file, 'r') as f:
            report = json.load(f)
        
        assert report['coarse_run_signature'] == 'coarse_sig_123'
        assert report['fine_run_signature'] == 'fine_sig_456'
        assert 'coarse_pool' in report
        assert 'fine_pool' in report
        assert 'analysis_notes' in report
    
    def test_generate_final_report(self):
        """测试生成最终报告"""
        runner = PhaseRunner(str(self.config_file))
        
        coarse_manifest = {
            'run_signature': 'coarse_final_sig',
            'output_dir': str(self.temp_dir / 'final_coarse'),
            'pool_signature': {'pool_json': 'static_pool_200.json'},
            'data_signature': {'start': '2024-01-01', 'end': '2024-06-30'}
        }
        
        fine_manifest = {
            'run_signature': 'fine_final_sig',
            'output_dir': str(self.temp_dir / 'final_fine'),
            'pool_signature': {'pool_json': 'static_pool_800.json'},
            'data_signature': {'start': '2024-01-01', 'end': '2024-12-31'}
        }
        
        final_report = runner._generate_final_report(coarse_manifest, fine_manifest)
        
        # 验证最终报告
        assert isinstance(final_report, dict)
        assert final_report['pipeline_type'] == 'coarse_to_fine'
        assert 'coarse_phase' in final_report
        assert 'fine_phase' in final_report
        assert 'recommendations' in final_report
        
        # 验证报告文件已创建
        report_file = Path(fine_manifest['output_dir']) / "final_pipeline_report.json"
        assert report_file.exists()
        
        # 验证文件内容
        with open(report_file, 'r') as f:
            file_report = json.load(f)
        
        assert file_report['pipeline_type'] == 'coarse_to_fine'
        assert len(file_report['recommendations']) > 0
    
    def test_run_phase_runner_invalid_phase(self):
        """测试运行无效阶段"""
        with pytest.raises(ValueError, match="Unknown phase"):
            run_phase_runner(str(self.config_file), phase="invalid_phase")
    
    def test_load_coarse_manifest(self):
        """测试加载粗筛manifest"""
        runner = PhaseRunner(str(self.config_file))
        
        # 测试不存在的情况
        manifest = runner._load_coarse_manifest()
        assert manifest is None
        
        # 创建测试manifest文件
        test_manifest = {
            'run_signature': 'test_coarse_sig',
            'output_dir': str(self.temp_dir / 'test_coarse'),
            'pool_signature': {'pool_json': 'static_pool_200.json'}
        }
        
        coarse_output_dir = Path(self.temp_dir / 'test_coarse')
        coarse_output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest_file = coarse_output_dir / "manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(test_manifest, f)
        
        # 修改配置以指向正确的目录
        runner.config['phase_config']['coarse']['out_dir'] = str(coarse_output_dir)
        
        # 测试存在的情况
        manifest = runner._load_coarse_manifest()
        assert manifest is not None
        assert manifest['run_signature'] == 'test_coarse_sig'
    
    def test_phase_runner_integration_mock(self):
        """测试阶段运行器的集成（模拟模式）"""
        # 这是一个集成测试的框架，实际运行可能需要更多设置
        # 这里主要验证函数调用和配置传递
        
        runner = PhaseRunner(str(self.config_file))
        
        # 验证配置加载正确
        assert runner.config is not None
        assert 'base_config' in runner.config
        assert 'phase_config' in runner.config
        
        # 验证阶段配置存在
        assert 'coarse' in runner.config['phase_config']
        assert 'fine' in runner.config['phase_config']
        
        print("PhaseRunner integration test framework completed")
        print("Note: Full integration test would require actual data and longer execution time")