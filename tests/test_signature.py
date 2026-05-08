"""
测试签名系统功能
"""
import pytest
import tempfile
from pathlib import Path
import json
import hashlib

from src.alpha_mvp.signature import (
    file_sha256, get_git_hash_or_unknown, compute_pool_signature,
    compute_template_yaml_hash, compute_data_signature, compute_field_signature,
    compute_grammar_signature, compute_eval_signature, build_complete_manifest,
    compute_run_signature, build_manifest_from_config
)

class TestSignature:
    """测试签名系统功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # 创建测试文件
        self.test_file = self.temp_dir / "test_file.txt"
        self.test_file.write_text("test content for signature")
        
        # 创建测试股票池
        self.test_pool = {
            'name': 'test_pool',
            'description': 'Test stock pool for unit tests',
            'codes': [f'STCK{i:04d}' for i in range(10)],
            'creation_date': '2024-01-01',
            'criteria': {
                'min_listing_days': 252,
                'min_market_cap': 1e9,
                'max_st_days_ratio': 0.05
            }
        }
        self.pool_file = self.temp_dir / "test_pool.json"
        with open(self.pool_file, 'w') as f:
            json.dump(self.test_pool, f)
        
        # 创建测试模板配置
        self.test_config = {
            'version': 'test_templates_v1',
            'windows': [10, 20, 30],
            'short_windows': [5, 10],
            'long_windows': [20, 30],
            'budgets': {
                '1': {'max_depth': 3, 'max_nodes': 10, 'max_ts_ops': 2, 'max_pair_ops': 1, 'max_binary_ops': 1, 'max_unary_ops': 2},
                '2': {'max_depth': 4, 'max_nodes': 15, 'max_ts_ops': 3, 'max_pair_ops': 1, 'max_binary_ops': 2, 'max_unary_ops': 3}
            },
            'families': [
                {
                    'name': 'single_ts_test',
                    'family': 'single',
                    'order': 1,
                    'enabled': True,
                    'unary_pre': ['Id', 'Abs'],
                    'ts_ops': ['TsMean', 'TsStd', 'TsRank'],
                    'outer_transforms': ['Rank'],
                    'complexity_tier': 1
                },
                {
                    'name': 'binary_same_ts_test',
                    'family': 'binary_same_ts',
                    'order': 2,
                    'enabled': True,
                    'ts_ops': ['TsMean', 'TsDelta'],
                    'binary_ops': ['Sub', 'Div', 'Mul'],
                    'outer_transforms': ['Rank'],
                    'complexity_tier': 2
                }
            ]
        }
        self.config_file = self.temp_dir / "test_templates.yaml"
        import yaml
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_file_sha256(self):
        """测试文件SHA256计算"""
        # 测试存在的文件
        hash1 = file_sha256(str(self.test_file))
        assert len(hash1) == 64  # SHA256哈希长度
        assert hash1 != "file_not_found"
        
        # 测试不存在的文件
        hash2 = file_sha256(str(self.temp_dir / "nonexistent.txt"))
        assert hash2 == "file_not_found"
        
        # 相同内容应该产生相同的哈希
        same_file = self.temp_dir / "same_content.txt"
        same_file.write_text("test content for signature")
        hash3 = file_sha256(str(same_file))
        assert hash1 == hash3
    
    def test_get_git_hash_or_unknown(self):
        """测试获取git哈希"""
        # 这个测试可能返回"unknown"，取决于测试环境是否有git
        result = get_git_hash_or_unknown()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_compute_pool_signature(self):
        """测试计算股票池签名"""
        signature = compute_pool_signature(str(self.pool_file))
        
        assert isinstance(signature, dict)
        assert 'pool_json' in signature
        assert 'pool_file_hash' in signature
        assert 'n_codes' in signature
        
        assert signature['pool_json'] == str(self.pool_file)
        assert signature['pool_file_hash'] != "file_not_found"
        assert signature['n_codes'] == 10
        
        # 测试不存在的文件
        with pytest.raises(FileNotFoundError):
            compute_pool_signature(str(self.temp_dir / "nonexistent.json"))
    
    def test_compute_template_yaml_hash(self):
        """测试计算模板YAML哈希"""
        hash_value = compute_template_yaml_hash(str(self.config_file))
        
        assert len(hash_value) == 64  # SHA256哈希长度
        assert hash_value != "file_not_found"
    
    def test_compute_field_signature(self):
        """测试计算字段签名"""
        fields = ['ret_1d', 'ret_5d', 'vol_log']
        
        # 测试默认版本
        signature1 = compute_field_signature(fields)
        assert isinstance(signature1, dict)
        assert 'field_set_hash' in signature1
        assert 'field_formula_version' in signature1
        assert len(signature1['field_set_hash']) > 0
        assert signature1['field_formula_version'].startswith('2026-')  # 默认使用时间戳
        
        # 测试指定版本
        version = "2024-01-01-v1"
        signature2 = compute_field_signature(fields, field_formula_version=version)
        assert signature2['field_formula_version'] == version
        
        # 不同字段应该产生不同的哈希
        different_fields = ['pe_ratio', 'pb_ratio']
        signature3 = compute_field_signature(different_fields)
        assert signature3['field_set_hash'] != signature1['field_set_hash']
    
    def test_compute_grammar_signature(self):
        """测试计算语法签名"""
        signature = compute_grammar_signature(
            template_yaml="configs/templates_v1.yaml",
            template_yaml_hash="abc123",
            seed=42,
            max_exprs=10000
        )
        
        assert isinstance(signature, dict)
        assert signature['template_yaml'] == "configs/templates_v1.yaml"
        assert signature['template_yaml_hash'] == "abc123"
        assert signature['seed'] == 42
        assert signature['max_exprs'] == 10000
    
    def test_compute_eval_signature(self):
        """测试计算评估签名"""
        signature = compute_eval_signature(
            forward_days=5,
            train_end="2024-08-31",
            test_start="2024-09-01",
            min_daily_valid_names=30
        )
        
        assert isinstance(signature, dict)
        assert signature['forward_days'] == 5
        assert signature['train_end'] == "2024-08-31"
        assert signature['test_start'] == "2024-09-01"
        assert signature['min_daily_valid_names'] == 30
    
    def test_build_complete_manifest(self):
        """测试构建完整清单"""
        manifest = build_complete_manifest(
            project_version="test-v1",
            code_hash="abc123def456",
            data_signature={"start": "2024-01-01", "end": "2024-12-31"},
            pool_signature={"pool_json": "test.json", "n_codes": 100},
            field_signature={"field_set_hash": "field123", "field_formula_version": "v1"},
            grammar_signature={"template_yaml": "templates.yaml", "seed": 42},
            eval_signature={"forward_days": 5, "train_end": "2024-08-31"},
            extra_info={"test_key": "test_value"}
        )
        
        assert isinstance(manifest, dict)
        assert manifest['project_version'] == "test-v1"
        assert manifest['code_hash'] == "abc123def456"
        assert 'data_signature' in manifest
        assert 'pool_signature' in manifest
        assert 'field_signature' in manifest
        assert 'grammar_signature' in manifest
        assert 'eval_signature' in manifest
        assert 'created_at' in manifest
        assert 'extra_info' in manifest
        assert manifest['extra_info']['test_key'] == "test_value"
    
    def test_compute_run_signature(self):
        """测试计算运行签名"""
        manifest = {
            "project_version": "test-v1",
            "code_hash": "abc123",
            "data_signature": {"start": "2024-01-01"},
            "pool_signature": {"n_codes": 100},
            "field_signature": {"field_set_hash": "field123"},
            "grammar_signature": {"seed": 42},
            "eval_signature": {"forward_days": 5},
            "created_at": "2024-01-01T00:00:00",
            "job_id": "test_job_123"
        }
        
        signature1 = compute_run_signature(manifest)
        
        assert isinstance(signature1, str)
        assert len(signature1) == 64  # SHA256哈希长度
        
        # 相同的稳定内容应该产生相同的签名（忽略时间戳和job_id）
        manifest2 = manifest.copy()
        manifest2["created_at"] = "2024-01-02T00:00:00"
        manifest2["job_id"] = "different_job"
        
        signature2 = compute_run_signature(manifest2)
        assert signature1 == signature2, "Run signature should be stable (ignore time-dependent fields)"
        
        # 不同的内容应该产生不同的签名
        manifest3 = manifest.copy()
        manifest3["project_version"] = "different-version"
        
        signature3 = compute_run_signature(manifest3)
        assert signature3 != signature1, "Different manifest should produce different signature"
    
    def test_build_manifest_from_config_mock(self):
        """测试从配置构建清单（使用模拟数据）"""
        from src.alpha_mvp.config import RunConfig, EvalConfig
        
        # 创建测试配置
        eval_config = EvalConfig(
            forward_days=5,
            windows=[10, 20, 30],
            min_daily_valid_names=30
        )
        
        cfg = RunConfig(
            duckdb_path=str(self.temp_dir / "test.duckdb"),
            pool_json=str(self.pool_file),
            start="2024-01-01",
            end="2024-12-31",
            fields=['ret_1d', 'ret_5d'],
            max_exprs=1000,
            seed=42,
            eval=eval_config,
            use_simulated=True,  # 使用模拟数据避免文件依赖
            train_end="2024-08-31",
            test_start="2024-09-01"
        )
        
        selected_fields = ['ret_1d', 'ret_5d', 'vol_log']
        
        # 构建清单
        manifest = build_manifest_from_config(cfg, str(self.config_file), selected_fields)
        
        assert isinstance(manifest, dict)
        assert 'project_version' in manifest
        assert 'code_hash' in manifest
        assert 'data_signature' in manifest
        assert 'pool_signature' in manifest
        assert 'field_signature' in manifest
        assert 'grammar_signature' in manifest
        assert 'eval_signature' in manifest
        
        # 验证池签名
        assert manifest['pool_signature']['pool_json'] == str(self.pool_file)
        assert manifest['pool_signature']['n_codes'] == 10
        
        # 验证字段签名
        assert manifest['field_signature']['field_formula_version'] is not None
        
        # 验证语法签名
        assert manifest['grammar_signature']['template_yaml'] == str(self.config_file)
        assert manifest['grammar_signature']['seed'] == 42
        assert manifest['grammar_signature']['max_exprs'] == 1000