"""
测试模板配置加载功能
"""
import pytest
import tempfile
from pathlib import Path
import yaml

from src.alpha_mvp.template_config import (
    load_template_config, save_template_config, validate_template_config,
    parse_complexity_budgets, parse_template_families, get_template_by_family,
    get_enabled_templates, validate_template_consistency
)
from src.alpha_mvp.template_spec import TemplateSpec, ComplexityBudget

class TestTemplateConfig:
    """测试模板配置相关功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_config_path = self.temp_dir / "test_templates.yaml"
        
        # 创建测试配置
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
        with open(self.test_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.test_config, f)
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_template_config(self):
        """测试加载模板配置"""
        specs, budgets, raw = load_template_config(str(self.test_config_path))
        
        # 验证返回结果
        assert isinstance(specs, list)
        assert isinstance(budgets, dict)
        assert isinstance(raw, dict)
        
        # 验证规格数量
        assert len(specs) == 2  # 测试配置中有2个模板族
        
        # 验证复杂度预算
        assert 1 in budgets
        assert 2 in budgets
        assert isinstance(budgets[1], ComplexityBudget)
        
        # 验证模板规格
        for spec in specs:
            assert isinstance(spec, TemplateSpec)
            assert spec.name != ""
            assert spec.family != ""
            assert spec.order > 0
    
    def test_validate_template_config(self):
        """测试配置验证"""
        # 有效配置
        valid_config = {
            'version': 'test',
            'budgets': {'1': {'max_depth': 3, 'max_nodes': 10, 'max_ts_ops': 2, 'max_pair_ops': 1, 'max_binary_ops': 1, 'max_unary_ops': 2}},
            'families': [{'name': 'test', 'family': 'single', 'order': 1, 'enabled': True}]
        }
        
        # 不应该抛出异常
        validate_template_config(valid_config)
        
        # 无效配置 - 缺少必需字段
        invalid_config = {
            'budgets': {'1': {'max_depth': 3}},
            'families': [{'name': 'test'}]
        }
        
        with pytest.raises(ValueError, match="Missing required key"):
            validate_template_config(invalid_config)
    
    def test_parse_complexity_budgets(self):
        """测试解析复杂度预算"""
        budgets_raw = {
            '1': {'max_depth': 3, 'max_nodes': 10, 'max_ts_ops': 2, 'max_pair_ops': 1, 'max_binary_ops': 1, 'max_unary_ops': 2},
            '2': {'max_depth': 5, 'max_nodes': 15, 'max_ts_ops': 3, 'max_pair_ops': 1, 'max_binary_ops': 2, 'max_unary_ops': 3}
        }
        
        budgets = parse_complexity_budgets(budgets_raw)
        
        assert len(budgets) == 2
        assert 1 in budgets
        assert 2 in budgets
        assert isinstance(budgets[1], ComplexityBudget)
        assert budgets[1].max_depth == 3
        assert budgets[2].max_depth == 5
    
    def test_parse_template_families(self):
        """测试解析模板族"""
        families_raw = [
            {
                'name': 'single_test',
                'family': 'single',
                'order': 1,
                'enabled': True,
                'unary_pre': ['Id', 'Abs'],
                'ts_ops': ['TsMean', 'TsStd'],
                'outer_transforms': ['Rank'],
                'complexity_tier': 1
            }
        ]
        
        specs = parse_template_families(families_raw)
        
        assert len(specs) == 1
        spec = specs[0]
        assert isinstance(spec, TemplateSpec)
        assert spec.name == 'single_test'
        assert spec.family == 'single'
        assert spec.order == 1
        assert spec.enabled == True
        assert 'Id' in spec.unary_pre
        assert 'TsMean' in spec.ts_ops
    
    def test_save_template_config(self):
        """测试保存模板配置"""
        # 创建测试规格和预算
        specs = [
            TemplateSpec(
                name='test_single',
                family='single',
                order=1,
                enabled=True,
                unary_pre=('Id', 'Abs'),
                ts_ops=('TsMean', 'TsStd'),
                outer_transforms=('Rank',),
                complexity_tier=1
            )
        ]
        
        budgets = {
            1: ComplexityBudget(tier=1, max_depth=3, max_nodes=10, max_ts_ops=2, max_pair_ops=1, max_binary_ops=1, max_unary_ops=2)
        }
        
        output_path = self.temp_dir / "saved_templates.yaml"
        save_template_config(str(output_path), specs, budgets, version="test_save")
        
        # 验证文件已创建
        assert output_path.exists()
        
        # 加载保存的配置
        loaded_specs, loaded_budgets, loaded_raw = load_template_config(str(output_path))
        
        assert len(loaded_specs) == 1
        assert len(loaded_budgets) == 1
        assert loaded_raw['version'] == 'test_save'
    
    def test_get_template_by_family(self):
        """测试按族获取模板"""
        specs = [
            TemplateSpec(name='single1', family='single', order=1, enabled=True),
            TemplateSpec(name='binary1', family='binary', order=2, enabled=True)
        ]
        
        single_spec = get_template_by_family(specs, 'single')
        assert single_spec.name == 'single1'
        assert single_spec.family == 'single'
        
        binary_spec = get_template_by_family(specs, 'binary')
        assert binary_spec.name == 'binary1'
        assert binary_spec.family == 'binary'
        
        # 测试不存在的族
        with pytest.raises(ValueError, match="Template family 'nonexistent' not found"):
            get_template_by_family(specs, 'nonexistent')
    
    def test_get_enabled_templates(self):
        """测试获取启用的模板"""
        specs = [
            TemplateSpec(name='enabled1', family='single', order=1, enabled=True),
            TemplateSpec(name='disabled1', family='binary', order=2, enabled=False),
            TemplateSpec(name='enabled2', family='triple', order=3, enabled=True)
        ]
        
        enabled_specs = get_enabled_templates(specs)
        
        assert len(enabled_specs) == 2
        assert all(spec.enabled for spec in enabled_specs)
        assert enabled_specs[0].name == 'enabled1'
        assert enabled_specs[1].name == 'enabled2'
    
    def test_validate_template_consistency(self):
        """测试模板一致性验证"""
        # 创建测试预算和规格
        budgets = {
            1: ComplexityBudget(tier=1, max_depth=3, max_nodes=10, max_ts_ops=2, max_pair_ops=1, max_binary_ops=1, max_unary_ops=2)
        }
        
        # 有效的规格
        valid_specs = [
            TemplateSpec(name='test', family='single', order=1, enabled=True, complexity_tier=1)
        ]
        
        # 不应该抛出异常
        validate_template_consistency(valid_specs, budgets)
        
        # 无效的规格 - 引用了不存在的复杂度层级
        invalid_specs = [
            TemplateSpec(name='test', family='single', order=1, enabled=True, complexity_tier=99)
        ]
        
        # 应该抛出异常
        with pytest.raises(ValueError, match="invalid complexity_tier"):
            validate_template_consistency(invalid_specs, budgets)
    
    def test_load_nonexistent_config(self):
        """测试加载不存在的配置文件"""
        nonexistent_path = self.temp_dir / "nonexistent.yaml"
        
        with pytest.raises(FileNotFoundError, match="Template config file not found"):
            load_template_config(str(nonexistent_path))
    
    def test_parse_complexity_budgets_invalid_tier(self):
        """测试解析无效的复杂度预算层级"""
        invalid_budgets = {
            'invalid': {'max_depth': 3, 'max_nodes': 10, 'max_ts_ops': 2, 'max_pair_ops': 1, 'max_binary_ops': 1, 'max_unary_ops': 2}
        }
        
        with pytest.raises(ValueError, match="Invalid tier key"):
            parse_complexity_budgets(invalid_budgets)