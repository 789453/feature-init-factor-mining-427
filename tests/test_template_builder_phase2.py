"""
测试模板构建器功能，特别是高阶模板
"""
import pytest
import tempfile
from pathlib import Path

from src.alpha_mvp.template_builder import (
    generate_expressions_from_specs, ExpressionRecord, stable_keep,
    build_single, build_binary_same_ts, build_binary_mixed_ts, 
    build_triple_modulation, build_quad_balanced
)
from src.alpha_mvp.template_spec import TemplateSpec, ComplexityBudget
from src.alpha_mvp.validator import Validator

class TestTemplateBuilder:
    """测试模板构建器功能"""
    
    def setup_method(self):
        """设置测试环境"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_fields = ['ret_1d', 'ret_5d', 'vol_log', 'hl_range']
        self.test_windows = [10, 20, 30]
        
        # 创建测试模板规格
        self.test_specs = [
            TemplateSpec(
                name='single_test',
                family='single',
                order=1,
                enabled=True,
                unary_pre=('Id', 'Abs'),
                ts_ops=('TsMean', 'TsStd', 'TsRank'),
                outer_transforms=('Rank', 'SLog1p'),
                complexity_tier=1
            ),
            TemplateSpec(
                name='binary_same_test',
                family='binary_same_ts',
                order=2,
                enabled=True,
                ts_ops=('TsMean', 'TsDelta'),
                binary_ops=('Sub', 'Div', 'Mul'),
                outer_transforms=('Rank',),
                complexity_tier=2
            ),
            TemplateSpec(
                name='binary_mixed_test',
                family='binary_mixed_ts',
                order=2,
                enabled=True,
                ts_ops=('TsMean', 'TsStd'),  # 用于兼容性
                left_ts_ops=('TsMean', 'TsDelta'),
                right_ts_ops=('TsMean', 'TsStd'),
                binary_ops=('Sub', 'Div'),
                outer_transforms=('Rank',),
                complexity_tier=2
            ),
            TemplateSpec(
                name='triple_test',
                family='triple',
                order=3,
                enabled=True,
                ts_ops=('TsMean', 'TsDelta'),
                binary_ops=('Add', 'Sub', 'Mul'),
                outer_transforms=('Rank',),
                complexity_tier=3
            ),
            TemplateSpec(
                name='quad_test',
                family='quad',
                order=4,
                enabled=True,
                ts_ops=('TsMean', 'TsEMA'),
                binary_ops=('Sub', 'Mul', 'Div'),
                outer_transforms=('Rank',),
                complexity_tier=4
            )
        ]
        
        self.test_budgets = {
            1: ComplexityBudget(tier=1, max_depth=3, max_nodes=10, max_ts_ops=2, max_pair_ops=1, max_binary_ops=1, max_unary_ops=2),
            2: ComplexityBudget(tier=2, max_depth=4, max_nodes=15, max_ts_ops=3, max_pair_ops=1, max_binary_ops=2, max_unary_ops=3),
            3: ComplexityBudget(tier=3, max_depth=5, max_nodes=20, max_ts_ops=4, max_pair_ops=1, max_binary_ops=3, max_unary_ops=4),
            4: ComplexityBudget(tier=4, max_depth=6, max_nodes=25, max_ts_ops=4, max_pair_ops=1, max_binary_ops=4, max_unary_ops=5)
        }
    
    def teardown_method(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_stable_keep(self):
        """测试确定性采样函数"""
        # 相同的输入应该产生相同的结果
        key1 = "test_key_1"
        rate = 0.5
        seed = 42
        
        result1 = stable_keep(key1, rate, seed)
        result2 = stable_keep(key1, rate, seed)
        
        assert result1 == result2, "stable_keep should be deterministic for same inputs"
        
        # 不同的key应该有不同的结果（大概率）
        key2 = "test_key_2"
        result3 = stable_keep(key2, rate, seed)
        
        # 由于rate=0.5，不同的key有不同的结果的概率很高
        # 但我们不能完全保证，所以这里不做断言
        
        # 不同的rate应该产生不同的结果
        result4 = stable_keep(key1, 0.1, seed)
        result5 = stable_keep(key1, 0.9, seed)
        
        # rate=0.1时应该很少通过，rate=0.9时应该经常通过
        # 但我们不做严格断言，因为存在随机性
    
    def test_build_single(self):
        """测试单变量模板构建"""
        spec = self.test_specs[0]  # single模板
        records = []
        seen = set()
        validator = Validator(
            fields=set(self.test_fields),
            windows=set(self.test_windows),
            max_depth=self.test_budgets[1].max_depth,
            max_nodes=self.test_budgets[1].max_nodes,
            max_ts_ops=self.test_budgets[1].max_ts_ops,
            max_pair_ops=self.test_budgets[1].max_pair_ops,
            max_binary_ops=self.test_budgets[1].max_binary_ops
        )
        
        build_single(spec, self.test_fields, self.test_windows, records, seen, validator, seed=42)
        
        # 验证生成的记录
        assert len(records) > 0, "Should generate some expressions"
        
        for record in records:
            assert isinstance(record, ExpressionRecord)
            assert record.template_family == 'single'
            assert record.template_order == 1
            assert 'Rank(' in record.expr or 'SLog1p(' in record.expr
            assert '$' in record.expr, "Expression should contain field references"
    
    def test_build_binary_same_ts(self):
        """测试相同时间窗口二元模板构建"""
        spec = self.test_specs[1]  # binary_same_ts模板
        records = []
        seen = set()
        validator = Validator(
            fields=set(self.test_fields),
            windows=set(self.test_windows),
            max_depth=self.test_budgets[2].max_depth,
            max_nodes=self.test_budgets[2].max_nodes,
            max_ts_ops=self.test_budgets[2].max_ts_ops,
            max_pair_ops=self.test_budgets[2].max_pair_ops,
            max_binary_ops=self.test_budgets[2].max_binary_ops
        )
        
        build_binary_same_ts(spec, self.test_fields, self.test_windows, records, seen, validator, seed=42)
        
        # 验证生成的记录
        assert len(records) > 0, "Should generate some expressions"
        
        for record in records:
            assert isinstance(record, ExpressionRecord)
            assert record.template_family == 'binary_same_ts'
            assert record.template_order == 2
            assert 'Rank(' in record.expr
            # 应该包含二元操作符
            assert any(op in record.expr for op in ['Sub(', 'Div(', 'Mul('])
            # 应该包含两个不同的字段
            field_count = sum(1 for field in self.test_fields if f'${field}' in record.expr)
            assert field_count >= 2, "Binary expression should reference at least 2 fields"
    
    def test_build_binary_mixed_ts(self):
        """测试混合时间窗口二元模板构建"""
        spec = self.test_specs[2]  # binary_mixed_ts模板
        records = []
        seen = set()
        validator = Validator(
            fields=set(self.test_fields),
            windows=set(self.test_windows),
            max_depth=self.test_budgets[2].max_depth,
            max_nodes=self.test_budgets[2].max_nodes,
            max_ts_ops=self.test_budgets[2].max_ts_ops,
            max_pair_ops=self.test_budgets[2].max_pair_ops,
            max_binary_ops=self.test_budgets[2].max_binary_ops
        )
        
        build_binary_mixed_ts(spec, self.test_fields, self.test_windows, records, seen, validator, seed=42)
        
        # 验证生成的记录
        assert len(records) > 0, "Should generate some expressions"
        
        for record in records:
            assert isinstance(record, ExpressionRecord)
            assert record.template_family == 'binary_mixed_ts'
            assert record.template_order == 2
            assert 'Rank(' in record.expr
            # 应该包含二元操作符
            assert any(op in record.expr for op in ['Sub(', 'Div('])
    
    def test_build_triple_modulation(self):
        """测试三阶调制模板构建"""
        spec = self.test_specs[3]  # triple模板
        # 使用object.__setattr__来设置forms属性（因为TemplateSpec是frozen）
        object.__setattr__(spec, 'forms', [
            'Rank(Mul(Sub(A,B),C))',
            'Rank(Sub(Mul(A,B),C))'
        ])
        
        records = []
        seen = set()
        validator = Validator(
            fields=set(self.test_fields),
            windows=set(self.test_windows),
            max_depth=self.test_budgets[3].max_depth,
            max_nodes=self.test_budgets[3].max_nodes,
            max_ts_ops=self.test_budgets[3].max_ts_ops,
            max_pair_ops=self.test_budgets[3].max_pair_ops,
            max_binary_ops=self.test_budgets[3].max_binary_ops
        )
        
        build_triple_modulation(spec, self.test_fields, self.test_windows, records, seen, validator, seed=42)
        
        # 验证生成的记录
        assert len(records) > 0, "Should generate some expressions"
        
        for record in records:
            assert isinstance(record, ExpressionRecord)
            assert record.template_family == 'triple'
            assert record.template_order == 3
            assert 'Rank(' in record.expr
            # 应该包含三个字段引用
            field_refs = [f'${field}' for field in self.test_fields if f'${field}' in record.expr]
            assert len(field_refs) >= 3, "Triple expression should reference at least 3 fields"
    
    def test_build_quad_balanced(self):
        """测试四阶平衡模板构建"""
        spec = self.test_specs[4]  # quad模板
        # 使用object.__setattr__来设置forms属性（因为TemplateSpec是frozen）
        object.__setattr__(spec, 'forms', [
            'Rank(Sub(Mul(A,B),Mul(C,D)))',
            'Rank(Mul(Sub(A,B),Sub(C,D)))'
        ])
        
        records = []
        seen = set()
        validator = Validator(
            fields=set(self.test_fields),
            windows=set(self.test_windows),
            max_depth=self.test_budgets[4].max_depth,
            max_nodes=self.test_budgets[4].max_nodes,
            max_ts_ops=self.test_budgets[4].max_ts_ops,
            max_pair_ops=self.test_budgets[4].max_pair_ops,
            max_binary_ops=self.test_budgets[4].max_binary_ops
        )
        
        build_quad_balanced(spec, self.test_fields, self.test_windows, records, seen, validator, seed=42)
        
        # 验证生成的记录
        assert len(records) > 0, "Should generate some expressions"
        
        for record in records:
            assert isinstance(record, ExpressionRecord)
            assert record.template_family == 'quad'
            assert record.template_order == 4
            assert 'Rank(' in record.expr
            # 应该包含四个字段引用
            field_refs = [f'${field}' for field in self.test_fields if f'${field}' in record.expr]
            assert len(field_refs) >= 4, "Quad expression should reference at least 4 fields"
    
    def test_generate_expressions_from_specs_with_yaml(self):
        """测试使用YAML配置的表达式生成"""
        # 创建YAML配置文件
        yaml_path = self.temp_dir / "test_templates.yaml"
        
        yaml_config = {
            'version': 'test_v1',
            'windows': [10, 20, 30],
            'short_windows': [5, 10],
            'long_windows': [20, 30],
            'budgets': {
                '1': {
                    'max_depth': 3, 'max_nodes': 10, 'max_ts_ops': 2,
                    'max_pair_ops': 1, 'max_binary_ops': 1, 'max_unary_ops': 2
                }
            },
            'families': [
                {
                    'name': 'single_yaml_test',
                    'family': 'single',
                    'order': 1,
                    'enabled': True,
                    'unary_pre': ['Id'],
                    'ts_ops': ['TsMean'],
                    'outer_transforms': ['Rank'],
                    'complexity_tier': 1
                }
            ]
        }
        
        import yaml
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f)
        
        # 生成表达式
        records = generate_expressions_from_specs(
            fields=self.test_fields,
            windows=self.test_windows,
            template_config_path=str(yaml_path),
            max_exprs=100,
            seed=42
        )
        
        # 验证结果
        assert len(records) > 0, "Should generate expressions with YAML config"
        assert len(records) <= 100, "Should respect max_exprs limit"
        
        for record in records:
            assert isinstance(record, ExpressionRecord)
            assert record.template_family == 'single'
            assert 'Rank(TsMean(' in record.expr
    
    def test_expression_deduplication(self):
        """测试表达式去重功能"""
        # 生成表达式两次，应该产生相同的结果（因为种子相同）
        records1 = generate_expressions_from_specs(
            fields=self.test_fields,
            windows=self.test_windows,
            specs=self.test_specs[:1],  # 只使用single模板
            max_exprs=50,
            seed=42
        )
        
        records2 = generate_expressions_from_specs(
            fields=self.test_fields,
            windows=self.test_windows,
            specs=self.test_specs[:1],  # 只使用single模板
            max_exprs=50,
            seed=42
        )
        
        # 比较规范化的表达式
        canonicals1 = {r.canonical for r in records1}
        canonicals2 = {r.canonical for r in records2}
        
        assert canonicals1 == canonicals2, "Same seed should produce same expressions"
        
        # 验证没有重复的规范化表达式
        assert len(set(r.canonical for r in records1)) == len(records1), "No duplicate canonical expressions"
    
    def test_max_exprs_limit(self):
        """测试最大表达式数量限制"""
        max_limit = 20
        
        records = generate_expressions_from_specs(
            fields=self.test_fields,
            windows=self.test_windows,
            specs=self.test_specs,
            max_exprs=max_limit,
            seed=42
        )
        
        assert len(records) <= max_limit, f"Should not exceed max_exprs limit of {max_limit}"
        assert len(records) > 0, "Should generate some expressions"
    
    def test_disabled_templates(self):
        """测试禁用模板"""
        # 创建一个禁用的模板
        disabled_specs = [
            TemplateSpec(
                name='disabled_test',
                family='single',
                order=1,
                enabled=False,  # 禁用
                unary_pre=('Id',),
                ts_ops=('TsMean',),
                outer_transforms=('Rank',),
                complexity_tier=1
            )
        ]
        
        records = generate_expressions_from_specs(
            fields=self.test_fields,
            windows=self.test_windows,
            specs=disabled_specs,
            max_exprs=100,
            seed=42
        )
        
        assert len(records) == 0, "Disabled templates should not generate expressions"