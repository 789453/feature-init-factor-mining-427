"""
YAML模板配置加载模块
"""
from __future__ import annotations
import yaml
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .template_spec import TemplateSpec, ComplexityBudget

def validate_template_config(raw: dict) -> None:
    """验证YAML配置文件的必需字段"""
    required_keys = ["version", "budgets", "families"]
    for key in required_keys:
        if key not in raw:
            raise ValueError(f"Missing required key '{key}' in template config")
    
    # 验证budgets
    if not isinstance(raw["budgets"], dict):
        raise ValueError("budgets must be a dictionary")
    
    # 验证families
    if not isinstance(raw["families"], list):
        raise ValueError("families must be a list")
    
    for i, family in enumerate(raw["families"]):
        family_required = ["name", "family", "order", "enabled"]
        for key in family_required:
            if key not in family:
                raise ValueError(f"Missing required key '{key}' in family {i}")

def parse_complexity_budgets(budgets_raw: dict) -> Dict[int, ComplexityBudget]:
    """解析复杂度预算配置"""
    budgets = {}
    for tier_str, budget_config in budgets_raw.items():
        try:
            tier = int(tier_str)
        except ValueError:
            raise ValueError(f"Invalid tier key: {tier_str}")
        
        if not isinstance(budget_config, dict):
            raise ValueError(f"Budget for tier {tier} must be a dictionary")
        
        required_fields = [
            "max_depth", "max_nodes", "max_ts_ops", 
            "max_pair_ops", "max_binary_ops", "max_unary_ops"
        ]
        
        for field in required_fields:
            if field not in budget_config:
                raise ValueError(f"Missing required field '{field}' in budget tier {tier}")
        
        budgets[tier] = ComplexityBudget(
            tier=tier,
            max_depth=budget_config["max_depth"],
            max_nodes=budget_config["max_nodes"],
            max_ts_ops=budget_config["max_ts_ops"],
            max_pair_ops=budget_config["max_pair_ops"],
            max_binary_ops=budget_config["max_binary_ops"],
            max_unary_ops=budget_config["max_unary_ops"]
        )
    
    return budgets

def parse_template_families(families_raw: list) -> List[TemplateSpec]:
    """解析模板族配置"""
    specs = []
    
    for family_config in families_raw:
        # 基础字段
        name = family_config["name"]
        family = family_config["family"]
        order = family_config["order"]
        enabled = family_config.get("enabled", True)
        complexity_tier = family_config.get("complexity_tier", 1)
        max_count = family_config.get("max_count")
        
        # 序列字段（转换为元组）
        outer_transforms = tuple(family_config.get("outer_transforms", ("Rank",)))
        unary_pre = tuple(family_config.get("unary_pre", ("Id",)))
        ts_ops = tuple(family_config.get("ts_ops", ()))
        binary_ops = tuple(family_config.get("binary_ops", ()))
        pair_ops = tuple(family_config.get("pair_ops", ()))
        windows = tuple(family_config.get("windows", ()))
        short_windows = tuple(family_config.get("short_windows", ()))
        long_windows = tuple(family_config.get("long_windows", ()))
        
        # 特殊处理：forms（用于triple和quad模板）
        forms = family_config.get("forms", [])
        
        spec = TemplateSpec(
            name=name,
            family=family,
            order=order,
            enabled=enabled,
            max_count=max_count,
            outer_transforms=outer_transforms,
            unary_pre=unary_pre,
            ts_ops=ts_ops,
            binary_ops=binary_ops,
            pair_ops=pair_ops,
            windows=windows,
            short_windows=short_windows,
            long_windows=long_windows,
            complexity_tier=complexity_tier
        )
        
        # 将forms存储在spec的额外属性中（需要修改TemplateSpec类）
        if forms:
            # 由于TemplateSpec是frozen，我们需要使用object.__setattr__
            object.__setattr__(spec, '_forms', forms)
        
        specs.append(spec)
    
    return specs

def load_template_config(path: str) -> Tuple[List[TemplateSpec], Dict[int, ComplexityBudget], Dict[str, Any]]:
    """
    加载YAML模板配置文件
    
    Args:
        path: YAML文件路径
        
    Returns:
        (模板规格列表, 复杂度预算字典, 原始配置字典)
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Template config file not found: {path}")
    
    with path_obj.open('r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    
    # 验证配置
    validate_template_config(raw)
    
    # 解析复杂度预算
    budgets = parse_complexity_budgets(raw["budgets"])
    
    # 解析模板族
    specs = parse_template_families(raw["families"])
    
    return specs, budgets, raw

def save_template_config(
    path: str,
    specs: List[TemplateSpec],
    budgets: Dict[int, ComplexityBudget],
    version: str = "templates_v1"
) -> None:
    """将模板配置保存为YAML文件"""
    config = {
        "version": version,
        "windows": [10, 20, 30, 40, 50],
        "short_windows": [5, 10, 20],
        "long_windows": [40, 60, 80]
    }
    
    # 转换预算
    budgets_dict = {}
    for tier, budget in budgets.items():
        budgets_dict[str(tier)] = {
            "max_depth": budget.max_depth,
            "max_nodes": budget.max_nodes,
            "max_ts_ops": budget.max_ts_ops,
            "max_pair_ops": budget.max_pair_ops,
            "max_binary_ops": budget.max_binary_ops,
            "max_unary_ops": budget.max_unary_ops
        }
    config["budgets"] = budgets_dict
    
    # 转换模板族
    families = []
    for spec in specs:
        family = {
            "name": spec.name,
            "family": spec.family,
            "order": spec.order,
            "enabled": spec.enabled,
            "complexity_tier": spec.complexity_tier
        }
        
        if spec.max_count:
            family["max_count"] = spec.max_count
        
        # 添加序列字段
        if spec.outer_transforms:
            family["outer_transforms"] = list(spec.outer_transforms)
        if spec.unary_pre:
            family["unary_pre"] = list(spec.unary_pre)
        if spec.ts_ops:
            family["ts_ops"] = list(spec.ts_ops)
        if spec.binary_ops:
            family["binary_ops"] = list(spec.binary_ops)
        if spec.pair_ops:
            family["pair_ops"] = list(spec.pair_ops)
        if spec.windows:
            family["windows"] = list(spec.windows)
        if spec.short_windows:
            family["short_windows"] = list(spec.short_windows)
        if spec.long_windows:
            family["long_windows"] = list(spec.long_windows)
        
        families.append(family)
    
    config["families"] = families
    
    # 保存为YAML
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with path_obj.open('w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

def create_default_templates_yaml(output_path: str) -> None:
    """创建默认的模板YAML配置文件"""
    from .template_spec import TEMPLATE_SPECS, COMPLEXITY_BUDGETS
    
    save_template_config(
        output_path,
        TEMPLATE_SPECS,
        COMPLEXITY_BUDGETS,
        version="templates_v1"
    )

def get_template_by_family(specs: List[TemplateSpec], family: str) -> TemplateSpec:
    """根据family名称获取模板规格"""
    for spec in specs:
        if spec.family == family:
            return spec
    raise ValueError(f"Template family '{family}' not found")

def get_enabled_templates(specs: List[TemplateSpec]) -> List[TemplateSpec]:
    """获取所有启用的模板规格"""
    return [spec for spec in specs if spec.enabled]

def validate_template_consistency(specs: List[TemplateSpec], budgets: Dict[int, ComplexityBudget]) -> None:
    """验证模板规格与复杂度预算的一致性"""
    for spec in specs:
        if spec.complexity_tier not in budgets:
            raise ValueError(f"Template '{spec.name}' has invalid complexity_tier: {spec.complexity_tier}")
        
        # 可以添加更多验证逻辑
        if spec.family in ["triple", "quad"] and not hasattr(spec, '_forms'):
            print(f"Warning: Template '{spec.name}' is high-order but has no forms defined")