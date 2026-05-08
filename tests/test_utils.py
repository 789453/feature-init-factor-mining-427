"""
测试基础模块
提供测试用的工具函数和模拟数据
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any

def create_test_data(n_codes: int = 50, n_days: int = 100, seed: int = 42) -> pd.DataFrame:
    """创建测试用的市场数据"""
    np.random.seed(seed)
    
    # 生成日期
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    dates = dates[dates.weekday < 5]  # 只保留工作日
    
    # 生成股票代码
    codes = [f'STCK{i:04d}' for i in range(n_codes)]
    
    # 生成基础数据
    data = []
    for date in dates:
        for code in codes:
            # 基础价格数据
            base_price = 10 + np.random.randn() * 5
            high = base_price * (1 + np.random.uniform(0.01, 0.05))
            low = base_price * (1 - np.random.uniform(0.01, 0.05))
            open_price = np.random.uniform(low, high)
            close = np.random.uniform(low, high)
            
            # 成交量
            volume = np.random.randint(100000, 1000000)
            
            # 收益率
            ret_1d = np.random.randn() * 0.02
            ret_5d = np.random.randn() * 0.05
            ret_20d = np.random.randn() * 0.1
            
            # 技术指标
            hl_range = (high - low) / high
            oc_ret = (close - open_price) / open_price
            co_ret = (open_price - close) / close
            vol_log = np.log(volume)
            
            data.append({
                'date': date,
                'code': code,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
                'ret_1d': ret_1d,
                'ret_5d': ret_5d,
                'ret_20d': ret_20d,
                'hl_range': hl_range,
                'oc_ret': oc_ret,
                'co_ret': co_ret,
                'vol_log': vol_log,
                'vol_std_20': np.random.uniform(0.1, 0.3),
                'vol_ratio': np.random.uniform(0.5, 2.0),
                'rsi_14': np.random.uniform(0, 100),
                'macd_diff': np.random.randn() * 0.02,
                'boll_position': np.random.uniform(-2, 2),
                'pe_ratio': np.random.uniform(5, 50),
                'pb_ratio': np.random.uniform(0.5, 5),
                'market_cap': np.random.uniform(1e9, 1e11)
            })
    
    return pd.DataFrame(data)

def create_test_pool(n_codes: int = 50) -> Dict[str, Any]:
    """创建测试用的股票池配置"""
    codes = [f'STCK{i:04d}' for i in range(n_codes)]
    
    return {
        'name': 'test_pool',
        'description': 'Test stock pool for unit tests',
        'codes': codes,
        'creation_date': '2024-01-01',
        'criteria': {
            'min_listing_days': 252,
            'min_market_cap': 1e9,
            'max_st_days_ratio': 0.05
        }
    }

def create_temp_directory() -> Path:
    """创建临时目录用于测试"""
    return Path(tempfile.mkdtemp())

def cleanup_temp_directory(path: Path) -> None:
    """清理临时目录"""
    if path.exists():
        shutil.rmtree(path)

def create_test_template_config() -> Dict[str, Any]:
    """创建测试用的模板配置"""
    return {
        'version': 'test_templates_v1',
        'windows': [10, 20, 30],
        'short_windows': [5, 10],
        'long_windows': [20, 30],
        'budgets': {
            '1': {
                'max_depth': 3,
                'max_nodes': 8,
                'max_ts_ops': 1,
                'max_pair_ops': 0,
                'max_binary_ops': 1,
                'max_unary_ops': 2
            },
            '2': {
                'max_depth': 4,
                'max_nodes': 12,
                'max_ts_ops': 2,
                'max_pair_ops': 1,
                'max_binary_ops': 2,
                'max_unary_ops': 3
            }
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
            },
            {
                'name': 'binary_mixed_ts_test',
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
                'name': 'triple_modulation_test',
                'family': 'triple',
                'order': 3,
                'enabled': True,
                'ts_ops': ['TsMean', 'TsDelta'],
                'forms': [
                    'Rank(Mul(Sub(A,B),C))',
                    'Rank(Sub(Mul(A,B),C))'
                ],
                'max_count': 1000,
                'complexity_tier': 2
            }
        ]
    }

def create_test_expressions() -> List[str]:
    """创建测试用的表达式列表"""
    return [
        "Rank(TsMean($ret_1d,10))",
        "Rank(TsStd($vol_log,20))",
        "Rank(Sub(TsMean($close,10),TsMean($close,20)))",
        "Rank(Div(TsMean($ret_1d,5),TsStd($ret_1d,20)))",
        "Rank(Mul(Sub(TsMean($ret_1d,10),TsMean($ret_5d,10)),$ret_20d))",
        "Rank(Sub(Mul($ret_1d,$ret_5d),$ret_20d))",
        "SLog1p(TsMean($hl_range,15))",
        "Rank(Abs(TsDelta($close,10)))",
        "Rank(TsRank($volume,20))",
        "Rank(TsIr($ret_1d,30))"
    ]

def create_test_run_config() -> Dict[str, Any]:
    """创建测试用的运行配置"""
    return {
        'duckdb_path': 'test_market.duckdb',
        'pool_json': 'test_pool.json',
        'start': '2024-01-01',
        'end': '2024-12-31',
        'fields': ['ret_1d', 'ret_5d', 'ret_20d', 'vol_log', 'hl_range'],
        'exclude_fields': [],
        'max_exprs': 1000,
        'batch_size': 32,
        'write_every': 10,
        'progress_min_interval_sec': 5,
        'out_dir': 'test_output',
        'use_simulated': True,
        'seed': 42,
        'force_rerun': False,
        'eval': {
            'forward_days': 5,
            'train_end': '2024-08-31',
            'test_start': '2024-09-01',
            'windows': [10, 20, 30],
            'min_daily_valid_names': 20
        }
    }

def assert_file_exists(file_path: Path, message: str = None) -> None:
    """断言文件存在"""
    assert file_path.exists(), message or f"File not found: {file_path}"

def assert_file_contains(file_path: Path, content: str, message: str = None) -> None:
    """断言文件包含指定内容"""
    assert file_path.exists(), f"File not found: {file_path}"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        file_content = f.read()
    
    assert content in file_content, message or f"File {file_path} does not contain: {content}"

def count_lines(file_path: Path) -> int:
    """计算文件行数"""
    if not file_path.exists():
        return 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def load_csv_as_df(file_path: Path) -> pd.DataFrame:
    """加载CSV文件为数据框"""
    assert file_path.exists(), f"CSV file not found: {file_path}"
    return pd.read_csv(file_path)