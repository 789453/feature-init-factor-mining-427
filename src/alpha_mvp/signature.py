"""
完整的run signature系统，用于严谨区分不同实验配置
"""
from __future__ import annotations
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import subprocess

def file_sha256(path: str) -> str:
    """计算文件的SHA256哈希值"""
    path_obj = Path(path)
    if not path_obj.exists():
        return "file_not_found"
    
    sha256_hash = hashlib.sha256()
    with path_obj.open("rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_git_hash_or_unknown() -> str:
    """获取git commit hash，如果失败则返回'unknown'"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

def compute_pool_signature(pool_json: str) -> dict:
    """计算股票池配置的签名"""
    pool_path = Path(pool_json)
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_json}")
    
    with pool_path.open('r', encoding='utf-8') as f:
        pool_data = json.load(f)
    
    # Handle both dict format (with 'codes' key) and list format (direct list)
    if isinstance(pool_data, dict):
        n_codes = len(pool_data.get('codes', []))
    elif isinstance(pool_data, list):
        n_codes = len(pool_data)
    else:
        n_codes = 0
    
    pool_file_hash = file_sha256(pool_json)
    
    return {
        "pool_json": pool_json,
        "pool_file_hash": pool_file_hash,
        "n_codes": n_codes
    }

def compute_template_yaml_hash(path: str) -> str:
    """计算YAML模板文件的哈希值"""
    return file_sha256(path)

def compute_data_signature(
    duckdb_path: str,
    start: str,
    end: str,
    table_name: str = "market_data"
) -> dict:
    """计算数据配置的签名"""
    import duckdb
    
    try:
        con = duckdb.connect(duckdb_path)
        # 获取表的基本信息
        result = con.execute(f"""
            SELECT 
                COUNT(*) as row_count,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT code) as unique_codes
            FROM {table_name}
            WHERE date >= '{start}' AND date <= '{end}'
        """).fetchone()
        
        if result:
            row_count, min_date, max_date, unique_codes = result
            # 计算数据指纹（基于关键统计信息）
            fingerprint_data = f"{row_count}_{min_date}_{max_date}_{unique_codes}_{start}_{end}"
            table_fingerprint = hashlib.sha256(fingerprint_data.encode()).hexdigest()[:16]
        else:
            table_fingerprint = "no_data"
            
        con.close()
    except Exception:
        table_fingerprint = "error"
    
    return {
        "duckdb_path": duckdb_path,
        "start": start,
        "end": end,
        "table_fingerprint": table_fingerprint,
        "adjustment": "qfq"  # 默认前复权，可以根据需要调整
    }

def compute_field_signature(selected_fields: list[str], field_formula_version: str = None) -> dict:
    """计算字段配置的签名"""
    from .field_registry import get_field_set_hash
    
    field_set_hash = get_field_set_hash(selected_fields)
    
    # 如果没有提供版本，使用时间戳作为默认版本
    if field_formula_version is None:
        from datetime import datetime
        field_formula_version = datetime.now().strftime("%Y-%m-%d")
    
    return {
        "field_set_hash": field_set_hash,
        "field_formula_version": field_formula_version
    }

def compute_grammar_signature(
    template_yaml: str,
    template_yaml_hash: str,
    seed: int,
    max_exprs: int
) -> dict:
    """计算语法配置的签名"""
    return {
        "template_yaml": template_yaml,
        "template_yaml_hash": template_yaml_hash,
        "seed": seed,
        "max_exprs": max_exprs
    }

def compute_eval_signature(
    forward_days: int,
    train_end: str,
    test_start: str,
    min_daily_valid_names: int = 30
) -> dict:
    """计算评估配置的签名"""
    return {
        "forward_days": forward_days,
        "train_end": train_end,
        "test_start": test_start,
        "min_daily_valid_names": min_daily_valid_names
    }

def build_complete_manifest(
    project_version: str,
    code_hash: str,
    data_signature: dict,
    pool_signature: dict,
    field_signature: dict,
    grammar_signature: dict,
    eval_signature: dict,
    extra_info: Dict[str, Any] = None
) -> dict:
    """构建完整的实验配置清单"""
    manifest = {
        "project_version": project_version,
        "code_hash": code_hash,
        "data_signature": data_signature,
        "pool_signature": pool_signature,
        "field_signature": field_signature,
        "grammar_signature": grammar_signature,
        "eval_signature": eval_signature,
        "created_at": datetime.now().isoformat()
    }
    
    if extra_info:
        manifest["extra_info"] = extra_info
    
    return manifest

def compute_run_signature(manifest: dict) -> str:
    """计算运行签名（基于完整配置清单的哈希）"""
    # 移除时间戳等可变字段，确保相同配置产生相同签名
    stable_manifest = {k: v for k, v in manifest.items() if k not in ["created_at", "job_id"]}
    content = json.dumps(stable_manifest, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()

def build_manifest_from_config(
    cfg,
    template_yaml_path: str,
    selected_fields: list[str]
) -> dict:
    """从配置对象构建完整的manifest"""
    from datetime import datetime
    
    # 1. 代码版本
    code_hash = get_git_hash_or_unknown()
    
    # 2. 数据签名
    data_signature = compute_data_signature(
        cfg.duckdb_path if not cfg.use_simulated else "simulated",
        cfg.start,
        cfg.end
    )
    
    # 3. 股票池签名
    pool_signature = compute_pool_signature(cfg.pool_json)
    
    # 4. 字段签名
    field_signature = compute_field_signature(selected_fields)
    
    # 5. 语法签名
    template_yaml_hash = compute_template_yaml_hash(template_yaml_path)
    grammar_signature = compute_grammar_signature(
        template_yaml_path,
        template_yaml_hash,
        cfg.seed,
        cfg.max_exprs
    )
    
    # 6. 评估签名
    eval_signature = compute_eval_signature(
        cfg.eval.forward_days,
        cfg.train_end,
        cfg.test_start,
        cfg.eval.min_daily_valid_names
    )
    
    # 7. 构建完整manifest
    manifest = build_complete_manifest(
        project_version="phase2-v2",
        code_hash=code_hash,
        data_signature=data_signature,
        pool_signature=pool_signature,
        field_signature=field_signature,
        grammar_signature=grammar_signature,
        eval_signature=eval_signature
    )
    
    return manifest

# 兼容性函数，用于现有代码
def compute_legacy_run_signature(manifest: dict) -> str:
    """兼容旧版本的运行签名计算"""
    content = json.dumps(manifest, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()