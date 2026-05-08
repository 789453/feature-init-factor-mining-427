"""
模板组合目录管理模块
用于跟踪所有模板因子组合的生成和使用情况
"""
from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
import duckdb

@dataclass
class TemplateCombo:
    """模板组合信息"""
    grammar_hash: str
    combo_hash: str
    template_name: str
    template_family: str
    template_order: int
    form: str
    left_ts_op: Optional[str] = None
    right_ts_op: Optional[str] = None
    ts_ops: List[str] = None
    binary_ops: List[str] = None
    outer_transform: Optional[str] = None
    windows: List[int] = None
    n_generated: int = 0
    n_valid: int = 0
    n_sampled: int = 0
    
    def __post_init__(self):
        if self.ts_ops is None:
            self.ts_ops = []
        if self.binary_ops is None:
            self.binary_ops = []
        if self.windows is None:
            self.windows = []

class TemplateComboCatalog:
    """模板组合目录管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.con = duckdb.connect(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """初始化模板组合目录表"""
        self.con.execute("""
        CREATE TABLE IF NOT EXISTS template_combo_catalog (
            grammar_hash VARCHAR,
            combo_hash VARCHAR PRIMARY KEY,
            template_name VARCHAR,
            template_family VARCHAR,
            template_order INTEGER,
            form VARCHAR,
            left_ts_op VARCHAR,
            right_ts_op VARCHAR,
            ts_ops JSON,
            binary_ops JSON,
            outer_transform VARCHAR,
            windows JSON,
            n_generated INTEGER DEFAULT 0,
            n_valid INTEGER DEFAULT 0,
            n_sampled INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 创建索引
        self.con.execute("""
        CREATE INDEX IF NOT EXISTS idx_template_family ON template_combo_catalog(template_family)
        """)
        self.con.execute("""
        CREATE INDEX IF NOT EXISTS idx_grammar_hash ON template_combo_catalog(grammar_hash)
        """)
    
    def add_combo(self, combo: TemplateCombo) -> None:
        """添加或更新模板组合"""
        self.con.execute("""
        INSERT INTO template_combo_catalog (
            grammar_hash, combo_hash, template_name, template_family, template_order,
            form, left_ts_op, right_ts_op, ts_ops, binary_ops, outer_transform, windows,
            n_generated, n_valid, n_sampled
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (combo_hash) DO UPDATE SET
            n_generated = EXCLUDED.n_generated,
            n_valid = EXCLUDED.n_valid,
            n_sampled = EXCLUDED.n_sampled,
            updated_at = CURRENT_TIMESTAMP
        """, (
            combo.grammar_hash, combo.combo_hash, combo.template_name, combo.template_family,
            combo.template_order, combo.form, combo.left_ts_op, combo.right_ts_op,
            json.dumps(combo.ts_ops), json.dumps(combo.binary_ops), combo.outer_transform,
            json.dumps(combo.windows), combo.n_generated, combo.n_valid, combo.n_sampled
        ))
    
    def get_combo(self, combo_hash: str) -> Optional[TemplateCombo]:
        """获取指定的模板组合"""
        result = self.con.execute("""
        SELECT grammar_hash, combo_hash, template_name, template_family, template_order,
               form, left_ts_op, right_ts_op, ts_ops, binary_ops, outer_transform,
               windows, n_generated, n_valid, n_sampled
        FROM template_combo_catalog
        WHERE combo_hash = ?
        """, (combo_hash,)).fetchone()
        
        if result:
            return TemplateCombo(
                grammar_hash=result[0],
                combo_hash=result[1],
                template_name=result[2],
                template_family=result[3],
                template_order=result[4],
                form=result[5],
                left_ts_op=result[6],
                right_ts_op=result[7],
                ts_ops=json.loads(result[8]),
                binary_ops=json.loads(result[9]),
                outer_transform=result[10],
                windows=json.loads(result[11]),
                n_generated=result[12],
                n_valid=result[13],
                n_sampled=result[14]
            )
        return None
    
    def get_combos_by_family(self, template_family: str) -> List[TemplateCombo]:
        """获取指定模板族的所有组合"""
        results = self.con.execute("""
        SELECT grammar_hash, combo_hash, template_name, template_family, template_order,
               form, left_ts_op, right_ts_op, ts_ops, binary_ops, outer_transform,
               windows, n_generated, n_valid, n_sampled
        FROM template_combo_catalog
        WHERE template_family = ?
        ORDER BY template_order, combo_hash
        """, (template_family,)).fetchall()
        
        combos = []
        for result in results:
            combo = TemplateCombo(
                grammar_hash=result[0],
                combo_hash=result[1],
                template_name=result[2],
                template_family=result[3],
                template_order=result[4],
                form=result[5],
                left_ts_op=result[6],
                right_ts_op=result[7],
                ts_ops=json.loads(result[8]),
                binary_ops=json.loads(result[9]),
                outer_transform=result[10],
                windows=json.loads(result[11]),
                n_generated=result[12],
                n_valid=result[13],
                n_sampled=result[14]
            )
            combos.append(combo)
        return combos
    
    def update_combo_stats(self, combo_hash: str, n_generated: int = None, 
                          n_valid: int = None, n_sampled: int = None) -> None:
        """更新模板组合的统计信息"""
        updates = []
        params = []
        
        if n_generated is not None:
            updates.append("n_generated = ?")
            params.append(n_generated)
        
        if n_valid is not None:
            updates.append("n_valid = ?")
            params.append(n_valid)
        
        if n_sampled is not None:
            updates.append("n_sampled = ?")
            params.append(n_sampled)
        
        if not updates:
            return
        
        params.append(combo_hash)
        update_sql = f"""
        UPDATE template_combo_catalog 
        SET {', '.join(updates)}, updated_at = CURRENT_TIMESTAMP
        WHERE combo_hash = ?
        """
        self.con.execute(update_sql, params)
    
    def get_attribution_stats(self, grammar_hash: str) -> Dict[str, Any]:
        """获取模板组合的归因统计"""
        stats = self.con.execute("""
        SELECT 
            template_family,
            COUNT(*) as n_combos,
            SUM(n_generated) as total_generated,
            SUM(n_valid) as total_valid,
            SUM(n_sampled) as total_sampled,
            AVG(n_generated) as avg_generated,
            AVG(n_valid) as avg_valid,
            AVG(CASE WHEN n_generated > 0 THEN n_valid * 100.0 / n_generated ELSE 0 END) as avg_valid_rate
        FROM template_combo_catalog
        WHERE grammar_hash = ?
        GROUP BY template_family
        ORDER BY template_family
        """, (grammar_hash,)).fetchall()
        
        return {
            "by_family": [
                {
                    "template_family": row[0],
                    "n_combos": row[1],
                    "total_generated": row[2],
                    "total_valid": row[3],
                    "total_sampled": row[4],
                    "avg_generated": row[5],
                    "avg_valid": row[6],
                    "avg_valid_rate": row[7]
                }
                for row in stats
            ],
            "summary": {
                "total_combos": sum(row[1] for row in stats),
                "total_generated": sum(row[2] for row in stats),
                "total_valid": sum(row[3] for row in stats),
                "overall_valid_rate": (
                    sum(row[3] for row in stats) * 100.0 / sum(row[2] for row in stats)
                    if sum(row[2] for row in stats) > 0 else 0
                )
            }
        }
    
    def export_to_csv(self, output_path: str, grammar_hash: str = None) -> None:
        """导出模板组合目录到CSV文件"""
        import pandas as pd
        
        if grammar_hash:
            query = """
            SELECT * FROM template_combo_catalog 
            WHERE grammar_hash = ? 
            ORDER BY template_family, template_order, combo_hash
            """
            df = self.con.execute(query, (grammar_hash,)).df()
        else:
            query = """
            SELECT * FROM template_combo_catalog 
            ORDER BY grammar_hash, template_family, template_order, combo_hash
            """
            df = self.con.execute(query).df()
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
    def close(self):
        """关闭数据库连接"""
        self.con.close()

def compute_combo_hash(template_family: str, form: str, ts_ops: List[str], 
                      binary_ops: List[str], outer_transform: str, windows: List[int]) -> str:
    """计算模板组合的哈希值"""
    combo_data = {
        "template_family": template_family,
        "form": form,
        "ts_ops": sorted(ts_ops),
        "binary_ops": sorted(binary_ops),
        "outer_transform": outer_transform,
        "windows": sorted(windows)
    }
    combo_str = json.dumps(combo_data, sort_keys=True)
    return hashlib.sha256(combo_str.encode()).hexdigest()[:16]

def create_combo_from_expression(expr: str, template_spec, grammar_hash: str) -> TemplateCombo:
    """从表达式创建模板组合记录"""
    # 这里需要根据表达式解析出模板组合信息
    # 这是一个简化版本，实际实现需要更复杂的解析逻辑
    
    form = expr  # 简化处理，实际需要解析AST
    ts_ops = list(template_spec.ts_ops) if template_spec.ts_ops else []
    binary_ops = list(template_spec.binary_ops) if template_spec.binary_ops else []
    outer_transform = template_spec.outer_transforms[0] if template_spec.outer_transforms else None
    windows = []  # 需要从表达式中提取
    
    combo_hash = compute_combo_hash(
        template_spec.family, form, ts_ops, binary_ops, outer_transform, windows
    )
    
    return TemplateCombo(
        grammar_hash=grammar_hash,
        combo_hash=combo_hash,
        template_name=template_spec.name,
        template_family=template_spec.family,
        template_order=template_spec.order,
        form=form,
        ts_ops=ts_ops,
        binary_ops=binary_ops,
        outer_transform=outer_transform,
        windows=windows
    )