"""
Phase Runner - 粗筛/细筛流程自动化
实现coarse->fine的完整工作流
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from .config import RunConfig
from .phase2_pipeline import run_phase2
from .candidate_sampler_extended import select_for_fine_screen, export_candidate_expr_file, export_candidate_analysis
from .signature import build_manifest_from_config, compute_run_signature

class PhaseRunner:
    """阶段运行器，管理粗筛和细筛流程"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.phase_config = self.config.get('phase_config', {})
        self.phase_metadata = {}  # 存储额外的阶段配置信息
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def run_coarse_phase(self) -> Dict[str, Any]:
        """运行粗筛阶段"""
        coarse_cfg = self.phase_config.get('coarse', {})
        if not coarse_cfg:
            raise ValueError("No coarse phase configuration found")
        
        print("=" * 60)
        print("[PhaseRunner] Starting COARSE phase...")
        print("=" * 60)
        
        # 构建粗筛配置
        coarse_run_config = self._build_run_config(coarse_cfg, phase_type="coarse")
        
        # 运行粗筛
        start_time = time.time()
        coarse_manifest = run_phase2(coarse_run_config)
        coarse_time = time.time() - start_time
        
        print(f"[PhaseRunner] Coarse phase completed in {coarse_time:.1f}s")
        
        # 保存粗筛结果摘要
        self._save_phase_summary("coarse", coarse_manifest, coarse_time)
        
        return coarse_manifest
    
    def run_fine_phase(self, coarse_manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """运行细筛阶段"""
        fine_cfg = self.phase_config.get('fine', {})
        if not fine_cfg:
            raise ValueError("No fine phase configuration found")
        
        print("=" * 60)
        print("[PhaseRunner] Starting FINE phase...")
        print("=" * 60)
        
        # 如果没有提供粗筛manifest，尝试从输出目录加载
        if coarse_manifest is None:
            coarse_manifest = self._load_coarse_manifest()
        
        # 构建细筛配置
        fine_run_config = self._build_run_config(fine_cfg, phase_type="fine", coarse_manifest=coarse_manifest)
        
        # 运行细筛
        start_time = time.time()
        fine_manifest = run_phase2(fine_run_config)
        fine_time = time.time() - start_time
        
        print(f"[PhaseRunner] Fine phase completed in {fine_time:.1f}s")
        
        # 保存细筛结果摘要
        self._save_phase_summary("fine", fine_manifest, fine_time)
        
        # 生成粗筛到细筛的对比报告
        if coarse_manifest:
            self._generate_decay_report(coarse_manifest, fine_manifest)
        
        return fine_manifest
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """运行完整的粗筛+细筛流程"""
        print("=" * 60)
        print("[PhaseRunner] Starting FULL pipeline...")
        print("=" * 60)
        
        # 运行粗筛
        coarse_manifest = self.run_coarse_phase()
        
        # 运行细筛
        fine_manifest = self.run_fine_phase(coarse_manifest)
        
        # 生成最终报告
        final_report = self._generate_final_report(coarse_manifest, fine_manifest)
        
        print("=" * 60)
        print("[PhaseRunner] Full pipeline completed successfully!")
        print("=" * 60)
        
        return final_report
    
    def _build_run_config(self, phase_cfg: Dict[str, Any], phase_type: str, 
                         coarse_manifest: Optional[Dict[str, Any]] = None) -> RunConfig:
        """构建运行配置"""
        from .config import EvalConfig
        
        # 基础配置
        base_cfg = self.config.get('base_config', {})
        
        # 合并基础配置和阶段配置
        merged_cfg = {**base_cfg, **phase_cfg}
        
        # 设置阶段特定参数
        merged_cfg['phase_type'] = phase_type
        
        # 细筛阶段使用粗筛的候选表达式
        if phase_type == "fine" and coarse_manifest:
            coarse_output_dir = Path(coarse_manifest['output_dir'])
            candidate_file = coarse_output_dir / "exports" / "fine_candidates.expr"
            if candidate_file.exists():
                merged_cfg['expr_file'] = str(candidate_file)
                print(f"[PhaseRunner] Using candidate expressions from coarse phase: {candidate_file}")
            else:
                print("[PhaseRunner] Warning: No candidate file found from coarse phase")
        
        # 设置细筛阶段的粗筛签名（用于衰减分析）
        if phase_type == "fine" and coarse_manifest:
            merged_cfg['coarse_signature'] = coarse_manifest.get('run_signature')
        
        # 设置模板配置路径
        if 'template_config_path' not in merged_cfg:
            merged_cfg['template_config_path'] = self.config.get('template_config_path')
        
        # 设置排名策略
        if 'ranking_strategy' not in merged_cfg:
            merged_cfg['ranking_strategy'] = self.config.get('ranking_strategy', 'interleave')
        
        # 设置候选选择参数
        if 'candidate_selection' in self.config:
            candidate_cfg = self.config['candidate_selection']
            for key, value in candidate_cfg.items():
                if key not in merged_cfg:
                    merged_cfg[f'candidate_{key}'] = value
        
        # 创建评估配置
        eval_config = EvalConfig(
            forward_days=merged_cfg.get('forward_days', 5),
            windows=merged_cfg.get('windows', [10, 20, 30, 40, 50]),
            min_daily_valid_names=merged_cfg.get('min_daily_valid_names', 30)
        )
        
        # 创建运行配置
        run_config = RunConfig(
            duckdb_path=merged_cfg.get('duckdb_path'),
            pool_json=merged_cfg.get('pool_json'),
            start=merged_cfg.get('start'),
            end=merged_cfg.get('end'),
            fields=merged_cfg.get('fields'),
            exclude_fields=merged_cfg.get('exclude_fields'),
            field_file=merged_cfg.get('field_file'),
            expr_file=merged_cfg.get('expr_file'),
            max_exprs=merged_cfg.get('max_exprs', 10000),
            batch_size=merged_cfg.get('batch_size', 128),
            write_every=merged_cfg.get('write_every', 100),
            progress_min_interval_sec=merged_cfg.get('progress_min_interval_sec', 30),
            out_dir=merged_cfg.get('out_dir'),
            sqlite_path=merged_cfg.get('sqlite_path'),
            use_simulated=merged_cfg.get('use_simulated', False),
            seed=merged_cfg.get('seed', 42),
            force_rerun=merged_cfg.get('force_rerun', False),
            start_expr=merged_cfg.get('start_expr'),
            end_expr=merged_cfg.get('end_expr'),
            eval=eval_config,
            train_end=merged_cfg.get('train_end', '20250831'),
            test_start=merged_cfg.get('test_start', '20250901')
        )
        
        # 存储额外的配置信息（但不传递给RunConfig）
        self.phase_metadata[phase_type] = {
            'phase_type': merged_cfg.get('phase_type'),
            'template_config_path': merged_cfg.get('template_config_path'),
            'ranking_strategy': merged_cfg.get('ranking_strategy', 'interleave'),
            'top_n_display': merged_cfg.get('top_n_display', 100),
            'extended_attribution': merged_cfg.get('extended_attribution', True),
            'candidate_top_k': merged_cfg.get('candidate_top_k', 1000),
            'candidate_sample_n': merged_cfg.get('candidate_sample_n', 1000),
            'candidate_min_per_family': merged_cfg.get('candidate_min_per_family', 100),
            'candidate_min_per_field': merged_cfg.get('candidate_min_per_field', 20),
            'candidate_min_per_operator': merged_cfg.get('candidate_min_per_operator', 20),
            'candidate_alpha': merged_cfg.get('candidate_alpha', 0.85),
            'coarse_signature': merged_cfg.get('coarse_signature'),
            'fine_signature': merged_cfg.get('fine_signature')
        }
        
        return run_config
    
    def _load_coarse_manifest(self) -> Optional[Dict[str, Any]]:
        """加载粗筛阶段的manifest"""
        coarse_cfg = self.phase_config.get('coarse', {})
        if not coarse_cfg:
            return None
        
        output_dir = Path(coarse_cfg.get('out_dir', 'outputs/phase2/coarse'))
        manifest_file = output_dir / "manifest.json"
        
        if manifest_file.exists():
            with open(manifest_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def _save_phase_summary(self, phase: str, manifest: Dict[str, Any], duration: float) -> None:
        """保存阶段运行摘要"""
        output_dir = Path(manifest.get('output_dir', f'outputs/phase2/{phase}'))
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / f"{phase}_summary.json"
        
        summary = {
            "phase": phase,
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "duration_seconds": duration,
            "run_signature": manifest.get('run_signature'),
            "total_expressions": manifest.get('total_expressions', 0),
            "valid_expressions": manifest.get('valid_expressions', 0),
            "pool_info": manifest.get('pool_signature', {}),
            "field_info": manifest.get('field_signature', {}),
            "template_info": manifest.get('grammar_signature', {})
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[PhaseRunner] {phase.capitalize()} phase summary saved to {summary_file}")
    
    def _generate_decay_report(self, coarse_manifest: Dict[str, Any], fine_manifest: Dict[str, Any]) -> None:
        """生成粗筛到细筛的衰减报告"""
        coarse_output_dir = Path(coarse_manifest.get('output_dir', 'outputs/phase2/coarse'))
        fine_output_dir = Path(fine_manifest.get('output_dir', 'outputs/phase2/fine'))
        
        # 这里可以添加更详细的衰减分析逻辑
        decay_report = {
            "coarse_run_signature": coarse_manifest.get('run_signature'),
            "fine_run_signature": fine_manifest.get('run_signature'),
            "coarse_pool": coarse_manifest.get('pool_signature', {}).get('pool_json'),
            "fine_pool": fine_manifest.get('pool_signature', {}).get('pool_json'),
            "coarse_date_range": f"{coarse_manifest.get('data_signature', {}).get('start')} - {coarse_manifest.get('data_signature', {}).get('end')}",
            "fine_date_range": f"{fine_manifest.get('data_signature', {}).get('start')} - {fine_manifest.get('data_signature', {}).get('end')}",
            "analysis_notes": "Decay analysis can be performed by comparing factor performance across different pools and time periods"
        }
        
        decay_file = fine_output_dir / "coarse_to_fine_decay_report.json"
        fine_output_dir.mkdir(parents=True, exist_ok=True)
        with open(decay_file, 'w') as f:
            json.dump(decay_report, f, indent=2)
        
        print(f"[PhaseRunner] Decay report saved to {decay_file}")
    
    def _generate_final_report(self, coarse_manifest: Dict[str, Any], fine_manifest: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终报告"""
        final_report = {
            "pipeline_type": "coarse_to_fine",
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "coarse_phase": {
                "run_signature": coarse_manifest.get('run_signature'),
                "output_dir": coarse_manifest.get('output_dir'),
                "pool": coarse_manifest.get('pool_signature', {}).get('pool_json'),
                "date_range": f"{coarse_manifest.get('data_signature', {}).get('start')} - {coarse_manifest.get('data_signature', {}).get('end')}"
            },
            "fine_phase": {
                "run_signature": fine_manifest.get('run_signature'),
                "output_dir": fine_manifest.get('output_dir'),
                "pool": fine_manifest.get('pool_signature', {}).get('pool_json'),
                "date_range": f"{fine_manifest.get('data_signature', {}).get('start')} - {fine_manifest.get('data_signature', {}).get('end')}"
            },
            "recommendations": [
                "Review coarse_to_fine_decay_report.json for performance decay analysis",
                "Check candidate selection in fine phase exports directory",
                "Validate top performers using the validation pipeline"
            ]
        }
        
        # 保存最终报告
        output_dir = Path(fine_manifest.get('output_dir', 'outputs/phase2/fine'))
        final_report_file = output_dir / "final_pipeline_report.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(final_report_file, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"[PhaseRunner] Final report saved to {final_report_file}")
        
        return final_report

def create_default_phase_config(
    coarse_pool: str = "static_pool_200.json",
    fine_pool: str = "static_pool_800.json",
    coarse_start: str = "20240101",
    coarse_end: str = "20260430",
    fine_start: str = "20180101",
    fine_end: str = "20260430",
    output_base_dir: str = "outputs/phase2"
) -> Dict[str, Any]:
    """创建默认的粗筛/细筛配置"""
    return {
        "base_config": {
            "duckdb_path": "data/market.duckdb",
            "forward_days": 5,
            "train_end": "20250831",
            "test_start": "20250901",
            "windows": [10, 20, 30, 40, 50],
            "batch_size": 128,
            "write_every": 100,
            "progress_min_interval_sec": 30,
            "seed": 42,
            "extended_attribution": True,
            "ranking_strategy": "interleave"
        },
        "template_config_path": "configs/phase2/templates_v1.yaml",
        "candidate_selection": {
            "top_k": 1000,
            "sample_n": 1000,
            "min_per_family": 100,
            "min_per_field": 20,
            "min_per_operator": 20,
            "alpha": 0.85
        },
        "phase_config": {
            "coarse": {
                "pool_json": coarse_pool,
                "start": coarse_start,
                "end": coarse_end,
                "max_exprs": 80000,
                "out_dir": f"{output_base_dir}/coarse_{coarse_pool.replace('.json', '').replace('static_pool_', '')}_{coarse_start}",
                "phase_type": "coarse",
                "top_n_display": 100
            },
            "fine": {
                "pool_json": fine_pool,
                "start": fine_start,
                "end": fine_end,
                "max_exprs": 20000,  # 细筛阶段通常表达式数量较少
                "out_dir": f"{output_base_dir}/fine_{fine_pool.replace('.json', '').replace('static_pool_', '')}_{fine_start}",
                "phase_type": "fine",
                "top_n_display": 100
            }
        }
    }

def run_phase_runner(config_path: str, phase: str = "full") -> Dict[str, Any]:
    """
    运行阶段运行器
    
    Args:
        config_path: 配置文件路径
        phase: 运行阶段 ('coarse', 'fine', 'full')
        
    Returns:
        运行结果
    """
    runner = PhaseRunner(config_path)
    
    if phase == "coarse":
        return runner.run_coarse_phase()
    elif phase == "fine":
        return runner.run_fine_phase()
    elif phase == "full":
        return runner.run_full_pipeline()
    else:
        raise ValueError(f"Unknown phase: {phase}. Choose from 'coarse', 'fine', 'full'")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python phase_runner.py <config_path> [phase]")
        sys.exit(1)
    
    config_path = sys.argv[1]
    phase = sys.argv[2] if len(sys.argv) > 2 else "full"
    
    result = run_phase_runner(config_path, phase)
    print(f"Phase runner completed: {phase}")