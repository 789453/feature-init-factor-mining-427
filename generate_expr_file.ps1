Set-StrictMode -Version 2
$env:PATH = "D:\Total_Tools\miniforge3\Scripts;" + $env:PATH
$env:CONDA_DEFAULT_ENV = "universal"
$env:CONDA_PREFIX = "D:\Total_Tools\miniforge3\envs\universal"

python -c "
from src.alpha_mvp.grammar import save_all_expressions
from src.alpha_mvp.fields import DEFAULT_FEATURES
import json

result = save_all_expressions(
    fields=DEFAULT_FEATURES,
    windows=(10, 20, 30, 40, 50),
    allow_heavy_ops=False,
    out_dir='outputs/expressions'
)
print('Expression file generated:')
print('  Total:', result['total'])
print('  File:', result['expr_file'])
print('  Meta:', result['meta_file'])

with open(result['meta_file']) as f:
    meta = json.load(f)
print()
print('Stats breakdown:')
for k, v in meta.items():
    if k not in ('fields', 'windows'):
        print('  ', k, ':', v)
"