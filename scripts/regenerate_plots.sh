#!/bin/bash
# Regenerate all plots from cached validation results (no recalculation)

export PATH="D:/Total_Tools/miniforge3/Scripts:$PATH"
export CONDA_DEFAULT_ENV="universal"
export CONDA_PREFIX="D:/Total_Tools/miniforge3/envs/universal"

cd "d:/Trading/My_factor_mining_427"
python regenerate_plots.py
