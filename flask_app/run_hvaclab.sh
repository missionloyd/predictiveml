#!/bin/bash
export TZ="UTC"

source activate tf-gpu

time_step=24
datelevel="hour"
table="hvaclab"

python3 main.py --model_type xgboost --prune --run_all --save_predictions --time_step "$time_step" --datelevel "$datelevel" --table "$table" --results_file "${table}_${datelevel}.csv"

# python3 main.py --prune --table "$table"
