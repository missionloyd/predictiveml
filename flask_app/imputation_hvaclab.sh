#!/bin/bash'
export TZ="UTC"

source activate tf-gpu

table="hvaclab"

python3 main.py --preprocess --save_preprocessed_files --table "$table"