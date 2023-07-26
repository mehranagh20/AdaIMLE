#!/bin/bash

python train.py --hps fewshot \
    --data_root ./datasets/100-shot-obama \
    --change_coef 0.01 \
    --force_factor 100 \
    --imle_staleness 5 \
    --imle_force_resample 15  \
    --lr 0.00005 \