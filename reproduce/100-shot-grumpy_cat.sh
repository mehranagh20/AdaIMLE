#!/bin/bash

python train.py --hps fewshot \
    --data_root ./datasets/100-shot-grumpy_cat \
    --change_coef 0.02 \
    --force_factor 100 \
    --imle_staleness 5 \
    --imle_force_resample 25  \
    --lr 0.0001 \