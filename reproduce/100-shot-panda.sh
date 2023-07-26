#!/bin/bash

python train.py --hps fewshot \
    --data_root ./datasets/100-shot-panda \
    --change_coef 0.01 \
    --force_factor 200 \
    --imle_staleness 5 \
    --imle_force_resample 30  \
    --lr 0.0001 \