#!/bin/bash

# python train.py --hps fewshot \
#     --change_coef 0.02 \
#     --force_factor 100 \
#     --imle_staleness 5 \
#     --imle_force_resample 15  \
#     --lr 0.0001 \


name=ffhq
lr=0.0001
l2=0.05
ldim=256
wand_name="new-2-snoise-block-imle-ffhq-l2${l2}-lr${lr}-ldim${ldim}"
wandb_project="block-imle-ffhq"

python train.py --hps fewshot \
    --data_root ./datasets/ffhq \
    --l2_coef $l2 \
    --change_coef 0.01 \
    --max_hierarchy 0 \
    --force_factor 100 \
    --imle_staleness 5 \
    --imle_force_resample 10  \
    --lr $lr \
    --wandb_name $wand_name \
    --wandb_project $wandb_project \
    --fid_freq 10 --fid_factor 5 \
    --use_wandb 1 --wandb_mode online \
    --use_wandb 0 \
    --latent_dim $ldim
