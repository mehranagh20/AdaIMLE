#!/bin/bash


name=100-shot-panda
change=0.0
factor=20
force=10
lr=0.00001
stal=10
wand_name="$name-chg-${change}-fac-${factor}-frc-${force}-lr-${lr}-stl-${stal}"

save_dir=/home/mehranag/scratch/saved_models/vdimle-reproduce/$wand_name
data_root=/home/mehranag/projects/rrg-keli/data/few-shot-images/${name}
restore_latent_path=/home/mehranag/scratch/saved_models/archived/4-ada3-2048-50p-full/test/latent/0-latest.npy
restore_path=/home/mehranag/scratch/saved_models/panda-naive/test/iter-450000-

#cd dciknn_cuda
#python setup.py install
#cd ..
cp /home/mehranag/inception-2015-12-05.pt /tmp

ssh -D 9050 -q -C -N narval1 &
python train.py --hps fewshot --save_dir $save_dir --data_root $data_root --lpips_coef 1 --l2_coef 0.1 \
    --change_threshold 1 --change_coef $change --force_factor $factor --imle_db_size 5000 --imle_staleness $stal \
    --imle_force_resample $force --latent_epoch 0 --latent_lr 0.0 --imle_factor 0 --lr $lr --n_batch 4 \
    --proj_dim 800 --imle_batch 20 --iters_per_save 1000 --iters_per_images 500 --image_size 256 \
    --proj_proportion 1 --latent_dim 1024 --iters_per_ckpt 5000 \
    --dec_blocks '1x4,4m1,4x4,8m4,8x4,16m8,16x3,32m16,32x2,64m32,64x2,128m64,128x2,256m128' \
    --max_hierarchy 256 --image_size 256 --use_wandb 1 --wandb_project $name-keep --wandb_name $wand_name --wandb_mode offline \
    --fid_freq 10 --fid_factor 5
