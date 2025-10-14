#!/bin/bash

LATENT_DIM=16
WIDTH=64
LOSS='mse'
DECODER=hyper
DECAY_STEPS=200
N_SAMPLES=100
MAX_STEP=25
EPOCHS=24000
DATASET='diffusion_42x42'
LR_DECODER=1e-3

for SEED in 101;
    do
    python3 ./script_diffusion_CROM.py --decay_steps=$DECAY_STEPS \
        --activation="sin" \
        --decay_rate=0.985 \
        --num_samples=$N_SAMPLES \
        --max_step=$MAX_STEP \
        --seed=$SEED \
        --latent_dim=$LATENT_DIM \
        --width=$WIDTH \
        --epochs=$EPOCHS \
        --decoder_arch=$DECODER \
        --loss=$LOSS \
        --dataset=$DATASET \
        --prefix="3L";
    done
