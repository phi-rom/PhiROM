#!/bin/bash

LATENT_DIM=4
WIDTH=32
LOSS='mse'
DECODER=mlp
DECAY_STEPS=100
N_SAMPLES=8
MAX_STEP=100
EPOCHS=12000
LR_DECODER=1e-3

for SEED in 102;
    do
    python3 ./script_burgers_CROM.py --decay_steps=$DECAY_STEPS \
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
        --prefix="4L";
    done
