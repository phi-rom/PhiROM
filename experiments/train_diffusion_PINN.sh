#!/bin/bash

LATENT_DIM=16
WIDTH=64
NODE_WIDTH=64
LOSS='nmse'
EVOLVE_START=400
DECODER=hyper
NODE=mlp
TRAINING_MODE=jacobian_inverse
BATCH_SIZE=1250
DECAY_STEPS=200
N_SAMPLES=100
MAX_STEP=25
EPOCHS=24000
DATASET='diffusion_42x42'
LR_DECODER=5e-3
LR_NODE=-1
LR_LATENT=-1
ODE_SOLVER="bosh3"
GAMMA=0.1

for SEED in 101;
    do
    python3 ./script_diffusion_NODE.py --decay_steps=$DECAY_STEPS \
        --gamma=$GAMMA \
        --activation="sin" \
        --decay_rate=0.985 \
        --num_samples=$N_SAMPLES \
        --max_step=$MAX_STEP \
        --seed=$SEED \
        --latent_dim=$LATENT_DIM \
        --width=$WIDTH \
        --node_width=$NODE_WIDTH \
        --epochs=$EPOCHS \
        --decoder_arch=$DECODER \
        --node_arch=$NODE \
        --node_training_mode=$TRAINING_MODE \
        --loss=$LOSS \
        --dataset=$DATASET \
        --learning_rate_decoder=$LR_DECODER \
        --learning_rate_node=$LR_NODE \
        --learning_rate_latent=$LR_LATENT \
        --batch_size=$BATCH_SIZE \
        --evolve_start=$EVOLVE_START \
        --ode_solver=$ODE_SOLVER \
        --normalize \
        --loss_lambda=0.5 \
        --pinn \
        --prefix="3L3L_FIXED";
    done
