#!/bin/bash

LATENT_DIM=16
WIDTH=64
NODE_WIDTH=64
LOSS='mse'
EVOLVE_START=0
DECODER=hyper
NODE=mlp
TRAINING_MODE=labels
BATCH_SIZE=50
DECAY_STEPS=200
N_SAMPLES=100
MAX_STEP=25
EPOCHS=24000
DATASET='diffusion_42x42'
LR_DECODER=1e-2
LR_NODE=1e-3
LR_LATENT=1e-3
ODE_SOLVER="bosh3"
GAMMA=0.99      # For DINo, GAMMA is the decay rate for the integration cutoff probability
GAMMA_EPOCHS=20

for SEED in 101;
    do
    python3 ./script_diffusion_NODE.py --decay_steps=$DECAY_STEPS \
        --gamma=$GAMMA \
        --gamma_epochs=$GAMMA_EPOCHS \
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
        --dino \
        --prefix="3L3L";
    done
