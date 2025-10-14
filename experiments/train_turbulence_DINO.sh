#!/bin/bash

LATENT_DIM=100
WIDTH=80
NODE_WIDTH=512
LOSS='mse'
EVOLVE_START=0
DECODER=hyper
NODE=mlp
TRAINING_MODE=labels
BATCH_SIZE=32
DECAY_STEPS=400
N_SAMPLES=256
MAX_STEP=50
EPOCHS=24000  
DATASET='ns_turbulence_new_64x64_ins=5'
LR_DECODER=1e-2
LR_NODE=1e-3
LR_LATENT=1e-3
ODE_SOLVER="bosh3"
GAMMA_EPOCHS=20

for SEED in 101;
    do
    python3 ./script_turbulence.py --decay_steps=$DECAY_STEPS \
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
        --prefix="3L4L" \
        --gamma_epochs=$GAMMA_EPOCHS \
        --gamma=0.99 \
        --dino;
    done
