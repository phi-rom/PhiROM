#!/bin/bash

LATENT_DIM=4
WIDTH=64
NODE_WIDTH=64
LOSS='nmse'
EVOLVE_START=0
DECODER=mlp
NODE=hyper_concat
TRAINING_MODE=labels
BATCH_SIZE=10
DECAY_STEPS=200
N_SAMPLES=40
MAX_STEP=100
EPOCHS=20000
DATASET='lbm_population_ins=5_N40_IRREGULAR0.02'
LR_DECODER=1e-2
LR_NODE=1e-3
LR_LATENT=1e-3
ODE_SOLVER="bosh3"
ACTIVATION='sin'

PREFIX="6L3L" 

for SEED in 102;
    do
    NVIDIA_TF32_OVERRIDE=0 python3 ./script_lbm_irregular.py --decay_steps=$DECAY_STEPS \
        --activation=$ACTIVATION \
        --gamma=0.99 \
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
        --prefix=$PREFIX;
    done;
