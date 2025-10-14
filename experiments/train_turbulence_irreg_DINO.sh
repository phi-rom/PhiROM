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
N_SAMPLES=256
MAX_STEP=50
EPOCHS=24000
# DATASET='ns_turbulence_fixed_irregular0.05_64x64_ins=5'
LR_DECODER=1e-2
LR_NODE=1e-3
LR_LATENT=1e-3
ODE_SOLVER="bosh3"
GAMMA=0.99
GAMMA_EPOCHS=20
DECAY_STEPS=400

for SEED in 101;
    do
    for DATASET in 'ns_turbulence_irregular0.02_64x64_ins=5';
        do
        python3 ./script_turbulence_irregular.py --gamma=$GAMMA \
            --decay_steps=$DECAY_STEPS \
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
            --gamma_epochs=$GAMMA_EPOCHS \
            --gamma_decay_rate=0.99 \
            --normalize \
            --dino \
            --prefix="3L4L";
        done
    done
