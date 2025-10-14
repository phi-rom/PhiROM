#!/bin/bash

LATENT_DIM=100
WIDTH=80
NODE_WIDTH=256
LOSS='nmse'
EVOLVE_START=400
DECODER=hyper
NODE=mlp
TRAINING_MODE=jacobian_inverse
BATCH_SIZE=768
DECAY_STEPS=80
N_SAMPLES=256
MAX_STEP=40
EPOCHS=24000
DATASET='kdv_2d_new_CUTOFF=2_64_ins=1'
LR_DECODER=5e-3
LR_NODE=-1
LR_LATENT=-1
ODE_SOLVER="bosh3"
GAMMA=0.1

for SEED in 101;
    do
    python3 ./script_kdv_2d.py --decay_steps=$DECAY_STEPS \
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
        --prefix="3L4L";
    done
