#!/bin/bash

LATENT_DIM=4
WIDTH=64
NODE_WIDTH=64
LOSS='nmse'
EVOLVE_START=2000 
DECODER=mlp
NODE=hyper_concat
TRAINING_MODE=jacobian_inverse
BATCH_SIZE=500

DECAY_STEPS=40 
N_SAMPLES=40
MAX_STEP=100  
EPOCHS=20000 
DATASET='cylinder_population_ins=5_N40_IRREGULAR0.02'
LR_DECODER=1e-3
SOLVER_STEPS=5
LR_NODE=-1
LR_LATENT=-1
ODE_SOLVER="bosh3"
ACTIVATION='sin'

GAMMA=0.1 # Subsampling rate for the QR decomposition 

PREFIX="6L3L_5STEPS" 

for SEED in 102;
    do
    NVIDIA_TF32_OVERRIDE=0 python3 ./script_cylinder_irregular.py --decay_steps=$DECAY_STEPS \
        --activation=$ACTIVATION \
        --node_activation=swish \
        --gamma=$GAMMA \
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
        --solver_steps=$SOLVER_STEPS \
        --loss_lambda=0.8 \
        --normalize \
        --adaptive \
        --max_ode_steps=100 \
        --final_lr=1e-6 \
        --prefix=$PREFIX;
    done;
