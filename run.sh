#!/bin/bash

export TASK=$1
export RUN=$2
export CUT=$3
export TRAINING_MODE=$4
export NUM_SAMPLES=$5
export TRAINING_TYPE=buffer
CUDA_VISIBLE_DEVICES=0 python train.py \
    --task $TASK \
    --prefix_MLP MLP1 \
    --lr 0.001 \
    --freeze_weights 1 \
    --freeze_except xxxx \
    --model_name t5-large \
    --early_stopping 1 \
    --test_eval_after_every_task 1 \
    --select_k_per_class 1000 \
    --batch_size 4 \
    --num_epochs 20 \
    --prefix_len $NUM_SAMPLES \
    --save_dir ./ \
    --save_name "$TRAINING_TYPE" \
    --cut $CUT \
    --db_name RUN_"$RUN"_CUT_"$CUT"_"$NUM_SAMPLES" \
    --training_mode $TRAINING_MODE
    

   # --select_k_per_class $NUM_SAMPLES \
   #    --prefix_len 100  \
