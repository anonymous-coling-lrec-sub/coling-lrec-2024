#!/bin/bash
export TASK=$1
export TASK_TYPE=$2 # gen or cls
export TRAINING_MODE=$3
export LR=$4
export TRAINING_TYPE=buffer
CUDA_VISIBLE_DEVICES=0 python multi_train.py \
    --task place_holder \
    --prefix_MLP MLP1 \
    --lr $LR \
    --freeze_weights 1 \
    --freeze_except xxxx \
    --model_name t5-large \
    --early_stopping 1 \
    --test_eval_after_every_task 1 \
    --select_k_per_class 4 \
    --batch_size 4 \
    --num_epochs 20 \
    --prefix_len 100  \
    --save_dir ./ \
    --save_name "$TRAINING_TYPE" \
    --cut 0 \
    --db_name test \
    --training_mode $TRAINING_MODE \
    --task_type $TASK_TYPE \
    --val_task $TASK

     
