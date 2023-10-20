 export CUT=$1
 CUDA_VISIBLE_DEVICES=0 python multi_train.py \
       --task rte \
       --prefix_MLP MLP1 \
       --lr 2e-7 \
       --freeze_weights 1 \
       --freeze_except xxxx \
       --model_name t5-large \
       --early_stopping 1 \
       --test_eval_after_every_task 1 \
       --select_k_per_class 8 \
       --batch_size 8 \
       --num_epochs 20 \
       --prefix_len 100  \
       --save_dir save_models/test \
       --save_name my_model_folder \
       --cut 0 \
       --db_name test \
       --training_mode bf \
       --task_type cls \
       --val_task sst

       # tasks cnndm, cola, sst, rte, xsum, dart, enro, snli
       # Classification: cola, sst, rte, snli
       # Generation: xsum, dart, enro, cnndm
       # training_mode: lora, bf, last_layer, fine_tune