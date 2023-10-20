CUDA_VISIBLE_DEVICES=0 
echo "start"
python train.py \
       --task dart \
       --prefix_MLP None \
       --lr 0.3 \
       --freeze_weights 1 \
       --freeze_except xxxx \
       --model_name t5-large \
       --early_stopping 1 \
       --test_eval_after_every_task 1 \
       --select_k_per_class 1 \
       --batch_size 8 \
       --num_epochs 10 \
       --prefix_len 100  \
       --save_dir save_models \
       --save_name my_model_folder \
       --cut 0 \
       --db_name test \
       --training_mode lora \

       # tasks cnndm, cola, sst, rte, xsum, dart, enro,snli
       # Classification: cola, sst, rte, snli
       # Generation: xsum, dart, enro, cnndm
       # training_mode: lora, bf, last_layer, fine_tune