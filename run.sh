CUDA_VISIBLE_DEVICES=0,1 python test.py --data_file data_new.csv \
    --model_name t5-large \
    --instruction "Write a poem :" \
    --max_source_length 600 \
    --max_target_length 512 \
    --num_epochs 10 \
    --batch_size 8 \
    --grad_steps 1 \
    --lr 5e-5