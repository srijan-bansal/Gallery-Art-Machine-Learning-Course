model_name=$1
output_file=$2

python run_poemization.py \
    --model_name_or_path $model_name \
    --do_train \
    --do_eval \
    --do_predict \
    --pad_to_max_length \
    --max_target_length 512 \
    --val_max_target_length 512 \
    --max_source_length 512 \
    --preprocessing_num_workers 4 \
    --train_file train.json \
    --validation_file test.json \
    --test_file test.json \
    --dataset_config 3.0.0 \
    --source_prefix "write a poem: " \
    --output_dir models/$output_file \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --text_column explanation \
    --summary_column text \
    --per_device_eval_batch_size 4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --num_train_epochs 10 \
    --save_strategy epoch \
    --logging_strategy epoch \
    --evaluation_strategy epoch \
    --load_best_model_at_end \
    --metric_for_best_model eval_rougeL
