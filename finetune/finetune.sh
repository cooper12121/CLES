
DATA_PATH=
OUTPUT_PATH=
MODEL_PATH=
TASK=

CURRENT_TIME=$(date +'%m-%d-%Y_%H:%M:%S')
OUTPUT_LOG_PATH=${OUTPUT_PATH}/${TASK}

mkdir -p $OUTPUT_LOG_PATH
cd finetune
export CMD="deepspeed --include localhost:0,1 finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_LOG_PATH \
    --num_train_epochs 1 \
    --model_max_length 1024 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy 'no' \
    --save_strategy 'steps' \
    --save_steps 0.5 \
    --save_total_limit 15 \
    --learning_rate 2e-6 \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --lr_scheduler_type 'cosine' \
    --gradient_checkpointing True \
    --report_to 'tensorboard' \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 False \
    --use_lora False"

echo $CMD
eval ${CMD} 2>&1 | tee -a ${OUTPUT_LOG_PATH}/log_${CURRENT_TIME}.txt
set +x