MODEL_PATH=
DATA_PATH=../dataset/CLES/training/test_process.json
OUTPUT_DIR=
DATASET=CLES
CURRENT_TIME=$(date +'%m-%d_%T')
LOGS_PATH=./logs

CUDA_VISIBLE_DEVICES=0,3 accelerate launch \
    --num_machines 1 \
    --num_processes 2 \
    --mixed_precision=bf16 \
    ./inference.py \
    --model ${MODEL_PATH} \
    --batch_size 1 \
    --data ${DATA_PATH} \
    --output_path ${OUTPUT_DIR}/outputs.json \
    --precision fp16 \
    --max_input_length 1206 \
    --max_output_length 512 \
    --temperature 0.0 \
    --top_p 0.0 \
    --repetition_penalty 1.0
