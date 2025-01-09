
CURRENT_TIME=$(date +'%m-%d-%Y-%H:%M:%S')
WORKSPACE=
export PYTHONPATH=${WORKSPACE}

MODEL_PATH=
FILE_PATH=../dataset/CLES/training/test_process.json


export CUDA_VISIBLE_DEVICES=0,2
# #  CUDA_VISIBLE_DEVICES=4,5,6,7
export CMD="python3 eval.py \
        --model_name_or_path_baseline ${MODEL_PATH} \
        --file_path ${FILE_PATH} \
        --output_path ./output \
        --use_vllm True"

echo $CMD
eval ${CMD} 2>&1 | tee -a $WORKSPACE/log/log_event_summarize_${CURRENT_TIME}.txt
set +x
