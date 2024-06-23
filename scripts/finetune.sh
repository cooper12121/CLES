


# # connection config
# # #---------------------------------------------------------------------------------
NET_TYPE="low"
export NCCL_IB_TIMEOUT=24
if [[ "${NET_TYPE}" = "low" ]]; then
    export NCCL_SOCKET_IFNAME=eth1
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_HCA=mlx5_2:1,mlx5_2:1
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
else
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_SL=3
    export NCCL_CHECK_DISABLE=1
    export NCCL_P2P_DISABLE=0
    export NCCL_IB_DISABLE=0
    export NCCL_LL_THRESHOLD=16384
    export NCCL_IB_CUDA_SUPPORT=1
    export NCCL_SOCKET_IFNAME=bond1
    export UCX_NET_DEVICES=bond1
    export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
    export NCCL_COLLNET_ENABLE=0
    export SHARP_COLL_ENABLE_SAT=0
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    export NCCL_IB_TC=160
    export NCCL_PXN_DISABLE=1
fi



# DEBUG=false
# # env config
# #---------------------------------------------------------------------------------
MASTER_ADDR=$CHIEF_IP
MASTER_PORT=6000
NUM_GPUS=$NODE_NUM


WORKSPACE=/apdcephfs_qy3/share_301372554/share_info/qianggao/Yi
export PYTHONPATH=${WORKSPACE}


MODEL_PATH="/apdcephfs_qy3/share_301372554/share_info/hencygao/llm/Qwen1.5-1.8B"



DATA_PATH="/apdcephfs_qy3/share_301372554/share_info/qianggao/dataset/t/test_example.jsonl"

EVAL_PATH="None"

#评测时的输出结果
EVAL_OUTPUT_PATH="None"  


MODEL_OUTPUT_DIR="/apdcephfs_qy3/share_301372554/share_info/qianggao/Yi/output"


MODEL_TYPE="Qwen-1.8b"
# MODEL_TYPE="llama3-8x8b-instruct_router_warmboot"
#Yi_8*6b是冷启、Yi_router是热启动
TASK="Qwen-1.8b-train_instruct_t_new"
# data config

FORMAT_MODE=t

MAX_RESPONSE=1





# training setups
#---------------------------------------------------------------------------------
# BATCH_SIZE=
MICRO_BATCH_SIZE=4
# NUM_GPUS=8
echo $NUM_GPUS
echo $MICRO_BATCH_SIZE
# GRADIENT_ACCUMULATION_STEP=$((BATCH_SIZE / NUM_GPUS / MICRO_BATCH_SIZE))
GRADIENT_ACCUMULATION_STEP=2

#广告数据400就够了
#alpaca714
MAX_LENGTH=1000


PADDING_SIDE="right"
TRUNCATION_SIDE="left"
POOLING_TYPE="last"

EPOCH=1
LEARNING_RATE=2e-6




# deepspeed setups
#---------------------------------------------------------------------------------
# DS_ZERO=3
# if [[ $DS_ZERO = 2 ]]; then
#     DEEPSPEED=${WORKSPACE}/configs/default_zero2_config.json
# else
#     DEEPSPEED=${WORKSPACE}/configs/default_offload_opt_param.json
# fi

TMP_DIR=${WORKSPACE}/tmp
mkdir -p $TMP_DIR


# NODE_IP_LIST='11.215.108.39,11.222.153.238'
echo $NODE_IP_LIST > ${TMP_DIR}/env.txt
 
# generate hostfile and pssh.hosts
sed "s/:/ slots=/g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/hostfile
sed "s/:.//g" ${TMP_DIR}/env.txt | sed "s/,/\n/g" >  ${TMP_DIR}/pssh.hosts

DEEPSPEED=${WORKSPACE}/config/configs/ds_config_zero2_no_offload.json


# output config
#----------------------------------------------------------------------------------
EXPERIMENT_NAME=$(date +'%m-%d')_${TASK}_${MODEL_NAME}_${DATA_NAME}_bs_${BATCH_SIZE}_maxlen_${MAX_LENGTH}_pad_${PADDING_SIDE}_lr_${LEARNING_RATE}_format_${FORMAT_MODE}

OUTPUT_DIR=${MODEL_OUTPUT_DIR}/${EXPERIMENT_NAME}
LOGS_PATH=${OUTPUT_DIR}/logs

mkdir -p $OUTPUT_DIR
mkdir -p $LOGS_PATH


echo "begin experiment ${EXPERIMENT_NAME}"


CURRENT_TIME=$(date +'%m-%d-%Y_%H:%M:%S')

cd finetune
export CMD="deepspeed   --hostfile ${TMP_DIR}/hostfile --master_addr ${MASTER_ADDR} --master_port=${MASTER_PORT} finetune.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --eval_path $EVAL_PATH \
    --eval_output_path $EVAL_OUTPUT_PATH \
    --output_dir $OUTPUT_DIR\
    --do_train True \
    --do_eval False \
    --padding_side $PADDING_SIDE \
    --num_train_epochs 1 \
    --model_max_length $MAX_LENGTH \
    --per_device_train_batch_size ${MICRO_BATCH_SIZE} \
    --per_device_eval_batch_size ${MICRO_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEP} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit 15 \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio 0.05 \
    --logging_steps 1 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing True \
    --report_to "tensorboard" \
    --deepspeed configs/ds_config_zero3.json \
    --bf16 True \
    --use_lora False"



echo $CMD
eval ${CMD} 2>&1 | tee -a ${LOGS_PATH}/log_${CURRENT_TIME}.txt
set +x
