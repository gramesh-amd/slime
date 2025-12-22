#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
SLIME_PATH=$(realpath "${SCRIPT_DIR}/../../../..")

HOSTNAME=$(hostname)

LOG_INFO() {
    if [ "$*" = "" ]; then
        echo ""
    else
        echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
    fi
}

LOG_INFO_RANK0() {
    if [ "$NODE_RANK" -eq 0 ]; then
        if [ "$*" = "" ]; then
            echo ""
        else
            echo "[NODE-$NODE_RANK($HOSTNAME)] [INFO] $*"
        fi
    fi
}

LOG_ERROR() {
    echo "[NODE-$NODE_RANK($HOSTNAME)] [ERROR] $*";
}

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-1234}
export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}

LOG_INFO_RANK0 "==========Training cluster info=========="
LOG_INFO_RANK0 "MASTER_ADDR: $MASTER_ADDR"
LOG_INFO_RANK0 "MASTER_PORT: $MASTER_PORT"
LOG_INFO_RANK0 "NNODES: $NNODES"
LOG_INFO_RANK0 "NODE_RANK: $NODE_RANK"
LOG_INFO_RANK0 "GPUS_PER_NODE: $GPUS_PER_NODE"
LOG_INFO_RANK0 ""

# ----------------- NCCL and Network Settings -----------------
# VERSION, WARN, INFO, DEBUG, TRACE
export NCCL_DEBUG=${NCCL_DEBUG:-}

# Disable NCCL internal checks to reduce overhead
export NCCL_CHECKS_DISABLE=1

# Using tensor model parallelism or context parallelism require 
# setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1


export NCCL_IB_GID_INDEX=3

# Disable cross NIC communication for NCCL
export NCCL_CROSS_NIC=0

# Dynamically get InfiniBand Host Channel Adapter index for NCCL if not set
if [ -z "${NCCL_IB_HCA}" ]; then
    NCCL_IB_HCA=$(bash "${SLIME_PATH}/examples/train_infer_mismatch_helper/tools/get_nccl_ib_hca.sh")
fi
export NCCL_IB_HCA

# Dynamically get network interface IP address for socket communication if not set
if [ -z "${IP_INTERFACE}" ]; then
    IP_INTERFACE=$(bash "${SLIME_PATH}/examples/train_infer_mismatch_helper/tools/get_ip_interface.sh")
fi
export IP_INTERFACE

# Set network interfaces for NCCL and Gloo, fallback to detected IP_INTERFACE
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$IP_INTERFACE}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$IP_INTERFACE}

LOG_INFO_RANK0 "==========NCCL and Network Settings=========="
LOG_INFO_RANK0 "NCCL_DEBUG: $NCCL_DEBUG"
LOG_INFO_RANK0 "NCCL_CHECKS_DISABLE: $NCCL_CHECKS_DISABLE"
LOG_INFO_RANK0 "CUDA_DEVICE_MAX_CONNECTIONS: $CUDA_DEVICE_MAX_CONNECTIONS"
LOG_INFO_RANK0 "NCCL_IB_GID_INDEX: $NCCL_IB_GID_INDEX"
LOG_INFO_RANK0 "NCCL_CROSS_NIC: $NCCL_CROSS_NIC"
LOG_INFO "NCCL_IB_HCA: $NCCL_IB_HCA"
LOG_INFO "NCCL_SOCKET_IFNAME: $NCCL_SOCKET_IFNAME"
LOG_INFO "GLOO_SOCKET_IFNAME: $GLOO_SOCKET_IFNAME"
LOG_INFO ""

# install slime
echo "RANK-${NODE_RANK}, Installing slime..."
cd $SLIME_PATH && pip install -e . 2>&1 > /dev/null
echo "RANK-${NODE_RANK}, Installing slime done..."

# disable torch.dist patch in megatron
echo "RANK-${NODE_RANK}, Disabling torch.dist patch in megatron..."
# TODO(wenx)
# cd $SLIME_PATH/bak && ./patch.sh && cd ..
echo "RANK-${NODE_RANK}, Disabling torch.dist patch in megatron done..."


DISTRIBUTED_ARGS=(
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

echo "RANK-${NODE_RANK}, Converting Qwen3-235B-A22B-FP8..."

cd $SLIME_PATH
source ${SLIME_PATH}/scripts/models/qwen3-235B-A22B.sh
PYTHONPATH=/app/Megatron-LM torchrun \
    ${DISTRIBUTED_ARGS[*]} \
    ${SLIME_PATH}/tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --expert-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --decoder-last-pipeline-num-layers 22 \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint ${SLIME_PATH}/models/Qwen/Qwen3-235B-A22B-FP8 \
    --save ${SLIME_PATH}/models/Qwen/Qwen3-235B-A22B-FP8_torch_dist \
    --trust-remote-code \
    2>&1 | tee ${LOG_DIR}/log_convert_qwen3-235B-A22B-FP8-${NODE_RANK}.log
    # --tensor-model-parallel-size 4 \