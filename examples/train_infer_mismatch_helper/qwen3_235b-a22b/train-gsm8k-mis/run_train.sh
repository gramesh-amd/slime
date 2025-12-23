#!/bin/bash
# =============================================================================
# QWEN3-235B-A22B MULTINODE TRAINING SCRIPT - GSM8K
# =============================================================================
# 
# This script runs distributed training for Qwen3-235B-A22B on a multinode
# Ray cluster with the GSM8K dataset (simpler math problems).
#
# Cluster Requirements:
#   - 4 nodes with 8 GPUs each (32 GPUs total)
#   - AMD MI355 GPUs with 275 GB HBM each
#   - ~1.6 TB CPU RAM per node
#   - Shared filesystem (WekaFS) accessible from all nodes
#
# Usage:
#   NNODES=4 bash examples/train_infer_mismatch_helper/qwen3_235b-a22b/train-gsm8k-mis/run_train.sh
#
# =============================================================================


# Qwen3-235B-A22B (235B total, 22B activated) MoE training script for GSM8K
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

# set -ex

HOSTNAME=$(hostname)
SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
SLIME_PATH=$(realpath "${SCRIPT_DIR}/../../../..")

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

if [ -z "${MASTER_ADDR}" ]; then
  LOG_ERROR "MASTER_ADDR is not set. Please set it to the master node address."
  exit 1
fi

# For AMD GPU
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# Trust remote code for HuggingFace models (needed for Moonlight)
export HF_HUB_TRUST_REMOTE_CODE=1

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
LOG_INFO "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

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
# cd $SLIME_PATH/bak && ./patch.sh && cd ..
echo "RANK-${NODE_RANK}, Disabling torch.dist patch in megatron done..."


source ${SLIME_PATH}/scripts/models/qwen3-235B-A22B.sh

  #  --load ${SLIME_PATH}/models/Qwen/Qwen3-235B-A22B_slime/
CKPT_ARGS=(
   --hf-checkpoint ${SLIME_PATH}/models/Qwen/Qwen3-235B-A22B
   --ref-load ${SLIME_PATH}/models/Qwen/Qwen3-235B-A22B_torch_dist
   --save ${BASE_DIR}/models/Qwen/Qwen3-235B-A22B_gsm8k_slime_$(date +%Y%m%d_%H%M%S)/
   --save-interval 50
)

# =============================================================================
# ROLLOUT / DATA ARGUMENTS - GSM8K
# =============================================================================
ROLLOUT_ARGS=(
   --prompt-data ${SLIME_PATH}/data/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle

   # Use deepscaler reward model - extracts answer after </think> tag
   # Qwen3 supports thinking mode with <think>...</think> tags
   --rm-type deepscaler
   
   --num-rollout 64
  #  --rollout-batch-size 4           # 4 * n_samples(4) = 16 = global_batch_size
   --rollout-batch-size 8           # 8 * n_samples(4) = 32 = global_batch_size
   --n-samples-per-prompt 4         # Standard for GRPO
   # GSM8K is simpler - 1024 tokens sufficient for responses
   --rollout-max-response-len 1024
   --rollout-temperature 0.8
   --global-batch-size 32           # Must be divisible by DP=8
  #  --global-batch-size 16           # Must be divisible by DP=8
  
  #  --num-steps-per-rollout 2       # use multiple steps per rollout if OOM

  #  --num-rollout 64
  #  --rollout-batch-size 32
  #  --n-samples-per-prompt 8
  #  # GSM8K problems are simpler - 1024 tokens is sufficient for responses
  #  --rollout-max-response-len 1024
  #  --rollout-temperature 0.8
  #  --num-steps-per-rollout 4
  #  # global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout
  #  # 64 = 32 * 8 // 4
  #  --global-batch-size 64

   --balance-data
)

EVAL_ARGS=(
   --eval-interval 50
   --eval-prompt-data gsm8k ${SLIME_PATH}/data/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   # GSM8K eval - 1024 tokens is sufficient
   --eval-max-response-len 1024
   --eval-top-k 1
)

# =============================================================================
# PARALLELISM CONFIGURATION (32 GPUs across 4 nodes)
# =============================================================================
# Qwen3-235B-A22B: 94 layers, 128 experts, 4 query groups (GQA)
# 
# CONSTRAINT: TP must divide num_query_groups (4), so TP ∈ {1, 2, 4}
#
# For 32 GPUs (4 nodes x 8 GPUs):
# - Tensor Parallel (TP): 4 (max allowed by 4 query groups)
# - Pipeline Parallel (PP): 1 (no pipeline parallelism)
# - Expert Model Parallel (EP): 8 (128 experts / 8 = 16 experts per GPU group)
# - Context Parallel (CP): 1 (no sequence splitting)
# - Data Parallel (DP): 32 / (4 * 1) = 8
#
# Training uses: 4 nodes x 8 GPUs = 32 GPUs
# Rollout uses: 32 GPUs with SGLang DP/EP

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096        # Adequate with 1.6TB RAM per node
  #  --max-tokens-per-gpu 8192        
)

# =============================================================================
# GRPO / RL ALGORITHM ARGUMENTS
# =============================================================================
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# =============================================================================
# OPTIMIZER ARGUMENTS
# =============================================================================
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-7                         # Lower LR for large model stability
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   
   # CPU offload disabled - causes CPU RAM OOM during initialization
   # With 275 GB GPU memory and colocate mode, keep optimizer on GPU
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# =============================================================================
# WANDB LOGGING
# =============================================================================
export WANDB_API_KEY=2306e9cea020b222412e4a4c0912e05c0722f4e2
WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-qwen235
   --wandb-group qwen3-235b-a22b-gsm8k-multinode
   --wandb-key ${WANDB_API_KEY}
)

# =============================================================================
# SGLANG INFERENCE ENGINE CONFIGURATION
# =============================================================================
# SGLANG_ARGS=(
#    --rollout-num-gpus-per-engine 32  # 4 nodes x 8 GPUs = 32 total
#    --sglang-mem-fraction-static 0.45 # More headroom with 1.6TB RAM
#    --sglang-enable-dp-attention
#    --sglang-dp-size 4                # 4 DP replicas (32/4=8 GPUs each)
#    --sglang-ep-size 8                # EP=8 within each 8-GPU replica
#    --sglang-enable-dp-lm-head
#    --sglang-disable-cuda-graph       # Disable for AMD compatibility
#    --sglang-disable-custom-all-reduce # Skip JIT compilation that causes crash
   
#    # MoE-specific settings
#    --use-slime-router
#    # Use aiter attention backend for MLA models on ROCm (required for fused MLA)
#   #  --sglang-attention-backend aiter
# )
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 32
  #  --rollout-num-gpus-per-engine 16
   --sglang-mem-fraction-static 0.5
   # --sglang-enable-dp-attention
   # --sglang-dp-size 4
   # --sglang-ep-size 32
  #  --sglang-expert-parallel-size 16 # OOM
   --sglang-expert-parallel-size 32
   --sglang-max-running-requests 64
   --use-slime-router
   # --sglang-enable-dp-lm-head
   # --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   # --sglang-moe-a2a-backend deepep
   # --sglang-deepep-mode auto
   --sglang-disable-cuda-graph
   --sglang-disable-custom-all-reduce # Skip JIT compilation that causes crash
   # Use aiter attention backend for MLA models on ROCm (required for fused MLA)
  #  --sglang-attention-backend aiter
)

# =============================================================================
# MISCELLANEOUS ARGUMENTS
# =============================================================================
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
   --custom-config-path $SLIME_PATH/examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path $SLIME_PATH/examples/train_infer_mismatch_helper/mis.compute_mis_weights_with_cp
)


# launch the master node of ray in container
# export no_proxy="127.0.0.1,${MASTER_ADDR}"

export RAY_HEAD_MASTER_PORT=1235

# Check if this is the master node (NODE_RANK=0)
if [ "${NODE_RANK}" = "0" ]; then
  LOG_INFO "========== Master Node (NODE_RANK=${NODE_RANK}) =========="
  LOG_INFO "Starting Ray head on ${MASTER_ADDR}"
  ray start --head --node-ip-address ${MASTER_ADDR} --port=${RAY_HEAD_MASTER_PORT} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
  
  # SSH to all worker nodes and start ray workers
  worker_id=0
  for WORKER_IP in $SLURM_NODELIST; do
    if [[ "$WORKER_IP" == "$MASTER_ADDR" ]]; then
      LOG_INFO "Skipping master node: ${WORKER_IP} (worker_id=${worker_id})"
      worker_id=$((worker_id+1))
      continue
    fi

    LOG_INFO "Starting Ray worker on ${WORKER_IP} (worker_id=${worker_id})"
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null root@"${WORKER_IP}" bash --norc --noprofile -c '
      echo "  >[WORKER-'"${worker_id}"' $(hostname)] Starting Ray worker on '"${WORKER_IP}"'..."
      while ! docker ps --format "{{.Names}}" | grep -q "^dev_train$"; do
        echo "  >[WORKER-'"${worker_id}"' $(hostname)] Waiting for dev_train container..."
        sleep 2
      done
      echo "  >[WORKER-'"${worker_id}"' $(hostname)] Container found, starting Ray worker..."
      docker exec dev_train bash -c "pkill -9 sglang ; ray stop --force ; pkill -9 python ; ray start --address='"${MASTER_ADDR}"':'"${RAY_HEAD_MASTER_PORT}"' --num-gpus 8 --node-ip-address '"${WORKER_IP}"' --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265"
      echo "  >[WORKER-'"${worker_id}"' $(hostname)] Ray worker async started successfully!"
    '
    # ' &
    worker_id=$((worker_id+1))
  done
  
  LOG_INFO "Waiting for all Ray workers to start..."
  wait
  LOG_INFO "All Ray workers started!"
  ray status
  
else
  LOG_INFO "========== Worker Node (NODE_RANK=${NODE_RANK}, HOSTNAME=${HOSTNAME}) =========="
  LOG_INFO "This is a worker node. Waiting for master node to initialize Ray cluster..."
  LOG_INFO "Master node will SSH into this container and start Ray worker."
  LOG_INFO "Keeping container alive..."
  
  # Worker nodes just sleep and wait for master to SSH in
  while true; do
    sleep 60
  done
fi


# Only master node submits the Ray job
if [ "${NODE_RANK}" = "0" ]; then
  # =============================================================================
  # LAUNCH TRAINING JOB
  # =============================================================================
  LOG_INFO "============================================================"
  LOG_INFO "LAUNCHING QWEN3-235B-A22B MULTINODE TRAINING (GSM8K)"
  LOG_INFO "============================================================"
  LOG_INFO "Submitting Ray Job from Master Node"
  LOG_INFO "Ray Address: ${RAY_ADDRESS}"
  LOG_INFO "Model: Qwen3-235B-A22B (235B total, 22B activated)"
  LOG_INFO "Dataset: GSM8K (grade school math)"
  LOG_INFO "Cluster: 4 nodes x 8 GPUs = 32 GPUs (colocated training+rollout)"
  LOG_INFO "Parallelism: TP=4, PP=1, EP=8, DP=8"
  LOG_INFO "Training Script: ${TRAIN_SCRIPT}"
  LOG_INFO "============================================================"
  
      # \"no_proxy\": \"${no_proxy}\",
      # \"SGLANG_USE_AITER\": \"1\",
      # \"SGLANG_ROCM_FUSED_DECODE_MLA\": \"1\",
  # Build the runtime environment JSON with proper variable substitution
  RUNTIME_ENV_JSON="{
    \"env_vars\": {
      \"PYTHONPATH\": \"/app/Megatron-LM/\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"${CUDA_DEVICE_MAX_CONNECTIONS:-1}\",
      \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
      \"SLIME_HOST_IP\": \"${MASTER_ADDR}\",
      \"MASTER_ADDR\": \"${MASTER_ADDR}\",
      \"HF_HUB_TRUST_REMOTE_CODE\": \"1\",
      \"NCCL_DEBUG\": \"${NCCL_DEBUG}\",
      \"NCCL_CHECKS_DISABLE\": \"${NCCL_CHECKS_DISABLE}\",
      \"NCCL_IB_GID_INDEX\": \"${NCCL_IB_GID_INDEX}\",
      \"NCCL_CROSS_NIC\": \"${NCCL_CROSS_NIC}\",
      \"NCCL_IB_HCA\": \"${NCCL_IB_HCA}\",
      \"NCCL_SOCKET_IFNAME\": \"${NCCL_SOCKET_IFNAME}\",
      \"GLOO_SOCKET_IFNAME\": \"${GLOO_SOCKET_IFNAME}\",
      \"RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES\": \"${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES}\",
      \"HIP_VISIBLE_DEVICES\": \"${HIP_VISIBLE_DEVICES}\"
    }
  }"

  cd $SLIME_PATH
    #  --rollout-num-gpus 32 \
    #  --update-weight-buffer-size $(( 1024 * 1024 * 1024 * 4 )) \
    #  ${CUSTOM_ARGS[@]} \
  ray job submit --address="http://${MASTER_ADDR}:8265" \
     --runtime-env-json="${RUNTIME_ENV_JSON}" \
     -- python3 $SLIME_PATH/train.py \
     --actor-num-nodes 4 \
     --actor-num-gpus-per-node 8 \
     --colocate \
     --no-offload-rollout \
     --no-offload-train \
     ${MODEL_ARGS[@]} \
     ${CKPT_ARGS[@]} \
     ${ROLLOUT_ARGS[@]} \
     ${OPTIMIZER_ARGS[@]} \
     ${GRPO_ARGS[@]} \
     ${PERF_ARGS[@]} \
     ${EVAL_ARGS[@]} \
     ${SGLANG_ARGS[@]} \
     ${MISC_ARGS[@]} \
     ${WANDB_ARGS[@]}
  
  LOG_INFO "========== Ray Job Submitted Successfully =========="
else
  LOG_INFO "========== Worker Node: No job submission (handled by master) =========="
fi

