#!/bin/bash

# Qwen3-30B-A3B (30.5B total, 3.3B activated) MoE training script
# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# For AMD GPU
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES:-"1"} # Must set to 1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"} #You can choose which gpus to use

# RCCL/NCCL settings - set before Ray starts so they're inherited by all workers
export NCCL_DEBUG=INFO
export RCCL_MSCCL_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export HSA_FORCE_FINE_GRAIN_PCIE=1
# Disable memory saver which can cause hangs (torch_memory_saver is CUDA-only)
export SGLANG_ENABLE_MEMORY_SAVER=0
export TORCH_MEMORY_SAVER_DISABLE=1

# Create libcuda stubs for ROCm compatibility (prevents "libcuda.so.1" errors)
if [ ! -f /usr/local/lib/libcuda.so.1 ]; then
    ln -sf /opt/rocm/lib/libamdhip64.so /usr/local/lib/libcuda.so.1 2>/dev/null || true
    ln -sf /opt/rocm/lib/libamdhip64.so /usr/local/lib/libcudart.so.12 2>/dev/null || true
    ldconfig 2>/dev/null || true
fi

# Disable torch_memory_saver preload library (incompatible with ROCm)
TMS_LIB="/opt/venv/lib/python3.10/site-packages/torch_memory_saver_hook_mode_preload.abi3.so"
if [ -f "$TMS_LIB" ]; then
    mv "$TMS_LIB" "${TMS_LIB}.disabled" 2>/dev/null || true
fi

export LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/hip/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "/home/goramesh/slime/scripts/models/qwen3-30B-A3B.sh"

# Base directory for checkpoints
BASE_DIR="/home/goramesh/slime/models"

CKPT_ARGS=(
   --hf-checkpoint ${BASE_DIR}/Qwen3-30B-A3B
   --ref-load ${BASE_DIR}/Qwen3-30B-A3B_torch_dist
   # --load ${BASE_DIR}/Qwen3-30B-A3B_slime/
   --save ${BASE_DIR}/Qwen3-30B-A3B_slime/
   --save-interval 200
)

ROLLOUT_ARGS=(
   --prompt-data /home/goramesh/slime/data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 10
   --eval-prompt-data aime /home/goramesh/slime/data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

# Qwen3-30B-A3B: 48 layers, 128 experts, 30.5B total params
# MoE parallelism: distribute 128 experts across GPUs
PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 4
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 12288
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project mi355-slime-mis
   --wandb-group qwen3-30b-a3b-mis
   --wandb-key fb94b9f175c5ed0d600273fbe3da2dbf8a440671
)

# MoE model: use full node for inference engine
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static 0.7
   --sglang-expert-parallel-size 8
   --use-slime-router
   --sglang-disable-cuda-graph
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # Qwen3 MoE uses standard attention (not MLA)
   --attention-backend flash
   --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# Use /dev/shm for Ray temp directory (root partition is full, /dev/shm has 1.5TB free)
export RAY_TMPDIR=/dev/shm/ray_tmp_${USER}
mkdir -p ${RAY_TMPDIR}

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir=${RAY_TMPDIR}

# Build the runtime environment JSON with proper variable substitution
# Note: Most env vars are now exported before ray start and inherited automatically
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/app/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --no-offload-rollout \
   --no-offload-train \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]}

