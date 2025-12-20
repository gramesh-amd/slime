#!/bin/bash

# Moonlight-16B-A3B (16B total, 3B activated) MoE training script

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
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source ${SCRIPT_DIR}/../../scripts/models/moonlight.sh

CKPT_ARGS=(
   --hf-checkpoint models/moonshotai/Moonlight-16B-A3B
   # Use original TP=1 checkpoint
   --ref-load models/moonshotai/Moonlight-16B-A3B_torch_dist
   # Can also load from previous checkpoint to resume training:
   # --load models/moonshotai/Moonlight-16B-A3B_slime/
   --save models/moonshotai/Moonlight-16B-A3B_slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   # We need to use math reward model (extracts \boxed{} without requiring </think> tags) as deepscaler reward model needs </think> tags
   --rm-type math
   --num-rollout 64
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   # Moonlight has 8192 context - leave room for input (~500 tokens)
   --rollout-max-response-len 7168
   --rollout-temperature 0.8

   --num-steps-per-rollout 4

   --global-batch-size 32
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data data/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 8
   # Moonlight has 8192 context - leave room for input
   --eval-max-response-len 7168
   --eval-top-p 0.7
)

# Moonlight-16B-A3B: 27 layers, 64 experts, 16B total params, 3B activated
# NOTE: EP=1 to avoid ALLTOALL NCCL timeout on ROCm

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   # Can try increasing if GPU memory allows (official uses 8192)
   --max-tokens-per-gpu 8192
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

   # CPU offload like Qwen3-30B (saves GPU memory)
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project mi355-slime-mis
   --wandb-group moonlight-16b-a3b-mis-testing
   --wandb-key fb94b9f175c5ed0d600273fbe3da2dbf8a440671
)

# currently test everything with tp=1
# sglang 0.5.6 has inference output differences with biggers tps [TODO: fix this]
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # Reduced from 0.7 to 0.5 to leave more GPU memory for training with EP=1
   --sglang-mem-fraction-static 0.5
   --sglang-expert-parallel-size 1
   --sglang-max-running-requests 64
   --use-slime-router
   --sglang-disable-cuda-graph
   # Use aiter attention backend for MLA models on ROCm (required for fused MLA)
   --sglang-attention-backend aiter
)

MISC_ARGS=(
   --trust-remote-code
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --no-gradient-accumulation-fusion
)

CUSTOM_ARGS=(
   --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
# Added NCCL debugging and timeout settings
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/app/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_USE_AITER\": \"1\",
    \"SGLANG_ROCM_FUSED_DECODE_MLA\": \"1\",
    \"HF_HUB_TRUST_REMOTE_CODE\": \"1\"
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
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${WANDB_ARGS[@]}

