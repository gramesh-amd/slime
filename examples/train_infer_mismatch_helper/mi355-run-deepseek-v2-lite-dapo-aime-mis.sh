#!/bin/bash

# DeepSeek-V2-Lite (16B total, 2.4B activated) MoE training script for DAPO AIME
# Uses DAPO reward model and AIME/DAPO-math datasets for competition-level math training

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

# Trust remote code for HuggingFace models (needed for DeepSeek-V2-Lite)
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
source ${SCRIPT_DIR}/../../scripts/models/deepseek-v2-lite.sh

CKPT_ARGS=(
   # Use Chat model for chat-formatted data
   --hf-checkpoint models/deepseek-ai/DeepSeek-V2-Lite-Chat
   # Use original TP=1 checkpoint
   --ref-load models/deepseek-ai/DeepSeek-V2-Lite-Chat_torch_dist
   # Can also load from previous checkpoint to resume training:
   # --load models/deepseek-ai/DeepSeek-V2-Lite-Chat_dapo_slime/
   --save models/deepseek-ai/DeepSeek-V2-Lite-Chat_dapo_aime_slime/
   --save-interval 20000
)

# Choose dataset: dapo-math-17k for training, aime-2024 for evaluation
# DAPO-math contains 17k competition-level math problems
PROMPT_SET=data/dapo-math-17k/dapo-math-17k.jsonl
# For AIME-2024 evaluation only (30 problems):
# PROMPT_SET=data/aime-2024/aime-2024.jsonl

ROLLOUT_ARGS=(
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   
   # DAPO reward model - uses Answer: pattern and strict \boxed{} matching
   # Better for competition math than standard math RM
   --rm-type dapo
   --reward-key score
   
   # Standard DAPO training hyperparameters (from codebase best practices)
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   
   # Competition math needs longer responses - 8192 is the standard
   --rollout-max-response-len 8192
   --rollout-temperature 0.8

   --num-steps-per-rollout 2

   # global_batch_size = rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout
   # 128 = 32 * 8 // 2
   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   # Use AIME-2024 as evaluation benchmark
   --eval-prompt-data aime-2024 data/aime-2024/aime-2024.jsonl
   # Multiple samples for pass@k evaluation on hard problems
   --n-samples-per-eval-prompt 16
   # AIME evaluation - allow very long responses for complex reasoning
   --eval-max-response-len 16384
   --eval-top-k 1
)

# DeepSeek-V2-Lite: 27 layers, 64 experts, 16B total params, 2.4B activated
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
   # Standard tokens per GPU for long sequence training
   --max-tokens-per-gpu 9216
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

   # CPU offload (saves GPU memory for longer sequences)
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project mi355-slime-mis
   --wandb-group deepseek-v2-lite-dapo-aime-testing
   --wandb-key fb94b9f175c5ed0d600273fbe3da2dbf8a440671
)

# AITER requires TP=1 for DeepSeek-V2-Lite (16 attention heads)
# Using aiter backend for fused MLA decode kernels
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   # Memory for inference - balance between KV cache and training
   --sglang-mem-fraction-static 0.5
   --sglang-expert-parallel-size 1
   # Standard max running requests for long sequences
   --sglang-max-running-requests 64
   --use-slime-router
   --sglang-disable-cuda-graph
   # Use aiter attention backend for MLA models on ROCm (required for fused MLA)
   --sglang-attention-backend aiter
   # Longer timeout for router requests (competition math takes longer)
   --sglang-router-request-timeout-secs 28800
   # Stagger engine startup to avoid HIP/ROCm race conditions (15 seconds per engine)
   # Total startup time = 8 engines * 15s = ~2 minutes
   --sglang-engine-startup-delay 15
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

ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265 --temp-dir=/dev/shm
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

