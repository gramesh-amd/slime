#!/bin/bash

export NNODES=${NNODES:-1}
export MASTER_PORT=${MASTER_PORT:-12345}

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
SLIME_PATH=$(realpath "${SCRIPT_DIR}/../../../..")
echo "SLIME_PATH: $SLIME_PATH"

export LOG_DIR=${LOG_DIR:-"${SLIME_PATH}/logs/qwen3_235b-a22b/train-gsm8k-mis"}
LOG_FILE="${LOG_DIR}/log_slurm_train-gsm8k-mis.txt"
mkdir -p "$LOG_DIR"

srun -N "${NNODES}" \
     --exclusive \
     --export ALL \
     --ntasks-per-node=1 \
     --cpus-per-task="${CPUS_PER_TASK:-96}" \
     bash -c "
          readarray -t node_array < <(scontrol show hostnames \"\$SLURM_JOB_NODELIST\")
          if [ \"\$SLURM_NODEID\" = \"0\" ]; then
              echo \"========== Slurm cluster info ==========\"
              echo \"SLURM_NODELIST: \${node_array[*]}\"
              echo \"SLURM_NNODES: \${SLURM_NNODES}\"
              echo \"SLURM_GPUS_ON_NODE: \${SLURM_GPUS_ON_NODE}\"
              echo \"\"
          fi
          export SLURM_NODELIST=\${node_array[*]}
          echo \"SLURM_NODELIST: \${SLURM_NODELIST}\"
          export MASTER_ADDR=\${node_array[0]}
          export MASTER_PORT=\${MASTER_PORT}
          export NNODES=\${SLURM_NNODES}
          export NODE_RANK=\${SLURM_PROCID}
          export GPUS_PER_NODE=\${SLURM_GPUS_ON_NODE}
          bash ${SCRIPT_DIR}/run_local_train.sh \"\$@\" 2>&1 | tee ${LOG_FILE}
     " bash "$@"