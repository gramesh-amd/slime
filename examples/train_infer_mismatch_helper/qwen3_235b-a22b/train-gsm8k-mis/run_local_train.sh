#!/bin/bash

SCRIPT_DIR=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
SLIME_PATH=$(realpath "${SCRIPT_DIR}/../../../..")

export DOCKER_IMAGE=${DOCKER_IMAGE:-"docker.io/grameshamd/miles-slime-rocm7-mi35x:mla-fix"}
export CLEAN_DOCKER_CONTAINER=${CLEAN_DOCKER_CONTAINER:-1}

# ------------------ Cluster Env Defaults ------------------
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-1234}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

if [ "$NODE_RANK" = "0" ]; then
    echo "========== Cluster info =========="
    echo "MASTER_ADDR: $MASTER_ADDR"
    echo "MASTER_PORT: $MASTER_PORT"
    echo "NNODES: $NNODES"
    echo "GPUS_PER_NODE: $GPUS_PER_NODE"
    echo ""
fi

VOLUME_ARGS=(
    -v "$SLIME_PATH":"$SLIME_PATH"
    -v "$HOME/.ssh":"$HOME/.ssh:ro"
)

# ------------------ Optional Container Cleanup ------------------
docker_podman_proxy() {
    if command -v podman &>/dev/null; then
        podman "$@"
    elif command -v docker &>/dev/null; then
        docker "$@"
    else
        echo "Neither Docker nor Podman found!" >&2
        return 1
    fi
}

if [[ "${CLEAN_DOCKER_CONTAINER:-0}" == "1" ]]; then
    echo "Node-${NODE_RANK}: Cleaning up existing containers..."
    CONTAINERS=$(docker_podman_proxy ps -aq)
    if [[ -n "$CONTAINERS" ]]; then
        for cid in $CONTAINERS; do
            docker_podman_proxy rm -f "$cid"
        done
        echo "Node-${NODE_RANK}: Removed containers: $CONTAINERS"
    else
        echo "Node-${NODE_RANK}: No containers to remove."
    fi
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
    echo "Node-${NODE_RANK}: Skipping training container launch."
    exit 0
else
    echo "Node-${NODE_RANK}: Launching training container."
fi

# ------------------ Launch Training Container ------------------
docker_podman_proxy run --rm \
    --name dev_train \
    --env MASTER_ADDR \
    --env MASTER_PORT \
    --env NNODES \
    --env NODE_RANK \
    --env GPUS_PER_NODE \
    --env SLURM_NODELIST \
    --env LOG_DIR \
    "${ENV_ARGS[@]}" \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --cap-add=SYS_PTRACE --cap-add=CAP_SYS_ADMIN \
    --security-opt seccomp=unconfined --group-add video \
    --privileged --device=/dev/infiniband \
    "${VOLUME_ARGS[@]}" \
    "$DOCKER_IMAGE" /bin/bash -c "\
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: begin, time=$(date +"%Y.%m.%d %H:%M:%S")' && \
        rm /etc/apt/sources.list.d/rocm.list && sudo apt update 2>&1 > /dev/null && \
        sudo apt install iproute2 openssh-client -y 2>&1 > /dev/null && \
        sed -i '/import torch/a import warnings' /app/Megatron-LM/megatron/core/model_parallel_config.py && \
        cd $SLIME_PATH && \
        bash ${SCRIPT_DIR}/run_train.sh \"\$@\" 2>&1 && \
        echo '[NODE-${NODE_RANK}(${HOSTNAME})]: end, time=$(date +"%Y.%m.%d %H:%M:%S")'
    " bash "${ARGS[@]}"