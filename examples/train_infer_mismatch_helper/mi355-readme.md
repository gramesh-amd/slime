# Docker

You can setup the docker image from slime/docker/Dockerfile.rocm7.gfx950:

```bash
cd <slime-repo>/docker
docker build -f Dockerfile.rocm7.gfx950 -t slime-rocm7-mi355:latest .
```

# Quick start

Download the docker image from docker hub:
```bash
docker pull rlsys/miles:MI350-355-latest
```
or
 
Start your docker container based on the above built image:
```bash
docker run -it --rm --device=/dev/kfd --device=/dev/dri --ipc=host \
  --name slime-session2 \
  -v /home/goramesh:/home/goramesh \
  slime-rocm7-mi355:latest bash

```

Then download and install slime:

```bash
git clone <slime-repo>
cd slime
pip install -e .
```

# Data and model setup

Download the models and datasets:
```bash
# model downloads
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen/Qwen3-4B
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir models/Qwen/Qwen3-30B-A3B

# train/eval data download
# dapo
huggingface-cli download --repo-type dataset zhuzilin/dapo-math-17k --local-dir data/dapo-math-17k

# aime(eval)
huggingface-cli download --repo-type dataset zhuzilin/aime-2024 --local-dir data/aime-2024

# gsm8k
huggingface-cli download --repo-type dataset zhuzilin/gsm8k --local-dir data/gsm8k
```

Convert model weights from hf to torch_dist
```bash
cd <slime-repo>
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}') # /app/Megatron-LM in our docker
source <slime-repo>/scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]}     \
--no-gradient-accumulation-fusion    \
--hf-checkpoint models/Qwen3-30B-A3B     \
--save models/Qwen/Qwen3-30B-A3B_torch_dist


MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}') # /app/Megatron-LM in our docker
source <slime-repo>/scripts/models/qwen3-4B.sh
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]}     \
--no-gradient-accumulation-fusion    \
--hf-checkpoint models/Qwen/Qwen3-4B    \
--save models/Qwen/Qwen3-4B_torch_dist


# convert hf checkpoint to torch dist for Moonlight-16B-A3B
source scripts/models/moonlight.sh
PYTHONPATH=/app/Megatron-LM python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} \
    --no-gradient-accumulation-fusion \
    --hf-checkpoint models/moonshotai/Moonlight-16B-A3B \
    --save models/moonshotai/Moonlight-16B-A3B_torch_dist \
    --trust-remote-code
```


# Example

```bash
cd <slime-repo>

# test qwen3-4b (Dense model)
bash examples/train_infer_mismatch_helper/mi355-run-qwen3-4b-mis.sh


# test qwen3-30b-a3b (GQA MoE model)
bash examples/train_infer_mismatch_helper/mi355-run-qwen3-30b-a3b-mis.sh

# test moonlight-16b-a3b (MLA MoE model)
# gsm8k train + eval
bash examples/train_infer_mismatch_helper/mi355-run-moonlight-16b-gsm8k-mis.sh
# dapo17k train + aime2024 eval
bash examples/train_infer_mismatch_helper/mi355-run-moonlight-16b-a3b-mis.sh

```

^ Make sure to double check the data/model paths, gpu-memory settings before launching. Currently the scripts use 
```
--no-offload-rollout \
   --no-offload-train \
```
to disable use of torch_memory_saver on mi355 as this causes a memory leak or hang during synchrous training [WIP: to fix].
