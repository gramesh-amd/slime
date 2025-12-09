# Docker

You can setup the docker image from slime/docker/Dockerfile.rocm7.gfx950:

```bash
cd <slime-repo>/docker
docker build -f Dockerfile.rocm7.gfx950 -t slime-rocm7-mi355:latest .
```

# Quick start

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
```

Convert model weights from hf to torch_dist
```bash
cd <slime-repo>
MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}') # /app/Megatron-LM in our docker
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]}     \
--no-gradient-accumulation-fusion    \
--hf-checkpoint models/Qwen3-30B-A3B     \
--save models/Qwen/Qwen3-30B-A3B_torch_dist


MEGATRON_LM_PATH=$(pip list | grep megatron-core | awk '{print $NF}') # /app/Megatron-LM in our docker
PYTHONPATH=${MEGATRON_LM_PATH} python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]}     \
--no-gradient-accumulation-fusion    \
--hf-checkpoint models/Qwen/Qwen3-4B    \
--save models/Qwen/Qwen3-4B_torch_dist
```


# Example

```bash
cd <slime-repo>
bash examples/train_infer_mismatch_helper/mi355-run-qwen3-4b-mis.sh

or 
cd <slime-repo>
bash examples/train_infer_mismatch_helper/mi355-run-qwen3-30b-a3b-mis.sh
```
^ Make sure to double check the data/model paths, gpu-memory settings before launching. Currently the scripts use 
```
--no-offload-rollout \
   --no-offload-train \
```
to disable use of torch_memory_saver on mi355 as this causes a memory leak or hang during synchrous training [WIP: to fix].
