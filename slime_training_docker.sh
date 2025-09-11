# docker pull rlsys/slime:slime_ubuntu22.04_rocm6.3.4-patch-numa-patch_sglang0.4.9_megatron-patch_ray2.47.1_apex_torch-memory-saver0.0.8-patch-vim
# docker tag rlsys/slime:slime_ubuntu22.04_rocm6.3.4-patch-numa-patch_sglang0.4.9_megatron-patch_ray2.47.1_apex_torch-memory-saver0.0.8-patch-vim slime-rocm:v1.0

# docker pull rlsys/april:slime_ubuntu22.04_rocm6.3.4-patch-numa_vllm0.8.5-patch_sglang0.4.7_megatron-core-patch_ray0.47-patch_apex_vim
 
docker pull rlsys/slime:latest



NAME=slime_training
DOCKER=rlsys/slime:latest

# DOCKER=slime_training_image
# if ! docker image inspect $DOCKER >/dev/null 2>&1; then
#   gunzip -c /shared-aig/sapmajum/codebases/PROJECTS/code-gen/milestone2/RL/RL/slime/slime_training_image.tar.gz | docker load
# fi

# if docker container inspect $NAME >/dev/null 2>&1; then
#   docker rm -f $NAME
# fi


 


# DOCKER=rlsys/april:slime_ubuntu22.04_rocm6.3.4-patch-numa_vllm0.8.5-patch_sglang0.4.7_megatron-core-patch_ray0.47-patch_apex_vim

MY_UID=$(id -u)
MY_GID=$(id -g)



 
mkdir -p /mnt/m2m_nobackup/sapmajum/

docker run -dit --name $NAME \
--device /dev/kfd \
--device /dev/dri \
-v ~/.ssh:/root/.ssh \
--network=host \
--group-add video \
--cap-add=SYS_PTRACE \
--security-opt seccomp=unconfined \
--shm-size=128g \
--ulimit memlock=-1 \
--ulimit stack=67108864 \
-v /shared-aig/sapmajum/codebases/PROJECTS/code-gen/milestone2/RL:/root/sapmajum \
-v /mnt/m2m_nobackup/sapmajum/:/mnt/m2m_nobackup/sapmajum/ \
-e HF_HOME=/root/sapmajum \
-w /workspace \
$DOCKER

docker exec -it $NAME bash
