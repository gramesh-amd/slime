if ! docker images -q verl-sandbox-img | grep -q .; then
  gunzip -c /shared-aig/sapmajum/codebases/PROJECTS/code-gen/milestone2/RL/RL/slime/verl-sandbox-img.tar.gz | docker load
fi

# Get your user and group IDs from the host
MY_UID=$(id -u)
MY_GID=$(id -g)

# Announce what's happening
echo "--> Removing old container named '$NAME' (if it exists)..."
docker rm -f $NAME 2>/dev/null || true

echo "--> Starting new container named '$NAME'..."
echo "    User: $(whoami) (UID: $MY_UID, GID: $MY_GID)"
echo "    Home Directory: /home/sapmajum"



docker run -dit \
  --network=host \
  --user $MY_UID:$MY_GID \
  -v /shared-aig/sapmajum/codebases/PROJECTS/code-gen/milestone2/RL:/root/sapmajum \
  -e HF_HOME=/root/sapmajum \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --cap-add=SYS_RESOURCE \
  --security-opt seccomp=unconfined \
  --shm-size=128g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name verl-sandbox \
  verl-sandbox-img

docker exec -it verl-sandbox bash -c "
  source activate sandbox-runtime && \
  fuser -k 8080/tcp || true && \
  uvicorn sandbox.server.sandbox_api:app --host 0.0.0.0 --port 8080
"