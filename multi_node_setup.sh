#!/bin/bash


# COMMANDS
# cd /root/sapmajum/RL/verl&& pip install -e .
# cd /root/sapmajum/RL/tb-eval-RL && pip install -e . --no-deps && pip install tenacity loguru parse_llm_code rank_bm25
# cd /root/sapmajum/RL/slime && pip install -e .

# # Path to the MPI hostfile
HOSTFILE="/shared-aig/sapmajum/codebases/PROJECTS/code-gen/milestone2/RL/RL/slime/mpi_hostfile"
SCRIPT_PATH="/shared-aig/sapmajum/codebases/PROJECTS/code-gen/milestone2/RL/RL/slime/start_sandbox.sh"
SLIME_SCRIPT_PATH="/shared-aig/sapmajum/codebases/PROJECTS/code-gen/milestone2/RL/RL/slime/slime_training_docker.sh"
# Read hostnames from mpi_hostfile and extract IPs
if [[ ! -f "$HOSTFILE" ]]; then
  echo "Error: MPI hostfile not found at $HOSTFILE"
  exit 1
fi

# Extract IPs from hostfile (assuming format: ip slots=8)
IPS=$(awk '{print $1}' "$HOSTFILE")

# Copy SSH keys to each host
echo "Copying SSH keys to hosts..."
for ip in $IPS; do
  echo "Copying key to $ip"
  ssh-copy-id -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa.pub "$ip"
done

# # Execute start_sandbox.sh
# echo "Starting sandbox..."
# bash "$SCRIPT_PATH"


# # Execute slime_training_docker.sh
# echo "Starting slime training docker..."
# bash "$SLIME_SCRIPT_PATH"