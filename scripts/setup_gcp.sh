#!/usr/bin/env bash
# Create a Google Cloud GPU VM, install CUDA PyTorch on it, run the benchmarks, fetch results.
#
# Run from your own machine (Git Bash / Linux / macOS / Cloud Shell) with gcloud installed and
# authenticated (`gcloud auth login`, `gcloud config set project <id>`). Subcommands, in order:
#
#   bash scripts/setup_gcp.sh check     # read-only: image families, L4 zones, your L4 quota
#   bash scripts/setup_gcp.sh create    # create the VM            (starts billing)
#   bash scripts/setup_gcp.sh install   # clone repo + install PyTorch + prepare data on the VM
#   bash scripts/setup_gcp.sh run       # start scripts/run_benchmarks.sh in tmux on the VM
#   bash scripts/setup_gcp.sh attach    # watch the run (detach: Ctrl-b then d)
#   bash scripts/setup_gcp.sh fetch     # copy results/ back here
#   bash scripts/setup_gcp.sh stop      # stop the VM (GPU billing stops, disk is still billed)
#   bash scripts/setup_gcp.sh delete    # delete the VM and its disk
#   bash scripts/setup_gcp.sh print     # print every gcloud command above without running any
#
# `vm-install` is what `install` runs ON the VM; you don't call it yourself.
#
# Settings can be overridden with environment variables, e.g.
#   ZONE=us-east4-c SPOT=1 bash scripts/setup_gcp.sh create
#
# Prerequisites you have to handle in the Cloud Console (one-time):
#   - Compute Engine API enabled on the project.
#   - GPU quota: "NVIDIA L4 GPUs" >= 4 in the VM's region, and "GPUs (all regions)" >= 4.
#     New projects usually start at 0 and need a quota increase request.
# Check current prices for g2-standard-48 (and Spot) before creating: GPU VMs are expensive to
# leave running. `stop` or `delete` when done.

set -euo pipefail

# ---------------------------------------------------------------- settings
VM_NAME="${VM_NAME:-gpt2-scaling}"
ZONE="${ZONE:-us-central1-a}"                 # must offer L4s; `check` lists the zones that do
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-48}" # 4x NVIDIA L4 (24 GB each), 48 vCPUs. G2 machine
                                               # types come with their GPUs attached, so no --accelerator flag.
# Deep Learning VM image with the NVIDIA driver preinstalled. Family names change over time:
# confirm this one appears in `check`, otherwise pick a current common-cu12* Ubuntu family.
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu128-ubuntu-2204-nvidia-570}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
DISK_SIZE="${DISK_SIZE:-200GB}"
SPOT="${SPOT:-0}"   # 1 = Spot VM: much cheaper, but can be preempted mid-run (the run log shows it)
REPO_URL="${REPO_URL:-https://github.com/JawadurRahman/multi-gpu-llm-scaling.git}"
REPO_DIR="multi-gpu-llm-scaling"               # on the VM, relative to $HOME
# PyTorch: same version as the CPU smoke tests. The cu126 wheel bundles the CUDA 12.6 runtime and
# only needs a driver that supports CUDA 12.x. Wheels exist for Python 3.10-3.13+.
TORCH_VERSION="${TORCH_VERSION:-2.14.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
OWT_SHARDS="${OWT_SHARDS:-1}"                  # OpenWebText parquet shards (~300 MB each)

DRY_RUN="${DRY_RUN:-0}"

# ---------------------------------------------------------------- helpers
run() {
  # Print the command (copy-pasteable: args with spaces/specials are single-quoted), then run
  # it unless DRY_RUN=1.
  local out="+" a
  for a in "$@"; do
    if [[ "$a" =~ ^[A-Za-z0-9_./:=@,+-]+$ ]]; then out+=" $a"; else out+=" '${a//\'/\'\\\'\'}'"; fi
  done
  echo "$out"
  if [[ "$DRY_RUN" != "1" ]]; then "$@"; fi
}

region() { echo "${ZONE%-*}"; }
ssh_vm() { run gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "$1"; }

# ---------------------------------------------------------------- laptop-side subcommands
cmd_check() {
  echo "## Deep Learning VM image families (CUDA 12, Ubuntu):"
  run gcloud compute images list --project "$IMAGE_PROJECT" --no-standard-images \
      --filter="family~^common-cu12.*ubuntu" --format="value(family)"
  echo "## Zones offering NVIDIA L4:"
  run gcloud compute accelerator-types list --filter="name=nvidia-l4" --format="value(zone)"
  echo "## L4 quota in $(region):"
  run gcloud compute regions describe "$(region)" --flatten=quotas \
      --filter="quotas.metric~NVIDIA_L4" --format="table(quotas.metric,quotas.limit,quotas.usage)"
}

cmd_create() {
  local spot_flags=()
  if [[ "$SPOT" == "1" ]]; then
    spot_flags=(--provisioning-model=SPOT --instance-termination-action=STOP)
  fi
  # --maintenance-policy=TERMINATE is required for VMs with GPUs (they can't live-migrate).
  # install-nvidia-driver=True makes older DLVM images install the driver on first boot;
  # images with the driver baked in ignore it.
  run gcloud compute instances create "$VM_NAME" \
      --zone "$ZONE" \
      --machine-type "$MACHINE_TYPE" \
      --maintenance-policy TERMINATE \
      --image-family "$IMAGE_FAMILY" --image-project "$IMAGE_PROJECT" \
      --boot-disk-size "$DISK_SIZE" --boot-disk-type pd-balanced \
      --metadata install-nvidia-driver=True \
      "${spot_flags[@]}"
}

cmd_install() {
  # Clone (or update) the public repo on the VM, then run vm-install there. Results are only
  # trustworthy from committed code: benchmark.py records the commit and a git_dirty flag.
  ssh_vm "set -e; if [ -d ~/$REPO_DIR ]; then git -C ~/$REPO_DIR pull --ff-only; \
else git clone $REPO_URL ~/$REPO_DIR; fi; \
TORCH_VERSION=$TORCH_VERSION TORCH_INDEX=$TORCH_INDEX OWT_SHARDS=$OWT_SHARDS \
bash ~/$REPO_DIR/scripts/setup_gcp.sh vm-install"
}

cmd_run() {
  # tmux keeps the run alive if the SSH connection drops.
  ssh_vm "cd ~/$REPO_DIR && mkdir -p logs && tmux new-session -d -s bench \
'bash scripts/run_benchmarks.sh 2>&1 | tee logs/run_benchmarks_\$(date +%Y%m%d-%H%M%S).log' \
&& echo 'started in tmux session \"bench\"'"
}

cmd_attach() {
  run gcloud compute ssh "$VM_NAME" --zone "$ZONE" -- -t "tmux attach -t bench"
}

cmd_fetch() {
  # Everything under results/ (CSVs, per-step times, profiler summaries, env info, and the large
  # traces, which stay git-ignored locally). Extracted over the local results/ directory.
  ssh_vm "tar czf ~/results.tgz -C ~/$REPO_DIR results"
  run gcloud compute scp --zone "$ZONE" "$VM_NAME:~/results.tgz" ./results_from_vm.tgz
  run tar xzf ./results_from_vm.tgz
  echo "Results extracted into ./results. Review them, then commit results/*.csv etc."
}

cmd_stop()   { run gcloud compute instances stop   "$VM_NAME" --zone "$ZONE"; }
cmd_delete() { run gcloud compute instances delete "$VM_NAME" --zone "$ZONE"; }

cmd_print() {
  DRY_RUN=1
  for c in check create install run attach fetch stop delete; do
    echo; echo "### $c"; "cmd_$c"
  done
}

# ---------------------------------------------------------------- runs ON the VM
cmd_vm_install() {
  local repo="$HOME/$REPO_DIR"
  echo "## GPU driver check"
  nvidia-smi   # fails here if the driver isn't installed yet (DLVM first boot can take a few minutes)

  echo "## System packages"
  sudo apt-get update -y
  # python3-venv: isolated env; build-essential: gcc for torch.compile's Inductor (Triton itself
  # ships with the PyTorch wheel); tmux: keeps runs alive across SSH disconnects
  sudo apt-get install -y python3-venv python3-dev build-essential git tmux

  echo "## Python env + PyTorch $TORCH_VERSION (CUDA wheel from $TORCH_INDEX)"
  # /usr/bin/python3 is used explicitly so the venv doesn't depend on any conda env the image ships.
  /usr/bin/python3 -m venv "$HOME/venv"
  "$HOME/venv/bin/pip" install --upgrade pip
  "$HOME/venv/bin/pip" install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
  "$HOME/venv/bin/pip" install -r "$repo/requirements.txt"

  echo "## Verify CUDA, bf16 and NCCL from PyTorch"
  "$HOME/venv/bin/python" - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch can't see a GPU"
n = torch.cuda.device_count()
print(f"torch {torch.__version__}, CUDA runtime {torch.version.cuda}, NCCL {torch.cuda.nccl.version()}")
for i in range(n):
    print(f"  cuda:{i} {torch.cuda.get_device_name(i)}, bf16={torch.cuda.is_bf16_supported()}")
EOF

  echo "## Data: OpenWebText subset ($OWT_SHARDS shard(s))"
  cd "$repo"
  "$HOME/venv/bin/python" scripts/prepare_data.py --dataset openwebtext --owt_shards "$OWT_SHARDS"
  echo "## vm-install done"
}

# ---------------------------------------------------------------- dispatch
sub="${1:-}"
case "$sub" in
  check|create|install|run|attach|fetch|stop|delete|print|vm-install) "cmd_${sub//-/_}" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
