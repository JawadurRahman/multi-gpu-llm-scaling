#!/usr/bin/env bash
# The experiment matrix. Runs ON the GPU VM (started by `setup_gcp.sh run`) from the repo root.
#
# Every benchmark appends a row to results/benchmarks.csv. Every command, including failures such
# as out-of-memory, is also recorded with its exit code in results/run_log.csv, so failed
# configurations are reported, not silently dropped.
#
#   bash scripts/run_benchmarks.sh                       # all stages
#   STAGES="env scaling" bash scripts/run_benchmarks.sh  # a subset
#   DRY_RUN=1 bash scripts/run_benchmarks.sh             # print the commands only
#   EXTRA_ARGS="--model tiny --steps 5" ...              # append args to every run (smoke tests)
#
# Stages:
#   env        hardware/software snapshot: nvidia-smi, GPU topology, NCCL transport, pip freeze
#   scaling    baseline on 1, 2, 4 GPUs, repeated REPEATS times (interleaved, to spread out drift)
#   profile    one profiled run per GPU count (compute / exposed comm / idle breakdown)
#   ablations  one change at a time vs the baseline, on 1 and 4 GPUs
#   fsdp       DDP vs FSDP (experimental): throughput, memory, and a loss-equivalence check

set -uo pipefail  # deliberately no -e: one failed run (e.g. OOM) must not stop the rest

cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venv/bin/python}"
TORCHRUN="${TORCHRUN:-$HOME/venv/bin/torchrun}"
DATASET="${DATASET:-openwebtext}"
BS="${BS:-8}"                 # micro-batch per GPU for the baseline (sequences of 1024 tokens)
DTYPE="${DTYPE:-bf16}"        # baseline precision (L4 supports bf16)
REPEATS="${REPEATS:-3}"
STAGES="${STAGES:-env scaling profile ablations fsdp}"
DRY_RUN="${DRY_RUN:-0}"

RUN_LOG=results/run_log.csv
if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p results logs
  [[ -s "$RUN_LOG" ]] || echo "utc_time,num_gpus,exit_code,args,log_file" > "$RUN_LOG"
fi

# bench NGPU ARGS...  -> runs scripts/benchmark.py on the first NGPU GPUs
bench() {
  local ngpu="$1"; shift
  local gpus; gpus="$(seq -s, 0 $((ngpu - 1)))"
  local cmd
  if [[ "$ngpu" == "1" ]]; then
    # 1 GPU = plain single process, no DDP wrapper: the honest baseline for speedup.
    cmd=(env CUDA_VISIBLE_DEVICES="$gpus" "$PY" scripts/benchmark.py)
  else
    cmd=(env CUDA_VISIBLE_DEVICES="$gpus" "$TORCHRUN" --standalone --nproc_per_node="$ngpu" scripts/benchmark.py)
  fi
  # EXTRA_ARGS is appended last (so it wins), e.g. EXTRA_ARGS="--model tiny" for a CPU smoke test
  # shellcheck disable=SC2206
  cmd+=(--dataset "$DATASET" "$@" ${EXTRA_ARGS:-})
  echo; echo "### [$(date -u +%H:%M:%S)] ${cmd[*]}"
  [[ "$DRY_RUN" == "1" ]] && return 0
  local log; log="logs/$(date +%Y%m%d-%H%M%S)_g${ngpu}.log"
  "${cmd[@]}" 2>&1 | tee "$log"
  local status="${PIPESTATUS[0]}"
  echo "$(date -u +%FT%TZ),$ngpu,$status,\"$* ${EXTRA_ARGS:-}\",$log" >> "$RUN_LOG"
  [[ "$status" == "0" ]] || echo "!!! exit code $status (recorded in $RUN_LOG)"
}

BASE=(--batch_size "$BS" --dtype "$DTYPE")

stage_env() {
  local dir; dir="results/env/$(date +%Y%m%d-%H%M%S)"
  echo "### env snapshot -> $dir"
  [[ "$DRY_RUN" == "1" ]] && return 0
  mkdir -p "$dir"
  git rev-parse HEAD > "$dir/git_commit.txt"
  nvidia-smi > "$dir/nvidia-smi.txt"
  nvidia-smi -q > "$dir/nvidia-smi-q.txt"
  nvidia-smi topo -m > "$dir/gpu_topology.txt"   # how the GPUs connect (PCIe switch / host bridge)
  lscpu > "$dir/lscpu.txt"
  free -g > "$dir/memory.txt"
  "$PY" -m pip freeze > "$dir/pip_freeze.txt"
  # Short 4-GPU run with NCCL_DEBUG=INFO, to record which transport NCCL picks between the L4s
  # (P2P over PCIe vs shared host memory). It writes to its own CSV, not benchmarks.csv.
  NCCL_DEBUG=INFO CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
    scripts/benchmark.py --dataset "$DATASET" "${BASE[@]}" --warmup_steps 2 --steps 5 \
    --results_csv "$dir/nccl_probe.csv" --run_name nccl_probe > "$dir/nccl_info.txt" 2>&1
  echo "env snapshot written to $dir"
}

stage_scaling() {
  for r in $(seq 1 "$REPEATS"); do
    for n in 1 2 4; do
      bench "$n" "${BASE[@]}" --run_name "scaling_r$r"
    done
  done
}

stage_profile() {
  for n in 1 2 4; do
    bench "$n" "${BASE[@]}" --profile --run_name profile
  done
}

stage_ablations() {
  # Each run changes exactly one thing from the baseline (BASE, 1 run each; the variance of the
  # baseline itself comes from the repeated scaling runs).
  for n in 1 4; do
    bench "$n" --batch_size "$BS" --dtype fp32                       --run_name abl_fp32_tf32
    bench "$n" --batch_size "$BS" --dtype fp32 --no-tf32             --run_name abl_fp32_notf32
    bench "$n" "${BASE[@]}" --compile                                --run_name abl_compile
    bench "$n" "${BASE[@]}" --num_workers 4 --pin_memory             --run_name abl_loader
    bench "$n" "${BASE[@]}" --vocab_size 50257                       --run_name abl_vocab50257
    bench "$n" "${BASE[@]}" --head_dim 128                           --run_name abl_headdim128
    bench "$n" "${BASE[@]}" --head_dim 32                            --run_name abl_headdim32
    bench "$n" "${BASE[@]}" --grad_accum 4                           --run_name abl_accum4
    bench "$n" "${BASE[@]}" --compile --num_workers 4 --pin_memory   --run_name abl_combined
  done
  # micro-batch size sweep on 1 GPU (the largest may run out of memory; that gets logged)
  for bs in 4 16 32; do
    bench 1 --batch_size "$bs" --dtype "$DTYPE" --run_name abl_bs
  done
}

stage_fsdp() {
  for n in 2 4; do
    bench "$n" "${BASE[@]}" --parallel ddp                                  --run_name fsdp_cmp
    bench "$n" "${BASE[@]}" --parallel fsdp                                 --run_name fsdp_cmp
    bench "$n" "${BASE[@]}" --parallel fsdp --no-fsdp_reshard_after_forward --run_name fsdp_cmp
  done
  bench 4 "${BASE[@]}" --parallel fsdp --profile --run_name fsdp_profile
  # correctness: fp32, same seed -> loss_last should match DDP's (see docs/fsdp.md)
  bench 2 --batch_size "$BS" --dtype fp32 --parallel ddp  --warmup_steps 3 --steps 20 --run_name fsdp_equiv
  bench 2 --batch_size "$BS" --dtype fp32 --parallel fsdp --warmup_steps 3 --steps 20 --run_name fsdp_equiv
}

echo "stages: $STAGES | baseline: batch_size=$BS dtype=$DTYPE | repeats=$REPEATS | started $(date -u +%FT%TZ)"
for s in $STAGES; do
  echo; echo "================ stage: $s ================"
  "stage_$s"
done
echo; echo "finished $(date -u +%FT%TZ)"
if [[ "$DRY_RUN" != "1" ]]; then
  echo "failed runs (exit code != 0):"
  awk -F, 'NR > 1 && $3 != 0' "$RUN_LOG" || true
fi
