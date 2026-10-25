# FSDP mode (experimental)

`scripts/benchmark.py --parallel fsdp` shards the model with PyTorch's FSDP2 API
(`torch.distributed.fsdp.fully_shard`) instead of replicating it with DDP.

**Status: experimental.** It is correct on CPU (see below) but has not yet been run on GPUs.
Nothing in this repo claims FSDP is faster or slower, or uses more or less memory, until
`results/benchmarks.csv` has GPU rows for both modes.

## What it does

| | DDP (default) | FSDP (`--parallel fsdp`) |
|---|---|---|
| Parameters per GPU | full copy | 1/N shard; each block's full params are all-gathered just before it runs |
| Gradients per GPU | full copy | 1/N shard (reduce-scattered) |
| AdamW state per GPU | full copy | 1/N shard |
| Communication per step | all-reduce of gradients (overlapped with backward) | all-gather of params in forward, again in backward (unless `--no-fsdp_reshard_after_forward`), and reduce-scatter of gradients |
| Mixed precision | autocast, fp32 weights | autocast, plus params all-gathered in bf16 (`MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32)`) |

Wrapping: each of the 12 transformer blocks is one FSDP unit. The root unit holds the
embeddings, final LayerNorm and `lm_head`. `lm_head` is weight-tied to the token embedding,
so the two have to be in the same unit.

The questions to answer on GPUs:
- How much lower is peak memory per GPU at the same batch size?
- What does the extra communication cost in tokens/s at 2 and 4 GPUs, on L4s connected over PCIe?
- Does the memory saved allow a larger per-GPU batch, and does that larger batch recover the throughput?

GPT-2 small (124M params) is small for FSDP. Its weights, gradients and AdamW state take roughly
2 GB in fp32 (arithmetic: 124M × (4 B weights + 4 B grads + 8 B Adam moments); activations
not included), on a 24 GB L4. So the result may well be that FSDP doesn't pay off at
this size. That would be a legitimate finding, as long as it's measured.

## Validated so far (CPU, 2 ranks, gloo, tiny model)

Same seed, so the same init and data on each rank, fp32, 3 warmup + 20 timed steps:

| run | loss after 23 steps |
|---|---|
| DDP | 9.6228 |
| FSDP | 9.6228 |
| FSDP, `--no-fsdp_reshard_after_forward` | 9.6228 |
| FSDP + `--compile --compile_backend aot_eager` | 9.6228 |
| DDP, `--grad_accum 2` | 9.583 |
| FSDP, `--grad_accum 2` | 9.583 |

Identical losses mean FSDP computes the same updates as DDP, including the
accumulation path (`set_requires_gradient_sync`). These are smoke-test values from the git-ignored
`results/smoke/` and are correctness checks only, not performance results.

## Not validated yet

- Anything on CUDA/NCCL: memory, throughput, and the NCCL all-gather/reduce-scatter overlap.
- Fused AdamW on sharded DTensor parameters (only the non-fused CPU path has run).
- `torch.compile` with the Inductor backend (only `aot_eager` was tested; Windows has no C++ compiler).
- `--dtype fp16` is rejected in FSDP mode: GradScaler with sharded gradients isn't validated here.
  L4 supports bf16, so the target hardware doesn't need it.
- The DDP-only `replicas_match` check doesn't apply (ranks hold different shards by design). It's
  left empty in the CSV. Repeat the loss-equivalence check above on the GPUs instead.

## Running the comparison on the VM

```bash
# same per-GPU batch, bf16, DDP vs FSDP vs FSDP without resharding, on 2 and 4 GPUs
for n in 2 4; do
  for mode in "--parallel ddp" "--parallel fsdp" "--parallel fsdp --no-fsdp_reshard_after_forward"; do
    torchrun --standalone --nproc_per_node=$n scripts/benchmark.py --dataset openwebtext \
      --batch_size 8 --dtype bf16 $mode --run_name fsdp_cmp
  done
done
# GPU loss-equivalence check (fp32, short)
torchrun --standalone --nproc_per_node=2 scripts/benchmark.py --dataset openwebtext --dtype fp32 \
  --parallel ddp  --warmup_steps 3 --steps 20 --run_name fsdp_equiv
torchrun --standalone --nproc_per_node=2 scripts/benchmark.py --dataset openwebtext --dtype fp32 \
  --parallel fsdp --warmup_steps 3 --steps 20 --run_name fsdp_equiv
```

Compare `peak_mem_allocated_mb`, `tokens_per_sec` and `loss_last` in `results/benchmarks.csv`.
Adding `--profile` shows how much of the all-gather / reduce-scatter time is exposed.
`scripts/setup_gcp.sh` includes these runs.
