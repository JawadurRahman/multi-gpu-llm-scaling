"""
Throughput benchmark for GPT-2 training on 1..N GPUs (single node) with DDP.

Runs `--warmup_steps` untimed steps, then `--steps` timed steps, and appends one summary row
to a CSV in results/ plus a per-step timing file in results/steps/. Nothing is estimated:
every number in the CSV is measured during the run.

Training loop adapted from nanoGPT's train.py (https://github.com/karpathy/nanoGPT, MIT).

Scaling mode is *weak scaling*: --batch_size is the micro-batch per GPU, so the global batch
grows with the GPU count. This is how DDP is normally used, and it makes
speedup = tokens/s(N GPUs) / tokens/s(1 GPU) a fair comparison.

Examples:
  # 1 GPU (or CPU) without torchrun
  python scripts/benchmark.py --dataset openwebtext --batch_size 8
  # 4 GPUs on one node
  torchrun --standalone --nproc_per_node=4 scripts/benchmark.py --dataset openwebtext --batch_size 8
  # tiny CPU smoke test
  python scripts/benchmark.py --model tiny --dataset shakespeare --warmup_steps 2 --steps 5
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import GPT, GPTConfig

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Architecture presets. "gpt2" is GPT-2 small (124M). "tiny" exists only so the full code path
# can run on a laptop CPU in seconds; its numbers say nothing about GPU performance.
MODEL_PRESETS = {
    "gpt2": dict(n_layer=12, n_head=12, n_embd=768, block_size=1024),
    "tiny": dict(n_layer=2, n_head=2, n_embd=64, block_size=64),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # model
    p.add_argument("--model", choices=MODEL_PRESETS, default="gpt2")
    p.add_argument("--n_layer", type=int, help="override the preset")
    p.add_argument("--n_head", type=int, help="override the preset")
    p.add_argument("--n_embd", type=int, help="override the preset")
    p.add_argument("--block_size", type=int, help="override the preset (sequence length)")
    p.add_argument("--vocab_size", type=int, default=50304,
                   help="50304 = GPT-2's 50257 padded to a multiple of 64 (nanoGPT default)")
    # data / batch
    p.add_argument("--dataset", default="openwebtext", help="reads data/<dataset>/train.bin")
    p.add_argument("--batch_size", type=int, default=8, help="micro-batch size PER GPU (sequences)")
    # run length
    p.add_argument("--warmup_steps", type=int, default=20, help="untimed steps (allocator, cuDNN/NCCL setup)")
    p.add_argument("--steps", type=int, default=200, help="timed steps")
    # optimizer: AdamW with nanoGPT's GPT-2 settings, constant LR (no schedule needed for a benchmark)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0, help="0 disables clipping")
    # system
    p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                   help="allow TF32 for fp32 matmuls on Ampere+ GPUs (nanoGPT default)")
    p.add_argument("--seed", type=int, default=1337)
    # output
    p.add_argument("--run_name", default="", help="free-form label stored in the CSV")
    p.add_argument("--results_csv", default=None,
                   help="default: results/benchmarks.csv on GPU, results/smoke/cpu_smoke.csv on CPU")
    p.add_argument("--log_interval", type=int, default=20)
    args = p.parse_args()

    # fill unset architecture fields from the preset
    for k, v in MODEL_PRESETS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.results_csv is None:
        # CPU runs are smoke tests only. They go to a git-ignored folder so they can never be
        # mixed into the GPU results that the plots and README are built from.
        args.results_csv = (os.path.join(REPO_ROOT, "results", "benchmarks.csv") if args.device == "cuda"
                            else os.path.join(REPO_ROOT, "results", "smoke", "cpu_smoke.csv"))
    return args


def setup_distributed(device_type):
    """torchrun sets RANK/LOCAL_RANK/WORLD_SIZE. Without them we run as a single process."""
    if "RANK" not in os.environ:
        return 0, 0, 1, torch.device("cuda:0" if device_type == "cuda" else "cpu")
    rank, local_rank, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    if device_type == "cuda":
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
    else:
        # gloo lets the multi-process code path be smoke-tested on a CPU-only machine
        device = torch.device("cpu")
        dist.init_process_group(backend="gloo")
    return rank, local_rank, world, device


class RandomWindowSampler:
    """nanoGPT's "poor man's data loader": sample random block_size windows from a
    memory-mapped token file. Each rank uses its own seed so ranks see different data."""

    def __init__(self, path, batch_size, block_size, seed, device):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        assert len(self.data) > block_size + 1, f"{path} is too small for block_size={block_size}"
        self.batch_size, self.block_size, self.device = batch_size, block_size, device
        self.gen = torch.Generator().manual_seed(seed)

    def next_batch(self):
        ix = torch.randint(len(self.data) - self.block_size - 1, (self.batch_size,), generator=self.gen)
        x = torch.stack([torch.from_numpy(self.data[i:i + self.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(self.data[i + 1:i + 1 + self.block_size].astype(np.int64)) for i in ix])
        if self.device.type == "cuda":
            # pinned host memory allows a non_blocking (async) host-to-device copy
            return x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        return x.to(self.device), y.to(self.device)


def sync(device):
    # CUDA kernels run asynchronously; without this, the timer would only measure kernel launch.
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        # a dirty tree means the numbers came from code that isn't in any commit
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             cwd=REPO_ROOT, text=True).strip())
        return commit, dirty
    except Exception:
        return "unknown", None


def reduce_max(value, device):
    """Max of a float across ranks (identity when not distributed)."""
    if not dist.is_initialized():
        return value
    t = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    if exists:
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        if header != list(row):
            raise SystemExit(f"{path} has different columns than this version of the script. "
                             f"Use --results_csv to write to a new file.")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    rank, local_rank, world, device = setup_distributed(args.device)
    master = rank == 0
    torch.manual_seed(args.seed)  # same init on all ranks (DDP also broadcasts rank 0's weights)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = args.tf32
        torch.backends.cudnn.allow_tf32 = args.tf32

    # ---- data ----
    data_path = os.path.join(REPO_ROOT, "data", args.dataset, "train.bin")
    if not os.path.exists(data_path):
        raise SystemExit(f"{data_path} not found. Run: python scripts/prepare_data.py --dataset {args.dataset}")
    loader = RandomWindowSampler(data_path, args.batch_size, args.block_size, args.seed + rank, device)

    # ---- model ----
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       block_size=args.block_size, vocab_size=args.vocab_size)
    model = GPT(config).to(device)
    n_params = model.get_num_params()
    flops_per_token = model.flops_per_token()
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95), device.type)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    tokens_per_step = world * args.batch_size * args.block_size
    if master:
        print(f"world_size={world} device={device} params={n_params / 1e6:.2f}M "
              f"tokens/step={tokens_per_step:,} warmup={args.warmup_steps} steps={args.steps}")

    # ---- train loop ----
    step_times = []
    first_loss = last_loss = None
    total_steps = args.warmup_steps + args.steps
    model.train()
    for step in range(total_steps):
        if step == args.warmup_steps and device.type == "cuda":
            # measure peak memory over the timed steps only (steady state)
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        t0 = time.perf_counter()

        # Step time is end-to-end: data loading + forward + backward (incl. DDP all-reduce,
        # which overlaps with backward) + optimizer.
        x, y = loader.next_batch()
        _, loss = model(x, y)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        sync(device)
        dt = time.perf_counter() - t0
        if step >= args.warmup_steps:
            step_times.append(dt)

        if step == 0 or step == total_steps - 1 or step % args.log_interval == 0:
            lossf = loss.item()  # rank-local loss; .item() is a sync point, so only on log steps
            first_loss = lossf if first_loss is None else first_loss
            last_loss = lossf
            if master:
                phase = "warmup" if step < args.warmup_steps else "timed"
                print(f"step {step:4d} ({phase}) loss {lossf:.4f} time {dt * 1000:.1f} ms")

    # ---- correctness check: DDP replicas must still hold identical weights ----
    # If gradients were not all-reduced, ranks would diverge after the first step on different data.
    replicas_match = True
    if world > 1:
        with torch.no_grad():
            checksum = sum(p.double().sum() for p in model.parameters()).item()
        replicas_match = reduce_max(checksum, device) == -reduce_max(-checksum, device)
        if master and not replicas_match:
            print("WARNING: model weights differ across ranks; gradients were not synchronised")

    # ---- aggregate across ranks ----
    # Ranks are synchronised every step by the gradient all-reduce, so their wall-clock totals
    # are close; taking the max is the conservative choice (the slowest rank sets the pace).
    total_time = reduce_max(sum(step_times), device)
    peak_alloc_mb = peak_reserved_mb = None
    if device.type == "cuda":
        peak_alloc_mb = reduce_max(torch.cuda.max_memory_allocated(device) / 2**20, device)
        peak_reserved_mb = reduce_max(torch.cuda.max_memory_reserved(device) / 2**20, device)

    if master:
        tokens_per_sec = tokens_per_step * args.steps / total_time
        ms = [t * 1000 for t in step_times]
        commit, dirty = git_info()
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_id = f"{timestamp}_{args.model}_g{world}_b{args.batch_size}" + (f"_{args.run_name}" if args.run_name else "")

        row = {
            "run_id": run_id,
            "run_name": args.run_name,
            "git_commit": commit,
            "git_dirty": dirty,
            "device_type": device.type,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "num_gpus": world if device.type == "cuda" else 0,
            "world_size": world,
            "model": args.model,
            "params_m": round(n_params / 1e6, 3),
            "n_layer": args.n_layer, "n_head": args.n_head, "n_embd": args.n_embd,
            "head_dim": args.n_embd // args.n_head,
            "block_size": args.block_size, "vocab_size": args.vocab_size,
            "dataset": args.dataset,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_tokens": tokens_per_step,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.steps,
            "step_time_ms_mean": round(statistics.mean(ms), 3),
            "step_time_ms_median": round(statistics.median(ms), 3),
            "step_time_ms_std": round(statistics.stdev(ms), 3) if len(ms) > 1 else 0.0,
            "step_time_ms_min": round(min(ms), 3),
            "step_time_ms_max": round(max(ms), 3),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "tokens_per_sec_per_gpu": round(tokens_per_sec / world, 1),
            # achieved model TFLOP/s per GPU from the PaLM 6N + attention formula. MFU (divide by the
            # GPU's peak for the dtype used) is computed later, citing the peak's source.
            "model_tflops_per_gpu": round(flops_per_token * tokens_per_sec / world / 1e12, 3),
            "peak_mem_allocated_mb": round(peak_alloc_mb, 1) if peak_alloc_mb is not None else "",
            "peak_mem_reserved_mb": round(peak_reserved_mb, 1) if peak_reserved_mb is not None else "",
            "loss_first": round(first_loss, 4),
            "loss_last": round(last_loss, 4),
            "replicas_match": replicas_match,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "",
            "nccl_version": ".".join(map(str, torch.cuda.nccl.version())) if device.type == "cuda" else "",
            "flags": json.dumps({k: v for k, v in sorted(vars(args).items())
                                 if k not in ("results_csv", "run_name", "log_interval")}),
        }
        append_csv(args.results_csv, row)

        # per-step times, so variance and outliers can be inspected later
        steps_path = os.path.join(os.path.dirname(args.results_csv), "steps", f"{run_id}.csv")
        os.makedirs(os.path.dirname(steps_path), exist_ok=True)
        with open(steps_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timed_step", "step_time_ms"])
            w.writerows((i, round(t, 3)) for i, t in enumerate(ms))

        print(f"\n{run_id}")
        print(f"  tokens/s:   {tokens_per_sec:,.0f}  ({tokens_per_sec / world:,.0f} per rank)")
        print(f"  step time:  mean {row['step_time_ms_mean']} ms, median {row['step_time_ms_median']} ms, "
              f"std {row['step_time_ms_std']} ms")
        if peak_alloc_mb is not None:
            print(f"  peak mem:   {peak_alloc_mb:,.0f} MB allocated, {peak_reserved_mb:,.0f} MB reserved (max over ranks)")
        print(f"  wrote {args.results_csv} and {steps_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
