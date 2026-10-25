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
  # add a profiler pass (traces -> results/traces/<run_id>/, summary -> results/profiles/<run_id>/)
  torchrun --standalone --nproc_per_node=4 scripts/benchmark.py --dataset openwebtext --profile
"""

import argparse
import csv
import json
import os
import platform
import statistics
import subprocess
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from model import GPT, GPTConfig
from profiling import PROFILER_WARMUP_STEPS, format_summary, make_profiler, phase, summarize, top_ops_table

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
                   help="50304 = GPT-2's 50257 padded to a multiple of 64 (nanoGPT default); "
                        "50257 = unpadded. Padding makes the lm_head / embedding matmuls tile evenly.")
    p.add_argument("--head_dim", type=int,
                   help="attention head dimension; sets n_head = n_embd // head_dim (GPT-2 small: 64). "
                        "Params are unchanged, only how attention is split into heads.")
    # data / batch
    p.add_argument("--dataset", default="openwebtext", help="reads data/<dataset>/train.bin")
    p.add_argument("--batch_size", type=int, default=8, help="micro-batch size PER GPU (sequences)")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="micro-batches per optimizer step (per GPU). DDP all-reduces only on the last one.")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes; 0 = load batches in the training process")
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False,
                   help="page-locked host batches, enabling async (non_blocking) host-to-GPU copies")
    # precision / compiler
    p.add_argument("--dtype", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                   help="autocast dtype. auto: bf16 on GPUs that support it (L4, A100), fp16 otherwise "
                        "(T4), fp32 on CPU. fp16 uses a GradScaler to avoid gradient underflow.")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (compile time is in warmup)")
    p.add_argument("--compile_backend", default="inductor",
                   help="torch.compile backend. 'inductor' for real runs; 'aot_eager' traces the graph "
                        "without codegen, to test the compile path on machines without a C++ compiler")
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
    # profiling
    p.add_argument("--profile", action="store_true",
                   help="after the timed steps, profile extra steps with torch.profiler: saves traces "
                        "and prints a compute / communication / data-loading breakdown")
    p.add_argument("--profile_steps", type=int, default=10, help="steps recorded by the profiler")
    # output
    p.add_argument("--run_name", default="", help="free-form label stored in the CSV")
    p.add_argument("--results_csv", default=None,
                   help="default: results/benchmarks.csv on GPU, results/smoke/cpu_smoke.csv on CPU")
    p.add_argument("--log_interval", type=int, default=20)
    args = p.parse_args()

    if args.warmup_steps < 1 or args.steps < 1 or args.grad_accum < 1:
        p.error("--warmup_steps, --steps and --grad_accum must be >= 1")
    if args.head_dim is not None and args.n_head is not None:
        p.error("pass --head_dim or --n_head, not both")
    # fill unset architecture fields from the preset
    for k, v in MODEL_PRESETS[args.model].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.head_dim is not None:
        if args.n_embd % args.head_dim:
            p.error(f"--head_dim {args.head_dim} must divide n_embd={args.n_embd}")
        args.n_head = args.n_embd // args.head_dim
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "auto":
        if args.device == "cuda":
            args.dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            args.dtype = "fp32"
    if args.dtype == "bf16" and args.device == "cuda" and not torch.cuda.is_bf16_supported():
        p.error("this GPU has no bf16 support (e.g. T4); use --dtype fp16")
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


class RandomWindows(IterableDataset):
    """Endless stream of random (x, y) windows from a memory-mapped token file: nanoGPT's
    sampling scheme, wrapped as a Dataset so DataLoader workers / pin_memory can be toggled.
    Every (rank, worker) pair gets its own random stream, so no two see the same sequence."""

    def __init__(self, path, block_size, seed, rank):
        self.path, self.block_size, self.seed, self.rank = path, block_size, seed, rank
        n_tokens = os.path.getsize(path) // 2
        assert n_tokens > block_size + 1, f"{path} is too small for block_size={block_size}"

    def __iter__(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        # Open the memmap here, inside the process that reads it: worker processes are spawned
        # and a memmap should not be pickled across to them.
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        rng = np.random.default_rng([self.seed, self.rank, worker_id])
        high = len(data) - self.block_size - 1
        while True:
            i = int(rng.integers(high))
            chunk = torch.from_numpy(data[i:i + self.block_size + 1].astype(np.int64))
            yield chunk[:-1], chunk[1:]


class BatchLoader:
    def __init__(self, args, path, rank, device):
        self.device = device
        # non_blocking only makes the host-to-device copy async when the source is pinned
        self.non_blocking = args.pin_memory and device.type == "cuda"
        loader = DataLoader(RandomWindows(path, args.block_size, args.seed, rank),
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=args.pin_memory and device.type == "cuda",
                            persistent_workers=args.num_workers > 0)
        self.it = iter(loader)

    def next_batch(self):
        x, y = next(self.it)
        return x.to(self.device, non_blocking=self.non_blocking), y.to(self.device, non_blocking=self.non_blocking)


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


def shared_run_id(args, world):
    """Rank 0's timestamped id, broadcast so every rank writes files under the same name."""
    run_id = [f"{time.strftime('%Y%m%d-%H%M%S')}_{args.model}_g{world}_b{args.batch_size}"
              + (f"_{args.run_name}" if args.run_name else "")]
    if dist.is_initialized():
        dist.broadcast_object_list(run_id, src=0)
    return run_id[0]


def _fmt(summary, key):
    """Profiler value for the CSV, or empty when not profiled / not available on this device."""
    if summary is None or key not in summary:
        return ""
    return round(summary[key], 4)


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
    loader = BatchLoader(args, data_path, rank, device)

    # ---- model ----
    config = GPTConfig(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       block_size=args.block_size, vocab_size=args.vocab_size)
    model = GPT(config).to(device)
    n_params = model.get_num_params()
    flops_per_token = model.flops_per_token()
    optimizer = model.configure_optimizers(args.weight_decay, args.lr, (0.9, 0.95), device.type)
    if args.compile:
        # compile before wrapping in DDP (as nanoGPT does); DDP then splits the compiled graph at
        # gradient-bucket boundaries so all-reduce can still overlap with backward
        model = torch.compile(model, backend=args.compile_backend)
    if world > 1:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # ---- mixed precision ----
    # Weights and optimizer state stay fp32; autocast runs matmuls etc. in bf16/fp16.
    pt_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    autocast = nullcontext() if args.dtype == "fp32" else torch.autocast(device.type, dtype=pt_dtype)
    # fp16 has a narrow exponent range, so small gradients underflow to 0 unless the loss is scaled
    # up first. bf16 has fp32's exponent range and needs no scaler. Disabled = transparent no-op.
    scaler = torch.amp.GradScaler(device.type, enabled=args.dtype == "fp16")

    tokens_per_step = world * args.batch_size * args.grad_accum * args.block_size
    if master:
        print(f"world_size={world} device={device} dtype={args.dtype} compile={args.compile} "
              f"params={n_params / 1e6:.2f}M n_head={args.n_head} vocab={args.vocab_size} "
              f"tokens/step={tokens_per_step:,} warmup={args.warmup_steps} steps={args.steps}")

    run_id = shared_run_id(args, world)
    results_dir = os.path.dirname(args.results_csv)

    def train_step():
        # One optimizer step over grad_accum micro-batches. The phase labels show up as named
        # regions in profiler traces.
        for micro in range(args.grad_accum):
            # DDP all-reduces gradients on every backward by default. With accumulation only the
            # last micro-batch needs it; no_sync() skips the others (gradients just accumulate locally).
            last = micro == args.grad_accum - 1
            sync_ctx = model.no_sync() if (world > 1 and not last) else nullcontext()
            with phase("data_loading"):
                x, y = loader.next_batch()
            with sync_ctx:
                with phase("forward"), autocast:
                    _, loss = model(x, y)
                    loss = loss / args.grad_accum  # so accumulated grads equal the mean over micro-batches
                with phase("backward"):
                    scaler.scale(loss).backward()  # under DDP, all-reduce overlaps with this backward
        with phase("optimizer"):
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)  # clip the true gradients, not the scaled ones
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)  # with fp16, skips the step if any gradient is inf/nan
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        return loss.detach() * args.grad_accum  # last micro-batch's loss, unscaled for logging

    model.train()

    # ---- warmup (untimed) ----
    # The first step includes torch.compile's compilation (if on) and CUDA/NCCL initialisation,
    # so it is timed separately and reported as first_step_s.
    sync(device)
    t_first = time.perf_counter()
    for step in range(args.warmup_steps):
        loss = train_step()
        if step == 0:
            first_loss = loss.item()
            first_step_s = reduce_max(time.perf_counter() - t_first, device)
            if master:
                print(f"warmup step 0 loss {first_loss:.4f} ({first_step_s:.1f} s incl. setup/compile)")
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)  # peak memory over the timed steps only

    # ---- timed steps ----
    # GPU: per-step times come from CUDA events recorded on the stream between steps. We do NOT
    # call cuda.synchronize() every step: that would stop the CPU from preparing step i+1 while the
    # GPU runs step i, which no real training loop does, and would inflate data-loading cost.
    # Throughput uses host wall-clock over the whole window, synchronised only at both ends.
    is_cuda = device.type == "cuda"
    marks = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps + 1)] if is_cuda else []
    wall_t0 = time.perf_counter()
    for i in range(args.steps):
        if is_cuda:
            marks[i].record()
        else:
            marks.append(time.perf_counter())
        loss = train_step()
        if (i + 1) % args.log_interval == 0:
            lossf = loss.item()  # sync point, so only on log steps
            if master:
                avg_ms = (time.perf_counter() - wall_t0) * 1000 / (i + 1)
                print(f"timed step {i + 1:4d}/{args.steps} loss {lossf:.4f} avg {avg_ms:.1f} ms/step")
    if is_cuda:
        marks[-1].record()
    else:
        marks.append(time.perf_counter())
    sync(device)
    wall_time = time.perf_counter() - wall_t0
    last_loss = loss.item()
    if is_cuda:
        step_ms = [marks[i].elapsed_time(marks[i + 1]) for i in range(args.steps)]
    else:
        step_ms = [(marks[i + 1] - marks[i]) * 1000 for i in range(args.steps)]

    # ---- optional profiling window (after the timed steps, so it can't skew them) ----
    profile_summary = None
    if args.profile:
        trace_dir = os.path.join(results_dir, "traces", run_id)
        n = PROFILER_WARMUP_STEPS + args.profile_steps
        with make_profiler(device, trace_dir, rank, args.profile_steps) as prof:
            for i in range(n):
                train_step()
                if i == n - 1:
                    sync(device)  # make sure the last step's kernels finish inside the window
                prof.step()
        profile_summary = summarize(prof, device, args.profile_steps)
        prof_dir = os.path.join(results_dir, "profiles", run_id)
        os.makedirs(prof_dir, exist_ok=True)
        with open(os.path.join(prof_dir, f"rank{rank}.json"), "w") as f:
            json.dump({"run_id": run_id, "rank": rank, **profile_summary}, f, indent=2)
        if master:
            print("\n" + top_ops_table(prof, device))
            print(format_summary(profile_summary, rank))
            print(f"  traces: {trace_dir}  (all ranks' summaries: {prof_dir})")

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
    total_time = reduce_max(wall_time, device)
    peak_alloc_mb = peak_reserved_mb = None
    if device.type == "cuda":
        peak_alloc_mb = reduce_max(torch.cuda.max_memory_allocated(device) / 2**20, device)
        peak_reserved_mb = reduce_max(torch.cuda.max_memory_reserved(device) / 2**20, device)

    if master:
        tokens_per_sec = tokens_per_step * args.steps / total_time
        ms = step_ms
        commit, dirty = git_info()

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
            "dtype": args.dtype,
            "compile": args.compile_backend if args.compile else False,
            "batch_size_per_gpu": args.batch_size,
            "grad_accum": args.grad_accum,
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "global_batch_tokens": tokens_per_step,
            "warmup_steps": args.warmup_steps,
            "timed_steps": args.steps,
            "first_step_s": round(first_step_s, 3),
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
            # profiler breakdown (rank 0) if --profile was set; all ranks are in results/profiles/
            "prof_gpu_compute_frac": _fmt(profile_summary, "gpu_compute_frac"),
            "prof_gpu_comm_exposed_frac": _fmt(profile_summary, "gpu_comm_exposed_frac"),
            "prof_gpu_idle_frac": _fmt(profile_summary, "gpu_idle_frac"),
            "prof_cpu_data_loading_ms": _fmt(profile_summary, "cpu_data_loading_ms"),
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
              f"std {row['step_time_ms_std']} ms (timed steps, profiler off)")
        if peak_alloc_mb is not None:
            print(f"  peak mem:   {peak_alloc_mb:,.0f} MB allocated, {peak_reserved_mb:,.0f} MB reserved (max over ranks)")
        print(f"  wrote {args.results_csv} and {steps_path}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
