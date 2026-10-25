"""
PyTorch Profiler helpers for benchmark.py: build the profiler, and turn its events into a
short compute / communication / data-loading breakdown.

How the GPU breakdown works
---------------------------
Every CUDA kernel the profiler records is put in one of three groups by name:
  communication  NCCL kernels (DDP gradient all-reduce, FSDP all-gather / reduce-scatter)
  memcpy         Memcpy / Memset (host-to-device batch copies, etc.)
  compute        everything else (matmuls, attention, layernorm, optimizer, ...)
The profiled window on the GPU timeline (first kernel start -> last kernel end) is then split
into four parts that add up to 100%:
  compute        some compute kernel is running
  exposed comm   NCCL is running and no compute kernel is, i.e. communication NOT hidden
                 behind backward. This is the part that hurts scaling.
  memcpy only    only copies are running
  idle           nothing is running on the GPU, e.g. it is waiting on the CPU (Python
                 overhead, kernel launch, or data loading that isn't keeping up)
The total NCCL time, including the part overlapped with compute, is reported separately.

On CPU (smoke tests) there is no kernel timeline, so only the CPU-side phase times
(data_loading / forward / backward / optimizer) and communication op times are reported.
"""

import os
from contextlib import contextmanager

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler

PHASES = ("data_loading", "forward", "backward", "optimizer")
COMM_KEYWORDS = ("nccl", "gloo", "all_reduce", "allreduce", "all_gather", "allgather",
                 "reduce_scatter", "broadcast")
PROFILER_WARMUP_STEPS = 2  # first profiled steps carry profiler start-up overhead; discarded


@contextmanager
def phase(name):
    # Labels a region in the trace. Cheap enough to leave on when no profiler is active
    # (it only checks a flag), so timed and profiled steps run the same code.
    with record_function(name):
        yield


def make_profiler(device, trace_dir, rank, active_steps):
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    os.makedirs(trace_dir, exist_ok=True)
    return profile(
        activities=activities,
        # repeat=1: one cycle only. The profiler clears events between cycles.
        schedule=schedule(wait=0, warmup=PROFILER_WARMUP_STEPS, active=active_steps, repeat=1),
        # gzip'd Chrome trace, one file per rank; open in https://ui.perfetto.dev or TensorBoard
        on_trace_ready=tensorboard_trace_handler(trace_dir, worker_name=f"rank{rank}", use_gzip=True),
        record_shapes=False,  # shapes/stacks make traces much bigger and add overhead
        with_stack=False,
    )


# ---- interval arithmetic on (start, end) lists, in microseconds ----

def _union(intervals):
    merged = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _length(merged):
    return sum(e - s for s, e in merged)


def _subtract(a, b):
    """Parts of merged list a not covered by merged list b."""
    out, j = [], 0
    for s, e in a:
        cur = s
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < e:
            if b[k][0] > cur:
                out.append([cur, b[k][0]])
            cur = max(cur, b[k][1])
            k += 1
        if cur < e:
            out.append([cur, e])
    return out


def _is_comm(name):
    n = name.lower()
    return any(k in n for k in COMM_KEYWORDS)


def summarize(prof, device, n_steps):
    """Return a dict of per-step times in ms (and GPU fractions) from the profiler's events."""
    events = prof.events()
    cpu = [e for e in events if e.device_type == DeviceType.CPU]
    per_step_ms = lambda us: us / 1000 / n_steps

    s = {"profiled_steps": n_steps}
    step_spans = [e for e in cpu if e.name.startswith("ProfilerStep")]
    s["cpu_step_wall_ms"] = per_step_ms(sum(e.time_range.end - e.time_range.start for e in step_spans))
    for name in PHASES:
        spans = [(e.time_range.start, e.time_range.end) for e in cpu if e.name == name]
        s[f"cpu_{name}_ms"] = per_step_ms(_length(_union(spans)))
    s["cpu_comm_ops_ms"] = per_step_ms(_length(_union(
        [(e.time_range.start, e.time_range.end) for e in cpu if _is_comm(e.name)])))

    if device.type == "cuda":
        # Kineto mirrors record_function labels (our phases, ProfilerStep#N) onto the GPU timeline
        # as "user annotation" spans. They are not kernels and would cover whole steps, so drop them.
        kernels = [e for e in events if e.device_type == DeviceType.CUDA
                   and e.name not in PHASES and not e.name.startswith("ProfilerStep")]
        if not kernels:
            s["gpu_note"] = "no CUDA events recorded"
            return s
        groups = {"comm": [], "memcpy": [], "compute": []}
        for e in kernels:
            n = e.name.lower()
            g = "comm" if "nccl" in n else "memcpy" if ("memcpy" in n or "memset" in n) else "compute"
            groups[g].append((e.time_range.start, e.time_range.end))
        C, M, P = (_union(groups[g]) for g in ("compute", "comm", "memcpy"))
        window = max(e.time_range.end for e in kernels) - min(e.time_range.start for e in kernels)

        compute = _length(C)
        exposed_comm = _length(_subtract(M, C))
        memcpy_only = _length(_subtract(_subtract(P, C), M))
        busy = _length(_union(C + M + P))
        s.update({
            "gpu_window_ms": per_step_ms(window),
            "gpu_compute_ms": per_step_ms(compute),
            "gpu_comm_total_ms": per_step_ms(_length(M)),
            "gpu_comm_exposed_ms": per_step_ms(exposed_comm),
            "gpu_memcpy_only_ms": per_step_ms(memcpy_only),
            "gpu_idle_ms": per_step_ms(window - busy),
        })
        for k in ("compute", "comm_exposed", "memcpy_only", "idle"):
            s[f"gpu_{k}_frac"] = s[f"gpu_{k}_ms"] / s["gpu_window_ms"]
    return s


def format_summary(s, rank):
    lines = [f"Profiler breakdown (rank {rank}, {s['profiled_steps']} steps, per-step averages; "
             f"profiler overhead inflates absolute times):"]
    if "gpu_window_ms" in s:
        w = s["gpu_window_ms"]
        lines += [
            f"  GPU timeline          {w:8.2f} ms/step",
            f"    compute             {s['gpu_compute_ms']:8.2f} ms  ({100 * s['gpu_compute_frac']:5.1f}%)",
            f"    exposed comm (NCCL) {s['gpu_comm_exposed_ms']:8.2f} ms  ({100 * s['gpu_comm_exposed_frac']:5.1f}%)"
            f"   [total NCCL {s['gpu_comm_total_ms']:.2f} ms, rest overlapped with compute]",
            f"    memcpy only         {s['gpu_memcpy_only_ms']:8.2f} ms  ({100 * s['gpu_memcpy_only_frac']:5.1f}%)",
            f"    idle                {s['gpu_idle_ms']:8.2f} ms  ({100 * s['gpu_idle_frac']:5.1f}%)",
        ]
    elif "gpu_note" in s:
        lines.append(f"  GPU: {s['gpu_note']}")
    lines.append(f"  CPU side (wall time inside each labelled region; step {s['cpu_step_wall_ms']:.2f} ms):")
    for name in PHASES:
        lines.append(f"    {name:<18}  {s[f'cpu_{name}_ms']:8.2f} ms")
    lines.append(f"    communication ops   {s['cpu_comm_ops_ms']:8.2f} ms  (CPU-side time of c10d/NCCL/gloo calls)")
    if "gpu_window_ms" in s:
        lines.append("  CPU times overlap the GPU (kernels run asynchronously); data loading only costs "
                     "throughput if the GPU sits idle waiting for it.")
    return "\n".join(lines)


def top_ops_table(prof, device, row_limit=12):
    sort_by = "self_device_time_total" if device.type == "cuda" else "self_cpu_time_total"
    return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
