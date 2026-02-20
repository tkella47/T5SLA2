"""
Benchmark: T5SLA2Attention vs standard T5Attention (HuggingFace)

Measures wall-clock time and peak memory for a forward pass across
multiple sequence lengths.

Usage:
    python benchmark.py
    python benchmark.py --device mps   # Apple Silicon GPU
    python benchmark.py --reps 20      # more repetitions for stability
"""

import argparse
import time
import copy

import torch
import torch.nn as nn

from transformers.models.t5.modeling_t5 import T5Attention, T5Config

from t5_sla2 import T5SLA2Attention, SLA2Config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_t5_config(d_model=512, d_kv=64, num_heads=8) -> T5Config:
    cfg = T5Config(
        d_model=d_model,
        d_kv=d_kv,
        num_heads=num_heads,
        d_ff=2048,
        dropout_rate=0.0,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
    )
    return cfg


def build_standard_attn(cfg: T5Config, is_decoder=False) -> T5Attention:
    return T5Attention(cfg, has_relative_attention_bias=True)


def build_sla2_attn(cfg: T5Config, sla2_cfg: SLA2Config) -> T5SLA2Attention:
    attn = T5SLA2Attention(cfg, has_relative_attention_bias=True, sla2_config=sla2_cfg)
    return attn


def call_module(module: nn.Module, hidden: torch.Tensor):
    """Call either T5Attention or T5SLA2Attention with the right args."""
    L = hidden.shape[1]
    if isinstance(module, T5Attention):
        # transformers 5.x requires query_length or cache_position
        return module(hidden, query_length=L)
    return module(hidden)


def time_forward(
    module: nn.Module,
    hidden: torch.Tensor,
    reps: int,
    device: str,
) -> float:
    """Return median wall-clock seconds per forward pass."""
    module.eval()
    with torch.no_grad():
        # Warm-up
        for _ in range(3):
            call_module(module, hidden)
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            call_module(module, hidden)
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    times.sort()
    return times[len(times) // 2]  # median


def peak_mem_mb(module: nn.Module, hidden: torch.Tensor, device: str) -> float:
    """Return peak memory (MB) during a single forward pass."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        module.eval()
        with torch.no_grad():
            call_module(module, hidden)
        return torch.cuda.max_memory_allocated() / 1e6
    elif device == "mps":
        torch.mps.empty_cache()
        module.eval()
        with torch.no_grad():
            call_module(module, hidden)
        return torch.mps.current_allocated_memory() / 1e6
    return float("nan")  # CPU: not tracked


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(args):
    device = args.device
    reps = args.reps
    batch = args.batch

    cfg = make_t5_config(d_model=args.d_model, d_kv=args.d_kv, num_heads=args.num_heads)
    sla2_cfg = SLA2Config(
        block_size=args.block_size,
        top_k_frac=args.top_k_frac,
        min_seq_len=args.min_seq_len,
    )

    seq_lens = args.seq_lens

    print(f"\n{'='*70}")
    print(f"  T5SLA2 vs Standard T5Attention Benchmark")
    print(f"  device={device}  batch={batch}  d_model={args.d_model}  "
          f"heads={args.num_heads}  reps={reps}")
    print(f"  SLA2: block_size={args.block_size}  top_k_frac={args.top_k_frac}  "
          f"min_seq_len={args.min_seq_len}")
    print(f"{'='*70}")
    print(f"{'SeqLen':>8}  {'Standard(ms)':>14}  {'SLA2(ms)':>12}  "
          f"{'Speedup':>9}  {'Mem-Std(MB)':>12}  {'Mem-SLA2(MB)':>13}")
    print(f"{'-'*70}")

    for L in seq_lens:
        hidden = torch.randn(batch, L, args.d_model, device=device)

        # --- Standard T5Attention ---
        std_attn = build_standard_attn(cfg).to(device)
        t_std = time_forward(std_attn, hidden, reps, device) * 1000  # ms
        mem_std = peak_mem_mb(std_attn, hidden, device)

        # --- SLA2Attention ---
        sla2_attn = build_sla2_attn(cfg, copy.deepcopy(sla2_cfg)).to(device)
        t_sla2 = time_forward(sla2_attn, hidden, reps, device) * 1000  # ms
        mem_sla2 = peak_mem_mb(sla2_attn, hidden, device)

        speedup = t_std / t_sla2 if t_sla2 > 0 else float("inf")

        mem_std_s  = f"{mem_std:.1f}" if not torch.tensor(mem_std).isnan() else "  N/A"
        mem_sla2_s = f"{mem_sla2:.1f}" if not torch.tensor(mem_sla2).isnan() else "  N/A"

        print(f"{L:>8}  {t_std:>14.2f}  {t_sla2:>12.2f}  "
              f"{speedup:>8.2f}x  {mem_std_s:>12}  {mem_sla2_s:>13}")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5SLA2 speed benchmark")

    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps"],
                        help="Compute device")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_kv", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--reps", type=int, default=10,
                        help="Repetitions per measurement (median is reported)")
    parser.add_argument("--seq_lens", type=int, nargs="+",
                        default=[128, 256, 512, 1024, 2048],
                        help="Sequence lengths to test")
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--top_k_frac", type=float, default=0.15)
    parser.add_argument("--min_seq_len", type=int, default=256,
                        help="Below this length SLA2 falls back to full attention")

    args = parser.parse_args()

    # Validate device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU.")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU.")
        args.device = "cpu"

    run(args)
