"""
Step-by-step timing breakdown of T5SLA2Attention forward pass.

Each sub-step is timed individually with MPS/CUDA synchronization so we get
accurate wall-clock cost per stage.

Usage:
    python profile_breakdown.py                        # MPS, L=1024
    python profile_breakdown.py --device cpu --seq 512
"""

import argparse
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from t5_sla2 import (
    T5SLA2Attention, SLA2Config,
    pool_position_bias, _expand_block_mask,
)


# ---------------------------------------------------------------------------
# Synchronization helper
# ---------------------------------------------------------------------------

def sync(device: str):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def timed(fn, device: str, reps: int = 20):
    """Run fn reps times, return median ms."""
    for _ in range(3):
        fn()
    sync(device)

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Main breakdown
# ---------------------------------------------------------------------------

def run(args):
    device = args.device
    L = args.seq
    B = args.batch
    reps = args.reps

    class MockConfig:
        d_model = 512
        d_kv = 64
        num_heads = 8
        dropout_rate = 0.0
        is_decoder = False
        relative_attention_num_buckets = 32
        relative_attention_max_distance = 128

    cfg = MockConfig()
    sla2_cfg = SLA2Config(block_size=64, top_k_frac=0.15, min_seq_len=256)
    D = cfg.d_model

    attn = T5SLA2Attention(cfg, has_relative_attention_bias=True, sla2_config=sla2_cfg)
    attn.eval().to(device)

    x = torch.randn(B, L, D, device=device)

    print(f"\n{'='*62}")
    print(f"  T5SLA2 Step Breakdown  |  device={device}  B={B}  L={L}")
    print(f"{'='*62}")

    results = {}

    # ---- 1. Q/K/V projections ----
    def step_qkv():
        _ = attn._shape(attn.q(x), L, B)
        _ = attn._shape(attn.k(x), L, B)
        _ = attn._shape(attn.v(x), L, B)

    results["1  QKV projections"] = timed(step_qkv, device, reps)

    with torch.no_grad():
        q = attn._shape(attn.q(x), L, B)
        k = attn._shape(attn.k(x), L, B)
        v = attn._shape(attn.v(x), L, B)

    # ---- 2. Position bias ----
    def step_bias():
        _ = attn.compute_bias(L, L, device=device)

    results["2  Position bias"] = timed(step_bias, device, reps)

    with torch.no_grad():
        position_bias = attn.compute_bias(L, L, device=device)

    # ---- 3. Pool position bias (for router) ----
    bs = sla2_cfg.block_size

    def step_pool():
        _ = pool_position_bias(position_bias.detach(), bs, bs)

    results["3  Pool bias (router prep)"] = timed(step_pool, device, reps)

    with torch.no_grad():
        pooled_bias = pool_position_bias(position_bias.detach(), bs, bs)

    # ---- 4. Router (block selection) ----
    def step_router():
        _ = attn.router(
            q, k,
            block_size=bs,
            pooled_bias=pooled_bias,
            k_frac=sla2_cfg.top_k_frac,
            tau=sla2_cfg.tau,
            max_iter=sla2_cfg.max_iter,
            hard=True,
        )

    results["4  Router (block selection)"] = timed(step_router, device, reps)

    with torch.no_grad():
        block_mask = attn.router(
            q, k,
            block_size=bs,
            pooled_bias=pooled_bias,
            k_frac=sla2_cfg.top_k_frac,
            tau=sla2_cfg.tau,
            max_iter=sla2_cfg.max_iter,
            hard=True,
        )

    # ---- 5a. Expand block mask ----
    def step_expand():
        _ = _expand_block_mask(block_mask, bs, bs, L, L)

    results["5a Expand block mask"] = timed(step_expand, device, reps)

    with torch.no_grad():
        full_mask = _expand_block_mask(block_mask, bs, bs, L, L)

    # ---- 5b. Sparse attention (masked-dense: FULL L×L matmul + mask) ----
    def step_sparse():
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        scores = scores + position_bias
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores + (1.0 - full_mask) * neg_inf
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        _ = torch.matmul(weights, v)

    results["5b Sparse attn (masked-dense, O(L^2))"] = timed(step_sparse, device, reps)

    # ---- 6. Linear attention ----
    def step_linear():
        phi_q = attn.phi(q)
        phi_k = attn.phi(k)
        kv_sum = torch.matmul(phi_k.transpose(-1, -2), v)
        k_sum = phi_k.sum(dim=-2, keepdim=True)
        numerator = torch.matmul(phi_q, kv_sum)
        denominator = (phi_q * k_sum).sum(dim=-1, keepdim=True).clamp(min=1e-6)
        _ = numerator / denominator

    results["6  Linear attn (O(ND^2))"] = timed(step_linear, device, reps)

    # ---- 7. Alpha blend ----
    with torch.no_grad():
        o_sparse = attn._sparse_attention(q, k, v, block_mask, position_bias, None)
        o_linear = attn._linear_attention(q, k, v)

    def step_blend():
        a = attn.alpha()
        _ = a * o_sparse + (1.0 - a) * o_linear

    results["7  Alpha blend"] = timed(step_blend, device, reps)

    # ---- 8. Output projection ----
    with torch.no_grad():
        a = attn.alpha()
        merged = a * o_sparse + (1.0 - a) * o_linear
        merged_2d = merged.transpose(1, 2).contiguous().view(B, L, attn.inner_dim)

    def step_out():
        _ = attn.o(merged_2d)

    results["8  Output projection"] = timed(step_out, device, reps)

    # ---- Full SLA2 forward (end-to-end) ----
    def step_full():
        attn(x)

    results["── FULL SLA2 forward"] = timed(step_full, device, reps)

    # ---- Standard attention (bare matmul, no mask) ----
    def step_std_attn():
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.scale
        scores = scores + position_bias
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        _ = torch.matmul(weights, v)

    results["── Std attn core (O(L^2), no overhead)"] = timed(step_std_attn, device, reps)

    # ---- Print table ----
    total = results["── FULL SLA2 forward"]
    print(f"\n  {'Step':<42}  {'ms':>8}  {'% of SLA2':>10}")
    print(f"  {'-'*42}  {'-'*8}  {'-'*10}")
    for name, ms in results.items():
        is_ref = "FULL" in name or "Std" in name
        pct_str = f"{ms/total*100:8.1f}%" if not is_ref else "        —"
        print(f"  {name:<42}  {ms:>8.2f}  {pct_str}")

    sparse_ms = results["5b Sparse attn (masked-dense, O(L^2))"]
    linear_ms = results["6  Linear attn (O(ND^2))"]
    router_ms = results["4  Router (block selection)"]
    std_ms    = results["── Std attn core (O(L^2), no overhead)"]

    print(f"\n  KEY FINDINGS:")
    print(f"  - Sparse branch ({sparse_ms:.2f} ms) does a FULL L\u00d7L matmul then masks it")
    print(f"    -> Same memory/compute as standard attention ({std_ms:.2f} ms) + masking overhead")
    print(f"  - Linear branch ({linear_ms:.2f} ms) runs in parallel, adding more compute")
    print(f"  - SLA2 = sparse + linear + router + blend \u2248 2\u00d7 baseline")
    print(f"\n  TO FIX THIS with Triton kernels:")
    print(f"  - Only compute the ~{sla2_cfg.top_k_frac*100:.0f}% selected blocks (skip the rest entirely)")
    print(f"  - Fuse masking + softmax + matmul into one kernel (no L\u00d7L materialization)")
    print(f"  - Expected real speedup: ~{1/sla2_cfg.top_k_frac:.1f}x on sparse branch alone")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--reps", type=int, default=20)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    with torch.no_grad():
        run(args)
