# SLA2 Session Notes

## What We Built
- `t5_sla2.py`: Drop-in replacement for HuggingFace T5Attention using SLA2 (arXiv 2602.12675)
- `MOTIVATION.md`: Design rationale document
- Repo: https://github.com/tkella47/T5SLA2.git

## Environment
- Python 3.12.9, venv at `.venv/` — activate with `source .venv/bin/activate`
- torch 2.10.0, transformers 5.2.0 installed via `pip install -r requirements.txt`
- All 10 smoke tests pass: `python t5_sla2.py`

## Architecture Recap
Two branches blended by learned alpha per head:
- **Sparse branch**: Router picks top 15% of key blocks per query block, full attention only on those
- **Linear branch**: Feature map (ELU+1) on Q and K, then kernel trick `phi(Q) @ (phi(K)^T @ V)` — O(N*D^2) instead of O(N^2)
- **Output**: `alpha * sparse + (1-alpha) * linear`

## Key Concept: Linear Attention Order of Operations
- Standard attention: dot product (Q@K^T) → softmax → multiply by V
- Linear attention: feature map phi FIRST on Q and K independently → THEN dot product → multiply by V
- Phi goes first, which is what allows rearranging multiplication order (associative trick)
- Default phi = ELU+1 (always positive). Softmax-as-phi is a different thing — just an alternative option, not used by default.

## Benchmarking (done — see benchmark.py and profile_breakdown.py)
- `benchmark.py`: wall-clock comparison SLA2 vs standard T5Attention across seq lengths
- `profile_breakdown.py`: per-step timing breakdown of SLA2 forward pass
- Ran on MPS (Apple Silicon), results at L=1024:
  - Standard attn core: ~14ms, SLA2 full forward: ~31ms (~0.5x, i.e. 2x slower)
  - Step breakdown shows sparse branch = 58.6% of total time
- **Root cause**: the masked-dense sparse path computes the FULL L×L score matrix
  then masks it — no actual compute savings. The gather path also expands to O(L²)
  before gathering. Both paths are paying full dense attention cost + overhead.
- The linear branch (O(ND²)) only costs ~5% — not the bottleneck

## Why It's Slow: The Core Problem
SLA2 runs TWO full branches (sparse + linear) and blends them:
- Sparse "branch" = full L×L matmul + masking = dense attention + overhead
- Linear branch = correct O(ND²) but adds on top
- Net result: ~2× the compute of standard attention

## How to Fix: Triton Kernel (next step, needs CUDA GPU)
Two options, in order of pragmatism:

**Option A — xformers (faster to get working)**
- `xformers.ops` has pre-built block-sparse Triton kernels (`BlockSparseCS`)
- Adapt router mask output → xformers format
- No kernel writing needed

**Option B — Custom Triton kernel (more control)**
- Write a block-sparse attention kernel using FlashAttention-style online softmax
- Only iterate over router-selected blocks — never touch unselected ones
- Fuse: QK matmul + position bias + softmax + V matmul into one kernel
- Expected speedup on sparse branch alone: ~6.7× (1/top_k_frac=0.15)
- Key challenge: online softmax must accumulate across all selected key blocks
  per query block (not per-block softmax, which would be wrong)
- Drop in as replacement for `_sparse_attention()` in t5_sla2.py

**Triton kernel paths (not implemented yet)**
- Sparse branch: custom Triton OR xformers block-sparse
- Linear branch: could swap in flash-linear-attention (fla-org) for further gains
- No fused SLA2 kernel exists anywhere yet

## Training Pipeline
1. **Stage 1**: Freeze model, train only router + alpha via MSE distillation against full attention output (~500 steps)
2. **Stage 2**: Switch router to hard mode, fine-tune on task loss

## What's Left / Next Steps
- **[NEXT]** On CUDA machine: implement Triton block-sparse kernel (Option A or B above)
  - Start with xformers option to validate speedup is real before writing custom kernel
  - Target: sparse branch should be ~6-7× faster, overall SLA2 ~2-3× faster than std attn
- Test with actual HuggingFace T5 model (`patch_t5_with_sla2`)
- Run Stage 1 router training on real data
- Re-run `benchmark.py` and `profile_breakdown.py` on CUDA after kernel swap
