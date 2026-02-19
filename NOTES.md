# SLA2 Session Notes

## What We Built
- `t5_sla2.py`: Drop-in replacement for HuggingFace T5Attention using SLA2 (arXiv 2602.12675)
- `MOTIVATION.md`: Design rationale document
- Repo: https://github.com/tkella47/T5SLA2.git

## Environment
- Python 3.9.6 (use `python3`, no `python` binary)
- torch 2.8.0 installed via `pip3 install torch`
- No conda, no numpy
- All 10 smoke tests pass: `python3 t5_sla2.py`

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

## Triton Kernel Paths (not implemented yet)
- Sparse branch: could swap in FlexAttention (`torch.nn.attention.flex_attention`) or hbsattn
- Linear branch: could swap in flash-linear-attention (fla-org)
- No one has written a fused SLA2 kernel yet

## Training Pipeline
1. **Stage 1**: Freeze model, train only router + alpha via MSE distillation against full attention output (~500 steps)
2. **Stage 2**: Switch router to hard mode, fine-tune on task loss

## What's Left / Next Steps
- Test with actual HuggingFace T5 model (`patch_t5_with_sla2`)
- Run Stage 1 router training on real data
- Benchmark against standard T5 attention
- Optionally integrate Triton kernels for real speedup
