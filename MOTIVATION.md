# SLA2 for T5: Motivation and Design Rationale

## Why SLA2?

Standard attention computes scores for every (query, key) pair — N^2 scores total. But after softmax, 90-97% of attention weights are near zero. The model only cares about a small fraction of keys for any given query. SLA2 exploits this by skipping the parts that would softmax to ~0 anyway.

## Why Learned Routing?

Prior sparse attention approaches each have drawbacks:

- **Fixed patterns** (Longformer, BigBird): Hand-designed local windows + global tokens. Don't adapt to the input.
- **Hash-based** (Reformer): LSH to find similar Q/K pairs. Noisy — hash collisions miss relevant pairs.
- **Top-k on raw scores** (Routing Transformer): Must compute full N^2 scores first, defeating the purpose.

SLA2 operates at **block granularity** with **learned projections**. Tokens are pooled into blocks of 64, and a small router predicts which block pairs matter. Router cost is O((N/B)^2) — for N=4096 with B=64, that's 4K scores instead of 16M.

The router has its own `proj_q` and `proj_k` projections, separate from attention's Q/K. The attention projections optimize for accurate scores; the router projections optimize for predicting which blocks will have high aggregate attention. These are related but distinct tasks. Stage 1 distillation training teaches the router by minimizing MSE between SLA2 output and full attention output.

## Why Two Branches?

Sparse attention alone completely zeros out unselected blocks. Even when individual scores are small, their aggregate contribution can matter for tasks requiring broad context.

Linear attention via the kernel trick — `phi(Q) @ (phi(K)^T @ V)` — provides O(N*D^2) approximate coverage of all keys. It's less accurate than full attention but captures the broad, diffuse signal that sparse attention misses.

The combination `alpha * sparse + (1 - alpha) * linear` provides:

- **Sparse branch**: Precise attention to the most important blocks, with position bias
- **Linear branch**: Coarse coverage of everything else, no position bias (incompatible with the kernel trick)
- **Learned alpha**: Per-head blending — peaky heads favor sparse, broad-context heads favor linear

## Why It Should Work for T5

The SLA2 paper demonstrated results on diffusion models, but the mechanism is architecture-agnostic:

- T5's attention exhibits the same sparsity pattern — most weights near zero, especially for longer sequences
- T5's relative position bias helps the router — pooled to block level, it gives a strong prior that nearby blocks matter
- Safe fallback — short sequences and generation steps use standard attention, so we never do worse than baseline
- Identical projection structure — same Q/K/V/O shapes, same lack of bias terms

## Known Limitations

- **Router quality is everything.** Poor Stage 1 convergence means the router drops important blocks. Needs sufficient data and training steps.
- **Masked-dense doesn't save memory.** The default path still materializes the full N^2 score matrix (just masks non-selected blocks to -inf). The gather path saves memory but has complex indexing. Real speedups need Triton kernels.
- **Linear attention is a lossy approximation.** The ELU+1 feature map is not a perfect softmax substitute. Tasks requiring precise long-range attention may suffer.
- **Block granularity is coarse.** If a critical key is at position 65 but the rest of its block is irrelevant, all 64 positions are included. Block size 64 trades selection precision for router efficiency.

## Summary

1. Attention is sparse in practice — exploit it
2. The sparse pattern is input-dependent — learn it, don't hardcode it
3. Block-level routing is cheap — O((N/B)^2) instead of O(N^2)
4. Sparse alone is lossy — linear attention covers the gaps at O(N*D^2)
5. Different heads need different strategies — learned alpha per head

The paper reports 18.6x speedup with negligible quality loss. Whether that holds for T5 on a specific task is an empirical question — Stage 1 training and downstream evaluation would answer it.
