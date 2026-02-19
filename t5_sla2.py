"""
T5-SLA2: Sparse-Linear Attention with Learnable Routing for T5

Adaptation of SLA2 (arXiv 2602.12675) for HuggingFace T5Attention.
Drop-in replacement that combines learned block-sparse routing with
linear attention via a learnable per-position combination ratio alpha.

Usage:
    from t5_sla2 import T5SLA2Attention, patch_t5_with_sla2

    # Option A: patch an existing T5 model
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    patch_t5_with_sla2(model, top_k_frac=0.15, block_size=64)

    # Option B: use directly
    attn = T5SLA2Attention(config, has_relative_attention_bias=True)
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SLA2Config:
    """Configuration for SLA2-specific hyperparameters."""
    block_size: int = 64
    top_k_frac: float = 0.15
    tau: float = 0.1
    max_iter: int = 20
    feature_map: str = "elu"          # "elu", "relu", or "softmax"
    min_seq_len: int = 256
    sparse_method: str = "masked"     # "masked" or "gather"
    scale_attn: bool = False          # T5 default: no sqrt(d) scaling
    use_qat: bool = False
    qat_bits: int = 8
    router_hard: bool = False         # True after Stage 1 training

# ---------------------------------------------------------------------------
# Utility: SoftTop-k with per-row binary-searched lambda
# ---------------------------------------------------------------------------

def soft_topk(
    scores: torch.Tensor,
    k_frac: float,
    tau: float = 0.1,
    max_iter: int = 20,
) -> torch.Tensor:
    """Differentiable soft top-k selection via sigmoid with per-row lambda.

    For each row, binary-searches for lambda_i such that
        sum_j sigmoid((scores[..., i, j] / tau) + lambda_i) ≈ k_frac * n_cols

    Args:
        scores: (..., n_rows, n_cols) block-level scores.
        k_frac: Fraction of columns to select per row.
        tau: Temperature for sigmoid sharpness.
        max_iter: Binary search iterations.

    Returns:
        Soft mask of same shape, values in (0, 1).
    """
    target = k_frac * scores.shape[-1]
    scaled = scores / tau  # (..., n_rows, n_cols)

    lo = torch.full(scores.shape[:-1], -1e4, device=scores.device, dtype=scores.dtype)
    hi = torch.full_like(lo, 1e4)

    for _ in range(max_iter):
        mid = (lo + hi) * 0.5                          # (..., n_rows)
        cur = torch.sigmoid(scaled + mid.unsqueeze(-1)) # (..., n_rows, n_cols)
        total = cur.sum(dim=-1)                         # (..., n_rows)
        lo = torch.where(total < target, mid, lo)
        hi = torch.where(total >= target, mid, hi)

    lam = (lo + hi) * 0.5
    return torch.sigmoid(scaled + lam.unsqueeze(-1))


def hard_topk(
    scores: torch.Tensor,
    k_frac: float,
) -> torch.Tensor:
    """Hard top-k selection (inference). Returns binary mask."""
    k = max(1, int(k_frac * scores.shape[-1]))
    _, idx = scores.topk(k, dim=-1)
    mask = torch.zeros_like(scores)
    mask.scatter_(-1, idx, 1.0)
    return mask

# ---------------------------------------------------------------------------
# Utility: Block mean-pooling
# ---------------------------------------------------------------------------

def mean_pool_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Pool last-but-one dim into blocks of `block_size`, taking the mean.

    Args:
        x: (B, H, N, D)

    Returns:
        (B, H, ceil(N / block_size), D)
    """
    B, H, N, D = x.shape
    n_blocks = math.ceil(N / block_size)
    pad = n_blocks * block_size - N
    if pad > 0:
        x = F.pad(x, (0, 0, 0, pad))  # pad sequence dim
    x = x.view(B, H, n_blocks, block_size, D)
    if pad > 0:
        # Mask padded positions so they don't contribute to mean
        counts = torch.full((n_blocks,), block_size, device=x.device, dtype=x.dtype)
        counts[-1] = block_size - pad
        x = x.sum(dim=3) / counts.view(1, 1, -1, 1)
    else:
        x = x.mean(dim=3)
    return x

# ---------------------------------------------------------------------------
# Utility: Pool position bias to block level
# ---------------------------------------------------------------------------

def pool_position_bias(
    bias: torch.Tensor,
    bq: int,
    bk: int,
) -> torch.Tensor:
    """Mean-pool a (1, H, Lq, Lk) position bias tensor to block level.

    Returns: (1, H, ceil(Lq/bq), ceil(Lk/bk))
    """
    _, H, Lq, Lk = bias.shape
    nq = math.ceil(Lq / bq)
    nk = math.ceil(Lk / bk)

    padq = nq * bq - Lq
    padk = nk * bk - Lk
    b = bias
    if padq > 0 or padk > 0:
        b = F.pad(b, (0, padk, 0, padq))

    b = b.view(1, H, nq, bq, nk * bk)
    b = b.mean(dim=3)                       # pool query blocks
    b = b.view(1, H, nq, nk, bk)
    b = b.mean(dim=4)                       # pool key blocks
    return b

# ---------------------------------------------------------------------------
# Utility: Feature maps for linear attention
# ---------------------------------------------------------------------------

def _phi_elu(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0

def _phi_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x)

def _phi_softmax(x: torch.Tensor) -> torch.Tensor:
    return F.softmax(x, dim=-1)

_FEATURE_MAPS = {
    "elu": _phi_elu,
    "relu": _phi_relu,
    "softmax": _phi_softmax,
}

# ---------------------------------------------------------------------------
# Utility: FakeQuantize (STE) for optional QAT
# ---------------------------------------------------------------------------

class FakeQuantize(torch.autograd.Function):
    """Fake quantization with straight-through estimator."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int = 8) -> torch.Tensor:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
        x_abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = x_abs_max / qmax
        x_q = (x / scale).round().clamp(qmin, qmax) * scale
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradients straight through
        return grad_output, None


def fake_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Apply fake quantization if training, identity otherwise."""
    if x.requires_grad:
        return FakeQuantize.apply(x, bits)
    # Inference path
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    x_abs_max = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = x_abs_max / qmax
    return (x / scale).round().clamp(qmin, qmax) * scale

# ---------------------------------------------------------------------------
# Learnable Router
# ---------------------------------------------------------------------------

class LearnableRouter(nn.Module):
    """Block-level learnable routing for SLA2.

    Pools Q and K to block level, applies learned projections, computes
    block-level affinity scores, and selects top-k% key blocks per query block.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_head = d_model // num_heads
        self.proj_q = nn.Linear(self.d_head, self.d_head, bias=False)
        self.proj_k = nn.Linear(self.d_head, self.d_head, bias=False)
        # Learnable scale for incorporating position bias into routing
        self.bias_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        block_size: int,
        pooled_bias: Optional[torch.Tensor] = None,
        k_frac: float = 0.15,
        tau: float = 0.1,
        max_iter: int = 20,
        hard: bool = False,
    ) -> torch.Tensor:
        """Compute routing mask.

        Args:
            q: (B, H, Lq, d_head)
            k: (B, H, Lk, d_head)
            block_size: Block size for pooling.
            pooled_bias: (1, H, nq, nk) pooled position bias (optional).
            k_frac: Fraction of key blocks to keep.
            tau: SoftTop-k temperature.
            max_iter: Binary search iterations.
            hard: Use hard top-k (inference).

        Returns:
            mask: (B, H, nq, nk) with values in {0,1} (hard) or (0,1) (soft).
        """
        q_pooled = mean_pool_blocks(q, block_size)  # (B, H, nq, d)
        k_pooled = mean_pool_blocks(k, block_size)  # (B, H, nk, d)

        q_proj = self.proj_q(q_pooled)               # (B, H, nq, d)
        k_proj = self.proj_k(k_pooled)               # (B, H, nk, d)

        scores = torch.matmul(q_proj, k_proj.transpose(-1, -2))  # (B, H, nq, nk)
        scores = scores / math.sqrt(self.d_head)

        if pooled_bias is not None:
            scores = scores + self.bias_scale * pooled_bias

        if hard:
            return hard_topk(scores, k_frac)
        else:
            return soft_topk(scores, k_frac, tau=tau, max_iter=max_iter)

# ---------------------------------------------------------------------------
# Combination Ratio (alpha)
# ---------------------------------------------------------------------------

class CombinationRatio(nn.Module):
    """Learnable per-head alpha that blends sparse and linear branches.

    O = alpha * O_sparse + (1 - alpha) * O_linear
    """

    def __init__(self, num_heads: int):
        super().__init__()
        # One logit per head, initialized so alpha ≈ 0.7 (favor sparse initially)
        self.logits = nn.Parameter(torch.full((1, num_heads, 1, 1), 0.85))

    def forward(self) -> torch.Tensor:
        """Returns alpha in (0, 1) with shape (1, H, 1, 1)."""
        return torch.sigmoid(self.logits)

# ---------------------------------------------------------------------------
# Expand block mask to full sequence-level mask
# ---------------------------------------------------------------------------

def _expand_block_mask(
    block_mask: torch.Tensor,
    bq: int,
    bk: int,
    Lq: int,
    Lk: int,
) -> torch.Tensor:
    """Expand (B, H, nq, nk) block mask to (B, H, Lq, Lk).

    Each block entry is replicated to its constituent positions.
    """
    B, H, nq, nk = block_mask.shape
    # Expand query dim
    mask = block_mask.unsqueeze(3).expand(B, H, nq, bq, nk)  # (B,H,nq,bq,nk)
    mask = mask.reshape(B, H, nq * bq, nk)
    # Expand key dim
    mask = mask.unsqueeze(4).expand(B, H, nq * bq, nk, bk)
    mask = mask.reshape(B, H, nq * bq, nk * bk)
    # Trim to actual lengths
    return mask[:, :, :Lq, :Lk]

# ---------------------------------------------------------------------------
# Gather-based sparse attention
# ---------------------------------------------------------------------------

def _gather_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
    position_bias: Optional[torch.Tensor],
    attn_mask: Optional[torch.Tensor],
    scale: float,
    use_qat: bool,
    qat_bits: int,
    dropout_p: float,
    training: bool,
) -> torch.Tensor:
    """Gather-based block-sparse attention for actual memory savings.

    Instead of materializing the full N^2 score matrix, gathers only the
    selected K/V blocks per query block and computes attention on the subset.
    """
    B, H, Lq, D = q.shape
    Lk = k.shape[2]
    bq = bk = block_size
    nq = math.ceil(Lq / bq)
    nk = math.ceil(Lk / bk)

    # Pad Q, K, V to multiples of block_size
    padq = nq * bq - Lq
    padk = nk * bk - Lk
    if padq > 0:
        q = F.pad(q, (0, 0, 0, padq))
    if padk > 0:
        k = F.pad(k, (0, 0, 0, padk))
        v = F.pad(v, (0, 0, 0, padk))

    # Reshape into blocks
    q_blocks = q.view(B, H, nq, bq, D)     # (B, H, nq, bq, D)
    k_blocks = k.view(B, H, nk, bk, D)     # (B, H, nk, bk, D)
    v_blocks = v.view(B, H, nk, bk, D)     # (B, H, nk, bk, D)

    # Get indices of selected key blocks per query block
    # block_mask: (B, H, nq, nk) -> get top-k indices
    k_per_q = max(1, round(block_mask.sum(dim=-1).float().max().item()))
    _, sel_idx = block_mask.topk(k_per_q, dim=-1)  # (B, H, nq, k_per_q)

    # Gather selected key/value blocks: expand k_blocks along query-block dim
    # k_blocks: (B, H, nk, bk, D) -> (B, H, 1, nk, bk, D) -> (B, H, nq, nk, bk, D)
    idx_k = sel_idx.unsqueeze(-1).unsqueeze(-1).expand(B, H, nq, k_per_q, bk, D)
    k_exp = k_blocks.unsqueeze(2).expand(B, H, nq, nk, bk, D)
    k_sel = torch.gather(k_exp, 3, idx_k)  # (B, H, nq, k_per_q, bk, D)
    v_exp = v_blocks.unsqueeze(2).expand(B, H, nq, nk, bk, D)
    v_sel = torch.gather(v_exp, 3, idx_k)  # (B, H, nq, k_per_q, bk, D)

    # Flatten selected blocks: (B, H, nq, k_per_q * bk, D)
    k_flat = k_sel.reshape(B, H, nq, k_per_q * bk, D)
    v_flat = v_sel.reshape(B, H, nq, k_per_q * bk, D)

    # QAT
    if use_qat:
        q_blocks = fake_quantize(q_blocks, qat_bits)
        k_flat = fake_quantize(k_flat, qat_bits)

    # Compute scores: (B, H, nq, bq, k_per_q * bk)
    scores = torch.matmul(q_blocks, k_flat.transpose(-1, -2))
    if scale != 1.0:
        scores = scores * scale

    # Gather and add position bias for selected blocks
    if position_bias is not None:
        pb = position_bias
        if padq > 0 or padk > 0:
            pb = F.pad(pb, (0, padk, 0, padq))
        # Reshape bias into blocks: (1, H, nq, bq, nk, bk)
        pb = pb.view(pb.shape[0], H, nq, bq, nk, bk)
        # Expand batch dim to match sel_idx if needed
        if pb.shape[0] < B:
            pb = pb.expand(B, -1, -1, -1, -1, -1)
        # Gather along nk dim (dim=4): need index (B, H, nq, bq, k_per_q, bk)
        idx_pb = sel_idx.unsqueeze(3).unsqueeze(5).expand(B, H, nq, bq, k_per_q, bk)
        pb_sel = torch.gather(pb, 4, idx_pb)  # (B, H, nq, bq, k_per_q, bk)
        pb_sel = pb_sel.reshape(B, H, nq, bq, k_per_q * bk)
        scores = scores + pb_sel

    # Gather and add causal/padding mask
    if attn_mask is not None:
        am = attn_mask
        if padq > 0 or padk > 0:
            am = F.pad(am, (0, padk, 0, padq), value=torch.finfo(am.dtype).min)
        Bam = am.shape[0]
        am = am.view(Bam, am.shape[1], nq, bq, nk, bk)
        if Bam < B:
            am = am.expand(B, -1, -1, -1, -1, -1)
        idx_am = sel_idx.unsqueeze(3).unsqueeze(5).expand(B, H, nq, bq, k_per_q, bk)
        am_sel = torch.gather(am, 4, idx_am)
        am_sel = am_sel.reshape(B, H, nq, bq, k_per_q * bk)
        scores = scores + am_sel

    # Softmax + dropout
    weights = F.softmax(scores, dim=-1)
    if use_qat:
        weights = fake_quantize(weights, qat_bits)
        v_flat = fake_quantize(v_flat, qat_bits)
    if training and dropout_p > 0:
        weights = F.dropout(weights, p=dropout_p)

    # Weighted sum: (B, H, nq, bq, D)
    out = torch.matmul(weights, v_flat)
    # Reshape back: (B, H, nq*bq, D) -> trim
    out = out.reshape(B, H, nq * bq, D)[:, :, :Lq, :]
    return out

# ---------------------------------------------------------------------------
# Main Module: T5SLA2Attention
# ---------------------------------------------------------------------------

class T5SLA2Attention(nn.Module):
    """SLA2 attention as a drop-in replacement for HuggingFace T5Attention.

    Combines block-sparse attention (via learned router) with linear attention
    (via feature-mapped kernel trick), blended by a learnable alpha ratio.

    Falls back to standard attention for short sequences or during generation
    with KV cache.
    """

    def __init__(
        self,
        config,
        has_relative_attention_bias: bool = False,
        sla2_config: Optional[SLA2Config] = None,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.d_kv
        self.dropout = config.dropout_rate

        # SLA2 config
        self.sla2 = sla2_config or SLA2Config()
        self.phi = _FEATURE_MAPS[self.sla2.feature_map]

        # Standard T5 projections (no bias, as in T5)
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        # Relative position bias (only first layer per block)
        if self.has_relative_attention_bias:
            self.relative_attention_num_buckets = config.relative_attention_num_buckets
            self.relative_attention_max_distance = config.relative_attention_max_distance
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

        # SLA2 components
        self.router = LearnableRouter(self.d_model, self.n_heads)
        self.alpha = CombinationRatio(self.n_heads)

        # Attention scale
        if self.sla2.scale_attn:
            self.scale = 1.0 / math.sqrt(self.d_kv)
        else:
            self.scale = 1.0

        self.gradient_checkpointing = False

    # ------ T5 relative position bias (copied from T5Attention) ------

    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128,
    ) -> torch.Tensor:
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )
        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length: int, key_length: int, device: torch.device) -> torch.Tensor:
        """Compute T5-style relative position bias."""
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position

        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=not self.is_decoder,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # (1, H, Lq, Lk)
        return values

    # ------ Reshape helpers ------

    def _shape(self, x: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        """(B, L, inner_dim) -> (B, H, L, d_kv)"""
        return x.view(batch_size, seq_len, self.n_heads, self.d_kv).transpose(1, 2)

    # ------ Standard (fallback) attention ------

    def _standard_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Standard scaled dot-product attention (fallback path)."""
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        scores = scores * self.scale

        if position_bias is not None:
            scores = scores + position_bias
        if mask is not None:
            scores = scores + mask

        weights = F.softmax(scores.float(), dim=-1).to(query_states.dtype)
        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout)
        return torch.matmul(weights, value_states), weights

    # ------ Sparse branch ------

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor,
        position_bias: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Sparse attention restricted to router-selected blocks."""
        if self.sla2.sparse_method == "gather":
            return _gather_sparse_attention(
                q, k, v, block_mask,
                block_size=self.sla2.block_size,
                position_bias=position_bias,
                attn_mask=attn_mask,
                scale=self.scale,
                use_qat=self.sla2.use_qat,
                qat_bits=self.sla2.qat_bits,
                dropout_p=self.dropout,
                training=self.training,
            )

        # ----- Masked-dense path (default) -----
        B, H, Lq, D = q.shape
        Lk = k.shape[2]
        bs = self.sla2.block_size

        if self.sla2.use_qat:
            q = fake_quantize(q, self.sla2.qat_bits)
            k = fake_quantize(k, self.sla2.qat_bits)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, Lq, Lk)

        if position_bias is not None:
            scores = scores + position_bias
        if attn_mask is not None:
            scores = scores + attn_mask

        # Expand block mask and apply
        full_mask = _expand_block_mask(block_mask, bs, bs, Lq, Lk)
        # Where mask ≈ 0 (not selected), set to -inf
        neg_inf = torch.finfo(scores.dtype).min
        scores = scores + (1.0 - full_mask) * neg_inf

        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)

        if self.sla2.use_qat:
            weights = fake_quantize(weights, self.sla2.qat_bits)
            v = fake_quantize(v, self.sla2.qat_bits)

        if self.training and self.dropout > 0:
            weights = F.dropout(weights, p=self.dropout)

        return torch.matmul(weights, v)

    # ------ Linear branch ------

    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Linear attention via kernel trick: phi(Q) @ (phi(K)^T @ V).

        O(N * D^2) complexity. No position bias (incompatible with kernel trick).
        """
        phi_q = self.phi(q)  # (B, H, Lq, D)
        phi_k = self.phi(k)  # (B, H, Lk, D)

        # Associative trick: compute KV summary first
        kv_sum = torch.matmul(phi_k.transpose(-1, -2), v)  # (B, H, D, D)
        k_sum = phi_k.sum(dim=-2, keepdim=True)             # (B, H, 1, D)

        numerator = torch.matmul(phi_q, kv_sum)              # (B, H, Lq, D)
        denominator = (phi_q * k_sum).sum(dim=-1, keepdim=True)  # (B, H, Lq, 1)
        denominator = denominator.clamp(min=1e-6)

        return numerator / denominator

    # ------ Forward ------

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        query_length: Optional[int] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        cache_position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass matching T5Attention.forward() signature.

        Returns:
            (attn_output, position_bias)
            or (attn_output, attn_weights, position_bias) if output_attentions
        """
        batch_size, seq_length = hidden_states.shape[:2]
        is_cross = key_value_states is not None
        kv_input = key_value_states if is_cross else hidden_states

        # Project Q, K, V
        query_states = self._shape(self.q(hidden_states), seq_length, batch_size)

        if past_key_value is not None and not is_cross:
            # Self-attention with cache: only project new Q, reuse K/V
            key_states = self._shape(self.k(kv_input), kv_input.shape[1], batch_size)
            value_states = self._shape(self.v(kv_input), kv_input.shape[1], batch_size)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        elif past_key_value is not None and is_cross:
            # Cross-attention with cache: reuse K/V from encoder
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        else:
            key_states = self._shape(self.k(kv_input), kv_input.shape[1], batch_size)
            value_states = self._shape(self.v(kv_input), kv_input.shape[1], batch_size)

        present_key_value = (key_states, value_states) if use_cache else None

        Lq = query_states.shape[2]
        Lk = key_states.shape[2]

        # Compute position bias (self-attention with relative bias, first layer)
        if position_bias is None:
            if not is_cross and self.has_relative_attention_bias:
                position_bias = self.compute_bias(Lq, Lk, device=hidden_states.device)
            else:
                position_bias = torch.zeros(
                    (1, self.n_heads, Lq, Lk),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )
            if mask is not None:
                position_bias = position_bias + mask

        # ------ Decide: SLA2 or fallback ------
        use_sla2 = (
            Lq >= self.sla2.min_seq_len
            and Lq > 1
            and past_key_value is None
        )

        if use_sla2:
            attn_output, attn_weights = self._sla2_forward(
                query_states, key_states, value_states, position_bias
            )
        else:
            attn_output, attn_weights = self._standard_attention(
                query_states, key_states, value_states, position_bias, mask=None
            )

        # Apply head mask if provided
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        # Reshape (B, H, L, d_kv) -> (B, L, inner_dim) and project out
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, Lq, self.inner_dim
        )
        attn_output = self.o(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs = outputs + (attn_weights,)
        outputs = outputs + (position_bias,)
        if use_cache:
            outputs = outputs + (present_key_value,)
        return outputs

    def _sla2_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_bias: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """SLA2 combined sparse + linear attention path.

        Returns (output, None) — attn_weights not meaningful for combined output.
        """
        bs = self.sla2.block_size

        # Pool position bias for router (detach to avoid double-backward through bias)
        pooled_bias = pool_position_bias(position_bias.detach(), bs, bs)

        # Router: get block-level selection mask
        block_mask = self.router(
            q, k,
            block_size=bs,
            pooled_bias=pooled_bias,
            k_frac=self.sla2.top_k_frac,
            tau=self.sla2.tau,
            max_iter=self.sla2.max_iter,
            hard=self.sla2.router_hard or not self.training,
        )

        # Sparse branch
        o_sparse = self._sparse_attention(
            q, k, v,
            block_mask=block_mask,
            position_bias=position_bias,
            attn_mask=None,  # mask already folded into position_bias
        )

        # Linear branch (no position bias — incompatible with kernel trick)
        o_linear = self._linear_attention(q, k, v)

        # Combine with learnable alpha
        a = self.alpha()  # (1, H, 1, 1)
        output = a * o_sparse + (1.0 - a) * o_linear

        return output, None

# ---------------------------------------------------------------------------
# Helper: patch a T5 model to use SLA2 attention
# ---------------------------------------------------------------------------

def patch_t5_with_sla2(
    model: nn.Module,
    sla2_config: Optional[SLA2Config] = None,
    **sla2_kwargs,
) -> nn.Module:
    """Replace all T5Attention modules in a HuggingFace T5 model with T5SLA2Attention.

    Args:
        model: A HuggingFace T5 model (T5ForConditionalGeneration, T5Model, etc.)
        sla2_config: SLA2Config instance (overrides sla2_kwargs).
        **sla2_kwargs: Passed to SLA2Config if sla2_config is None.

    Returns:
        The same model with attention modules replaced.
    """
    if sla2_config is None:
        sla2_config = SLA2Config(**sla2_kwargs)

    config = model.config

    def _replace_attention(parent: nn.Module, name: str, module: nn.Module):
        has_rel_bias = getattr(module, "has_relative_attention_bias", False)
        new_attn = T5SLA2Attention(
            config,
            has_relative_attention_bias=has_rel_bias,
            sla2_config=copy.deepcopy(sla2_config),
        )

        # Copy weights from original projections
        with torch.no_grad():
            new_attn.q.weight.copy_(module.q.weight)
            new_attn.k.weight.copy_(module.k.weight)
            new_attn.v.weight.copy_(module.v.weight)
            new_attn.o.weight.copy_(module.o.weight)
            if has_rel_bias:
                new_attn.relative_attention_bias.weight.copy_(
                    module.relative_attention_bias.weight
                )

        setattr(parent, name, new_attn)

    # Walk all modules looking for T5Attention instances
    # Use class name check to avoid hard import dependency
    replacements = []
    for parent_name, parent in model.named_modules():
        for name, child in parent.named_children():
            cls_name = type(child).__name__
            if cls_name in ("T5Attention", "T5LayerSelfAttention") and hasattr(child, "q"):
                replacements.append((parent, name, child))

    for parent, name, child in replacements:
        _replace_attention(parent, name, child)

    n = len(replacements)
    if n == 0:
        warnings.warn(
            "patch_t5_with_sla2: No T5Attention modules found. "
            "Make sure you are passing a T5 model."
        )
    else:
        print(f"[SLA2] Patched {n} T5Attention modules.")

    return model

# ---------------------------------------------------------------------------
# Helper: Stage 1 router training
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_full_attention_output(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_bias: Optional[torch.Tensor],
    scale: float,
) -> torch.Tensor:
    """Compute standard full attention output (no routing) as training target."""
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if position_bias is not None:
        scores = scores + position_bias
    weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.matmul(weights, v)


def train_router_stage1(
    model: nn.Module,
    dataloader,
    num_steps: int = 500,
    lr: float = 1e-3,
    device: str = "cuda",
) -> list:
    """Stage 1 training: train router + alpha parameters via MSE distillation.

    Freezes all parameters except router and alpha. For each forward pass,
    computes full attention output as target, then minimizes MSE between
    SLA2 output and full attention output.

    Args:
        model: A T5 model already patched with SLA2 attention.
        dataloader: Yields batches of input_ids (and optionally attention_mask).
        num_steps: Number of training steps.
        lr: Learning rate for router/alpha parameters.
        device: Device string.

    Returns:
        List of loss values per step.
    """
    # Collect SLA2 modules
    sla2_modules = []
    for m in model.modules():
        if isinstance(m, T5SLA2Attention):
            sla2_modules.append(m)

    if not sla2_modules:
        raise ValueError("No T5SLA2Attention modules found in model.")

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze router + alpha
    router_alpha_params = []
    for m in sla2_modules:
        for p in m.router.parameters():
            p.requires_grad_(True)
            router_alpha_params.append(p)
        for p in m.alpha.parameters():
            p.requires_grad_(True)
            router_alpha_params.append(p)
        # Ensure soft routing during Stage 1
        m.sla2.router_hard = False

    optimizer = torch.optim.Adam(router_alpha_params, lr=lr)
    model.to(device)
    model.train()

    losses = []
    data_iter = iter(dataloader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if isinstance(batch, dict):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            attention_mask = None

        # Hook into each SLA2 module to capture intermediate Q/K/V and outputs
        # For simplicity, we run a forward pass and collect losses from encoder
        optimizer.zero_grad()

        # Forward pass through model — SLA2 modules compute their combined output
        # We attach hooks to compute MSE loss against full attention
        layer_losses = []

        def make_hook(sla2_mod):
            original_sla2_forward = sla2_mod._sla2_forward

            def hooked_sla2_forward(q, k, v, position_bias):
                # Full attention target (no grad through target)
                with torch.no_grad():
                    target = _compute_full_attention_output(
                        q, k, v, position_bias, sla2_mod.scale
                    )
                # SLA2 output
                output, w = original_sla2_forward(q, k, v, position_bias)
                # MSE loss for this layer
                loss = F.mse_loss(output, target)
                layer_losses.append(loss)
                return output, w

            return hooked_sla2_forward

        # Install hooks
        original_fns = {}
        for m in sla2_modules:
            original_fns[id(m)] = m._sla2_forward
            m._sla2_forward = make_hook(m)

        try:
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            if layer_losses:
                total_loss = torch.stack(layer_losses).mean()
                total_loss.backward()
                optimizer.step()
                losses.append(total_loss.item())
        finally:
            # Restore original methods
            for m in sla2_modules:
                m._sla2_forward = original_fns[id(m)]

        if (step + 1) % 50 == 0:
            print(f"[Stage1] Step {step+1}/{num_steps}, loss={losses[-1]:.6f}")

    # Switch to hard routing for Stage 2 / inference
    for m in sla2_modules:
        m.sla2.router_hard = True

    print(f"[Stage1] Done. Final loss: {losses[-1]:.6f}. Router set to hard mode.")
    return losses

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _make_mock_config():
    """Create a minimal config object mimicking T5Config."""

    class MockConfig:
        d_model = 512
        d_kv = 64
        num_heads = 8
        dropout_rate = 0.0
        is_decoder = False
        relative_attention_num_buckets = 32
        relative_attention_max_distance = 128

    return MockConfig()


def _smoke_test():
    print("=" * 60)
    print("T5-SLA2 Smoke Test")
    print("=" * 60)

    device = "cpu"
    config = _make_mock_config()
    B, L, D = 2, 512, config.d_model

    # ---------- Test 1: Self-attention with SLA2 path ----------
    print("\n[1] Self-attention (SLA2 path, L=512)...")
    attn = T5SLA2Attention(config, has_relative_attention_bias=True).to(device)
    x = torch.randn(B, L, D, device=device)
    out = attn(x)
    attn_out, pos_bias = out[0], out[-1]
    assert attn_out.shape == (B, L, D), f"Expected {(B, L, D)}, got {attn_out.shape}"
    assert pos_bias.shape == (1, config.num_heads, L, L), f"Bias shape: {pos_bias.shape}"
    print(f"   Output: {attn_out.shape}, bias: {pos_bias.shape} ✓")

    # ---------- Test 2: Cross-attention ----------
    print("\n[2] Cross-attention (SLA2 path)...")
    config_dec = _make_mock_config()
    config_dec.is_decoder = True
    attn_cross = T5SLA2Attention(config_dec, has_relative_attention_bias=False).to(device)
    dec_hidden = torch.randn(B, L, D, device=device)
    enc_hidden = torch.randn(B, L, D, device=device)
    out = attn_cross(dec_hidden, key_value_states=enc_hidden)
    assert out[0].shape == (B, L, D)
    print(f"   Output: {out[0].shape} ✓")

    # ---------- Test 3: Short sequence fallback ----------
    print("\n[3] Short sequence fallback (L=64 < min_seq=256)...")
    short_x = torch.randn(B, 64, D, device=device)
    out = attn(short_x)
    assert out[0].shape == (B, 64, D)
    print(f"   Output: {out[0].shape} ✓")

    # ---------- Test 4: KV cache fallback ----------
    print("\n[4] KV cache fallback (L_q=1)...")
    config_dec2 = _make_mock_config()
    config_dec2.is_decoder = True
    attn_dec = T5SLA2Attention(config_dec2, has_relative_attention_bias=True).to(device)
    q_one = torch.randn(B, 1, D, device=device)
    past_k = torch.randn(B, config.num_heads, 10, config.d_kv, device=device)
    past_v = torch.randn(B, config.num_heads, 10, config.d_kv, device=device)
    pos_bias_cached = torch.zeros(1, config.num_heads, 1, 11, device=device)
    out = attn_dec(
        q_one,
        position_bias=pos_bias_cached,
        past_key_value=(past_k, past_v),
        use_cache=True,
    )
    assert out[0].shape == (B, 1, D)
    print(f"   Output: {out[0].shape} ✓")

    # ---------- Test 5: Gradient flow ----------
    print("\n[5] Gradient flow check...")
    attn_grad = T5SLA2Attention(config, has_relative_attention_bias=True).to(device)
    x_grad = torch.randn(B, L, D, device=device, requires_grad=True)
    out = attn_grad(x_grad)
    loss = out[0].sum()
    loss.backward()
    # Check router gradients exist
    has_router_grad = attn_grad.router.proj_q.weight.grad is not None
    has_alpha_grad = attn_grad.alpha.logits.grad is not None
    has_q_grad = attn_grad.q.weight.grad is not None
    has_input_grad = x_grad.grad is not None
    assert has_router_grad, "Router has no gradient!"
    assert has_alpha_grad, "Alpha has no gradient!"
    assert has_q_grad, "Q projection has no gradient!"
    assert has_input_grad, "Input has no gradient!"
    print(f"   router.proj_q: {has_router_grad}, alpha: {has_alpha_grad}, "
          f"q: {has_q_grad}, input: {has_input_grad} ✓")

    # ---------- Test 6: Gather-based sparse method ----------
    print("\n[6] Gather-based sparse attention...")
    sla2_gather = SLA2Config(sparse_method="gather")
    attn_gather = T5SLA2Attention(config, has_relative_attention_bias=True, sla2_config=sla2_gather).to(device)
    out = attn_gather(x)
    assert out[0].shape == (B, L, D)
    print(f"   Output: {out[0].shape} ✓")

    # ---------- Test 7: QAT mode ----------
    print("\n[7] QAT mode...")
    sla2_qat = SLA2Config(use_qat=True, qat_bits=8)
    attn_qat = T5SLA2Attention(config, has_relative_attention_bias=True, sla2_config=sla2_qat).to(device)
    out = attn_qat(x)
    assert out[0].shape == (B, L, D)
    print(f"   Output: {out[0].shape} ✓")

    # ---------- Test 8: output_attentions flag ----------
    print("\n[8] output_attentions=True...")
    out = attn(x, output_attentions=True)
    assert len(out) == 3, f"Expected 3 outputs, got {len(out)}"
    print(f"   Outputs: attn_out={out[0].shape}, weights={out[1]}, bias={out[2].shape} ✓")

    # ---------- Test 9: Mock Stage 1 training (2 steps) ----------
    print("\n[9] Mock Stage 1 training (2 steps)...")
    attn_train = T5SLA2Attention(config, has_relative_attention_bias=True).to(device)
    attn_train.train()

    # Simulate 2 training steps directly on the module
    router_params = list(attn_train.router.parameters()) + list(attn_train.alpha.parameters())
    opt = torch.optim.Adam(router_params, lr=1e-3)
    for step in range(2):
        opt.zero_grad()
        x_t = torch.randn(B, L, D, device=device)
        q_t = attn_train._shape(attn_train.q(x_t), L, B)
        k_t = attn_train._shape(attn_train.k(x_t), L, B)
        v_t = attn_train._shape(attn_train.v(x_t), L, B)
        pb = attn_train.compute_bias(L, L, device=device)

        # Full attention target
        with torch.no_grad():
            target = _compute_full_attention_output(q_t, k_t, v_t, pb, attn_train.scale)

        # SLA2 output
        sla2_out, _ = attn_train._sla2_forward(q_t, k_t, v_t, pb)
        loss = F.mse_loss(sla2_out, target)
        loss.backward()
        opt.step()
        print(f"   Step {step+1}: loss={loss.item():.6f}")
    print("   Stage 1 mock training ✓")

    # ---------- Test 10: Non-square attention (different Lq, Lk) ----------
    print("\n[10] Non-square cross-attention (Lq=512, Lk=256)...")
    attn_ns = T5SLA2Attention(config_dec, has_relative_attention_bias=False).to(device)
    dec_h = torch.randn(B, 512, D, device=device)
    enc_h = torch.randn(B, 256, D, device=device)
    out = attn_ns(dec_h, key_value_states=enc_h)
    assert out[0].shape == (B, 512, D)
    print(f"   Output: {out[0].shape} ✓")

    print("\n" + "=" * 60)
    print("All smoke tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    _smoke_test()
