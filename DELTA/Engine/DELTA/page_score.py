import torch
import triton
import triton.language as tl


def _next_power_of_2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


@triton.jit
def _compute_page_scores_kernel(
    logits_ptr,
    lse_ptr,
    out_ptr,
    logits_stride_b,
    logits_stride_h,
    logits_stride_s,
    lse_stride_b,
    lse_stride_h,
    out_stride_b,
    out_stride_p,
    n_heads,
    seq_len,
    log2e,
    PAGE_SIZE: tl.constexpr,
    BLOCK_HEAD: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    page_idx = tl.program_id(1)

    offs_h = tl.arange(0, BLOCK_HEAD)
    offs_tok = tl.arange(0, PAGE_SIZE)

    tok_idx = page_idx * PAGE_SIZE + offs_tok
    head_mask = offs_h < n_heads
    tok_mask = tok_idx < seq_len
    load_mask = head_mask[:, None] & tok_mask[None, :]

    logits_ptrs = (
        logits_ptr
        + batch_idx * logits_stride_b
        + offs_h[:, None] * logits_stride_h
        + tok_idx[None, :] * logits_stride_s
    )
    logits = tl.load(logits_ptrs, mask=load_mask, other=-float("inf")).to(tl.float32)

    lse = tl.load(
        lse_ptr + batch_idx * lse_stride_b + offs_h * lse_stride_h,
        mask=head_mask,
        other=0.0,
    ).to(tl.float32)

    probs = tl.math.exp2((logits - lse[:, None]) * log2e)
    head_max = tl.max(probs, axis=0)
    page_score = tl.sum(head_max, axis=0)

    tl.store(out_ptr + batch_idx * out_stride_b + page_idx * out_stride_p, page_score)


def compute_page_scores_triton(
    logits: torch.Tensor,
    lse: torch.Tensor,
    out: torch.Tensor,
    *,
    log2e: float,
    page_size: int,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B, H, S], got shape {tuple(logits.shape)}")
    if lse.ndim != 2:
        raise ValueError(f"lse must be [B, H], got shape {tuple(lse.shape)}")
    if out.ndim != 2:
        raise ValueError(f"out must be [B, P], got shape {tuple(out.shape)}")

    batch_size, n_heads, seq_len = logits.shape
    if lse.shape != (batch_size, n_heads):
        raise ValueError(
            f"lse shape mismatch: expected {(batch_size, n_heads)}, got {tuple(lse.shape)}"
        )
    if seq_len != out.shape[1] * page_size:
        raise ValueError(
            f"logits seq len {seq_len} does not match out pages {out.shape[1]} * page size {page_size}"
        )
    if batch_size == 0 or out.shape[1] == 0:
        return out

    block_head = min(_next_power_of_2(n_heads), 64)
    grid = (batch_size, out.shape[1])
    _compute_page_scores_kernel[grid](
        logits,
        lse,
        out,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        lse.stride(0),
        lse.stride(1),
        out.stride(0),
        out.stride(1),
        n_heads,
        seq_len,
        log2e,
        PAGE_SIZE=page_size,
        BLOCK_HEAD=block_head,
    )
    return out
