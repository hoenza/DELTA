import torch
import triton
import triton.language as tl


@triton.jit
def _pack_page_indices_kernel(
    slot_block_table,
    active_slots,
    start_cols,
    counts,
    indptr,
    out_indices,
    row_block_table,
    pages_per_slot: tl.constexpr,
    write_row_block: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)

    count = tl.load(counts + row)
    slot = tl.load(active_slots + row).to(tl.int64)
    start_col = tl.load(start_cols + row).to(tl.int64)
    src = slot_block_table + slot * pages_per_slot + start_col + offsets

    mask = offsets < count
    vals = tl.load(src, mask=mask, other=0)

    dst_start = tl.load(indptr + row).to(tl.int64)
    tl.store(out_indices + dst_start + offsets, vals, mask=mask)

    if write_row_block:
        row_dst = row_block_table + row * pages_per_slot + offsets
        tl.store(row_dst, vals, mask=mask)


def _next_power_of_2(x: int) -> int:
    return 1 << (max(int(x), 1) - 1).bit_length()


def pack_page_indices(
    slot_block_table: torch.Tensor,
    active_slots: torch.Tensor,
    start_cols: torch.Tensor,
    counts: torch.Tensor,
    indptr: torch.Tensor,
    out_indices: torch.Tensor,
    row_block_table: torch.Tensor,
    n_rows: int,
    max_count: int,
    *,
    write_row_block: bool,
) -> None:
    if n_rows <= 0 or max_count <= 0:
        return

    block = _next_power_of_2(max_count)
    _pack_page_indices_kernel[(n_rows,)](
        slot_block_table,
        active_slots,
        start_cols,
        counts,
        indptr,
        out_indices,
        row_block_table,
        slot_block_table.shape[1],
        write_row_block,
        BLOCK=block,
    )
