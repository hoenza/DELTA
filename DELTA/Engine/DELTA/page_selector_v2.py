import torch


class PageSelectorV2:
    """Single-pass topk page selector (UniKV-style).

    Compared to PageSelector (v1) which uses a two-stage prefix-topk + last-L
    mask composition, this version:
      - Sets last-L page scores to +inf so they always survive topk
      - Sets invalid page scores to -inf
      - Runs a single ``scores.topk(k)``
      - Sorts the selected logical page positions
      - Gathers physical IDs via a 2D block table in logical order
      - Adds a small position bias (-1e-7 * page_idx) to break ties
    """

    def __init__(
        self,
        max_batch_size,
        pages_per_slot,
        enable_selective_cache,
        subset_cache_size,
        compression_ratio,
        L,
        device,
        position_bias_scale: float = 1e-7,
    ):
        self.max_batch_size = max_batch_size
        self.pages_per_slot = pages_per_slot
        self.enable_selective_cache = enable_selective_cache
        self.subset_cache_size = subset_cache_size
        self.compression_ratio = compression_ratio
        self.max_num_pages = pages_per_slot * max_batch_size
        self.max_num_subset_pages = subset_cache_size * max_batch_size
        self.L = L
        self.device = device
        self.position_bias_scale = float(position_bias_scale)

        self.qo_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        self.paged_kv_indices = torch.empty(self.max_num_pages, dtype=torch.int32, device=device)
        self.paged_kv_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
        self.paged_kv_last_page_len = torch.empty(max_batch_size, dtype=torch.int32, device=device)
        self.paged_subset_kv_indices = torch.empty(self.max_num_pages, dtype=torch.int32, device=device)
        self.paged_subset_kv_indptr = torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)

        if self.enable_selective_cache:
            self._block_table = torch.zeros(
                (max_batch_size, pages_per_slot), dtype=torch.int32, device=device)
            self._scores = torch.empty(
                (max_batch_size, pages_per_slot), dtype=torch.float32, device=device)
            self._pos_bias = None
            if self.position_bias_scale != 0.0:
                self._pos_bias = -torch.arange(
                    pages_per_slot, device=device, dtype=torch.float32
                ) * self.position_bias_scale
            self._j = torch.arange(pages_per_slot, dtype=torch.int64, device=device)
            self._col_ids = torch.arange(subset_cache_size, dtype=torch.int64, device=device)
            self._topk_vals = torch.empty(
                (max_batch_size, subset_cache_size), dtype=torch.float32, device=device)
            self._topk_idx = torch.empty(
                (max_batch_size, subset_cache_size), dtype=torch.int64, device=device)
            self._selected_pages_2d = torch.empty(
                (max_batch_size, subset_cache_size), dtype=torch.int32, device=device)
            self._selected_pages_flat = torch.empty(
                self.max_num_subset_pages, dtype=torch.int32, device=device)
            self._fixed_selected_indptr = (
                torch.arange(max_batch_size + 1, dtype=torch.int32, device=device)
                * subset_cache_size
            )

            self.tp = self.paged_kv_indptr[1:] - self.paged_kv_indptr[:-1]
            self.base_offsets = self.paged_kv_indptr[:-1]

    def plan(self, qo_indptr, paged_kv_indices, paged_kv_indptr,
             paged_kv_last_page_len, paged_subset_kv_indices,
             paged_subset_kv_indptr, L=None, block_table_prebuilt=False):
        self.qo_indptr = qo_indptr
        self.paged_kv_indices = paged_kv_indices
        self.paged_kv_indptr = paged_kv_indptr
        self.paged_kv_last_page_len = paged_kv_last_page_len
        self.paged_subset_kv_indices = paged_subset_kv_indices
        self.paged_subset_kv_indptr = paged_subset_kv_indptr

        if not self.enable_selective_cache:
            return
        if paged_kv_indices is paged_subset_kv_indices:
            return

        if L is not None:
            self.L = int(L)

        device = paged_kv_indptr.device
        B = paged_kv_indptr.numel() - 1
        self._active_B = B

        self.tp = paged_kv_indptr[1:] - paged_kv_indptr[:-1]          # [B]
        self.base_offsets = paged_kv_indptr[:-1]                       # [B]

        if block_table_prebuilt:
            return

        # Build 2D block table from CSR paged_kv_indices
        n_total = paged_kv_indices.numel()
        bt = self._block_table[:B]
        bt.zero_()
        if n_total > 0:
            row_ids = torch.repeat_interleave(
                torch.arange(B, device=device, dtype=torch.int64), self.tp.long())
            col_ids = torch.arange(n_total, device=device, dtype=torch.int64) - \
                      self.base_offsets[row_ids].long()
            bt[row_ids, col_ids] = paged_kv_indices

    def _prepare_scores(self, page_scores: torch.Tensor):
        B, Nmax = page_scores.shape
        scores = self._scores[:B, :Nmax]
        scores.copy_(torch.nan_to_num(page_scores, nan=float("-inf"), neginf=float("-inf")))
        if self._pos_bias is not None:
            scores.add_(self._pos_bias[:Nmax])

        j = self._j[:Nmax].unsqueeze(0)                              # [1, Nmax]
        tp = self.tp[:B].unsqueeze(1)                                 # [B, 1]

        # Force last-L pages to survive topk
        last_start = (self.tp[:B] - self.L).clamp(min=0).unsqueeze(1)
        last_mask = (j >= last_start) & (j < tp)
        scores[last_mask] = float('inf')

        # Invalidate pages beyond each row's actual count
        scores[j >= tp] = float('-inf')
        return scores, B, Nmax

    def _sort_selected_positions(self, selected_positions: torch.Tensor) -> torch.Tensor:
        # KV pages must stay in logical sequence order; physical page IDs are opaque.
        return selected_positions.sort(dim=1).values

    def _select_general_from_scores(self, scores: torch.Tensor, B: int, Nmax: int):
        device = scores.device
        k = self.subset_cache_size
        if k <= 0:
            selected_indptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
            return self.paged_subset_kv_indices[:0], selected_indptr

        # Single topk
        _, topk_idx = scores.topk(k, dim=1)                          # [B, k]
        topk_idx = self._sort_selected_positions(topk_idx)

        # Gather physical page IDs in logical page order.
        selected = torch.gather(
            self._block_table[:B, :Nmax], 1, topk_idx.to(torch.int64)
        ).to(torch.int32)                                             # [B, k]

        # Valid counts per row
        valid_counts = torch.clamp_max(self.tp[:B], k)               # [B]

        col = self._col_ids[:k].unsqueeze(0)                         # [1, k]
        vc32 = valid_counts.to(torch.int32)
        selected_indptr = torch.empty(B + 1, dtype=torch.int32, device=device)
        selected_indptr[0] = 0
        selected_indptr[1:] = torch.cumsum(vc32, dim=0)

        keep = col < valid_counts.unsqueeze(1)
        selected_indices = selected[keep].to(torch.int32)

        return selected_indices, selected_indptr

    def _select_fixed_count_from_scores(self, scores: torch.Tensor, B: int, Nmax: int):
        k = self.subset_cache_size
        if k <= 0:
            return self.paged_subset_kv_indices[:0], self._fixed_selected_indptr[:B + 1]

        torch.topk(
            scores,
            k,
            dim=1,
            largest=True,
            sorted=False,
            out=(self._topk_vals[:B, :k], self._topk_idx[:B, :k]),
        )
        sorted_topk_idx = self._sort_selected_positions(self._topk_idx[:B, :k])

        selected = torch.gather(
            self._block_table[:B, :Nmax],
            1,
            sorted_topk_idx,
        ).to(torch.int32)
        self._selected_pages_2d[:B, :k].copy_(selected)
        self._selected_pages_flat[:B * k].copy_(self._selected_pages_2d[:B, :k].reshape(-1))
        return self._selected_pages_flat[:B * k], self._fixed_selected_indptr[:B + 1]

    def select(self, page_scores: torch.Tensor, *, fixed_count: bool = False, debug_compare: bool = False):
        """Single-pass topk selection.

        Args:
            page_scores: [B, Nmax] float page scores (from attention buffer).
            fixed_count: when True, every row is expected to contribute exactly
                ``subset_cache_size`` pages and the selector returns static CSR.
            debug_compare: compare the fixed-count fast path against the general
                selector and raise on mismatch.

        Returns:
            selected_indices: int32 [nnz] physical page IDs in logical page order
            selected_indptr:  int32 [B+1] CSR row boundaries
        """
        scores, B, Nmax = self._prepare_scores(page_scores)
        if fixed_count:
            selected_indices, selected_indptr = self._select_fixed_count_from_scores(scores, B, Nmax)
            if debug_compare:
                ref_indices, ref_indptr = self._select_general_from_scores(scores, B, Nmax)
                if not torch.equal(selected_indptr, ref_indptr) or not torch.equal(selected_indices, ref_indices):
                    raise RuntimeError("PageSelectorV2 fixed-count fast path mismatch against general selector")
            return selected_indices, selected_indptr
        return self._select_general_from_scores(scores, B, Nmax)
