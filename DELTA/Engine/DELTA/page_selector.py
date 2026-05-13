import torch


class PageSelector:
    
    def __init__(self, max_batch_size, pages_per_slot, enable_selective_cache, subset_cache_size, compression_ratio, L, device):
        self.max_batch_size = max_batch_size
        self.pages_per_slot = pages_per_slot
        self.enable_selective_cache = enable_selective_cache
        self.subset_cache_size = subset_cache_size
        self.compression_ratio = compression_ratio
        self.max_num_pages = self.pages_per_slot * self.max_batch_size
        self.max_num_subset_pages = self.subset_cache_size * self.max_batch_size
        self.L = L
        self.device = device
        
        self.qo_indptr = torch.arange(self.max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_indices = torch.empty(self.max_num_pages, dtype=torch.int32, device=self.device)
        self.paged_kv_indptr = torch.arange(self.max_batch_size+1, dtype=torch.int32, device=self.device)
        self.paged_kv_last_page_len = torch.empty(self.max_batch_size, dtype=torch.int32, device=self.device)
        self.paged_subset_kv_indices = torch.empty(self.max_num_pages, dtype=torch.int32, device=self.device)
        self.paged_subset_kv_indptr = torch.arange(self.max_batch_size+1, dtype=torch.int32, device=self.device)
        
        if self.enable_selective_cache:
            self._selected_mask_buf = torch.empty(
                (max_batch_size, self.pages_per_slot), dtype=torch.bool, device=self.device
            )
            self._selected_mask_buf.zero_()
            self._j_buf = torch.arange(self.pages_per_slot, dtype=torch.int32, device=self.device)
            
            self.qo_lens = self.qo_indptr[1:] - self.qo_indptr[:-1]
            self.tp = (self.paged_kv_indptr[1:] - self.paged_kv_indptr[:-1])
            self.base_offsets = self.paged_kv_indptr[:-1]
            self.is_prefill = self.qo_lens > 1
            self.take_all = (self.is_prefill | (self.tp <= self.compression_ratio * self.subset_cache_size))
        
        
    def plan(self, qo_indptr, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len,
         paged_subset_kv_indices, paged_subset_kv_indptr, L=None):

        # Store references (no copies)
        self.qo_indptr               = qo_indptr
        self.paged_kv_indices        = paged_kv_indices
        self.paged_kv_indptr         = paged_kv_indptr
        self.paged_kv_last_page_len  = paged_kv_last_page_len
        self.paged_subset_kv_indices = paged_subset_kv_indices
        self.paged_subset_kv_indptr  = paged_subset_kv_indptr
        
        if self.enable_selective_cache and not (self.paged_kv_indices is self.paged_subset_kv_indices) :
            device = paged_kv_indptr.device
            B      = paged_kv_indptr.numel() - 1

            # Row-wise metadata (device tensors)
            self.qo_lens      = self.qo_indptr[1:] - self.qo_indptr[:-1]            # [B]
            self.tp           = (self.paged_kv_indptr[1:] - self.paged_kv_indptr[:-1])  # [B]
            self.base_offsets = self.paged_kv_indptr[:-1]                            # [B]
            self.is_prefill   = self.qo_lens > 1

            # Branching threshold
            if L is not None:
                self.L   = int(L)                           # Python int
            self.rem = max(0, int(self.subset_cache_size) - self.L)

            # Take-all mask (device)
            # NOTE: keep comparisons in float/int tensor domain; avoid python scalars here
            # compression_ratio * subset_cache_size is scalar -> multiply tensor to avoid scalar→device syncs
            # thresh = self.compression_ratio * float(self.subset_cache_size)
            thresh = float(self.subset_cache_size)
            self.take_all     = self.is_prefill | (self.tp <= thresh)
            self.need_top_last= ~self.take_all

            # prefix_len_full = clamp_max(tp - last_len, ...)
            # Use clamp_* with Python ints to avoid making tiny device scalars
            last_len_full         = torch.clamp_max(self.tp, self.L)                 # [B]
            self.prefix_len_full  = torch.clamp_min(self.tp - last_len_full, 0)      # [B]
            self.k_per_row_full   = torch.clamp_max(self.prefix_len_full, self.rem)  # [B]
            self.valid_rows       = (~self.take_all) & (self.k_per_row_full > 0)     # [B] bool

            # Pre-slice for case-B
            self.tp_need   = self.tp[self.need_top_last]
            self.last_len  = torch.clamp_max(self.tp_need, self.L)                   # [B_need]

            # ----- Cache CPU scalars so select() can branch without device syncs -----
            # (Avoid .any() / .max() in select())
            self.M_valid = int(self.valid_rows.sum().item())                          # CPU int
            if self.M_valid > 0 and self.rem > 0:
                self.rows_valid = torch.where(self.valid_rows)[0]                     # [M'] (device)
                self.k_valid    = self.k_per_row_full.index_select(0, self.rows_valid)      # [M']
                self.pl         = self.prefix_len_full.index_select(0, self.rows_valid)     # [M']

                # These are CPU ints; safe for Python if-branches in select()
                self.Pmax   = int(self.pl.max().item()) if self.pl.numel() > 0 else 0
                self.max_k  = int(self.k_valid.max().item()) if self.k_valid.numel() > 0 else 0

                # Small cached buffers on device
                if self.max_k > 0:
                    # Reuse a global arange buffer if you keep varying max_k; here we create exact size
                    self._ar   = torch.arange(self.max_k, device=device)
                    # "take" mask (M' x max_k): True for positions < per-row k
                    # This is cheap and saves recomputing in select()
                    self.take  = self._ar.unsqueeze(0) < self.k_valid.unsqueeze(1)
            else:
                self.rows_valid = None
                self.k_valid    = None
                self.pl         = None
                self.Pmax       = 0
                self.max_k      = 0

            # Optional: cache handy row indices to avoid rebuilding in select()
            # Set a reasonable max batch once (or grow-on-demand if you prefer)
            if not hasattr(self, "_rows_i64") or self._rows_i64.numel() < B:
                self._rows_i64 = torch.arange(B, device=device, dtype=torch.int64)
                self._rows_i32 = self._rows_i64.to(torch.int32)

            # Optional j buffer for columns; grow if Nmax increases at runtime
            # (we’ll still slice in select())
            if not hasattr(self, "_j_buf"):
                self._j_buf = torch.arange(0, max(int(self.tp.max().item()), 1), device=device, dtype=torch.int64)
            else:
                need = int(self.tp.max().item())
                if need > self._j_buf.numel():
                    self._j_buf = torch.arange(0, need, device=device, dtype=torch.int64)
  

    def select(self, page_scores: torch.Tensor):
        """
        Vectorized page selector:
        - TAKE-ALL rows: keep all tp pages.
        - NEED-TOP-LAST rows: select top-(subset_cache_size - L) from prefix (exclude last L),
            and always keep the last min(L, tp) pages.
        Returns:
        selected_indices: int32 [nnz]   (physical page IDs)
        selected_indptr:  int32 [B+1]   (CSR-style row boundaries)
        """
        device = page_scores.device
        B, Nmax = page_scores.shape

        # Robustify scores once (avoid NaNs poisoning topk)
        page_scores = torch.nan_to_num(page_scores, neginf=float('-inf'))

        # Ensure _j_buf length covers Nmax (grow-on-demand without syncs)
        if self._j_buf.numel() < Nmax:
            self._j_buf = torch.arange(0, Nmax, device=device, dtype=torch.int64)

        # Reuse mask buffer
        selected_mask = self._selected_mask_buf[:B, :Nmax]
        selected_mask.zero_()
        j = self._j_buf[:Nmax]  # [Nmax], int64

        # ---- TAKE-ALL (no .any() to avoid sync; indexing with empty mask is a no-op) ----
        selected_mask[self.take_all] = j.unsqueeze(0) < self.tp[self.take_all].unsqueeze(1)

        # ---- NEED-TOP-LAST: add last-L (branchless) ----
        # last_len and tp_need are aligned with need_top_last rows (computed in plan)
        last_mask = (j.unsqueeze(0) >= (self.tp_need - self.last_len).unsqueeze(1)) & \
                    (j.unsqueeze(0) <   self.tp_need.unsqueeze(1))
        selected_mask[self.need_top_last] |= last_mask

        # ---- Prefix TOP-K over rows_valid (skip heavy ops via CPU ints from plan) ----
        if self.M_valid > 0 and self.rem > 0 and self.Pmax > 0 and self.max_k > 0:
            rows_v = self.rows_valid                      # [M'] device
            # Slice compact prefix window and mask padding beyond each row's prefix length
            src = page_scores.index_select(0, rows_v)[:, :self.Pmax].contiguous()  # [M', Pmax]
            mask_pad = j[:self.Pmax].unsqueeze(0) >= self.pl.unsqueeze(1)          # [M', Pmax]
            src = src.masked_fill(mask_pad, float('-inf'))

            # Fixed max_k topk, then trim per row with 'take' (precomputed in plan)
            _, topk_idx = torch.topk(src, k=self.max_k, dim=1, largest=True, sorted=False)  # [M', max_k]

            # Build per-row picked mask in the prefix window and OR it in
            picked_mask = torch.zeros(src.shape, dtype=torch.bool, device=device)           # [M', Pmax]
            picked_mask.scatter_(1, topk_idx, self.take)                                    # mask valid ks
            prev = selected_mask.index_select(0, rows_v)                                    # [M', Nmax]
            prev[:, :self.Pmax] |= picked_mask
            selected_mask.index_copy_(0, rows_v, prev)

        # ---------------- Pack outputs WITHOUT CSR round-trip ----------------
        # Row counts & CSR indptr via cumsum (no sync)
        row_counts = selected_mask.sum(dim=1, dtype=torch.int32)              # [B]
        selected_indptr = torch.empty(B + 1, dtype=torch.int32, device=device)
        selected_indptr[0] = 0
        selected_indptr[1:] = torch.cumsum(row_counts, dim=0)

        # Nonzeros: get columns once; rows via repeat_interleave
        nz = torch.nonzero(selected_mask, as_tuple=False)                      # [nnz, 2] (int64)
        if nz.numel() == 0:
            # Nothing selected: return empty structures with correct shapes
            return self.paged_kv_indices.new_empty((0,), dtype=torch.int32), selected_indptr

        rows = nz[:, 0]                                                        # [nnz], int64
        cols = nz[:, 1]                                                        # [nnz], int64

        # Map (row, col) -> flat physical index in paged_kv_indices
        pos_in_flat = self.base_offsets.index_select(0, rows.to(torch.int32)).to(torch.int64) + cols
        selected_indices = self.paged_kv_indices.index_select(0, pos_in_flat).to(torch.int32)

        return selected_indices, selected_indptr

                    

    def select2(self, page_scores: torch.Tensor, L: int = 8):
        """
        Vectorized page selector with per-row ascending order:
        - If prefill OR tp <= compression_ratio * subset_cache_size: keep all pages of the row.
        - Else: keep last min(L, tp) pages AND top-(subset_cache_size - L) from the prefix (exclude those last pages).
        Returns pages in ascending index order per row.

        Returns:
            selected_indices: int32 [total_selected]  physical page IDs
            selected_indptr:  int32 [B+1]            CSR-style row boundaries
        """
        device = page_scores.device
        B, Nmax = page_scores.shape

        # Robustify scores
        page_scores = torch.nan_to_num(page_scores, neginf=float('-inf'))

        # Row metadata
        qo_lens      = self.page_selector.qo_indptr[1:] - self.page_selector.qo_indptr[:-1]                   # [B]
        tp           = (self.page_selector.paged_kv_indptr[1:] - self.page_selector.paged_kv_indptr[:-1])     # [B]
        base_offsets = self.page_selector.paged_kv_indptr[:-1]                                   # [B]
        is_prefill   = qo_lens > 1

        # Take-all predicate (integer threshold)
        thresh_val = int(self.compression_ratio * self.subset_cache_size)
        thresh     = torch.as_tensor(thresh_val, device=device, dtype=tp.dtype)
        take_all   = is_prefill | (tp <= thresh)                                   # [B] bool

        # Budget math
        L_t        = torch.as_tensor(L, device=device, dtype=tp.dtype)
        last_len   = torch.minimum(tp, L_t)                                        # [B]
        prefix_len = torch.clamp_min(tp - last_len, 0)                             # [B]
        rem        = max(0, self.subset_cache_size - L)                            # python int
        rem_t      = torch.as_tensor(rem, device=device, dtype=tp.dtype)
        k_prefix   = torch.minimum(prefix_len, rem_t).to(torch.int64)              # [B]
        need_rows  = ~take_all                                                     # [B] bool

        # Count kept pages per row
        counts = torch.where(take_all, tp.to(torch.int64), (k_prefix + last_len.to(torch.int64)))  # [B] int64

        # Build CSR indptr
        selected_indptr = torch.empty(B + 1, dtype=torch.int32, device=device)
        selected_indptr[0] = 0
        selected_indptr[1:] = torch.cumsum(counts.to(torch.int32), dim=0)
        total_selected = int(selected_indptr[-1].item())
        if total_selected == 0:
            return torch.empty(0, dtype=torch.int32, device=device), selected_indptr

        # Prepare a padded [B, maxC] matrix of column indices, fill with sentinel (Nmax)
        maxC   = int(counts.max().item())
        cols2d = torch.full((B, maxC), fill_value=Nmax, dtype=torch.int64, device=device)
        Tpos   = torch.arange(maxC, device=device, dtype=torch.int64).unsqueeze(0)    # [1, maxC]

        # ---- Fill TAKE-ALL rows (0..tp_i-1) ----
        rows_all = torch.where(take_all)[0]                                           # [M_all]
        if rows_all.numel() > 0:
            tp_all = tp.index_select(0, rows_all).to(torch.int64)                     # [M_all]
            # place positions 0..tp_i-1 with identical values 0..tp_i-1
            cols2d_all = cols2d.index_select(0, rows_all)                             # [M_all, maxC]
            keep_mask_all = Tpos < tp_all.unsqueeze(1)                                # [M_all, maxC]
            cols2d_all.copy_(torch.where(keep_mask_all, Tpos.expand_as(cols2d_all), cols2d_all))
            cols2d.index_copy_(0, rows_all, cols2d_all)

        # ---- Fill NEED-TOPLAST rows (prefix top-k + last L) ----
        rows_need = torch.where(need_rows)[0]                                         # [M_need]
        if rows_need.numel() > 0:
            k_need    = k_prefix.index_select(0, rows_need)                           # [M_need] int64
            last_need = last_len.index_select(0, rows_need).to(torch.int64)           # [M_need]
            tp_need   = tp.index_select(0, rows_need).to(torch.int64)                 # [M_need]
            pre_need  = prefix_len.index_select(0, rows_need).to(torch.int64)         # [M_need]

            # (A) Prefix Top-K over first pre_need columns
            Pmax = int(pre_need.max().item()) if rows_need.numel() > 0 else 0
            Kmax = int(k_need.max().item())   if rows_need.numel() > 0 else 0
            if Pmax > 0 and Kmax > 0 and rem > 0:
                src = page_scores.index_select(0, rows_need)[:, :Pmax]                # [M_need, Pmax]
                j = torch.arange(Pmax, device=device)
                src = src.masked_fill(j.unsqueeze(0) >= pre_need.unsqueeze(1), float('-inf'))
                topk_vals, topk_idx = torch.topk(src, Kmax, dim=1)                    # [M_need, Kmax]
                # Trim per-row to k_need using a mask
                A = torch.arange(Kmax, device=device, dtype=torch.int64)              # [Kmax]
                take_mask = A.unsqueeze(0) < k_need.unsqueeze(1)                      # [M_need, Kmax]
                # Write into first Kmax columns, padding with sentinel for extras
                prefix_block = torch.where(take_mask, topk_idx.to(torch.int64), torch.full_like(topk_idx, Nmax))
                cols2d_need = cols2d.index_select(0, rows_need)                       # [M_need, maxC]
                # Place prefix block at cols 0..Kmax-1
                cols2d_need[:, :Kmax] = torch.minimum(cols2d_need[:, :Kmax], torch.full_like(cols2d_need[:, :Kmax], Nmax))
                cols2d_need[:, :Kmax] = prefix_block
                cols2d.index_copy_(0, rows_need, cols2d_need)

            # (B) Last-L contiguous pages placed after the prefix picks
            if last_need.max().item() > 0:
                Lmax = int(last_need.max().item())
                t = torch.arange(Lmax, device=device, dtype=torch.int64)              # [Lmax]
                keep_mask_last = t.unsqueeze(0) < last_need.unsqueeze(1)              # [M_need, Lmax]
                # destinations per row: k_need + t
                dest_last = k_need.unsqueeze(1) + t.unsqueeze(0)                      # [M_need, Lmax]
                # values: (tp - last_need) + t
                vals_last = (tp_need - last_need).unsqueeze(1) + t.unsqueeze(0)       # [M_need, Lmax]

                # Filter to valid (kept) entries and scatter into cols2d
                rows_expand = rows_need.unsqueeze(1).expand_as(dest_last)[keep_mask_last]  # [sum(last_need)]
                dest_flat   = dest_last[keep_mask_last]                                     # [sum(last_need)]
                vals_flat   = vals_last[keep_mask_last]                                     # [sum(last_need)]
                cols2d.index_put_((rows_expand, dest_flat), vals_flat, accumulate=False)

        # ---- Enforce ascending order per row via sort, then pack ----
        cols2d_sorted, _ = torch.sort(cols2d, dim=1)                                   # ascending; sentinel Nmax to end
        # Keep only first counts[b] entries of each row (valid < Nmax)
        keep_mask = Tpos < counts.unsqueeze(1)                                          # [B, maxC]
        flat_cols = cols2d_sorted[keep_mask]                                            # [total_selected] int64

        # Map (row, col) -> physical page id
        rows_for_flat = torch.repeat_interleave(torch.arange(B, device=device, dtype=torch.int64), counts)  # [T]
        pos_in_flat = base_offsets.index_select(0, rows_for_flat.to(torch.int32)).to(torch.int64) + flat_cols
        selected_indices = self.page_selector.paged_kv_indices.index_select(0, pos_in_flat).to(torch.int32)

        return selected_indices, selected_indptr

    