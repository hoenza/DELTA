import torch
from DELTA.Engine.DELTA.model import Transformer
from DELTA.Engine.DELTA.scheduler import BatchScheduler, Request, PageManager
from DELTA.Engine.DELTA.page_metadata import pack_page_indices
from DELTA.Engine.DELTA.page_score import compute_page_scores_triton
from DELTA.Engine.DELTA.page_selector import PageSelector
from DELTA.Engine.DELTA.page_selector_v2 import PageSelectorV2
from DELTA.Engine.utils import load_model_DELTA
from DELTA.Engine.DELTA.Timer import Timer
import flashinfer, sys, math
import time
from tqdm import tqdm
import torch.distributed as dist
from collections import deque, defaultdict
import os, json, time
from pathlib import Path


def _get_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except Exception:
        return 0



class LMBackend:
    def __init__(
        self,
        dtype = torch.bfloat16,
        device: str = "cuda:0",
        dec_len: int = 1,
        draft_dec_len: int = None,
        cuda_graph_decode: bool = False,
        cuda_graph_delta_subset_segments: bool = True,
        delta_impl_profile: bool = False,
        delta_fused_page_scores: bool = True,
        delta_fixed_selector: bool = True,
        delta_fast_decode_page_info: bool = True,
        delta_v2_position_bias: bool = True,
        delta_debug_page_selection_parity: bool = False,
        delta_debug_fast_decode_page_info_parity: bool = False,
        delta_subset_plan_reuse: bool = True,
        delta_dump_buffer_dtype: str = "fp32",
        delta_page_score_impl: str = "del3_legacy_softmax",
        page_selector_version: str = "v2",
    ) -> None:
        page_selector_version = str(page_selector_version).lower()
        if page_selector_version not in {"v1", "v2"}:
            raise ValueError(
                f"Unsupported page_selector_version={page_selector_version!r}; expected 'v1' or 'v2'"
            )
        delta_dump_buffer_dtype = str(delta_dump_buffer_dtype).lower()
        if delta_dump_buffer_dtype not in {"fp16", "fp32"}:
            raise ValueError(
                f"Unsupported delta_dump_buffer_dtype={delta_dump_buffer_dtype!r}; expected 'fp16' or 'fp32'"
            )
        delta_page_score_impl = str(delta_page_score_impl).lower()
        if delta_page_score_impl not in {"delta_lse", "del3_legacy_softmax"}:
            raise ValueError(
                "Unsupported delta_page_score_impl="
                f"{delta_page_score_impl!r}; expected 'delta_lse' or 'del3_legacy_softmax'"
            )
        self.dtype = dtype
        self.device = device
        self.dec_len = dec_len
        self.model_forward = lambda model, x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen: model(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
        self._eager_model_forward = self.model_forward
        self._dynamic_compiled_model_forward = None
        self._baseline_compiled_model_forward = None
        self.cachelens = None
        
        # For distributed logging
        self.is_main_process = True
        self.tokenizer = None
        self.scheduler = None
        self.throughput_mode = False
        
        self.timer = None
        self._compiled = False
        self._pending_events = []
        self._step_records = []
        self.enable_cuda_graph_decode = cuda_graph_decode
        self.enable_cuda_graph_delta_subset_segments = cuda_graph_delta_subset_segments
        self.enable_delta_impl_profile = delta_impl_profile
        self.enable_delta_fused_page_scores = delta_fused_page_scores
        self.enable_delta_fixed_selector = delta_fixed_selector
        self.enable_delta_fast_decode_page_info = delta_fast_decode_page_info
        self.enable_delta_v2_position_bias = delta_v2_position_bias
        self.enable_delta_debug_page_selection_parity = delta_debug_page_selection_parity
        self.enable_delta_debug_fast_decode_page_info_parity = delta_debug_fast_decode_page_info_parity
        self.enable_delta_subset_plan_reuse = delta_subset_plan_reuse
        self.delta_dump_buffer_dtype = delta_dump_buffer_dtype
        self.delta_page_score_impl = delta_page_score_impl
        self.page_selector_version = page_selector_version
        self._use_cuda_graph_decode_this_step = False
        self._decode_graphs = {}
        self._decode_graph_outputs = {}
        self._decode_graph_capture_failed = set()
        self._decode_graph_capture_errors = {}
        self._decode_graph_capture_time_s = defaultdict(float)
        self._decode_graph_stats = defaultdict(int)
        self._decode_graph_fallback_reasons = defaultdict(int)
        self._delta_impl_profile_ms = defaultdict(float)
        self._delta_impl_profile_counts = defaultdict(int)
        self._delta_impl_profile_pending = []
        self._decode_graph_static_tokens = None
        self._decode_graph_static_input_pos = None
        self._decode_graph_qo_indptr = None
        self._delta_subset_segment_plan = []
        self._delta_subset_segment_graphs = {}
        self._delta_subset_segment_inputs = {}
        self._delta_subset_segment_outputs = {}
        self._delta_subset_segment_failed = set()
        self._delta_subset_segment_errors = {}
        self._delta_subset_segment_capture_time_s = defaultdict(float)
        self._delta_subset_segment_stats = defaultdict(int)
        self._delta_subset_segment_fallback_reasons = defaultdict(int)
        self.decode_wrapper_full_graph = None
        self.decode_wrapper_full_dump_graph = None
        self.decode_wrapper_subset_graph = None
        self._metadata_last_full_total = 0
        self._metadata_last_subset_total = 0
        self._metadata_last_subset_fixed = False
        self._baseline_static_qo_indptr = None
        self._baseline_static_paged_kv_indices = None
        self._baseline_static_paged_kv_indptr = None
        self._baseline_static_paged_kv_last_page_len = None
        self._baseline_static_decode_tokens = None
        self._baseline_static_decode_input_pos = None
        self.decode_wrapper_full_dump = None
        self._v2_prob_buf = None
        self._v2_head_max_buf = None
        self._v2_page_scores_buf = None
        self._metadata_decode_token_buf = None
        self._metadata_decode_tokens_host = None
        self._metadata_active_slots = None
        self._metadata_active_slots_host = None
        self._metadata_page_counts = None
        self._metadata_page_counts_host = None
        self._metadata_subset_counts = None
        self._metadata_subset_counts_host = None
        self._metadata_zero_starts = None
        self._metadata_subset_starts = None
        self._metadata_subset_starts_host = None
        self._metadata_subset_fixed_indptr = None
        self._metadata_lastlens_host = None
        self._metadata_slot_block_table = None
        self._metadata_qo_indptr_decode = None
        self._metadata_paged_kv_indices = None
        self._metadata_paged_kv_indptr = None
        self._metadata_paged_kv_last_page_len = None
        self._metadata_paged_subset_kv_indices = None
        self._metadata_paged_subset_kv_indptr = None
        self._postprocess_all_slots = None
        self._postprocess_all_slots_list = None
        self._postprocess_slot_ids = None
        self._postprocess_slot_ids_host = None
        self._postprocess_qo_lengths = None
        self._postprocess_qo_lengths_host = None
        self._postprocess_ones = None
        self._postprocess_last_token_positions = None
        self._postprocess_next_tokens = None
        self._postprocess_next_tokens_host = None
        self._using_subset_cache_this_step = False
        self._delta_fused_page_scores_failed = False
        self._wrapper_plan_cache = {}

    def _attention_dump_buffer_dtype(self):
        if self.delta_dump_buffer_dtype == "fp32":
            return torch.float32
        return torch.float16

    def _decode_dump_jit_args(self):
        if self.delta_dump_buffer_dtype == "fp32":
            buffer_type = "float"
            store_expr = "params.attention_weights[idx] = logits * params.sm_scale;"
        else:
            buffer_type = "half"
            store_expr = "params.attention_weights[idx] = __float2half(logits * params.sm_scale);"

        return [
            "batch_prefill_dump_attention_weights",
            self.dtype, self.dtype, self.dtype, torch.int32,
            self.model.config.head_dim, self.model.config.head_dim,
            ["attention_weights"], [buffer_type],
            ["sm_scale", "batch_size_val", "max_heads_val", "max_qo_len_val", "max_kv_len_val", "total_attention_elements"],
            ["double", "int64_t", "int64_t", "int64_t", "int64_t", "int64_t"],
            "BatchDumpAttentionWeights",
            f"""
struct BatchDumpAttentionWeights : AttentionVariantBase {{
static constexpr bool use_softmax = true;

uint32_t window_left, qo_len, kv_len;
float sm_scale_log2;

template <typename Params>
__device__ __host__ BatchDumpAttentionWeights(const Params& params, uint32_t batch_idx,
                                                uint8_t* smem_ptr) {{
    qo_len = params.get_qo_len(batch_idx);
    kv_len = params.get_kv_len(batch_idx);
    window_left = kv_len;
    sm_scale_log2 = params.sm_scale * math::log2e;
}}

REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {{
    if (qo_idx == (qo_len-1) && kv_idx < kv_len && batch_idx < params.batch_size_val) {{
    uint64_t total_elements = params.total_attention_elements;
    uint64_t idx = (batch_idx * params.max_heads_val * params.max_qo_len_val * params.max_kv_len_val) +
                    (qo_head_idx * params.max_qo_len_val * params.max_kv_len_val) +
                    (qo_idx * params.max_kv_len_val) + kv_idx;

    if (idx < total_elements) {{
        {store_expr}
    }}
    }}
    return logits;
}});
}};
""",
        ]

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def load_model(self, checkpoints: str, use_tp: bool, rank_group=None, group = None):
        self.model: Transformer = load_model_DELTA(checkpoint_path=checkpoints, device=self.device, precision=self.dtype, use_tp=use_tp, rank_group=rank_group, group=group)        

    def compile(self):
        import torch._inductor.config
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True
        torch._inductor.config.fx_graph_cache = True
        torch._functorch.config.enable_autograd_cache = True
        if hasattr(torch._inductor.config, "triton"):
            torch._inductor.config.triton.cudagraph_trees = False
        self._compiled = True
        self._dynamic_compiled_model_forward = torch.compile(
            self._eager_model_forward,
            mode="reduce-overhead",
            dynamic=True,
        )
        self.model_forward = self._dynamic_compiled_model_forward

    @torch.inference_mode()
    def setup_caches(self, max_batch_size: int = 1, max_seq_length: int = 2048):
        self.max_length = max_seq_length
        self.batch_size = max_batch_size
        # self.page_size = 1
        self.page_size = 16
        
        self.pages_per_slot = (max_seq_length + self.page_size - 1) // self.page_size
        self.max_num_pages = self.pages_per_slot * max_batch_size
        
        self.page_manager = PageManager(self.max_num_pages, seed=0)
        
        self.scheduler = BatchScheduler(
            max_batch_size, self.device, self.is_main_process,
            self.pages_per_slot, self.page_size, self.page_manager
        )
        
        self.cachelens = torch.zeros(max_batch_size, dtype=torch.int32, device=self.device)
        self._setup_postprocess_buffers(max_batch_size)
        
        self.enable_selective_cache = getattr(self.model.config, 'enable_selective_cache', False)
        self.subset_cache_size = getattr(self.model.config, 'subset_cache_size', 64)
        self.compression_ratio = getattr(self.model.config, 'compression_ratio', 1.0)
        self.L = getattr(self.model.config, 'L', 8)
     
        full_buf_size = 128 * 1024 * 1024 if self.enable_selective_cache else 512 * 1024 * 1024
        self.decode_buffer_full = torch.empty(full_buf_size, dtype=torch.uint8, device=self.device)
        self.decode_buffer_full_dump = (
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
            if self.enable_selective_cache else None
        )
        self.decode_buffer_subset = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)

        page_selector_cls = PageSelectorV2 if self.page_selector_version == "v2" else PageSelector
        selector_kwargs = {}
        if self.page_selector_version == "v2":
            selector_kwargs["position_bias_scale"] = 1e-7 if self.enable_delta_v2_position_bias else 0.0
        self.page_selector = page_selector_cls(
            self.batch_size,
            self.pages_per_slot,
            self.enable_selective_cache,
            self.subset_cache_size,
            self.compression_ratio,
            self.L,
            self.device,
            **selector_kwargs,
        )

        # Legacy mode: baseline (no selective cache) + torch.compile. The opt-in
        # cuda_graph_decode path below is pure CUDAGraph and intentionally avoids compile.
        self._cuda_graph_mode = (
            self._compiled
            and not self.enable_selective_cache
            and not self.enable_cuda_graph_decode
        )

        if self.enable_selective_cache:
            self._v2_prob_buf = torch.empty(
                (max_batch_size, self.model.config.n_head, max_seq_length),
                dtype=torch.float32,
                device=self.device,
            )
            self._setup_page_metadata_buffers(max_batch_size, max_seq_length)
            self._v2_head_max_buf = torch.empty(
                (max_batch_size, max_seq_length),
                dtype=torch.float32,
                device=self.device,
            )
            self._v2_page_scores_buf = torch.empty(
                (max_batch_size, self.pages_per_slot),
                dtype=torch.float32,
                device=self.device,
            )
        elif not self._cuda_graph_mode:
            self._setup_page_metadata_buffers(max_batch_size, max_seq_length)

        max_decode_len = 1
        max_heads = self.model.config.n_head
        print(f'max_heads: {max_heads}')
        decode_attention_elements = max_batch_size * max_heads * max_decode_len * max_seq_length

        if self._cuda_graph_mode:
            print("[CUDA graph mode] Enabled for baseline + compile")
        elif self.enable_cuda_graph_decode:
            print("[CUDA graph decode] Enabled for steady one-token decode (no torch.compile)")

        if self._cuda_graph_mode:
            # ---- Baseline + compile: simple wrappers, no JIT, use_cuda_graph=True ----
            self.attention_weights_buffer = None
            self.attention_weights_buffer_neg_inf = None
            self._lse_buffer = None
            self._log2e = None

            self.decode_wrapper_full = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.decode_buffer_full, "NHD", use_cuda_graph=True,
                qo_indptr_buf=self.page_selector.qo_indptr,
                paged_kv_indptr_buf=self.page_selector.paged_kv_indptr,
                paged_kv_indices_buf=self.page_selector.paged_kv_indices,
                paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
            )

            self.decode_wrapper_subset = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.decode_buffer_subset, "NHD", use_cuda_graph=True,
                qo_indptr_buf=self.page_selector.qo_indptr,
                paged_kv_indptr_buf=self.page_selector.paged_subset_kv_indptr,
                paged_kv_indices_buf=self.page_selector.paged_subset_kv_indices,
                paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
            )

            torch.library.define("mylib::target_decode", "(Tensor q, Tensor kv_cache) -> Tensor")
            @torch.library.impl("mylib::target_decode", "cuda")
            def target_decode(q, kv_cache):
                return self.decode_wrapper_full.run(q, kv_cache)

            @torch.library.register_fake("mylib::target_decode")
            def target_decode_abstract(q, kv_cache):
                return torch.empty_like(q)

            torch.library.define("mylib::target_subset_decode", "(Tensor q, Tensor kv_cache) -> Tensor")
            @torch.library.impl("mylib::target_subset_decode", "cuda")
            def target_subset_decode(q, kv_cache):
                return self.decode_wrapper_full.run(q, kv_cache)

            @torch.library.register_fake("mylib::target_subset_decode")
            def target_subset_decode_abstract(q, kv_cache):
                return torch.empty_like(q)

            torch.library.define("mylib::target_decode_plan", "(Tensor q, Tensor kv_cache) -> Tensor")
            @torch.library.impl("mylib::target_decode_plan", "cuda")
            def target_decode_plan(q, kv_cache):
                return self.decode_wrapper_full.run(q, kv_cache)

            @torch.library.register_fake("mylib::target_decode_plan")
            def target_decode_plan_abstract(q, kv_cache):
                return torch.empty_like(q)

        else:
            # ---- DELTA or non-compiled: use JIT dump only for the replan layer ----
            self.attention_weights_buffer = torch.full(
                (int(decode_attention_elements),),
                float("-inf"),
                dtype=self._attention_dump_buffer_dtype(),
                device=self.device,
            )
            self.attention_weights_buffer_neg_inf = torch.tensor(float("-inf"), dtype=self.attention_weights_buffer.dtype, device=self.attention_weights_buffer.device)

            if self.enable_selective_cache:
                self._lse_buffer = torch.zeros(max_batch_size, max_heads, dtype=torch.float32, device=self.device)
                self._log2e = math.log2(math.e)
            else:
                self._lse_buffer = None
                self._log2e = None

            decode_jit_args = self._decode_dump_jit_args()
            self.decode_wrapper_full = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.decode_buffer_full, "NHD", use_cuda_graph=False,
                qo_indptr_buf=self.page_selector.qo_indptr,
                paged_kv_indptr_buf=self.page_selector.paged_kv_indptr,
                paged_kv_indices_buf=self.page_selector.paged_kv_indices,
                paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
                backend="fa2",
            )

            if self.enable_selective_cache:
                self.decode_wrapper_full_dump = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.decode_buffer_full_dump, "NHD", use_cuda_graph=False,
                    qo_indptr_buf=self.page_selector.qo_indptr,
                    paged_kv_indptr_buf=self.page_selector.paged_kv_indptr,
                    paged_kv_indices_buf=self.page_selector.paged_kv_indices,
                    paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
                    backend="fa2",
                    jit_args=decode_jit_args,
                    jit_kwargs={}
                )

            self.decode_wrapper_subset = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.decode_buffer_subset, "NHD", use_cuda_graph=False,
                qo_indptr_buf=self.page_selector.qo_indptr,
                paged_kv_indptr_buf=self.page_selector.paged_subset_kv_indptr,
                paged_kv_indices_buf=self.page_selector.paged_subset_kv_indices,
                paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
                backend="fa2",
            )

            if self.enable_cuda_graph_decode:
                self.decode_wrapper_full_graph = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.decode_buffer_full, "NHD", use_cuda_graph=True,
                    qo_indptr_buf=self.page_selector.qo_indptr,
                    paged_kv_indptr_buf=self.page_selector.paged_kv_indptr,
                    paged_kv_indices_buf=self.page_selector.paged_kv_indices,
                    paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
                    backend="fa2",
                )

                if self.enable_selective_cache:
                    self.decode_wrapper_full_dump_graph = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                        self.decode_buffer_full_dump, "NHD", use_cuda_graph=True,
                        qo_indptr_buf=self.page_selector.qo_indptr,
                        paged_kv_indptr_buf=self.page_selector.paged_kv_indptr,
                        paged_kv_indices_buf=self.page_selector.paged_kv_indices,
                        paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
                        backend="fa2",
                        jit_args=decode_jit_args,
                        jit_kwargs={}
                    )

                self.decode_wrapper_subset_graph = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                    self.decode_buffer_subset, "NHD", use_cuda_graph=True,
                    qo_indptr_buf=self.page_selector.qo_indptr,
                    paged_kv_indptr_buf=self.page_selector.paged_subset_kv_indptr,
                    paged_kv_indices_buf=self.page_selector.paged_subset_kv_indices,
                    paged_kv_last_page_len_buf=self.page_selector.paged_kv_last_page_len,
                    backend="fa2",
                )

            torch.library.define("mylib::target_decode", "(Tensor q, Tensor kv_cache) -> Tensor")
            @torch.library.impl("mylib::target_decode", "cuda")
            def target_decode(q, kv_cache):
                wrapper = (
                    self.decode_wrapper_full_graph
                    if self._use_cuda_graph_decode_this_step and self.decode_wrapper_full_graph is not None
                    else self.decode_wrapper_full
                )
                if (not self.timer is None) and self.timer.get_timing_enabled():
                    stream = torch.cuda.current_stream(q.device)
                    start = torch.cuda.Event(enable_timing=True)
                    end   = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    result = wrapper.run(q, kv_cache)
                    end.record(stream)
                    end.synchronize()
                    self.timer.record_cuda_ms("target_decode", start, end)
                    return result
                else:
                    return wrapper.run(q, kv_cache)

            @torch.library.register_fake("mylib::target_decode")
            def target_decode_abstract(q, kv_cache):
                return torch.empty_like(q)

            torch.library.define("mylib::target_subset_decode", "(Tensor q, Tensor kv_cache) -> Tensor")
            @torch.library.impl("mylib::target_subset_decode", "cuda")
            def target_subset_decode(q, kv_cache):
                if self._using_subset_cache_this_step:
                    wrapper = (
                        self.decode_wrapper_subset_graph
                        if self._use_cuda_graph_decode_this_step and self.decode_wrapper_subset_graph is not None
                        else self.decode_wrapper_subset
                    )
                else:
                    wrapper = (
                        self.decode_wrapper_full_graph
                        if self._use_cuda_graph_decode_this_step and self.decode_wrapper_full_graph is not None
                        else self.decode_wrapper_full
                    )
                if (not self.timer is None) and self.timer.get_timing_enabled():
                    stream = torch.cuda.current_stream(q.device)
                    start = torch.cuda.Event(enable_timing=True)
                    end   = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    result = wrapper.run(q, kv_cache)
                    end.record(stream)
                    end.synchronize()
                    self.timer.record_cuda_ms("target_subset_decode", start, end)
                    return result
                else:
                    return wrapper.run(q, kv_cache)

            @torch.library.register_fake("mylib::target_subset_decode")
            def target_subset_decode_abstract(q, kv_cache):
                return torch.empty_like(q)

            torch.library.define("mylib::target_decode_plan", "(Tensor q, Tensor kv_cache) -> Tensor")
            @torch.library.impl("mylib::target_decode_plan", "cuda")
            def target_decode_plan(q, kv_cache):
                full_wrapper = (
                    self.decode_wrapper_full_graph
                    if self._use_cuda_graph_decode_this_step and self.decode_wrapper_full_graph is not None
                    else self.decode_wrapper_full
                )
                dump_wrapper = (
                    self.decode_wrapper_full_dump_graph
                    if self._use_cuda_graph_decode_this_step and self.decode_wrapper_full_dump_graph is not None
                    else self.decode_wrapper_full_dump
                )
                if (not self.timer is None) and self.timer.get_timing_enabled():
                    stream = torch.cuda.current_stream(q.device)
                    start = torch.cuda.Event(enable_timing=True)
                    end   = torch.cuda.Event(enable_timing=True)
                    start.record(stream)
                    if self.enable_selective_cache and self._using_subset_cache_this_step:
                        result, lse = self._profile_cuda_section(
                            "delta_planner_attention_dump",
                            lambda: dump_wrapper.run(
                                q, kv_cache,
                                self.attention_weights_buffer,
                                1.0 / math.sqrt(self.model.config.head_dim),
                                max_batch_size,
                                max_heads,
                                max_decode_len,
                                max_seq_length,
                                decode_attention_elements,
                                return_lse=True,
                            ),
                        )
                        if lse.shape[0] <= self._lse_buffer.shape[0]:
                            self._lse_buffer[:lse.shape[0]] = lse
                        self._profile_cuda_section("delta_replan_total", self._replan)
                    else:
                        result = full_wrapper.run(q, kv_cache)
                    end.record(stream)
                    end.synchronize()
                    self.timer.record_cuda_ms("target_decode_plan", start, end)
                    return result
                else:
                    if self.enable_selective_cache and self._using_subset_cache_this_step:
                        result, lse = self._profile_cuda_section(
                            "delta_planner_attention_dump",
                            lambda: dump_wrapper.run(
                                q, kv_cache,
                                self.attention_weights_buffer,
                                1.0 / math.sqrt(self.model.config.head_dim),
                                max_batch_size,
                                max_heads,
                                max_decode_len,
                                max_seq_length,
                                decode_attention_elements,
                                return_lse=True,
                            ),
                        )
                        if lse.shape[0] <= self._lse_buffer.shape[0]:
                            self._lse_buffer[:lse.shape[0]] = lse
                        self._profile_cuda_section("delta_replan_total", self._replan)
                        return result
                    else:
                        return full_wrapper.run(q, kv_cache)

            @torch.library.register_fake("mylib::target_decode_plan")
            def target_decode_plan_abstract(q, kv_cache):
                return torch.empty_like(q)
         
        with torch.device(self.device):
            self.model.setup_caches(num_pages=self.max_num_pages, page_size=self.page_size)
        if self.enable_cuda_graph_decode:
            self._setup_cuda_graph_decode_buffers(max_batch_size)
            self._setup_delta_subset_segment_plan()
        if self._cuda_graph_mode:
            self._setup_baseline_static_decode()

    def _make_host_tensor(self, *shape, dtype):
        try:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, dtype=dtype)

    def _setup_postprocess_buffers(self, max_batch_size: int):
        self._postprocess_all_slots = torch.arange(
            max_batch_size, dtype=torch.long, device=self.device)
        self._postprocess_all_slots_list = list(range(max_batch_size))
        self._postprocess_slot_ids = torch.empty(
            max_batch_size, dtype=torch.long, device=self.device)
        self._postprocess_slot_ids_host = self._make_host_tensor(
            max_batch_size, dtype=torch.long)
        self._postprocess_qo_lengths = torch.empty(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._postprocess_qo_lengths_host = self._make_host_tensor(
            max_batch_size, dtype=torch.int32)
        self._postprocess_ones = torch.ones(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._postprocess_last_token_positions = torch.empty(
            max_batch_size, dtype=torch.long, device=self.device)
        self._postprocess_next_tokens = torch.empty(
            max_batch_size, dtype=torch.long, device=self.device)
        self._postprocess_next_tokens_host = self._make_host_tensor(
            max_batch_size, dtype=torch.long)

    def _setup_cuda_graph_decode_buffers(self, max_batch_size: int):
        self._decode_graph_static_tokens = torch.zeros(
            (1, max_batch_size), dtype=torch.long, device=self.device)
        self._decode_graph_static_input_pos = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._decode_graph_qo_indptr = torch.arange(
            max_batch_size + 1, dtype=torch.int32, device=self.device)

    def _setup_delta_subset_segment_plan(self):
        self._delta_subset_segment_plan = []
        if not (
            self.enable_cuda_graph_decode
            and self.enable_cuda_graph_delta_subset_segments
            and self.enable_selective_cache
        ):
            return

        n_layer = self.model.config.n_layer
        full_layers = set(getattr(self.model.config, "full_cache_layers", []) or [])
        planner_layers = [
            layer_idx
            for layer_idx in sorted(full_layers)
            if layer_idx + 1 < n_layer and (layer_idx + 1) not in full_layers
        ]
        if not planner_layers:
            return

        cursor = 0
        for planner_layer in planner_layers:
            if cursor == 0 or cursor < planner_layer:
                self._delta_subset_segment_plan.append({
                    "kind": "segment",
                    "key": self._delta_subset_segment_key(cursor, planner_layer, cursor == 0, False),
                    "start": cursor,
                    "end": planner_layer,
                    "include_embed": cursor == 0,
                    "include_tail": False,
                })
            self._delta_subset_segment_plan.append({
                "kind": "planner_pre",
                "key": self._delta_subset_planner_key(planner_layer, "pre"),
                "layer": planner_layer,
            })
            self._delta_subset_segment_plan.append({
                "kind": "planner_attention",
                "layer": planner_layer,
            })
            self._delta_subset_segment_plan.append({
                "kind": "planner_post",
                "key": self._delta_subset_planner_key(planner_layer, "post"),
                "layer": planner_layer,
            })
            cursor = planner_layer + 1

        self._delta_subset_segment_plan.append({
            "kind": "segment",
            "key": self._delta_subset_segment_key(cursor, n_layer, cursor == 0, True),
            "start": cursor,
            "end": n_layer,
            "include_embed": cursor == 0,
            "include_tail": True,
        })

        if self.is_main_process:
            segments = [
                op["key"] if "key" in op else f"planner_attention_{op['layer']}"
                for op in self._delta_subset_segment_plan
            ]
            print(f"[CUDA graph decode] DELTA subset segment plan: {segments}")

    def _delta_subset_segment_key(self, start_layer: int, end_layer: int, include_embed: bool, include_tail: bool) -> str:
        if include_tail:
            prefix = "embed_tail" if include_embed else "tail"
        elif include_embed:
            prefix = "embed_layers"
        else:
            prefix = "layers"
        return f"delta_subset_{prefix}_{start_layer}_{end_layer}"

    def _delta_subset_planner_key(self, layer_idx: int, part: str) -> str:
        return f"delta_subset_planner_{part}_{layer_idx}"

    def _setup_page_metadata_buffers(self, max_batch_size: int, max_seq_length: int):
        self._metadata_decode_token_buf = torch.empty(
            (1, max_batch_size), dtype=torch.long, device=self.device)
        self._metadata_decode_tokens_host = self._make_host_tensor(
            max_batch_size, dtype=torch.long)

        self._metadata_active_slots = torch.empty(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._metadata_active_slots_host = self._make_host_tensor(
            max_batch_size, dtype=torch.int32)

        self._metadata_page_counts = torch.empty(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._metadata_page_counts_host = self._make_host_tensor(
            max_batch_size, dtype=torch.int32)

        self._metadata_subset_counts = torch.empty(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._metadata_subset_counts_host = self._make_host_tensor(
            max_batch_size, dtype=torch.int32)

        self._metadata_zero_starts = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._metadata_subset_starts = torch.empty(
            max_batch_size, dtype=torch.int32, device=self.device)
        self._metadata_subset_starts_host = self._make_host_tensor(
            max_batch_size, dtype=torch.int32)
        self._metadata_subset_fixed_indptr = (
            torch.arange(max_batch_size + 1, dtype=torch.int32, device=self.device)
            * self.subset_cache_size
        )
        self._metadata_lastlens_host = self._make_host_tensor(
            max_batch_size, dtype=torch.int32)

        self._metadata_slot_block_table = torch.empty(
            (max_batch_size, self.pages_per_slot), dtype=torch.int32, device=self.device)

        self._metadata_qo_indptr_decode = torch.arange(
            max_batch_size + 1, dtype=torch.int32, device=self.device)
        self._metadata_paged_kv_indices = self.page_selector.paged_kv_indices
        self._metadata_paged_kv_indptr = self.page_selector.paged_kv_indptr
        self._metadata_paged_kv_last_page_len = self.page_selector.paged_kv_last_page_len
        self._metadata_paged_subset_kv_indices = self.page_selector.paged_subset_kv_indices
        self._metadata_paged_subset_kv_indptr = self.page_selector.paged_subset_kv_indptr

    def _setup_baseline_static_decode(self):
        try:
            import torch._dynamo as dynamo
        except ImportError:
            dynamo = None

        self._baseline_static_qo_indptr = self.page_selector.qo_indptr
        self._baseline_static_paged_kv_indices = self.page_selector.paged_kv_indices
        self._baseline_static_paged_kv_indptr = self.page_selector.paged_kv_indptr
        self._baseline_static_paged_kv_last_page_len = self.page_selector.paged_kv_last_page_len
        self._baseline_static_decode_tokens = torch.zeros(
            (1, self.batch_size), dtype=torch.long, device=self.device
        )
        self._baseline_static_decode_input_pos = torch.zeros(
            self.batch_size, dtype=torch.int32, device=self.device
        )

        if dynamo is not None and hasattr(dynamo, "mark_static_address"):
            for tensor in (
                self._baseline_static_qo_indptr,
                self._baseline_static_paged_kv_indices,
                self._baseline_static_paged_kv_indptr,
                self._baseline_static_paged_kv_last_page_len,
                self._baseline_static_decode_tokens,
                self._baseline_static_decode_input_pos,
            ):
                dynamo.mark_static_address(tensor)

        self._baseline_compiled_model_forward = torch.compile(
            self._eager_model_forward,
            mode="reduce-overhead",
            fullgraph=True,
        )

    def reset_attention_weights_buffer(self):
        if self.attention_weights_buffer is not None:
            self.attention_weights_buffer.fill_(self.attention_weights_buffer_neg_inf)
    
    def _replan(self):
        if not (self.page_selector.paged_kv_indices is self.page_selector.paged_subset_kv_indices):
            page_scores = self._profile_cuda_section(
                "delta_page_score",
                self._compute_page_scores_from_attention_buffer,
            )
            use_fixed_selector = (
                self.page_selector_version == "v2"
                and
                self.enable_delta_fixed_selector
                and self._metadata_last_subset_fixed
            )
            if self.page_selector_version == "v2":
                select_pages = lambda: self.page_selector.select(
                    page_scores,
                    fixed_count=use_fixed_selector,
                    debug_compare=self.enable_delta_debug_page_selection_parity,
                )
            else:
                select_pages = lambda: self.page_selector.select(page_scores)
            selected_indices, selected_indptr = self._profile_cuda_section(
                "delta_page_select",
                select_pages,
            )

            if self.enable_delta_debug_page_selection_parity:
                ref_scores = self._compute_page_scores_reference()
                if self.page_selector_version == "v2":
                    ref_indices, ref_indptr = self.page_selector.select(ref_scores, fixed_count=False)
                else:
                    ref_indices, ref_indptr = self.page_selector.select(ref_scores)
                if not torch.equal(selected_indptr, ref_indptr) or not torch.equal(selected_indices, ref_indices):
                    raise RuntimeError("DELTA page selection mismatch against torch-score reference path")

            if self.enable_cuda_graph_decode:
                subset_indices = self.page_selector.paged_subset_kv_indices
                subset_indptr = self.page_selector.paged_subset_kv_indptr
                def copy_selected_subset():
                    subset_indices[:selected_indices.numel()].copy_(selected_indices)
                    subset_indptr[:selected_indptr.numel()].copy_(selected_indptr)
                self._profile_cuda_section("delta_subset_wrapper_copy", copy_selected_subset)
            else:
                self.page_selector.paged_subset_kv_indices = selected_indices
                self.page_selector.paged_subset_kv_indptr = selected_indptr
                subset_indices = self.page_selector.paged_subset_kv_indices
                subset_indptr = self.page_selector.paged_subset_kv_indptr

            # Later layers may run either eager or from segmented CUDA Graphs.
            # Keep both wrapper buffers current without rebinding tensor addresses.
            subset_wrappers = []
            if self.decode_wrapper_subset is not None:
                subset_wrappers.append(self.decode_wrapper_subset)
            if (
                self.decode_wrapper_subset_graph is not None
                and self.decode_wrapper_subset_graph is not self.decode_wrapper_subset
            ):
                subset_wrappers.append(self.decode_wrapper_subset_graph)

            def sync_subset_wrappers():
                for wrapper in subset_wrappers:
                    if wrapper._paged_kv_indices_buf.data_ptr() != subset_indices.data_ptr():
                        wrapper._paged_kv_indices_buf[:selected_indices.numel()].copy_(
                            subset_indices[:selected_indices.numel()])
                    if wrapper._paged_kv_indptr_buf.data_ptr() != subset_indptr.data_ptr():
                        wrapper._paged_kv_indptr_buf[:selected_indptr.numel()].copy_(
                            subset_indptr[:selected_indptr.numel()])
            self._profile_cuda_section("delta_subset_wrapper_sync", sync_subset_wrappers)
                
    def _compute_page_scores_from_attention_buffer_torch(self):
        bsz = self.batch_size
        n_heads = self.model.config.n_head
        seq_len = self.max_length
        page_sz = self.page_size
        max_pages = self.scheduler.max_pages
        active_bsz = self.scheduler.active_bsz

        if active_bsz == 0:
            return torch.empty(0, seq_len // page_sz, dtype=torch.float32, device=self.device)

        max_seq_len = max_pages * page_sz
        logits = self.attention_weights_buffer.reshape(bsz, n_heads, 1, seq_len)[:active_bsz, :, 0, :max_seq_len]
        lse = self._lse_buffer[:active_bsz]

        prob_buf = self._v2_prob_buf[:active_bsz, :, :max_seq_len]
        head_max = self._v2_head_max_buf[:active_bsz, :max_seq_len]
        page_scores = self._v2_page_scores_buf[:active_bsz, :max_pages]

        # The dump buffer already stores logits scaled by sm_scale, so reconstruct
        # probabilities directly from those scaled logits and the matching LSE.
        prob_buf.copy_(logits)
        prob_buf.sub_(lse[:, :, None])
        torch.exp(prob_buf, out=prob_buf)
        torch.amax(prob_buf, dim=1, out=head_max)
        torch.sum(
            head_max.reshape(active_bsz, max_pages, page_sz),
            dim=-1,
            out=page_scores,
        )
        return page_scores

    def _compute_page_scores_from_attention_buffer_legacy_softmax(self):
        bsz = self.batch_size
        n_heads = self.model.config.n_head
        seq_len = self.max_length
        page_sz = self.page_size
        max_pages = self.scheduler.max_pages
        active_bsz = self.scheduler.active_bsz

        if active_bsz == 0:
            return torch.empty(0, seq_len // page_sz, dtype=torch.float32, device=self.device)

        max_seq_len = max_pages * page_sz
        logits = self.attention_weights_buffer.reshape(
            bsz, n_heads, 1, seq_len
        )[:active_bsz, :, :, :max_seq_len].to(torch.float32)
        probs = torch.softmax(logits, dim=-1)
        scores_1d = probs.amax(dim=1).sum(dim=1)
        return scores_1d.reshape(active_bsz, max_pages, page_sz).sum(dim=-1)

    def _compute_page_scores_reference(self):
        if self.delta_page_score_impl == "del3_legacy_softmax":
            return self._compute_page_scores_from_attention_buffer_legacy_softmax()
        return self._compute_page_scores_from_attention_buffer_torch()

    def _compute_page_scores_from_attention_buffer(self):
        if self.delta_page_score_impl == "del3_legacy_softmax":
            return self._compute_page_scores_from_attention_buffer_legacy_softmax()

        bsz = self.batch_size
        n_heads = self.model.config.n_head
        seq_len = self.max_length
        page_sz = self.page_size
        max_pages = self.scheduler.max_pages
        active_bsz = self.scheduler.active_bsz

        if active_bsz == 0:
            return torch.empty(0, seq_len // page_sz, dtype=torch.float32, device=self.device)

        max_seq_len = max_pages * page_sz
        logits = self.attention_weights_buffer.reshape(
            bsz, n_heads, 1, seq_len
        )[:active_bsz, :, 0, :max_seq_len]
        lse = self._lse_buffer[:active_bsz]
        page_scores = self._v2_page_scores_buf[:active_bsz, :max_pages]

        if self.enable_delta_fused_page_scores and not self._delta_fused_page_scores_failed:
            try:
                return compute_page_scores_triton(
                    logits,
                    lse,
                    page_scores,
                    log2e=self._log2e,
                    page_size=page_sz,
                )
            except Exception as exc:
                self._delta_fused_page_scores_failed = True
                self._record_decode_graph_fallback("delta_fused_page_scores_failed")
                if self.is_main_process:
                    print(f"[DELTA] Fused page-score kernel failed; falling back to torch ops: {exc}")

        return self._compute_page_scores_from_attention_buffer_torch()

    def _record_decode_graph_fallback(self, reason: str):
        self._decode_graph_fallback_reasons[reason] += 1

    def _cuda_graph_decode_key(self) -> str:
        if not self.enable_selective_cache:
            return "baseline_full"
        if self._using_subset_cache_this_step:
            return "delta_subset"
        return "delta_full"

    def _can_use_delta_subset_segment_graph(self) -> tuple[bool, str]:
        if not self.enable_cuda_graph_delta_subset_segments:
            return False, "delta_subset_segments_disabled"
        if not self._delta_subset_segment_plan:
            return False, "delta_subset_segments_missing"
        if self.decode_wrapper_subset_graph is None:
            return False, "delta_subset_graph_wrapper_missing"
        if self.decode_wrapper_full_dump is None:
            return False, "delta_subset_dump_wrapper_missing"
        if not self._metadata_last_subset_fixed:
            return False, "delta_subset_unfixed_subset_csr"
        if self._metadata_last_subset_total <= 0:
            return False, "delta_subset_empty_subset_csr"
        return True, "delta_subset_segmented"

    def _can_use_cuda_graph_decode(self, active_slots: list, qo_lengths: list) -> tuple[bool, str]:
        if not self.enable_cuda_graph_decode:
            return False, "disabled"
        if self.page_selector_version != "v2":
            return False, f"page_selector_{self.page_selector_version}"
        if self._compiled:
            return False, "compiled_model"
        if (self.timer is not None) and self.timer.get_timing_enabled():
            return False, "timer_enabled"
        if self._metadata_slot_block_table is None:
            return False, "metadata_buffers_missing"
        if self._decode_graph_static_tokens is None or self._decode_graph_static_input_pos is None:
            return False, "static_buffers_missing"
        if len(active_slots) != self.batch_size:
            return False, "partial_batch"
        if active_slots != self._postprocess_all_slots_list[:self.batch_size]:
            return False, "non_contiguous_slots"
        if not all(qo_len == 1 for qo_len in qo_lengths):
            return False, "non_decode_step"
        if self.decode_wrapper_full_graph is None:
            return False, "graph_wrappers_missing"
        if self.enable_selective_cache and self._using_subset_cache_this_step:
            return self._can_use_delta_subset_segment_graph()

        key = self._cuda_graph_decode_key()
        if key in self._decode_graph_capture_failed:
            return False, f"capture_failed_{key}"
        return True, key

    def _prepare_cuda_graph_decode_inputs(self, input_tokens: torch.Tensor, input_pos: torch.Tensor):
        self._decode_graph_static_tokens.copy_(input_tokens)
        self._decode_graph_static_input_pos.copy_(input_pos)

        self.page_selector.qo_indptr = self._decode_graph_qo_indptr
        self.page_selector.paged_kv_indices = self._metadata_paged_kv_indices
        self.page_selector.paged_kv_indptr = self._metadata_paged_kv_indptr
        self.page_selector.paged_kv_last_page_len = self._metadata_paged_kv_last_page_len

        if self.enable_selective_cache and self._using_subset_cache_this_step:
            self.page_selector.paged_subset_kv_indices = self._metadata_paged_subset_kv_indices
            self.page_selector.paged_subset_kv_indptr = self._metadata_paged_subset_kv_indptr
        else:
            self.page_selector.paged_subset_kv_indices = self._metadata_paged_kv_indices
            self.page_selector.paged_subset_kv_indptr = self._metadata_paged_kv_indptr

        return (
            self._decode_graph_static_tokens,
            self._decode_graph_static_input_pos,
            self._decode_graph_qo_indptr,
            self._metadata_paged_kv_indices,
            self._metadata_paged_kv_indptr,
            self._metadata_paged_kv_last_page_len,
        )

    def _should_reuse_subset_wrapper_plan(
        self,
        wrapper,
        qo_indptr,
        paged_kv_indices,
        paged_kv_indptr,
        paged_kv_last_page_len,
    ) -> bool:
        if not self.enable_delta_subset_plan_reuse:
            return False
        if not self.enable_selective_cache or not self._using_subset_cache_this_step:
            return False
        if wrapper not in (self.decode_wrapper_subset, self.decode_wrapper_subset_graph):
            return False
        if not self._metadata_last_subset_fixed:
            return False
        if qo_indptr.shape[0] != self.batch_size + 1:
            return False
        if paged_kv_indptr.shape[0] != self.batch_size + 1:
            return False
        # Wrapper-plan reuse is only safe when the metadata tensors are the
        # long-lived in-place buffers that get refreshed every step. The generic
        # page-info builder allocates fresh tensors, and reusing a previous plan
        # there can leave flashinfer reading stale qo_indptr/last_page_len state.
        static_qo_ptrs = {self._metadata_qo_indptr_decode.data_ptr()}
        if self._decode_graph_qo_indptr is not None:
            static_qo_ptrs.add(self._decode_graph_qo_indptr.data_ptr())
        if qo_indptr.data_ptr() not in static_qo_ptrs:
            return False
        if paged_kv_indptr.data_ptr() != self._metadata_paged_subset_kv_indptr.data_ptr():
            return False
        if paged_kv_last_page_len.data_ptr() != self._metadata_paged_kv_last_page_len.data_ptr():
            return False
        # subset indices may be copied into the wrapper buffer after selection,
        # so pointer equality is not required here.
        return True

    def _plan_decode_wrapper(self, wrapper, qo_indptr, paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len, wrapper_name: str):
        if self._should_reuse_subset_wrapper_plan(
            wrapper,
            qo_indptr,
            paged_kv_indices,
            paged_kv_indptr,
            paged_kv_last_page_len,
        ):
            plan_key = (
                wrapper_name,
                int(qo_indptr.shape[0]),
                int(paged_kv_indptr.shape[0]),
                int(paged_kv_indices.numel()),
                self.model.config.n_head,
                self.model.config.n_local_heads,
                self.model.config.head_dim,
                self.page_size,
                str(self.dtype),
            )
            if self._wrapper_plan_cache.get(id(wrapper)) == plan_key:
                self._delta_impl_profile_counts[f"{wrapper_name}_plan_reused"] += 1
                return
            self._wrapper_plan_cache[id(wrapper)] = plan_key
        else:
            self._wrapper_plan_cache.pop(id(wrapper), None)

        self._profile_cuda_section(
            f"{wrapper_name}_plan",
            lambda: wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.model.config.n_head,
                num_kv_heads=self.model.config.n_local_heads,
                head_dim_qk=self.model.config.head_dim,
                page_size=self.page_size,
                q_data_type=self.dtype,
                causal=True,
            ),
        )

    def _run_eager_model_forward(self, input_tokens, input_pos, qo_indptr, paged_kv_indices,
                                 paged_kv_indptr, paged_kv_last_page_len):
        return self._eager_model_forward(
            model=self.model,
            x=input_tokens,
            input_pos=input_pos,
            kv_append_indptr=qo_indptr,
            kv_page_indices=paged_kv_indices,
            kv_page_indptr=paged_kv_indptr,
            kv_page_lastlen=paged_kv_last_page_len,
        )

    def _capture_cuda_graph_decode(self, key: str, input_tokens, input_pos, qo_indptr,
                                   paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        capture_start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        self._use_cuda_graph_decode_this_step = True
        try:
            stream = torch.cuda.Stream(device=torch.device(self.device))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._run_eager_model_forward(
                        input_tokens, input_pos, qo_indptr,
                        paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len)
            torch.cuda.current_stream().wait_stream(stream)

            with torch.cuda.graph(graph):
                static_output = self._run_eager_model_forward(
                    input_tokens, input_pos, qo_indptr,
                    paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len)
        except Exception as exc:
            self._decode_graph_capture_failed.add(key)
            self._decode_graph_capture_errors[key] = repr(exc)
            self._decode_graph_stats["capture_failures"] += 1
            self._record_decode_graph_fallback(f"capture_failed_{key}")
            self._use_cuda_graph_decode_this_step = False
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            if self.is_main_process:
                print(f"[CUDA graph decode] Capture failed for {key}; falling back to eager: {exc}")
            return None
        finally:
            self._use_cuda_graph_decode_this_step = False

        self._decode_graphs[key] = graph
        self._decode_graph_outputs[key] = static_output
        self._decode_graph_capture_time_s[key] += time.perf_counter() - capture_start
        self._decode_graph_stats["captures"] += 1
        graph.replay()
        self._decode_graph_stats["replays"] += 1
        if self.is_main_process:
            print(f"[CUDA graph decode] Captured {key}")
        return static_output

    def _run_cuda_graph_decode(self, key: str, input_tokens, input_pos, qo_indptr,
                               paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        if key not in self._decode_graphs:
            return self._capture_cuda_graph_decode(
                key, input_tokens, input_pos, qo_indptr,
                paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len)

        self._use_cuda_graph_decode_this_step = True
        try:
            self._decode_graphs[key].replay()
        except Exception as exc:
            self._decode_graph_capture_failed.add(key)
            self._decode_graph_capture_errors[key] = repr(exc)
            self._decode_graph_stats["replay_failures"] += 1
            self._record_decode_graph_fallback(f"replay_failed_{key}")
            if self.is_main_process:
                print(f"[CUDA graph decode] Replay failed for {key}; falling back to eager: {exc}")
            return None
        finally:
            self._use_cuda_graph_decode_this_step = False
        self._decode_graph_stats["replays"] += 1
        return self._decode_graph_outputs[key]

    def _record_delta_subset_segment_fallback(self, reason: str):
        self._delta_subset_segment_fallback_reasons[reason] += 1

    def _execute_delta_subset_graph_op(self, spec: dict, op_input, input_pos, qo_indptr,
                                       paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        kind = spec["kind"]
        if kind == "segment":
            if spec["include_embed"]:
                x = self.model.forward_decode_embed_layers(
                    op_input,
                    input_pos,
                    qo_indptr,
                    paged_kv_indices,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                    spec["start"],
                    spec["end"],
                )
            else:
                x = self.model.forward_decode_layers(
                    op_input,
                    input_pos,
                    qo_indptr,
                    paged_kv_indices,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                    spec["start"],
                    spec["end"],
                )
            if spec["include_tail"]:
                return self.model.forward_decode_finish(x)
            return x

        if kind == "planner_pre":
            return self.model.forward_decode_planner_pre(
                op_input,
                input_pos,
                qo_indptr,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
                spec["layer"],
            )

        if kind == "planner_post":
            residual, attn_out = op_input
            return self.model.forward_decode_planner_post(
                residual,
                attn_out,
                spec["layer"],
            )

        raise RuntimeError(f"Unsupported DELTA subset graph op kind: {kind}")

    def _run_delta_subset_graph_op_eager(self, spec: dict, op_input, input_pos, qo_indptr,
                                         paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        prev = self._use_cuda_graph_decode_this_step
        self._use_cuda_graph_decode_this_step = True
        try:
            return self._execute_delta_subset_graph_op(
                spec,
                op_input,
                input_pos,
                qo_indptr,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
            )
        finally:
            self._use_cuda_graph_decode_this_step = prev

    def _allocate_delta_subset_static_value(self, value):
        if torch.is_tensor(value):
            return torch.empty_like(value)
        if isinstance(value, tuple):
            return tuple(self._allocate_delta_subset_static_value(v) for v in value)
        raise TypeError(f"Unsupported DELTA subset static value type: {type(value)}")

    def _copy_delta_subset_static_value(self, key: str, static_value, value):
        if torch.is_tensor(value):
            if not torch.is_tensor(static_value):
                raise RuntimeError(f"Static input type mismatch for {key}")
            if static_value.shape != value.shape or static_value.dtype != value.dtype:
                raise RuntimeError(
                    f"Static input mismatch for {key}: have {tuple(static_value.shape)} {static_value.dtype}, "
                    f"got {tuple(value.shape)} {value.dtype}"
                )
            static_value.copy_(value)
            return static_value
        if isinstance(value, tuple):
            if not isinstance(static_value, tuple) or len(static_value) != len(value):
                raise RuntimeError(f"Static tuple mismatch for {key}")
            return tuple(
                self._copy_delta_subset_static_value(f"{key}[{idx}]", static_value[idx], value[idx])
                for idx in range(len(value))
            )
        raise TypeError(f"Unsupported DELTA subset static value type: {type(value)}")

    def _get_delta_subset_graph_static_input(self, key: str, value):
        static_input = self._delta_subset_segment_inputs.get(key)
        if static_input is None:
            static_input = self._allocate_delta_subset_static_value(value)
            self._delta_subset_segment_inputs[key] = static_input
        return self._copy_delta_subset_static_value(key, static_input, value)

    def _capture_delta_subset_segment_graph(self, spec: dict, op_input, input_pos, qo_indptr,
                                            paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        key = spec["key"]
        capture_start = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        prev = self._use_cuda_graph_decode_this_step
        self._use_cuda_graph_decode_this_step = True
        try:
            stream = torch.cuda.Stream(device=torch.device(self.device))
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._execute_delta_subset_graph_op(
                        spec,
                        op_input,
                        input_pos,
                        qo_indptr,
                        paged_kv_indices,
                        paged_kv_indptr,
                        paged_kv_last_page_len,
                    )
            torch.cuda.current_stream().wait_stream(stream)

            with torch.cuda.graph(graph):
                static_output = self._execute_delta_subset_graph_op(
                    spec,
                    op_input,
                    input_pos,
                    qo_indptr,
                    paged_kv_indices,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                )
        except Exception as exc:
            self._delta_subset_segment_failed.add(key)
            self._delta_subset_segment_errors[key] = repr(exc)
            self._delta_subset_segment_stats["capture_failures"] += 1
            self._record_delta_subset_segment_fallback(f"capture_failed_{key}")
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            if self.is_main_process:
                print(f"[CUDA graph decode] DELTA subset segment capture failed for {key}; using eager segment: {exc}")
            return None
        finally:
            self._use_cuda_graph_decode_this_step = prev

        self._delta_subset_segment_graphs[key] = graph
        self._delta_subset_segment_outputs[key] = static_output
        self._delta_subset_segment_capture_time_s[key] += time.perf_counter() - capture_start
        self._delta_subset_segment_stats["captures"] += 1
        graph.replay()
        self._delta_subset_segment_stats["replays"] += 1
        if self.is_main_process:
            print(f"[CUDA graph decode] Captured DELTA subset segment {key}")
        return static_output

    def _run_delta_subset_graph_op(self, spec: dict, op_input, input_pos, qo_indptr,
                                   paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        key = spec["key"]
        if key in self._delta_subset_segment_failed:
            self._record_delta_subset_segment_fallback(f"segment_disabled_{key}")
            return self._run_delta_subset_graph_op_eager(
                spec,
                op_input,
                input_pos,
                qo_indptr,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
            )

        if spec["kind"] == "segment" and spec["include_embed"]:
            static_input = op_input
        else:
            try:
                static_input = self._get_delta_subset_graph_static_input(key, op_input)
            except Exception as exc:
                self._delta_subset_segment_failed.add(key)
                self._delta_subset_segment_errors[key] = repr(exc)
                self._record_delta_subset_segment_fallback(f"static_input_failed_{key}")
                return self._run_delta_subset_graph_op_eager(
                    spec,
                    op_input,
                    input_pos,
                    qo_indptr,
                    paged_kv_indices,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                )

        if key not in self._delta_subset_segment_graphs:
            output = self._capture_delta_subset_segment_graph(
                spec,
                static_input,
                input_pos,
                qo_indptr,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
            )
            if output is None:
                return self._run_delta_subset_graph_op_eager(
                    spec,
                    op_input,
                    input_pos,
                    qo_indptr,
                    paged_kv_indices,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                )
            return output

        prev = self._use_cuda_graph_decode_this_step
        self._use_cuda_graph_decode_this_step = True
        try:
            self._delta_subset_segment_graphs[key].replay()
        except Exception as exc:
            self._delta_subset_segment_failed.add(key)
            self._delta_subset_segment_errors[key] = repr(exc)
            self._delta_subset_segment_stats["replay_failures"] += 1
            self._record_delta_subset_segment_fallback(f"replay_failed_{key}")
            if self.is_main_process:
                print(f"[CUDA graph decode] DELTA subset segment replay failed for {key}; using eager segment: {exc}")
            return self._run_delta_subset_graph_op_eager(
                spec,
                op_input,
                input_pos,
                qo_indptr,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
            )
        finally:
            self._use_cuda_graph_decode_this_step = prev

        self._delta_subset_segment_stats["replays"] += 1
        return self._delta_subset_segment_outputs[key]

    def _run_delta_subset_planner_attention(self, layer_idx: int, planner_state):
        residual, q = planner_state
        prev = self._use_cuda_graph_decode_this_step
        self._use_cuda_graph_decode_this_step = False
        try:
            return self._profile_cuda_section(
                "delta_planner_attention_mid",
                lambda: self.model.forward_decode_planner_attention(
                    q,
                    layer_idx,
                    residual.shape[0],
                    residual.shape[1],
                ),
            )
        finally:
            self._use_cuda_graph_decode_this_step = prev

    def _run_delta_subset_segmented_cuda_graph(self, input_tokens, input_pos, qo_indptr,
                                               paged_kv_indices, paged_kv_indptr, paged_kv_last_page_len):
        self._delta_subset_segment_stats["steps"] += 1
        x = None
        for op in self._delta_subset_segment_plan:
            if op["kind"] == "planner_attention":
                if x is None:
                    raise RuntimeError(f"Planner layer {op['layer']} has no hidden-state input")
                x = (
                    x[0],
                    self._run_delta_subset_planner_attention(
                        op["layer"],
                        x,
                    ),
                )
                continue

            op_input = input_tokens if op["kind"] == "segment" and op["include_embed"] else x
            if op_input is None:
                raise RuntimeError(f"Segment {op['key']} has no input")
            x = self._run_delta_subset_graph_op(
                op,
                op_input,
                input_pos,
                qo_indptr,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
            )
        return x
    
    def add_requests(self, input_ids_list: list, max_lengths: list, eot_token_ids: list):
        if self.scheduler is None:
            raise RuntimeError("Backend not initialized. Call setup_caches first.")
        self.scheduler.add_requests(input_ids_list, max_lengths, eot_token_ids)

    @torch.inference_mode()
    def run_scheduler_loop(self):
        """Main scheduler loop - unified encode/decode"""
        torch.cuda.synchronize()
        self.scheduler.stats['start_time'] = time.perf_counter()
        self._step_records = []
        
        if self.is_main_process:
            total_requests = len(self.scheduler.request_queue)
            request_progress = tqdm(total=total_requests, desc="Requests", unit="req")
            step_progress = tqdm(desc="Steps", unit="step")
        i = 0
        while True:
            i += 1
            finished_slots = self.scheduler.check_finished_requests()

            if finished_slots and self.throughput_mode:
                if self.is_main_process:
                    print(f"\n[throughput_mode] {len(finished_slots)} request(s) finished — stopping all to preserve fixed batch size")
                self._force_finish_all_requests()
                break

            for slot_id in finished_slots:
                self.cachelens[slot_id].fill_(0)

            newly_scheduled_slots, newly_scheduled_requests = self.scheduler.schedule_requests()
            if self.enable_selective_cache and (finished_slots or newly_scheduled_slots):
                self.reset_attention_weights_buffer()
            for slot_id, request in zip(newly_scheduled_slots, newly_scheduled_requests):
                request.initialize_pages(slot_id, self.pages_per_slot)
                self._sync_slot_block_table(slot_id, request, 0)
            
            if not self.scheduler.has_active_requests():
                break
            
            self._forward()

            torch.cuda.synchronize()
            self._flush_delta_impl_profile_events()
            for ev_dict in self._pending_events:
                timing_names = ["collect_pages", "plan", "model_forward", "postprocess", "step_total"]
                record = {"step": i}
                for name in timing_names:
                    start_ev, end_ev = ev_dict[name]
                    record[name + "_ms"] = start_ev.elapsed_time(end_ev)
                record["active_bsz"] = ev_dict["active_bsz"]
                record["total_pages"] = ev_dict["total_pages"]
                self._step_records.append(record)
            self._pending_events.clear()
            
            if self.is_main_process:
                finished_count = sum(1 for r in self.scheduler.request_queue if r.is_finished())
                request_progress.n = finished_count
                request_progress.refresh()
                
                total_pages = self.page_selector.paged_kv_indptr[-1]
                if self.enable_selective_cache:
                    sel_pages = self.page_selector.paged_subset_kv_indptr[-1]
                    step_progress.set_description(f"Steps (Pages: {sel_pages}/{total_pages})")
                else:
                    step_progress.set_description(f"Steps (Pages: {total_pages})")
                step_progress.update(1)

        if self.is_main_process:
            request_progress.close()
            step_progress.close()
            self._print_final_stats()
        
        return self.get_detailed_results()

    def _force_finish_all_requests(self):
        for slot_id in range(self.batch_size):
            req_id = self.scheduler.runner_slots[slot_id]
            if req_id != -1:
                request = self.scheduler.request_queue[req_id]
                if not request.is_finished():
                    request.status = "finished"
                self.scheduler.runner_slots[slot_id] = -1
                self.cachelens[slot_id].fill_(0)

    def _record_event(self):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        return ev

    def _profile_cuda_section(self, name: str, fn):
        if not self.enable_delta_impl_profile:
            return fn()
        start = self._record_event()
        out = fn()
        end = self._record_event()
        self._delta_impl_profile_pending.append((name, start, end))
        self._delta_impl_profile_counts[name] += 1
        return out

    def _flush_delta_impl_profile_events(self):
        if not self._delta_impl_profile_pending:
            return
        for name, start, end in self._delta_impl_profile_pending:
            self._delta_impl_profile_ms[name] += start.elapsed_time(end)
        self._delta_impl_profile_pending.clear()

    def _can_use_baseline_static_decode(self, active_slots: list) -> bool:
        if not self._cuda_graph_mode or self._baseline_compiled_model_forward is None:
            return False
        if len(active_slots) != self.batch_size:
            return False
        return all(
            len(self.scheduler.request_queue[self.scheduler.runner_slots[slot_id]].output_ids) > 0
            for slot_id in active_slots
        )

    def _collect_page_info_for_baseline_static_decode(self, active_slots: list):
        full_kv_indices = []
        total_pages = 0

        self._baseline_static_decode_input_pos.copy_(self.cachelens)
        self._baseline_static_qo_indptr[0] = 0
        self._baseline_static_qo_indptr[1:] = torch.arange(
            1, self.batch_size + 1, dtype=torch.int32, device=self.device
        )
        self._baseline_static_paged_kv_indptr[0] = 0

        for row_idx, slot_id in enumerate(active_slots):
            request = self.scheduler.request_queue[self.scheduler.runner_slots[slot_id]]
            request_qo_tokens = request.get_qo_tokens().to(self.device)
            if request_qo_tokens.numel() != 1:
                raise RuntimeError("Static baseline decode expects exactly one token per active slot.")

            request.plan(1, self.page_size)
            self._baseline_static_decode_tokens[0, row_idx] = request_qo_tokens[0]

            page_count = len(request.page_indices)
            full_kv_indices.extend(request.page_indices)
            total_pages += page_count
            self._baseline_static_paged_kv_indptr[row_idx + 1] = total_pages
            self._baseline_static_paged_kv_last_page_len[row_idx] = request.page_lastlen

        if total_pages > 0:
            indices_tensor = torch.tensor(full_kv_indices, dtype=torch.int32, device=self.device)
            self._baseline_static_paged_kv_indices[:total_pages].copy_(indices_tensor)

        pages_per_slot = self._baseline_static_paged_kv_indptr[1:] - self._baseline_static_paged_kv_indptr[:-1]
        self.scheduler.max_pages = int(pages_per_slot.max().item())
        self.scheduler.active_bsz = len(active_slots)

        # Keep progress reporting and wrapper planning pointed at the static buffers.
        self.page_selector.qo_indptr = self._baseline_static_qo_indptr
        self.page_selector.paged_kv_indices = self._baseline_static_paged_kv_indices
        self.page_selector.paged_kv_indptr = self._baseline_static_paged_kv_indptr
        self.page_selector.paged_kv_last_page_len = self._baseline_static_paged_kv_last_page_len
        self.page_selector.paged_subset_kv_indices = self._baseline_static_paged_kv_indices
        self.page_selector.paged_subset_kv_indptr = self._baseline_static_paged_kv_indptr

        return self._baseline_static_decode_tokens, [1] * len(active_slots)

    @torch.inference_mode()
    def _forward(self):
        """Single forward pass handling both new and continuing requests"""
        
        active_slots = []
        for slot_id in range(self.batch_size):
            request_id = self.scheduler.runner_slots[slot_id]
            if request_id != -1:
                active_slots.append(slot_id)
        if not active_slots:
            return

        ev_step_start = self._record_event()
        self._using_subset_cache_this_step = False

        use_static_decode = self._can_use_baseline_static_decode(active_slots)
        if use_static_decode:
            input_tokens, qo_lengths = self._collect_page_info_for_baseline_static_decode(active_slots)
            qo_indptr = self._baseline_static_qo_indptr
            paged_kv_indices = self._baseline_static_paged_kv_indices
            paged_kv_indptr = self._baseline_static_paged_kv_indptr
            paged_kv_last_page_len = self._baseline_static_paged_kv_last_page_len
            input_pos = self._baseline_static_decode_input_pos
        else:
            input_tokens, qo_lengths, input_pos = self._collect_page_info_for_slots(active_slots)
            qo_indptr = self.page_selector.qo_indptr
            paged_kv_indices = self.page_selector.paged_kv_indices
            paged_kv_indptr = self.page_selector.paged_kv_indptr
            paged_kv_last_page_len = self.page_selector.paged_kv_last_page_len

        ev_collect_done = self._record_event()

        use_cuda_graph_decode, graph_key_or_reason = self._can_use_cuda_graph_decode(
            active_slots, qo_lengths)
        if use_cuda_graph_decode:
            (
                input_tokens,
                input_pos,
                qo_indptr,
                paged_kv_indices,
                paged_kv_indptr,
                paged_kv_last_page_len,
            ) = self._prepare_cuda_graph_decode_inputs(input_tokens, input_pos)
            full_wrapper = self.decode_wrapper_full_graph
            full_dump_wrapper = (
                self.decode_wrapper_full_dump
                if graph_key_or_reason == "delta_subset_segmented"
                else self.decode_wrapper_full_dump_graph
            )
            subset_wrapper = self.decode_wrapper_subset_graph
        else:
            if self.enable_cuda_graph_decode:
                self._record_decode_graph_fallback(graph_key_or_reason)
            full_wrapper = self.decode_wrapper_full
            full_dump_wrapper = self.decode_wrapper_full_dump
            subset_wrapper = self.decode_wrapper_subset

        self._plan_decode_wrapper(
            full_wrapper, qo_indptr, paged_kv_indices, paged_kv_indptr,
            paged_kv_last_page_len, "decode_wrapper_full")
        
        if self.enable_selective_cache and self._using_subset_cache_this_step:
            self._plan_decode_wrapper(
                full_dump_wrapper, qo_indptr, paged_kv_indices, paged_kv_indptr,
                paged_kv_last_page_len, "decode_wrapper_full_dump")
            self._plan_decode_wrapper(
                subset_wrapper,
                self.page_selector.qo_indptr,
                self.page_selector.paged_subset_kv_indices,
                self.page_selector.paged_subset_kv_indptr,
                self.page_selector.paged_kv_last_page_len,
                "decode_wrapper_subset",
            )

        ev_plan_done = self._record_event()

        if use_cuda_graph_decode:
            if graph_key_or_reason == "delta_subset_segmented":
                tokens = self._run_delta_subset_segmented_cuda_graph(
                    input_tokens,
                    input_pos,
                    qo_indptr,
                    paged_kv_indices,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                )
            else:
                tokens = self._run_cuda_graph_decode(
                    graph_key_or_reason,
                    input_tokens,
                    input_pos,
                    qo_indptr,
                    paged_kv_indices,
                    paged_kv_indptr,
                    paged_kv_last_page_len,
                )
                if tokens is None:
                    full_wrapper = self.decode_wrapper_full
                    full_dump_wrapper = self.decode_wrapper_full_dump
                    subset_wrapper = self.decode_wrapper_subset
                    self._plan_decode_wrapper(
                        full_wrapper, qo_indptr, paged_kv_indices, paged_kv_indptr,
                        paged_kv_last_page_len, "decode_wrapper_full")
                    if self.enable_selective_cache and self._using_subset_cache_this_step:
                        self._plan_decode_wrapper(
                            full_dump_wrapper, qo_indptr, paged_kv_indices, paged_kv_indptr,
                            paged_kv_last_page_len, "decode_wrapper_full_dump")
                        self._plan_decode_wrapper(
                            subset_wrapper,
                            self.page_selector.qo_indptr,
                            self.page_selector.paged_subset_kv_indices,
                            self.page_selector.paged_subset_kv_indptr,
                            self.page_selector.paged_kv_last_page_len,
                            "decode_wrapper_subset",
                        )
                    tokens = self._run_eager_model_forward(
                        input_tokens, input_pos, qo_indptr, paged_kv_indices,
                        paged_kv_indptr, paged_kv_last_page_len)
        else:
            if use_static_decode:
                forward_fn = self._baseline_compiled_model_forward
            elif self._cuda_graph_mode:
                # Keep baseline prefill eager; only steady-state decode uses the static compiled path.
                forward_fn = self._eager_model_forward
            else:
                forward_fn = self.model_forward

            tokens = forward_fn(
                model=self.model,
                x=input_tokens,
                input_pos=input_pos,
                kv_append_indptr=qo_indptr,
                kv_page_indices=paged_kv_indices,
                kv_page_indptr=paged_kv_indptr,
                kv_page_lastlen=paged_kv_last_page_len
            )

        ev_model_done = self._record_event()
        
        self._postprocess_forward_outputs(active_slots, qo_lengths, qo_indptr, tokens)

        ev_step_end = self._record_event()

        self._pending_events.append({
            "collect_pages": (ev_step_start, ev_collect_done),
            "plan":          (ev_collect_done, ev_plan_done),
            "model_forward": (ev_plan_done, ev_model_done),
            "postprocess":   (ev_model_done, ev_step_end),
            "step_total":    (ev_step_start, ev_step_end),
            "active_bsz":    len(active_slots),
            "total_pages":   int(paged_kv_indptr[len(active_slots)]),
        })

    def _postprocess_forward_outputs(self, active_slots: list, qo_lengths: list, qo_indptr: torch.Tensor, tokens: torch.Tensor):
        active_bsz = len(active_slots)

        if active_slots == self._postprocess_all_slots_list[:active_bsz]:
            active_slot_ids = self._postprocess_all_slots[:active_bsz]
        else:
            for row_idx, slot_id in enumerate(active_slots):
                self._postprocess_slot_ids_host[row_idx] = slot_id
            self._postprocess_slot_ids[:active_bsz].copy_(
                self._postprocess_slot_ids_host[:active_bsz], non_blocking=True)
            active_slot_ids = self._postprocess_slot_ids[:active_bsz]

        if all(qo_len == 1 for qo_len in qo_lengths):
            qo_lengths_dev = self._postprocess_ones[:active_bsz]
        else:
            for row_idx, qo_len in enumerate(qo_lengths):
                self._postprocess_qo_lengths_host[row_idx] = qo_len
            self._postprocess_qo_lengths[:active_bsz].copy_(
                self._postprocess_qo_lengths_host[:active_bsz], non_blocking=True)
            qo_lengths_dev = self._postprocess_qo_lengths[:active_bsz]

        self.cachelens.index_add_(0, active_slot_ids, qo_lengths_dev)

        self._postprocess_last_token_positions[:active_bsz].copy_(qo_indptr[1:active_bsz + 1])
        self._postprocess_last_token_positions[:active_bsz].sub_(1)
        torch.index_select(
            tokens[0],
            0,
            self._postprocess_last_token_positions[:active_bsz],
            out=self._postprocess_next_tokens[:active_bsz],
        )
        self._postprocess_next_tokens_host[:active_bsz].copy_(
            self._postprocess_next_tokens[:active_bsz])
        next_tokens = self._postprocess_next_tokens_host[:active_bsz].tolist()

        for slot_id, next_token in zip(active_slots, next_tokens):
            request = self.scheduler.request_queue[self.scheduler.runner_slots[slot_id]]
            request.append_output(next_token)

        self.scheduler.stats['total_tokens_generated'] += sum(qo_lengths)

    def _sync_slot_block_table(self, slot_id: int, request: Request, old_page_count: int):
        if self._metadata_slot_block_table is None:
            return
        new_page_count = len(request.page_indices)
        if new_page_count <= old_page_count:
            return
        new_pages = torch.tensor(
            request.page_indices[old_page_count:new_page_count],
            dtype=torch.int32,
            device=self.device,
        )
        self._metadata_slot_block_table[
            slot_id, old_page_count:new_page_count
        ].copy_(new_pages)

    def _build_generic_page_info_snapshot(self, active_slots: list, *, already_planned: bool = False):
        all_qo_tokens = []
        qo_lengths = []
        full_kv_indices = []
        subset_kv_indices = []
        full_kv_indptr = [0]
        subset_kv_indptr = [0]
        lastlens = []

        for slot_id in active_slots:
            request = self.scheduler.request_queue[self.scheduler.runner_slots[slot_id]]
            request_qo_tokens = request.get_qo_tokens().to(self.device)
            if not already_planned:
                old_page_count = len(request.page_indices)
                request.plan(len(request_qo_tokens), self.page_size)
                self._sync_slot_block_table(slot_id, request, old_page_count)
            all_qo_tokens.append(request_qo_tokens)
            qo_lengths.append(len(request_qo_tokens))
            full_kv_indices.extend(request.page_indices)
            full_kv_indptr.append(full_kv_indptr[-1] + len(request.page_indices))

            if len(request_qo_tokens) == 1 and len(request.page_indices) > self.subset_cache_size:
                subset_kv_indices.extend(request.page_indices[-self.subset_cache_size:])
                subset_kv_indptr.append(subset_kv_indptr[-1] + self.subset_cache_size)
            else:
                subset_kv_indices.extend(request.page_indices)
                subset_kv_indptr.append(subset_kv_indptr[-1] + len(request.page_indices))

            lastlens.append(request.page_lastlen)

        active_bsz = len(active_slots)
        qo_indptr = torch.zeros(active_bsz + 1, dtype=torch.int32, device=self.device)
        if qo_lengths:
            qo_indptr[1:] = torch.cumsum(torch.tensor(qo_lengths, dtype=torch.int32), dim=0)

        paged_kv_indices = torch.tensor(full_kv_indices, dtype=torch.int32, device=self.device)
        paged_kv_indptr = torch.tensor(full_kv_indptr, dtype=torch.int32, device=self.device)
        paged_kv_last_page_len = torch.tensor(lastlens, dtype=torch.int32, device=self.device)

        pages_per_slot = paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        max_pages = int(pages_per_slot.max().item()) if active_bsz > 0 else 0
        subset_total = subset_kv_indptr[-1]
        full_total = full_kv_indptr[-1]
        use_subset = (
            self.enable_selective_cache
            and subset_total > 0
            and (full_total / subset_total) > self.compression_ratio
        )
        if use_subset:
            paged_subset_kv_indices = torch.tensor(subset_kv_indices, dtype=torch.int32, device=self.device)
            paged_subset_kv_indptr = torch.tensor(subset_kv_indptr, dtype=torch.int32, device=self.device)
        else:
            paged_subset_kv_indices = paged_kv_indices
            paged_subset_kv_indptr = paged_kv_indptr

        block_table = None
        if self.page_selector_version == "v2" and self.enable_selective_cache:
            block_table = torch.zeros(
                (active_bsz, self.pages_per_slot), dtype=torch.int32, device=self.device
            )
            for row_idx in range(active_bsz):
                start = int(full_kv_indptr[row_idx])
                end = int(full_kv_indptr[row_idx + 1])
                count = end - start
                if count > 0:
                    block_table[row_idx, :count].copy_(paged_kv_indices[start:end])

        return {
            "input_tokens": torch.cat(all_qo_tokens, dim=0).unsqueeze(0),
            "qo_lengths": qo_lengths,
            "input_pos": self.cachelens[active_slots].clone(),
            "qo_indptr": qo_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_last_page_len": paged_kv_last_page_len,
            "paged_subset_kv_indices": paged_subset_kv_indices,
            "paged_subset_kv_indptr": paged_subset_kv_indptr,
            "using_subset_cache": use_subset,
            "metadata_last_full_total": full_total,
            "metadata_last_subset_total": subset_total,
            "metadata_last_subset_fixed": (
                active_bsz == self.batch_size and subset_total == active_bsz * self.subset_cache_size
            ),
            "scheduler_max_pages": max_pages,
            "scheduler_active_bsz": active_bsz,
            "block_table": block_table,
        }

    def _apply_generic_page_info_snapshot(self, snapshot):
        self.scheduler.max_pages = snapshot["scheduler_max_pages"]
        self.scheduler.active_bsz = snapshot["scheduler_active_bsz"]
        self._using_subset_cache_this_step = snapshot["using_subset_cache"]
        self._metadata_last_full_total = snapshot["metadata_last_full_total"]
        self._metadata_last_subset_total = snapshot["metadata_last_subset_total"]
        self._metadata_last_subset_fixed = snapshot["metadata_last_subset_fixed"]

        if self.page_selector_version == "v2":
            self.page_selector.plan(
                snapshot["qo_indptr"],
                snapshot["paged_kv_indices"],
                snapshot["paged_kv_indptr"],
                snapshot["paged_kv_last_page_len"],
                snapshot["paged_subset_kv_indices"],
                snapshot["paged_subset_kv_indptr"],
                block_table_prebuilt=False,
            )
        else:
            self.page_selector.plan(
                snapshot["qo_indptr"],
                snapshot["paged_kv_indices"],
                snapshot["paged_kv_indptr"],
                snapshot["paged_kv_last_page_len"],
                snapshot["paged_subset_kv_indices"],
                snapshot["paged_subset_kv_indptr"],
            )

    def _debug_check_fast_decode_page_info_parity(self, active_slots: list, input_tokens, input_pos):
        ref = self._build_generic_page_info_snapshot(active_slots, already_planned=True)
        active_bsz = len(active_slots)

        checks = [
            ("input_tokens", input_tokens, ref["input_tokens"]),
            ("input_pos", input_pos, ref["input_pos"]),
            ("qo_indptr", self._metadata_qo_indptr_decode[:active_bsz + 1], ref["qo_indptr"]),
            ("paged_kv_indices", self.page_selector.paged_kv_indices[:self._metadata_last_full_total], ref["paged_kv_indices"]),
            ("paged_kv_indptr", self.page_selector.paged_kv_indptr[:active_bsz + 1], ref["paged_kv_indptr"]),
            ("paged_kv_last_page_len", self.page_selector.paged_kv_last_page_len[:active_bsz], ref["paged_kv_last_page_len"]),
        ]

        if self._using_subset_cache_this_step != ref["using_subset_cache"]:
            raise RuntimeError(
                "DELTA fast decode page-info mismatch: using_subset_cache "
                f"{self._using_subset_cache_this_step} != {ref['using_subset_cache']}"
            )

        if self._using_subset_cache_this_step:
            checks.extend(
                [
                    (
                        "paged_subset_kv_indices",
                        self.page_selector.paged_subset_kv_indices[:self._metadata_last_subset_total],
                        ref["paged_subset_kv_indices"],
                    ),
                    (
                        "paged_subset_kv_indptr",
                        self.page_selector.paged_subset_kv_indptr[:active_bsz + 1],
                        ref["paged_subset_kv_indptr"],
                    ),
                ]
            )

        if self.page_selector_version == "v2" and ref["block_table"] is not None:
            checks.append(
                (
                    "block_table",
                    self.page_selector._block_table[:active_bsz, :self.scheduler.max_pages],
                    ref["block_table"][:active_bsz, :ref["scheduler_max_pages"]],
                )
            )

        for name, actual, expected in checks:
            if not torch.equal(actual, expected):
                raise RuntimeError(f"DELTA fast decode page-info mismatch in {name}")

    def _can_use_fast_decode_page_info(self, active_slots: list) -> bool:
        if self._cuda_graph_mode:
            return False
        if not self.enable_delta_fast_decode_page_info:
            return False
        if self.page_selector_version != "v2":
            return False
        if self._metadata_slot_block_table is None:
            return False
        return all(
            len(self.scheduler.request_queue[self.scheduler.runner_slots[slot_id]].output_ids) > 0
            for slot_id in active_slots
        )

    def _copy_metadata_host_inputs(self, active_bsz: int):
        self._metadata_active_slots[:active_bsz].copy_(
            self._metadata_active_slots_host[:active_bsz], non_blocking=True)
        self._metadata_page_counts[:active_bsz].copy_(
            self._metadata_page_counts_host[:active_bsz], non_blocking=True)
        self._metadata_subset_counts[:active_bsz].copy_(
            self._metadata_subset_counts_host[:active_bsz], non_blocking=True)
        self._metadata_subset_starts[:active_bsz].copy_(
            self._metadata_subset_starts_host[:active_bsz], non_blocking=True)
        self._metadata_paged_kv_last_page_len[:active_bsz].copy_(
            self._metadata_lastlens_host[:active_bsz], non_blocking=True)

    def _collect_decode_page_info_fast(self, active_slots: list):
        active_bsz = len(active_slots)
        full_total = 0
        subset_total = 0
        max_pages = 0
        max_subset_pages = 0

        for row_idx, slot_id in enumerate(active_slots):
            request = self.scheduler.request_queue[self.scheduler.runner_slots[slot_id]]
            old_page_count = len(request.page_indices)
            request.plan(1, self.page_size)
            self._sync_slot_block_table(slot_id, request, old_page_count)

            page_count = len(request.page_indices)
            subset_count = min(page_count, self.subset_cache_size)
            subset_start = page_count - subset_count

            self._metadata_active_slots_host[row_idx] = slot_id
            self._metadata_page_counts_host[row_idx] = page_count
            self._metadata_subset_counts_host[row_idx] = subset_count
            self._metadata_subset_starts_host[row_idx] = subset_start
            self._metadata_lastlens_host[row_idx] = request.page_lastlen
            self._metadata_decode_tokens_host[row_idx] = request.output_ids[-1]

            full_total += page_count
            subset_total += subset_count
            max_pages = max(max_pages, page_count)
            max_subset_pages = max(max_subset_pages, subset_count)

        self._metadata_decode_token_buf[0, :active_bsz].copy_(
            self._metadata_decode_tokens_host[:active_bsz], non_blocking=True)
        self._copy_metadata_host_inputs(active_bsz)

        self._metadata_paged_kv_indptr[0] = 0
        torch.cumsum(
            self._metadata_page_counts[:active_bsz],
            dim=0,
            out=self._metadata_paged_kv_indptr[1:active_bsz + 1],
        )

        row_block_table = (
            self.page_selector._block_table
            if self.enable_selective_cache
            else self._metadata_paged_kv_indices
        )
        if self.enable_selective_cache and max_pages > 0:
            # The fast v2 path reuses row-major storage across steps while the
            # active-slot ordering can change. Clear the visible slice before
            # repacking so shorter rows do not inherit stale logical page IDs
            # from a previous row assignment.
            row_block_table[:active_bsz, :max_pages].zero_()
        pack_page_indices(
            self._metadata_slot_block_table,
            self._metadata_active_slots,
            self._metadata_zero_starts,
            self._metadata_page_counts,
            self._metadata_paged_kv_indptr,
            self._metadata_paged_kv_indices,
            row_block_table,
            active_bsz,
            max_pages,
            write_row_block=self.enable_selective_cache,
        )

        use_subset = (
            self.enable_selective_cache
            and subset_total > 0
            and (full_total / subset_total) > self.compression_ratio
        )
        self._using_subset_cache_this_step = use_subset
        self._metadata_last_full_total = full_total
        self._metadata_last_subset_total = subset_total
        self._metadata_last_subset_fixed = (
            active_bsz == self.batch_size
            and subset_total == active_bsz * self.subset_cache_size
        )

        full_indices = self._metadata_paged_kv_indices[:full_total]
        full_indptr = self._metadata_paged_kv_indptr[:active_bsz + 1]
        last_page_len = self._metadata_paged_kv_last_page_len[:active_bsz]

        if use_subset:
            if self._metadata_last_subset_fixed:
                self._metadata_paged_subset_kv_indptr[:active_bsz + 1].copy_(
                    self._metadata_subset_fixed_indptr[:active_bsz + 1]
                )
            else:
                self._metadata_paged_subset_kv_indptr[0] = 0
                torch.cumsum(
                    self._metadata_subset_counts[:active_bsz],
                    dim=0,
                    out=self._metadata_paged_subset_kv_indptr[1:active_bsz + 1],
                )
            pack_page_indices(
                self._metadata_slot_block_table,
                self._metadata_active_slots,
                self._metadata_subset_starts,
                self._metadata_subset_counts,
                self._metadata_paged_subset_kv_indptr,
                self._metadata_paged_subset_kv_indices,
                self.page_selector._block_table,
                active_bsz,
                max_subset_pages,
                write_row_block=False,
            )
            subset_indices = self._metadata_paged_subset_kv_indices[:subset_total]
            subset_indptr = self._metadata_paged_subset_kv_indptr[:active_bsz + 1]
        else:
            subset_indices = full_indices
            subset_indptr = full_indptr

        self.scheduler.max_pages = max_pages
        self.scheduler.active_bsz = active_bsz

        self.page_selector.plan(
            self._metadata_qo_indptr_decode[:active_bsz + 1],
            full_indices,
            full_indptr,
            last_page_len,
            subset_indices,
            subset_indptr,
            block_table_prebuilt=True,
        )

        input_pos = self.cachelens[active_slots]
        if self.enable_delta_debug_fast_decode_page_info_parity:
            self._debug_check_fast_decode_page_info_parity(
                active_slots,
                self._metadata_decode_token_buf[:, :active_bsz].clone(),
                input_pos.clone(),
            )

        return self._metadata_decode_token_buf[:, :active_bsz], [1] * active_bsz, input_pos

    def _collect_page_info_for_slots(self, active_slots: list):
        """Collect page info only for specified slots"""
        if self._can_use_fast_decode_page_info(active_slots):
            return self._collect_decode_page_info_fast(active_slots)
        snapshot = self._build_generic_page_info_snapshot(active_slots)
        self._apply_generic_page_info_snapshot(snapshot)
        return snapshot["input_tokens"], snapshot["qo_lengths"], snapshot["input_pos"]

    def _print_final_stats(self):
        if not self.is_main_process:
            return
        torch.cuda.synchronize()
        total_time = time.perf_counter() - self.scheduler.stats['start_time']
        finished_requests = [r for r in self.scheduler.request_queue if r.is_finished()]
        
        print(f"\n{'='*60}")
        print(f"SCHEDULER PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Total requests: {len(self.scheduler.request_queue)}")
        print(f"Finished requests: {len(finished_requests)}")
        total_tokens = self.scheduler.stats['total_tokens_generated']
        print(f"Total tokens: {total_tokens:,}")
        print(f"Tokens per second: {total_tokens / total_time:.2f}")

        if self.enable_cuda_graph_decode:
            print("\nCUDA Graph Decode:")
            print(f"  captures: {self._decode_graph_stats.get('captures', 0)}")
            print(f"  replays: {self._decode_graph_stats.get('replays', 0)}")
            print(f"  capture failures: {self._decode_graph_stats.get('capture_failures', 0)}")
            print(f"  replay failures: {self._decode_graph_stats.get('replay_failures', 0)}")
            if self._decode_graph_capture_time_s:
                capture_s = {
                    key: round(value, 3)
                    for key, value in self._decode_graph_capture_time_s.items()
                }
                print(f"  capture time by graph: {capture_s}")
            if self._decode_graph_fallback_reasons:
                print(f"  fallback reasons: {dict(self._decode_graph_fallback_reasons)}")
            if self._decode_graph_capture_errors:
                print(f"  capture errors: {self._decode_graph_capture_errors}")
            if self._delta_subset_segment_stats:
                print("  DELTA subset segmented graphs:")
                print(f"    steps: {self._delta_subset_segment_stats.get('steps', 0)}")
                print(f"    captures: {self._delta_subset_segment_stats.get('captures', 0)}")
                print(f"    replays: {self._delta_subset_segment_stats.get('replays', 0)}")
                print(f"    capture failures: {self._delta_subset_segment_stats.get('capture_failures', 0)}")
                print(f"    replay failures: {self._delta_subset_segment_stats.get('replay_failures', 0)}")
                if self._delta_subset_segment_capture_time_s:
                    segment_capture_s = {
                        key: round(value, 3)
                        for key, value in self._delta_subset_segment_capture_time_s.items()
                    }
                    print(f"    capture time by segment: {segment_capture_s}")
                if self._delta_subset_segment_fallback_reasons:
                    print(f"    fallback reasons: {dict(self._delta_subset_segment_fallback_reasons)}")
                if self._delta_subset_segment_errors:
                    print(f"    errors: {self._delta_subset_segment_errors}")
        if self.enable_delta_impl_profile and self._delta_impl_profile_counts:
            print("\nDELTA Implementation Profile:")
            for name in sorted(self._delta_impl_profile_counts):
                count = self._delta_impl_profile_counts[name]
                total_ms = self._delta_impl_profile_ms.get(name, 0.0)
                avg_ms = total_ms / count if count else 0.0
                print(f"  {name:<32s} avg {avg_ms:8.3f} ms  total {total_ms:10.1f} ms  ({count} calls)")

        if self._step_records:
            timing_names = ["collect_pages", "plan", "model_forward", "postprocess", "step_total"]
            print(f"\n{'─'*60}")
            print(f"CUDA EVENT TIMING BREAKDOWN (per step avg)")
            print(f"{'─'*60}")
            n = len(self._step_records)
            for name in timing_names:
                key = name + "_ms"
                total_ms = sum(r[key] for r in self._step_records)
                avg_ms = total_ms / n
                print(f"  {name:<20s}  avg {avg_ms:8.3f} ms  total {total_ms:10.1f} ms  ({n} steps)")
            print(f"{'─'*60}")

            timing_path = Path("logs") / f"step_timings_{time.strftime('%Y%m%d_%H%M%S')}.json"
            timing_path.parent.mkdir(parents=True, exist_ok=True)
            meta = {
                "total_time_s": total_time,
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / total_time,
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "page_size": self.page_size,
                "selective_cache": self.enable_selective_cache,
                "num_steps": n,
                "cuda_graph_decode": self.enable_cuda_graph_decode,
                "cuda_graph_decode_stats": dict(self._decode_graph_stats),
                "cuda_graph_decode_fallback_reasons": dict(self._decode_graph_fallback_reasons),
                "cuda_graph_decode_capture_errors": dict(self._decode_graph_capture_errors),
                "cuda_graph_delta_subset_segments": self.enable_cuda_graph_delta_subset_segments,
                "cuda_graph_delta_subset_segment_stats": dict(self._delta_subset_segment_stats),
                "cuda_graph_delta_subset_segment_fallback_reasons": dict(self._delta_subset_segment_fallback_reasons),
                "cuda_graph_delta_subset_segment_errors": dict(self._delta_subset_segment_errors),
                "delta_impl_profile": self.enable_delta_impl_profile,
                "delta_impl_profile_ms": dict(self._delta_impl_profile_ms),
                "delta_impl_profile_counts": dict(self._delta_impl_profile_counts),
                "delta_fused_page_scores": self.enable_delta_fused_page_scores,
                "delta_fused_page_scores_failed": self._delta_fused_page_scores_failed,
                "delta_fixed_selector": self.enable_delta_fixed_selector,
                "delta_debug_page_selection_parity": self.enable_delta_debug_page_selection_parity,
                "delta_subset_plan_reuse": self.enable_delta_subset_plan_reuse,
                "delta_dump_buffer_dtype": self.delta_dump_buffer_dtype,
                "delta_page_score_impl": self.delta_page_score_impl,
            }
            with open(timing_path, "w") as f:
                json.dump({"meta": meta, "steps": self._step_records}, f, indent=2)
            print(f"  Step timings saved to: {timing_path}")

        print(f"{'='*60}")

    def get_detailed_results(self):
        if self.scheduler is None:
            return []
        return self.scheduler.get_detailed_results(self.tokenizer)
