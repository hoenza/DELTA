from dataclasses import dataclass
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import torch.distributed as dist
import flashinfer
import time
from DELTA.Engine.DELTA.Timer import Timer


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    scaling_factor:float = 1.0
    # llama 3.1 with high_freq_factor and low_freq_factor
    low_freq_factor: int = None # added new
    high_freq_factor: int = None  # added new
    original_max_position_embeddings: int = None   # added new
    qkv_bias: bool = False
    # KV cache configuration
    full_cache_layers: list = None  # Layer indices that use full KV cache
    subset_cache_ratio: float = 0.5  # Ratio of KV cache to use for subset layers
    subset_cache_size: int = 0  # Fixed size of KV cache for subset layers (overrides ratio if >0)
    L: int = 8

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head
        
        # Set default full_cache_layers if not specified
        if self.full_cache_layers is None:
            self.full_cache_layers = []

    @classmethod
    def from_name(cls, name: str):
        if name in transformer_configs:
            return cls(**transformer_configs[name])
        # fuzzy search
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]
        # We may have two or more configs matched (e.g. "7B" and "Mistral-7B"). Find the best config match,
        # take longer name (as it have more symbols matched)
        if len(config) > 1:
            config.sort(key=len, reverse=True)
            assert len(config[0]) != len(config[1]), name # make sure only one 'best' match
        print(config)
        return cls(**transformer_configs[config[0]])


transformer_configs = {
    "llama-2-7b": dict(block_size=4096, n_layer=32, n_head=32, dim=4096),
    'llama-2-7b-32k': dict(block_size=32768, n_layer=32, dim= 4096, vocab_size=32000, scaling_factor=8),
    "llama-2-13b": dict(block_size=4096, n_layer=40, n_head=40, dim=5120),
    "llama-2-70b": dict(block_size=4096, n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
    "llama-3-8b": dict(block_size=8192, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000),
    "llama-3-70b": dict(block_size=8192, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000),
    "68m": dict(block_size=2048, n_layer=2, n_head=12, n_local_heads=12, dim=768, intermediate_size=3072, vocab_size=32000),
    "tinyllama": dict(block_size =2048, n_layer=22, n_head=32, n_local_heads=4, dim=2048, intermediate_size=5632, vocab_size=32000),
    "llama-3.1-8b": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000.0, scaling_factor=8, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
    "llama-3.1-70b": dict(block_size=131072, n_layer=80, n_head=64, n_local_heads=8, dim=8192, intermediate_size=28672, vocab_size=128256, rope_base=500000.0, scaling_factor=8, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
    "llama-3.2-1b": dict(block_size=131072, n_layer=16, n_head=32, n_local_heads=8, dim=2048, intermediate_size=8192, vocab_size=128256, rope_base=500000.0, scaling_factor=32, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
    "Qwen2.5-7b": dict(block_size=131072, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6),
    "Qwen2.5-14b": dict(block_size=131072, n_layer=48, n_head=40, n_local_heads=8, dim=5120, intermediate_size=13824, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6),
    "Qwen2.5-32b": dict(block_size=131072, n_layer=64, n_head=40, n_local_heads=8, dim=5120, intermediate_size=27648, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6),
    "Yi-1.5-6b": dict(block_size=4096, n_layer=32, n_head=32, n_local_heads=4, dim=4096, intermediate_size=11008, vocab_size=64000, rope_base=500000.0),
    "Yi-1.5-34b-32k": dict(block_size=32768, n_layer=60, n_head=56, n_local_heads=8, dim=7168, intermediate_size=20480, vocab_size=64000, rope_base=500000.0),
    "Mistral-7B-v0.1": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32000),
    "Mistral-7B-v0.3": dict(n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=32768, rope_base=1000000.0),
    "DeepSeek-R1-Distill-Llama-8B": dict(block_size=131072, n_layer=32, n_head=32, n_local_heads=8, dim=4096, intermediate_size=14336, vocab_size=128256, rope_base=500000.0, scaling_factor=8, high_freq_factor=4, low_freq_factor=1, original_max_position_embeddings=8192),
    "DeepSeek-R1-Distill-Qwen-1.5B": dict(block_size=131072, n_layer=28, n_head=12, n_local_heads=2, dim=1536, intermediate_size=8960, vocab_size=151936, rope_base=10000.0, qkv_bias=True, norm_eps=1e-6),
    "DeepSeek-R1-Distill-Qwen-7B": dict(block_size=131072, n_layer=28, n_head=28, n_local_heads=4, dim=3584, intermediate_size=18944, vocab_size=152064, rope_base=10000.0, qkv_bias=True, norm_eps=1e-6),
    "DeepSeek-R1-Distill-Qwen-14B": dict(block_size=131072, n_layer=48, n_head=40, n_local_heads=8, dim=5120, intermediate_size=13824, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-5),
    "DeepSeek-R1-Distill-Qwen-32B": dict(block_size=131072, n_layer=64, n_head=40, n_local_heads=8, dim=5120, intermediate_size=27648, vocab_size=152064, rope_base=1000000.0, qkv_bias=True, norm_eps=1e-6),
    "Qwen3-4B-Thinking-2507": dict(block_size=262144, n_layer=36, n_head=32, n_local_heads=8, dim=2560, intermediate_size=9728, vocab_size=151936, rope_base=5000000.0, norm_eps=1e-6, head_dim=128, qkv_bias=False),
}

class KVCache(nn.Module):
    def __init__(self, max_num_pages, page_size, n_heads, head_dim, dtype=torch.bfloat16, spec=False, draft_max_num_pages=0):
        super().__init__()
        cache_shape = (max_num_pages, 2, page_size, n_heads, head_dim)
        self.register_buffer('kv_cache', torch.zeros(cache_shape, dtype=dtype))
        
    def update_target(self, k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen):
        torch.ops.mylib.update_kv(
            k,
            v,
            kv_append_indptr,
            self.kv_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        return self.kv_cache
    
    def update_draft(self, k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen):
        torch.ops.mylib.update_kv(
            k,
            v,
            kv_append_indptr,
            self.draft_cache,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        return self.draft_cache


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.world_size = None
        self.rank = None
        self.process_group = None
        self.timer = None
        self.register_buffer("_tp_all_max_value", None, persistent=False)
        self.register_buffer("_tp_all_max_indices", None, persistent=False)
        # self.timer = Timer(timing_enabled=True, timing_log_dir='/tmp/op_times/model/')
        
    def setup_caches(self, num_pages, page_size, spec=False, draft_num_pages = 0, draft_budget = 0, window_size = 32):
        
        head_dim = self.config.dim // self.config.n_head
        # dtype = self.output.weight.dtype
        dtype = self.output.weight.dtype if self.output.weight.dtype == torch.float16 else torch.bfloat16

        if (self.config.high_freq_factor is not None) and (self.config.low_freq_factor is not None):
            torch.library.define(
                "mylib::rope",
                "(Tensor q, Tensor k, Tensor indptr, Tensor offsets) -> (Tensor ropeq, Tensor ropek)",
            )
            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                return flashinfer.rope.apply_llama31_rope(q, k, indptr, offsets, interleave=True, rope_scale=self.config.scaling_factor, rope_theta=self.config.rope_base, low_freq_factor=self.config.low_freq_factor, high_freq_factor=self.config.high_freq_factor, old_context_len=self.config.original_max_position_embeddings)

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return torch.empty_like(q), torch.empty_like(k)
        else:
            torch.library.define(
                "mylib::rope",
                "(Tensor q, Tensor k, Tensor indptr, Tensor offsets) -> (Tensor ropeq, Tensor ropek)",
            )
            @torch.library.impl("mylib::rope", "cuda")
            def rope(q, k, indptr, offsets):
                return flashinfer.rope.apply_rope(q, k, indptr, offsets, interleave=True, rope_scale=self.config.scaling_factor, rope_theta=self.config.rope_base)

            @torch.library.register_fake("mylib::rope")
            def rope_abstract(q, k, indptr, offsets):
                return torch.empty_like(q), torch.empty_like(k)

        for i, b in enumerate(self.layers):
            # Determine if this layer uses full cache
            if self.config.full_cache_layers:
                use_full_cache = i in self.config.full_cache_layers
                next_use_full_cache = (i + 1) in self.config.full_cache_layers if (i + 1) < self.config.n_layer else True
            else:
                use_full_cache = True
                next_use_full_cache = True
            
            b.attention.kv_cache = KVCache(num_pages, page_size, self.config.n_local_heads, head_dim, dtype, spec, draft_num_pages)

            if use_full_cache:
                if next_use_full_cache:
                    b.attention.attn_decode = torch.ops.mylib.target_decode
                else:
                    b.attention.attn_decode = torch.ops.mylib.target_decode_plan
            else:
                b.attention.attn_decode = torch.ops.mylib.target_subset_decode

            b.attention.rope = torch.ops.mylib.rope

    def forward(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        x = self.tok_embeddings(idx)
        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, self.timer)
        if (not self.timer is None) and self.timer.get_timing_enabled():
            self.timer.maybe_autoflush_all_timings()
        x = self.norm(x)
        logits = self.output(x)
        return self._finalize_logits(logits)

    def forward_decode_embed_layers(self, idx: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, start_layer: int, end_layer: int) -> Tensor:
        x = self.tok_embeddings(idx)
        return self.forward_decode_layers(
            x,
            input_pos,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
            start_layer,
            end_layer,
        )

    def forward_decode_layers(self, x: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, start_layer: int, end_layer: int) -> Tensor:
        for i in range(start_layer, end_layer):
            x = self.layers[i](
                x,
                input_pos,
                kv_append_indptr,
                kv_page_indices,
                kv_page_indptr,
                kv_page_lastlen,
                self.timer,
            )
        return x

    def forward_decode_finish(self, x: Tensor) -> Tensor:
        if (not self.timer is None) and self.timer.get_timing_enabled():
            self.timer.maybe_autoflush_all_timings()
        x = self.norm(x)
        logits = self.output(x)
        return self._finalize_logits(logits)

    def forward_decode_tail(self, x: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, start_layer: int) -> Tensor:
        x = self.forward_decode_layers(
            x,
            input_pos,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
            start_layer,
            self.config.n_layer,
        )
        return self.forward_decode_finish(x)

    def forward_decode_planner_pre(self, x: Tensor, input_pos: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, layer_idx: int) -> tuple[Tensor, Tensor]:
        return self.layers[layer_idx].forward_decode_planner_pre(
            x,
            input_pos,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )

    def forward_decode_planner_attention(self, q: Tensor, layer_idx: int, batch_size: int, seqlen: int) -> Tensor:
        return self.layers[layer_idx].forward_decode_planner_attention(
            q,
            batch_size,
            seqlen,
        )

    def forward_decode_planner_post(self, residual: Tensor, attn_out: Tensor, layer_idx: int) -> Tensor:
        return self.layers[layer_idx].forward_decode_planner_post(
            residual,
            attn_out,
        )

    def _ensure_tp_argmax_workspace(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        expected_shape = (logits.shape[0], logits.shape[1], self.world_size)
        if (
            self._tp_all_max_value is None
            or self._tp_all_max_value.shape != expected_shape
            or self._tp_all_max_value.dtype != logits.dtype
            or self._tp_all_max_value.device != logits.device
        ):
            self._tp_all_max_value = torch.zeros(
                expected_shape,
                dtype=logits.dtype,
                device=logits.device,
            )
        if (
            self._tp_all_max_indices is None
            or self._tp_all_max_indices.shape != expected_shape
            or self._tp_all_max_indices.device != logits.device
        ):
            self._tp_all_max_indices = torch.zeros(
                expected_shape,
                dtype=torch.long,
                device=logits.device,
            )
        return self._tp_all_max_value, self._tp_all_max_indices

    def _distributed_argmax(self, logits: Tensor) -> Tensor:
        all_max_value, all_max_indices = self._ensure_tp_argmax_workspace(logits)
        all_max_value.zero_()
        all_max_indices.zero_()
        local_max_value, local_max_indices = torch.max(logits, dim=-1)
        all_max_value[:, :, self.rank].copy_(local_max_value)
        all_max_indices[:, :, self.rank].copy_(local_max_indices)
        all_max_indices[:, :, self.rank] += self.rank * logits.shape[-1]
        dist.all_reduce(all_max_value, group=self.process_group)
        dist.all_reduce(all_max_indices, group=self.process_group)
        global_select_indices = torch.argmax(all_max_value, dim=-1)
        global_indices = torch.gather(
            all_max_indices,
            dim=-1,
            index=global_select_indices.unsqueeze(-1),
        )
        return global_indices.squeeze(-1)

    def _finalize_logits(self, logits: Tensor) -> Tensor:
        if self.process_group is not None:
            return self._distributed_argmax(logits)
        return torch.argmax(logits, dim=-1)

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, timer: Timer) -> Tensor:
        h = x + self.attention(self.attention_norm(x), offsets, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen, timer)
        out = h + self.feed_forward(self.ffn_norm(h), timer)
        return out

    def forward_decode_planner_pre(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> tuple[Tensor, Tensor]:
        q = self.attention.prepare_decode_qkv(
            self.attention_norm(x),
            offsets,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        return x, q

    def forward_decode_planner_attention(self, q: Tensor, batch_size: int, seqlen: int) -> Tensor:
        return self.attention.decode_attention_only(q, batch_size, seqlen)

    def forward_decode_planner_post(self, residual: Tensor, attn_out: Tensor) -> Tensor:
        h = residual + self.attention.project_attention_output(attn_out)
        return h + self.feed_forward(self.ffn_norm(h), None)


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=config.qkv_bias)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None
        self.process_group = None
        self.attn_decode = None
        self.attn_prefill = None
        self.attn_draft = None
        self.rope = None
        self.is_spec = False

        self.window_size = None
        self.pooling = None
        self.kernel_size = None
        self.draft_budget = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        
        self.n_q_heads  = config.n_head
        self.n_kv_heads = config.n_local_heads
        self.dim = self.n_q_heads * self.head_dim
        
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

        if prefix + "wq.bias" in state_dict:    
            bq = state_dict.pop(prefix + "wq.bias")
            bk = state_dict.pop(prefix + "wk.bias")
            bv = state_dict.pop(prefix + "wv.bias")
            state_dict[prefix + "wqkv.bias"] = torch.cat([bq, bk, bv])

    def prepare_decode_qkv(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor) -> Tensor:
        bsz, seqlen, _ = x.shape
        q_size = self.n_q_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim

        q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
        q = q.view(bsz * seqlen, self.n_q_heads, self.head_dim)
        k = k.view(bsz * seqlen, self.n_kv_heads, self.head_dim)
        v = v.view(bsz * seqlen, self.n_kv_heads, self.head_dim)
        q, k = self.rope(q, k, kv_append_indptr, offsets)
        self.kv_cache.update_target(
            k,
            v,
            kv_append_indptr,
            kv_page_indices,
            kv_page_indptr,
            kv_page_lastlen,
        )
        return q

    def decode_attention_only(self, q: Tensor, batch_size: int, seqlen: int) -> Tensor:
        y = self.attn_decode(q, self.kv_cache.kv_cache)
        return y.contiguous().view(batch_size, seqlen, self.dim)

    def project_attention_output(self, y: Tensor) -> Tensor:
        y = self.wo(y)
        if self.process_group != None:
            dist.all_reduce(y, group=self.process_group)
        return y

    def forward(self, x: Tensor, offsets: Tensor, kv_append_indptr: Tensor, kv_page_indices: Tensor, kv_page_indptr: Tensor, kv_page_lastlen: Tensor, timer: Timer) -> Tensor:
        bsz, seqlen, _ = x.shape
        q_size  = self.n_q_heads * self.head_dim
        kv_size = self.n_kv_heads * self.head_dim
        
        if (not timer is None) and timer.get_timing_enabled():
            stream = torch.cuda.current_stream(x.device)
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
            q = q.view(bsz * seqlen, self.n_q_heads,  self.head_dim)
            k = k.view(bsz * seqlen, self.n_kv_heads, self.head_dim)
            v = v.view(bsz * seqlen, self.n_kv_heads, self.head_dim)
            q, k = self.rope(q, k, kv_append_indptr, offsets)
            end.record(stream)
            end.synchronize()  # make sure timing is complete
            timer.record_cuda_ms("ffn1", start, end)
        else:
            q, k, v = self.wqkv(x).split([q_size, kv_size, kv_size], dim=-1)
            q = q.view(bsz * seqlen, self.n_q_heads,  self.head_dim)
            k = k.view(bsz * seqlen, self.n_kv_heads, self.head_dim)
            v = v.view(bsz * seqlen, self.n_kv_heads, self.head_dim)
            q, k = self.rope(q, k, kv_append_indptr, offsets)
        
        if (not timer is None) and timer.get_timing_enabled():
            stream = torch.cuda.current_stream(x.device)
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            kv_cache = self.kv_cache.update_target(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
            y = self.attn_decode(q, kv_cache)
            y = y.contiguous().view(bsz, seqlen, self.dim)
            end.record(stream)
            end.synchronize()  # make sure timing is complete
            timer.record_cuda_ms("attn", start, end)
        else:
            kv_cache = self.kv_cache.update_target(k, v, kv_append_indptr, kv_page_indices, kv_page_indptr, kv_page_lastlen)
            y = self.attn_decode(q, kv_cache)
            y = y.contiguous().view(bsz, seqlen, self.dim)
        
        if (not timer is None) and timer.get_timing_enabled():
            stream = torch.cuda.current_stream(x.device)
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            y = self.wo(y)
            end.record(stream)
            end.synchronize()  # make sure timing is complete
            timer.record_cuda_ms("ffn2", start, end)
        else:
            y = self.wo(y)
        
        if self.process_group != None:
            dist.all_reduce(y, group = self.process_group)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)
        self.process_group = None

    def forward(self, x: Tensor, timer: Timer) -> Tensor:
        if (not timer is None) and timer.get_timing_enabled():
            stream = torch.cuda.current_stream(x.device)
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            y = self.w2(F.silu(self.w1(x)) * self.w3(x))
            end.record(stream)
            end.synchronize()  # make sure timing is complete
            timer.record_cuda_ms("ffn3", start, end)
        else:
            y = self.w2(F.silu(self.w1(x)) * self.w3(x))
            
        if self.process_group != None:
            dist.all_reduce(y, group = self.process_group)
        return y


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
