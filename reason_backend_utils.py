from __future__ import annotations


FINAL_DELTA_BACKEND_CONFIG = {
    "page_selector_version": "v2",
    "delta_dump_buffer_dtype": "fp32",
    "delta_page_score_impl": "del3_legacy_softmax",
    "cuda_graph_delta_subset_segments": True,
    "delta_fused_page_scores": True,
    "delta_fixed_selector": True,
    "delta_fast_decode_page_info": True,
    "delta_v2_position_bias": True,
    "delta_subset_plan_reuse": True,
    "delta_impl_profile": False,
}


def add_delta_backend_args(parser, **_unused):
    parser.add_argument(
        "--cuda_graph_decode",
        action="store_true",
        help="Use CUDA Graph replay for steady one-token decode on the supported DELTA backend.",
    )


def validate_delta_backend_args(args) -> None:
    if not hasattr(args, "cuda_graph_decode"):
        raise ValueError("Missing required DELTA runtime flag: cuda_graph_decode")


def build_delta_backend_kwargs(args) -> dict:
    return {
        "cuda_graph_decode": args.cuda_graph_decode,
        **FINAL_DELTA_BACKEND_CONFIG,
    }


def delta_backend_config_fields(args, *, selective_cache_enabled: bool) -> dict:
    return {
        "cuda_graph_decode": args.cuda_graph_decode,
        "delta_backend": "final_public_release",
        "page_selector_version": FINAL_DELTA_BACKEND_CONFIG["page_selector_version"] if selective_cache_enabled else None,
        "cuda_graph_delta_subset_segments": FINAL_DELTA_BACKEND_CONFIG["cuda_graph_delta_subset_segments"] if selective_cache_enabled else None,
        "delta_fused_page_scores": FINAL_DELTA_BACKEND_CONFIG["delta_fused_page_scores"] if selective_cache_enabled else None,
        "delta_fixed_selector": FINAL_DELTA_BACKEND_CONFIG["delta_fixed_selector"] if selective_cache_enabled else None,
        "delta_fast_decode_page_info": FINAL_DELTA_BACKEND_CONFIG["delta_fast_decode_page_info"] if selective_cache_enabled else None,
        "delta_v2_position_bias": FINAL_DELTA_BACKEND_CONFIG["delta_v2_position_bias"] if selective_cache_enabled else None,
        "delta_subset_plan_reuse": FINAL_DELTA_BACKEND_CONFIG["delta_subset_plan_reuse"] if selective_cache_enabled else None,
        "delta_dump_buffer_dtype": FINAL_DELTA_BACKEND_CONFIG["delta_dump_buffer_dtype"] if selective_cache_enabled else None,
        "delta_page_score_impl": FINAL_DELTA_BACKEND_CONFIG["delta_page_score_impl"] if selective_cache_enabled else None,
    }
