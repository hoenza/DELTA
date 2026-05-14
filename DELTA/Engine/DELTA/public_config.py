"""Shared validated defaults for the public DELTA release.

This module deliberately centralizes the one supported backend configuration
used by the public runners and CLI plumbing. Legacy branches still exist in the
backend for migration safety, but the documented workflow should resolve to the
values below.
"""

FINAL_PUBLIC_DELTA_BACKEND_CONFIG = {
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
