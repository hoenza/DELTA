from __future__ import annotations


def add_delta_backend_args(
    parser,
    *,
    include_impl_profile: bool = False,
    include_strict_repro: bool = False,
):
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Whether to compile the model.",
    )
    parser.add_argument(
        "--cuda_graph_decode",
        action="store_true",
        help="Use pure CUDA Graph replay for steady one-token decode without torch.compile.",
    )
    parser.add_argument(
        "--disable_cuda_graph_delta_subset_segments",
        action="store_true",
        help="Disable segmented CUDA Graph replay inside DELTA subset decode.",
    )
    if include_impl_profile:
        parser.add_argument(
            "--delta_impl_profile",
            action="store_true",
            help="Record low-overhead CUDA event timings for DELTA implementation hot spots.",
        )
    else:
        parser.set_defaults(delta_impl_profile=False)
    parser.add_argument(
        "--disable_delta_fused_page_scores",
        action="store_true",
        help="Disable the Triton DELTA page-score kernel and use the torch fallback.",
    )
    parser.add_argument(
        "--disable_delta_fixed_selector",
        action="store_true",
        help="Disable the fixed-count DELTA selector fast path for steady subset decode.",
    )
    parser.add_argument(
        "--disable_delta_fast_decode_page_info",
        action="store_true",
        help="Disable the v2-only fast decode page-info collector and use the generic collector instead.",
    )
    parser.add_argument(
        "--disable_delta_v2_position_bias",
        action="store_true",
        help="Disable the tiny v2 logical-position tie-break bias during page selection.",
    )
    parser.add_argument(
        "--debug_delta_page_selection_parity",
        action="store_true",
        help="Compare optimized DELTA page scoring/selection against the reference path and raise on mismatch.",
    )
    parser.add_argument(
        "--debug_delta_fast_decode_page_info_parity",
        action="store_true",
        help="Compare the v2 fast decode page-info collector against the generic collector and raise on mismatch.",
    )
    parser.add_argument(
        "--disable_delta_subset_plan_reuse",
        action="store_true",
        help="Disable subset wrapper plan reuse when the subset CSR is structurally fixed.",
    )
    parser.add_argument(
        "--delta_dump_buffer_dtype",
        type=str,
        default="fp32",
        choices=["fp16", "fp32"],
        help="Attention dump buffer dtype used by DELTA planner attention.",
    )
    parser.add_argument(
        "--delta_page_score_impl",
        type=str,
        default="del3_legacy_softmax",
        choices=["delta_lse", "del3_legacy_softmax"],
        help="Page-score implementation used after DELTA planner attention.",
    )
    parser.add_argument(
        "--page_selector_version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Page selector backend to use for selective cache (default: v2)",
    )
    parser.add_argument(
        "--page_selector_v2",
        action="store_true",
        help="Deprecated alias for --page_selector_version v2",
    )
    if include_strict_repro:
        parser.add_argument(
            "--strict_repro_mode",
            action="store_true",
            help="Favor reproducibility over performance by disabling CUDA-graph and DELTA fast paths.",
        )
    else:
        parser.set_defaults(strict_repro_mode=False)


def apply_strict_repro_backend_overrides(args) -> None:
    if not getattr(args, "strict_repro_mode", False):
        return

    args.compile = False
    args.cuda_graph_decode = False
    args.disable_cuda_graph_delta_subset_segments = True
    args.disable_delta_fused_page_scores = True
    args.disable_delta_fixed_selector = True
    args.disable_delta_fast_decode_page_info = True
    args.disable_delta_subset_plan_reuse = True
    args.page_selector_version = "v1"
    args.page_selector_v2 = False


def validate_delta_backend_args(args) -> None:
    if args.compile and args.cuda_graph_decode:
        raise ValueError(
            "--cuda_graph_decode is intended for the non-compiled path; do not combine it with --compile."
        )
    if args.page_selector_v2 and args.page_selector_version != "v2":
        raise ValueError("--page_selector_v2 conflicts with --page_selector_version v1.")
    if args.cuda_graph_decode and args.page_selector_version != "v2":
        raise ValueError("--cuda_graph_decode currently requires --page_selector_version v2.")
    if args.cuda_graph_decode and args.disable_delta_fast_decode_page_info:
        raise ValueError("--cuda_graph_decode requires the v2 fast decode page-info collector.")


def build_delta_backend_kwargs(args) -> dict:
    return {
        "cuda_graph_decode": args.cuda_graph_decode,
        "cuda_graph_delta_subset_segments": not args.disable_cuda_graph_delta_subset_segments,
        "delta_impl_profile": getattr(args, "delta_impl_profile", False),
        "delta_fused_page_scores": not args.disable_delta_fused_page_scores,
        "delta_fixed_selector": not args.disable_delta_fixed_selector,
        "delta_fast_decode_page_info": not args.disable_delta_fast_decode_page_info,
        "delta_v2_position_bias": not args.disable_delta_v2_position_bias,
        "delta_debug_page_selection_parity": args.debug_delta_page_selection_parity,
        "delta_debug_fast_decode_page_info_parity": args.debug_delta_fast_decode_page_info_parity,
        "delta_subset_plan_reuse": not args.disable_delta_subset_plan_reuse,
        "delta_dump_buffer_dtype": args.delta_dump_buffer_dtype,
        "delta_page_score_impl": args.delta_page_score_impl,
        "page_selector_version": args.page_selector_version,
    }


def delta_backend_config_fields(args, *, selective_cache_enabled: bool) -> dict:
    return {
        "page_selector_version": args.page_selector_version if selective_cache_enabled else None,
        "cuda_graph_decode": args.cuda_graph_decode,
        "cuda_graph_delta_subset_segments": not args.disable_cuda_graph_delta_subset_segments,
        "strict_repro_mode": getattr(args, "strict_repro_mode", False),
        "delta_impl_profile": getattr(args, "delta_impl_profile", False),
        "delta_fused_page_scores": (not args.disable_delta_fused_page_scores) if selective_cache_enabled else None,
        "delta_fixed_selector": (not args.disable_delta_fixed_selector) if selective_cache_enabled else None,
        "delta_fast_decode_page_info": (not args.disable_delta_fast_decode_page_info) if selective_cache_enabled else None,
        "delta_v2_position_bias": (not args.disable_delta_v2_position_bias) if selective_cache_enabled else None,
        "delta_debug_page_selection_parity": args.debug_delta_page_selection_parity if selective_cache_enabled else None,
        "delta_debug_fast_decode_page_info_parity": args.debug_delta_fast_decode_page_info_parity if selective_cache_enabled else None,
        "delta_subset_plan_reuse": (not args.disable_delta_subset_plan_reuse) if selective_cache_enabled else None,
        "delta_dump_buffer_dtype": args.delta_dump_buffer_dtype if selective_cache_enabled else None,
        "delta_page_score_impl": args.delta_page_score_impl if selective_cache_enabled else None,
    }
