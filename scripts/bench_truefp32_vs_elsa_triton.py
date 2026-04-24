#!/usr/bin/env python3
import argparse
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import cudnn

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency
    pd = None


ELSA_VARIANTS = ("baseline", "vit", "mem", "tensor", "new")


def _parse_tokens_from_csv(
    path: str,
    *,
    min_tokens: int,
    max_tokens: int,
    num_tokens: int,
) -> List[int]:
    if pd is None:
        raise RuntimeError("pandas is required for --tokens_csv")
    df = pd.read_csv(path)
    if "tokens" not in df.columns:
        raise ValueError(f"{path} is missing a 'tokens' column")
    tokens = sorted(set(int(x) for x in df["tokens"].tolist()))
    if min_tokens:
        tokens = [t for t in tokens if t >= min_tokens]
    if max_tokens:
        tokens = [t for t in tokens if t <= max_tokens]
    if num_tokens > 0:
        tokens = tokens[-num_tokens:]
    return tokens


def _parse_tokens(args: argparse.Namespace) -> List[int]:
    if args.tokens:
        tokens = [int(x.strip()) for x in args.tokens.split(",") if x.strip()]
    elif args.tokens_csv:
        tokens = _parse_tokens_from_csv(
            args.tokens_csv,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            num_tokens=args.num_tokens,
        )
    else:
        tokens = [4096, 16384, 65536]
    if not tokens:
        raise ValueError("No tokens selected; please check --tokens/--tokens_csv filters.")
    return tokens


def _resolve_elsa_fns(names: Iterable[str]) -> Dict[str, Callable]:
    import importlib

    module = importlib.import_module("timm.models.elsa_triton")
    mapping = {
        "baseline": getattr(module, "elsa_triton_baseline_fp32"),
        "vit": getattr(module, "elsa_triton_vit_fp32"),
        "mem": getattr(module, "elsa_triton_mem_fp32"),
        "tensor": getattr(module, "elsa_triton_tensor_fp32"),
        "new": getattr(module, "elsa_triton_new_fp32"),
    }
    chosen: Dict[str, Callable] = {}
    for name in names:
        if name == "all":
            for key, fn in mapping.items():
                chosen[key] = fn
            continue
        if name not in mapping:
            raise ValueError(f"Unknown ELSA variant '{name}'. Choices: {sorted(mapping)}")
        chosen[name] = mapping[name]
    if not chosen:
        raise ValueError("No ELSA variants selected.")
    return chosen


def _build_sdpa_graph(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    causal: bool,
    impl: cudnn.attention_implementation,
    handle,
) -> Tuple[cudnn.pygraph, cudnn.tensor, cudnn.tensor, cudnn.tensor, cudnn.tensor, torch.Tensor]:
    sm = torch.cuda.get_device_capability()
    sm_version = sm[0] * 100 + sm[1] * 10
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=handle,
        sm_version=sm_version,
    )
    q_g = graph.tensor_like(q)
    k_g = graph.tensor_like(k)
    v_g = graph.tensor_like(v)
    diag_right = 0 if causal else None
    o_g, _stats_g = graph.scaled_dot_product_flash_attention(
        name="sdpa",
        q=q_g,
        k=k_g,
        v=v_g,
        generate_stats=False,
        attn_scale=scale,
        diagonal_alignment=cudnn.diagonal_alignment.TOP_LEFT,
        diagonal_band_right_bound=diag_right,
        compute_data_type=cudnn.data_type.FLOAT,
        implementation=impl,
    )
    o_g.set_output(True).set_dim(q.shape).set_stride(q.stride())
    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()
    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)
    return graph, q_g, k_g, v_g, o_g, workspace


def _benchmark(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _format_ms(ms: Optional[float]) -> str:
    if ms is None:
        return "n/a"
    return f"{ms:.3f}"


def run_bench(
    *,
    tokens: List[int],
    batch: int,
    heads: int,
    head_dim: int,
    causal: bool,
    warmup: int,
    iters: int,
    elsa_variants: Dict[str, Callable],
    impl: cudnn.attention_implementation,
    fallback_impl: Optional[cudnn.attention_implementation],
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    device = torch.device("cuda")
    for n in tokens:
        q = torch.randn(batch, heads, n, head_dim, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        scale = 1.0 / math.sqrt(float(head_dim))

        cudnn_out = None
        cudnn_ms = None
        cudnn_impl_name = impl.name
        cudnn_error = None

        handle = cudnn.create_handle()
        cudnn.set_stream(handle, torch.cuda.current_stream().cuda_stream)
        try:
            graph, q_g, k_g, v_g, o_g, workspace = _build_sdpa_graph(
                q, k, v, scale, causal, impl, handle
            )
            cudnn_out = torch.empty_like(q)

            def run_cudnn():
                graph.execute({q_g: q, k_g: k, v_g: v, o_g: cudnn_out}, workspace, handle=handle)

            run_cudnn()
            cudnn_ms = _benchmark(run_cudnn, warmup, iters)
        except Exception as exc:
            cudnn_error = str(exc)
            if fallback_impl is not None:
                cudnn_impl_name = fallback_impl.name
                try:
                    graph, q_g, k_g, v_g, o_g, workspace = _build_sdpa_graph(
                        q, k, v, scale, causal, fallback_impl, handle
                    )
                    cudnn_out = torch.empty_like(q)

                    def run_cudnn():
                        graph.execute({q_g: q, k_g: k, v_g: v, o_g: cudnn_out}, workspace, handle=handle)

                    run_cudnn()
                    cudnn_ms = _benchmark(run_cudnn, warmup, iters)
                    cudnn_error = f"{cudnn_error} | fallback={cudnn_impl_name}"
                except Exception as exc_fallback:
                    cudnn_error = f"{cudnn_error} | fallback_failed={exc_fallback}"
        finally:
            try:
                cudnn.destroy_handle(handle)
            except Exception:
                pass

        print(f"\nshape=B{batch} H{heads} N{n} D{head_dim} causal={causal}")
        if cudnn_ms is None:
            print(f"  cudnn_fp32=failed impl={cudnn_impl_name} error={cudnn_error}")
        else:
            print(f"  cudnn_fp32_ms={cudnn_ms:.3f} impl={cudnn_impl_name}")

        best_name = None
        best_ms = None

        for name, fn in elsa_variants.items():
            max_abs = None
            max_rel = None
            status = "ok"
            elsa_ms = None
            try:
                with torch.no_grad():
                    out = fn(q, k, v, is_causal=causal, bias=None)
                if cudnn_out is not None:
                    diff = (out - cudnn_out).abs()
                    max_abs = diff.max().item()
                    max_rel = (diff / (cudnn_out.abs() + 1e-8)).max().item()

                def run_elsa():
                    with torch.no_grad():
                        fn(q, k, v, is_causal=causal, bias=None)

                elsa_ms = _benchmark(run_elsa, warmup, iters)
            except Exception as exc:
                status = f"error: {exc}"

            if elsa_ms is not None and (best_ms is None or elsa_ms < best_ms):
                best_ms = elsa_ms
                best_name = name

            speedup = None
            if cudnn_ms is not None and elsa_ms is not None and cudnn_ms > 0:
                speedup = elsa_ms / cudnn_ms
            speedup_str = "n/a" if speedup is None else f"{speedup:.2f}x"

            print(
                f"  elsa_{name}_ms={_format_ms(elsa_ms)} "
                f"speedup(elsa/cudnn)={speedup_str} "
                f"max_abs={max_abs if max_abs is not None else 'n/a'} "
                f"max_rel={max_rel if max_rel is not None else 'n/a'} "
                f"status={status}"
            )

        if best_name is None:
            print("  best_elsa=unavailable")
        else:
            ratio = "n/a"
            if cudnn_ms is not None and best_ms is not None and cudnn_ms > 0:
                ratio = f"{best_ms / cudnn_ms:.2f}x"
            print(f"  best_elsa={best_name} best_ms={best_ms:.3f} speedup(elsa/cudnn)={ratio}")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    parser = argparse.ArgumentParser(
        description="Benchmark ELSA Triton variants vs cuDNN True-FP32 fused flash attention."
    )
    parser.add_argument("--tokens", default="", help="Comma-separated token list (e.g., 4096,16384).")
    parser.add_argument("--tokens_csv", default="", help="CSV path with a 'tokens' column.")
    parser.add_argument("--min_tokens", type=int, default=0, help="Filter: tokens >= min_tokens.")
    parser.add_argument("--max_tokens", type=int, default=0, help="Filter: tokens <= max_tokens.")
    parser.add_argument("--num_tokens", type=int, default=3, help="Keep the largest N tokens from CSV.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--elsa",
        default="all",
        help="Comma-separated ELSA variants: baseline,vit,mem,tensor,new,all.",
    )
    parser.add_argument(
        "--impl",
        choices=["auto", "unified", "composite"],
        default="unified",
        help="cuDNN attention implementation to request.",
    )
    parser.add_argument(
        "--fallback_impl",
        choices=["auto", "unified", "composite", "none"],
        default="auto",
        help="Fallback cuDNN implementation if the primary fails.",
    )
    args = parser.parse_args()

    tokens = _parse_tokens(args)
    elsa_variants = _resolve_elsa_fns([x.strip() for x in args.elsa.split(",") if x.strip()])

    impl_map = {
        "auto": cudnn.attention_implementation.AUTO,
        "unified": cudnn.attention_implementation.UNIFIED,
        "composite": cudnn.attention_implementation.COMPOSITE,
    }
    impl = impl_map[args.impl]
    fallback_impl = None if args.fallback_impl == "none" else impl_map[args.fallback_impl]

    print(f"torch={torch.__version__} cudnn_backend={cudnn.backend_version_string()}")
    print(f"impl={impl.name} fallback={fallback_impl.name if fallback_impl else 'none'}")
    print(f"tokens={tokens} batch={args.batch} heads={args.heads} head_dim={args.head_dim}")

    run_bench(
        tokens=tokens,
        batch=args.batch,
        heads=args.heads,
        head_dim=args.head_dim,
        causal=args.causal,
        warmup=args.warmup,
        iters=args.iters,
        elsa_variants=elsa_variants,
        impl=impl,
        fallback_impl=fallback_impl,
    )


if __name__ == "__main__":
    main()
