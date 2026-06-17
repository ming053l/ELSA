"""Contention-proof full-model bench via CUDA graphs.

Full-model ViT/Swin at batch>=8 is CPU-dispatch-bound (GPU ~1.4ms vs CPU ~21ms/fwd), so
plain mean/min wall-clock on a saturated login node (load avg ~200) measures CPU contention,
not the model. CUDA graphs capture the whole forward into ONE replayable unit: no per-launch
CPU dispatch -> pure GPU time, independent of CPU load. This is ALSO the op-fusion the ELSA
shape-mismatch overhead calls for (all the per-layer launches/copies are captured once).

BOTH the native model (SDPA/FA) and the ELSA drop-in are graphed -> a fair pure-GPU compare.
Forward only (inference). Run under scripts/bench_clean.sh on a clean GPU.

Usage: full_model_graph.py --model <timm> --dtype fp16|fp32 --batch 8 --image-size 224 384 512
"""
from __future__ import annotations
import argparse, sys, copy, warnings
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import timm
from elsa_twopass_clean import patch_timm_attention

try:
    from bench_timm_dropin import _set_input_size  # reuse the same resizer
except Exception:
    from benchmarks.bench_timm_dropin import _set_input_size  # type: ignore


def graph_min_ms(model, x, *, warmup_capture=5, iters=80):
    """Capture model(x) into a CUDA graph; return MIN replay time (ms) over iters."""
    static_x = x.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.no_grad():
            for _ in range(5):
                model(static_x)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.no_grad():
        with torch.cuda.graph(g):
            model(static_x)
    for _ in range(warmup_capture):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); g.replay(); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return min(ts)


def min_ms_eager(model, x, *, warmup=8, iters=60):
    """Non-graph cuda-event MIN fallback (for kernels whose wide SMEM tile can't be
    captured into a CUDA graph). At long seq the model is GPU-bound so MIN is robust."""
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    torch.cuda.synchronize()
    ts = []
    with torch.no_grad():
        for _ in range(iters):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); model(x); e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
    return min(ts)


def peak_mb(model, x):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image-size", type=int, nargs="+", default=[224, 384, 512])
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--window-ratio", type=int, default=8)
    p.add_argument("--iters", type=int, default=80)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    dev = "cuda"
    print("mode,model,dtype,image,batch,base_ms,base_mb,elsa_ms,elsa_mb,lat_ratio,mem_ratio,max_abs")
    for img in args.image_size:
        torch.manual_seed(1234)
        base = timm.create_model(args.model, pretrained=False).eval()
        _set_input_size(base, img, window_size=args.window_size, window_ratio=args.window_ratio)
        elsa = copy.deepcopy(base).eval()
        patch_timm_attention(elsa)
        base = base.to(dev, dtype); elsa = elsa.to(dev, dtype)
        x = torch.randn(args.batch, 3, img, img, device=dev, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # graph both for the fair pure-GPU compare; if EITHER side's wide SMEM
                # tile can't be graph-captured, fall back to eager MIN for BOTH (fair).
                mode = "graph"
                try:
                    bt = graph_min_ms(base, x, iters=args.iters)
                    et = graph_min_ms(elsa, x, iters=args.iters)
                except Exception:
                    torch.cuda.empty_cache()
                    mode = "eagermin"
                    bt = min_ms_eager(base, x, iters=args.iters)
                    et = min_ms_eager(elsa, x, iters=args.iters)
                bm = peak_mb(base, x); em = peak_mb(elsa, x)
                with torch.no_grad():
                    mx = (elsa(x).float() - base(x).float()).abs().max().item()
                print(f"{mode},{args.model},{args.dtype},{img},{args.batch},"
                      f"{bt:.6f},{bm:.3f},{et:.6f},{em:.3f},{et/bt:.4f},{em/bm:.4f},{mx:.6g}", flush=True)
            except Exception as e:  # graph capture can fail on some ops -> report, continue
                print(f"graph,{args.model},{args.dtype},{img},{args.batch},FAILED,,,,,,{type(e).__name__}:{e}", flush=True)
        del base, elsa, x; torch.cuda.empty_cache()
    print("GRAPH_DONE")


if __name__ == "__main__":
    main()
