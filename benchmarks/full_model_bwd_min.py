"""Contention-robust full-model fwd+bwd (training step) bench: cuda-event MIN per iter.

Backward is GPU-compute-heavy (less CPU-dispatch-bound than fwd inference), so a per-iter
cuda-event MIN over many iters captures the cleanest training step and is robust to the
saturated login node. BOTH native and ELSA drop-in measured identically. Run under
scripts/bench_clean.sh on a clean GPU.
"""
from __future__ import annotations
import argparse, copy, sys, warnings
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import timm
from elsa_twopass_clean import patch_timm_attention_train
from bench_timm_dropin_bwd import _set_input_size


def min_ms_mem(step, *, warmup=8, iters=40):
    torch.cuda.empty_cache()
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); step(); e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return min(ts), torch.cuda.max_memory_allocated() / 1024**2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="vit_tiny_patch16_224.augreg_in21k_ft_in1k")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--image-size", type=int, nargs="+", default=[224, 384, 512])
    p.add_argument("--window-size", type=int, default=None)
    p.add_argument("--window-ratio", type=int, default=8)
    p.add_argument("--iters", type=int, default=40)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    print("mode,model,dtype,image,batch,base_ms,base_mb,elsa_ms,elsa_mb,lat_ratio,mem_ratio")
    for img in args.image_size:
        torch.manual_seed(2468 + img)
        template = timm.create_model(args.model, pretrained=False).eval()
        _set_input_size(template, img, window_size=args.window_size, window_ratio=args.window_ratio)
        base = copy.deepcopy(template).to("cuda", dtype)
        x = torch.randn(args.batch, 3, img, img, device="cuda", dtype=dtype)
        with torch.no_grad():
            dout = torch.randn_like(base(x)) * 0.1

        def run_base():
            base.zero_grad(set_to_none=True)
            xi = x.detach().requires_grad_(True)
            base(xi).backward(dout)
            return xi.grad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                bt, bm = min_ms_mem(run_base, iters=args.iters)
                del base; torch.cuda.empty_cache()
                elsa = copy.deepcopy(template).eval()
                patch_timm_attention_train(elsa)
                elsa = elsa.to("cuda", dtype)

                def run_elsa():
                    elsa.zero_grad(set_to_none=True)
                    xi = x.detach().requires_grad_(True)
                    elsa(xi).backward(dout)
                    return xi.grad

                et, em = min_ms_mem(run_elsa, iters=args.iters)
                print(f"bwdmin,{args.model},{args.dtype},{img},{args.batch},"
                      f"{bt:.6f},{bm:.3f},{et:.6f},{em:.3f},{et/bt:.4f},{em/bm:.4f}", flush=True)
                del elsa
            except Exception as e:
                print(f"bwdmin,{args.model},{args.dtype},{img},{args.batch},FAILED,,,,,{type(e).__name__}:{e}", flush=True)
        del x; torch.cuda.empty_cache()
    print("BWDMIN_DONE")


if __name__ == "__main__":
    main()
