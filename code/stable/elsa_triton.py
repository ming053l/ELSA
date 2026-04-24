import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import warnings
from typing import Optional, Tuple
from contextlib import contextmanager
import math, os
import importlib.util
from pathlib import Path

_SHORT_ATTENTION_COMPILED = None
_UNSTABLE_ROUTE_WARNED: set[str] = set()


def _warn_once(key: str, msg: str) -> None:
    if key in _UNSTABLE_ROUTE_WARNED:
        return
    _UNSTABLE_ROUTE_WARNED.add(key)
    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _allow_unstable_paths() -> bool:
    return os.environ.get("ELSA_TRITON_ALLOW_UNSTABLE_PATHS", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
        "force",
    )


def _sanitize_fp16_fwd_block(block: int, *, name: str) -> int:
    """Clamp known-problematic fp16 tile settings to stable defaults."""
    if block == 96 and not _allow_unstable_paths():
        _warn_once(
            f"fp16_block96_{name}",
            (
                f"{name}=96 is disabled by default due observed Triton CompilationError "
                "on this stack; using 64. Set ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1 to force."
            ),
        )
        return 64
    return block


def _as_fp32_contig(x: torch.Tensor) -> torch.Tensor:
    if x.dtype == torch.float32:
        return x if x.is_contiguous() else x.contiguous()
    return x.to(torch.float32).contiguous()


@contextmanager
def _tf32_context(enabled: Optional[bool]):
    if enabled is None:
        yield
        return
    prev_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul
        torch.backends.cudnn.allow_tf32 = prev_cudnn


@contextmanager
def _temp_env(overrides: dict[str, Optional[str]]):
    prev = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _short_attention_base(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    B, H, N, D = q.shape
    dv = v.shape[-1]
    q_flat = q.reshape(B * H, N, D)
    k_flat = k.reshape(B * H, N, D)
    v_flat = v.reshape(B * H, N, dv)
    scores = torch.bmm(q_flat, k_flat.transpose(-1, -2))

    scale = logit_scale.exp().clamp_min(1e-6)
    scores = scores.view(B, H, N, N) * scale.view(1, H, 1, 1)

    attn_bias = None
    if relative_position_bias is not None:
        attn_bias = relative_position_bias.unsqueeze(0).expand(B, -1, -1, -1)
    if mask is not None:
        mask_bias = mask
        if mask_bias.dim() == 4 and mask_bias.size(1) == 1:
            mask_bias = mask_bias.view(B, 1, N, N)
        attn_bias = mask_bias if attn_bias is None else attn_bias + mask_bias
    if attn_bias is not None:
        scores = scores + attn_bias

    # For short windows (e.g. Swin N=64), fused softmax is typically faster than
    # manual exp/sum normalization while preserving exact attention semantics.
    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn.reshape(B * H, N, N), v_flat)
    return out.view(B, H, N, dv)


def _short_attention_compiled():
    global _SHORT_ATTENTION_COMPILED
    if _SHORT_ATTENTION_COMPILED is not None:
        return _SHORT_ATTENTION_COMPILED
    if not bool(int(os.environ.get("ELSA_SWIN_SHORT_COMPILE", "1"))):
        return None
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        return None

    def _impl(q, k, v, logit_scale, relative_position_bias, mask):
        return _short_attention_base(q, k, v, logit_scale, relative_position_bias, mask)

    try:
        _SHORT_ATTENTION_COMPILED = compile_fn(_impl, mode="reduce-overhead", fullgraph=True)
    except Exception:
        _SHORT_ATTENTION_COMPILED = None
    return _SHORT_ATTENTION_COMPILED

def _choose_tile(N: int, dev_prop, prefer_large=True):
    """
    根據序列長度 N 與 GPU 性能，選出自適應 BLOCK_M/N。
    - dev_prop: torch.cuda.get_device_properties(device)
    - prefer_large: 是否優先選大 tile (對資料中心卡較好)
    """
    # 桌機卡頻寬 < 400GB/s 視為 bandwidth-bound
    is_bandwidth_bound = getattr(dev_prop, "memoryBusWidth", 0) * \
                         getattr(dev_prop, "memoryClockRate", 0) < 400_000

    # 排序策略：資料中心卡 128→96→64；桌機卡 96→64→128
    candidate = [128, 64] if prefer_large and not is_bandwidth_bound else [64, 128]
    for blk in candidate:
        if N % blk == 0:
            return blk
    # 仍無法整除：依 N 大小決定
    return 128 if N > 8192 else 64

_ELSA_FP32_TUNE_CACHE = {}
_ELSA_FP32_FAST_TUNE_CACHE = {}
_ELSA_FP32_INFER_TUNE_CACHE = {}
_ELSA_FP32_SPLITD_TUNE_CACHE = {}
_ELSA_FP32_TRAIN_TUNE_CACHE = {}
_ELSA_FP32_STABLE_MODULE = None


def _load_elsa_fp32_stable_module():
    global _ELSA_FP32_STABLE_MODULE
    if _ELSA_FP32_STABLE_MODULE is not None:
        return _ELSA_FP32_STABLE_MODULE

    root = Path(__file__).resolve().parents[3]
    stable_path = (
        root
        / "timm"
        / "elsa_cuda"
        / "versions"
        / "original_20251021_195305"
        / "elsa_cuda"
        / "versions"
        / "original_20251021_195305"
        / "sic_triton.py"
    )
    if not stable_path.is_file():
        raise FileNotFoundError(f"Stable sic_triton not found: {stable_path}")
    spec = importlib.util.spec_from_file_location("_elsa_fp32_stable_train", stable_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stable sic_triton module from {stable_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _ELSA_FP32_STABLE_MODULE = module
    return module


def _should_use_stable_train_fwd(
    *,
    needs_grad: bool,
    use_tf32: bool,
    n_ctx: int,
    d_head: int,
    bwd_impl: str,
) -> bool:
    if not needs_grad or use_tf32:
        return False
    if bwd_impl not in ("auto", "mem"):
        return False
    # Default to "auto": on current A100 + CUDA 12.6 stack, stable forward tends to
    # reduce fp32 train-step latency for medium/long ViT sequence lengths.
    mode = os.environ.get("ELSA_TRITON_FP32_STABLE_TRAIN_FWD", "auto").strip().lower()
    if mode in ("0", "off", "false", "disable", "disabled"):
        return False
    if mode in ("1", "on", "true", "force"):
        return True
    try:
        min_n = int(os.environ.get("ELSA_TRITON_FP32_STABLE_MIN_N", "896"))
    except ValueError:
        min_n = 896
    try:
        max_d = int(os.environ.get("ELSA_TRITON_FP32_STABLE_MAX_D", "128"))
    except ValueError:
        max_d = 128
    return n_ctx >= max(64, min_n) and d_head <= max(16, max_d)


def _resolve_fp16_fast_accum(
    *,
    q: torch.Tensor,
    n_ctx: int,
    d_head: int,
    needs_grad: bool,
    is_causal: bool = False,
    prefer_infer_fast: bool = False,
) -> bool:
    """Resolve fp16/bf16 accumulator policy.

    Default behavior is conservative for inference, but enables a faster train path
    for long-sequence fp16 where full-model step latency is typically dominated by
    forward attention cost.
    """
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False

    raw = os.environ.get("ELSA_TRITON_FP16_FAST_ACCUM", "").strip().lower()
    if raw in ("1", "true", "on", "yes", "force"):
        return True
    if raw in ("0", "false", "off", "no"):
        return False

    # Auto policy:
    # - training only (keep inference exactness defaults unchanged)
    # - fp16 only (bf16 keeps fp32 accumulator by default)
    # - medium/long context and common ViT/Swin head dims
    if q.dtype != torch.float16:
        return False
    if not needs_grad:
        return bool((not is_causal) and prefer_infer_fast)
    try:
        min_n = int(os.environ.get("ELSA_TRITON_FP16_FAST_ACCUM_AUTO_MIN_N", "1024"))
    except ValueError:
        min_n = 1024
    try:
        max_d = int(os.environ.get("ELSA_TRITON_FP16_FAST_ACCUM_AUTO_MAX_D", "128"))
    except ValueError:
        max_d = 128
    return n_ctx >= max(64, min_n) and d_head <= max(16, max_d)


def _fp16_kblock_auto_enabled(
    *,
    n_ctx: int,
    d_head: int,
    is_causal: bool,
) -> bool:
    """Auto-enable the fp16 K-block fused route for short/mid exact attention."""
    if is_causal:
        return False
    return d_head <= 64 and 4096 <= n_ctx < 16384


def _fp16_flat_auto_enabled(
    *,
    n_ctx: int,
    d_head: int,
    is_causal: bool,
) -> bool:
    """Auto-enable flattened fp16 launch order on medium/long exact attention.

    The flat nomask route is the current best-performing fused path for common
    ViT geometry on A100 once sequence length is large enough for launch order
    and reshape overheads to dominate, but it can regress very short contexts.
    """
    if is_causal:
        return False
    if d_head <= 64:
        return n_ctx >= 16384
    return n_ctx >= 327680


def _stable_can_fp32_forward_with_z(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run locked stable CAN fp32 kernel and also return per-row z statistics."""
    stable_mod = _load_elsa_fp32_stable_module()
    if not hasattr(stable_mod, "kernel_integral_mhsa_stable"):
        out = stable_mod.can_triton_baseline_fp32(q, k, v, is_causal=False, bias=None)
        out_z = torch.ones(
            (q.shape[0], q.shape[1], q.shape[2]),
            device=q.device,
            dtype=torch.float32,
        )
        return out, out_z

    B, H, N, D = q.shape
    q_ = q.contiguous().view(B * H, N, D)
    k_ = k.contiguous().view(B * H, N, D)
    v_ = v.contiguous().view(B * H, N, D)

    out_s = torch.empty_like(q_, dtype=q.dtype)
    out_z = torch.empty((B * H, N), dtype=q.dtype, device=q.device)

    try:
        block_q = int(os.environ.get("ELSA_TRITON_FP32_STABLE_BLOCK_Q", "64"))
    except ValueError:
        block_q = 64
    try:
        block_n = int(os.environ.get("ELSA_TRITON_FP32_STABLE_BLOCK_N", "64"))
    except ValueError:
        block_n = 64
    try:
        num_warps = int(os.environ.get("ELSA_TRITON_FP32_STABLE_WARPS", "4"))
    except ValueError:
        num_warps = 4
    try:
        num_stages = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STAGES", "2"))
    except ValueError:
        num_stages = 2

    block_q = max(16, (block_q // 16) * 16)
    block_n = max(16, (block_n // 16) * 16)
    num_warps = max(1, num_warps)
    num_stages = max(1, num_stages)

    grid = (triton.cdiv(N, block_q), B * H)
    stable_mod.kernel_integral_mhsa_stable[grid](
        q_,
        k_,
        v_,
        out_s,
        out_z,
        B * H,
        N,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        out_s.stride(0),
        out_s.stride(1),
        out_s.stride(2),
        out_z.stride(1),
        out_z.stride(0),
        out_z.stride(1),
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        D_HEAD=D,
        SCALE=scale,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    out = (out_s / out_z.unsqueeze(-1)).view(B, H, N, D).to(q.dtype)
    return out, out_z.view(B, H, N).to(torch.float32)


def _stable_local_fp32_forward_with_mz(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    allow_tf32: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run local stable fp32 kernel and return out + per-row (m, z)."""
    B, H, N, D = q.shape
    q_ = q.contiguous().view(B * H, N, D)
    k_ = k.contiguous().view(B * H, N, D)
    v_ = v.contiguous().view(B * H, N, D)

    out_s = torch.empty_like(q_, dtype=q.dtype)
    out_z = torch.empty((B * H, N), dtype=q.dtype, device=q.device)
    out_m = torch.empty((B * H, N), dtype=q.dtype, device=q.device)

    try:
        block_q = int(os.environ.get("ELSA_TRITON_FP32_STABLE_BLOCK_Q", "64"))
    except ValueError:
        block_q = 64
    try:
        block_n = int(os.environ.get("ELSA_TRITON_FP32_STABLE_BLOCK_N", "64"))
    except ValueError:
        block_n = 64
    try:
        num_warps = int(os.environ.get("ELSA_TRITON_FP32_STABLE_WARPS", "4"))
    except ValueError:
        num_warps = 4
    try:
        num_stages = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STAGES", "2"))
    except ValueError:
        num_stages = 2

    block_q = max(16, (block_q // 16) * 16)
    block_n = max(16, (block_n // 16) * 16)
    num_warps = max(1, num_warps)
    num_stages = max(1, num_stages)

    grid = (triton.cdiv(N, block_q), B * H)
    kernel_integral_mhsa_stable[grid](
        q_,
        k_,
        v_,
        out_s,
        out_z,
        out_m,
        B * H,
        N,
        q_.stride(0),
        q_.stride(1),
        q_.stride(2),
        k_.stride(0),
        k_.stride(1),
        k_.stride(2),
        v_.stride(0),
        v_.stride(1),
        v_.stride(2),
        out_s.stride(0),
        out_s.stride(1),
        out_s.stride(2),
        out_z.stride(1),
        out_z.stride(0),
        out_z.stride(1),
        out_m.stride(1),
        out_m.stride(0),
        out_m.stride(1),
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        D_HEAD=D,
        SCALE=scale,
        ALLOW_TF32=allow_tf32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out = (out_s / out_z.unsqueeze(-1)).view(B, H, N, D).to(q.dtype)
    return out, out_m.view(B, H, N).to(torch.float32), out_z.view(B, H, N).to(torch.float32)


def _normalize_scan_accumulator_(
    out_acc: torch.Tensor,
    z_acc: torch.Tensor,
) -> torch.Tensor:
    """Convert scan numerator accumulator to final attention output in-place."""
    inv_z = z_acc.to(torch.float32).clamp_min_(1e-6).reciprocal_()
    out_acc.mul_(inv_z.unsqueeze(-1).to(out_acc.dtype))
    return out_acc


def _resolve_fp32_kernel_allow_tf32(
    *,
    requested_tf32: bool,
    needs_grad: bool,
) -> bool:
    """Guard TF32 in ELSA fp32 kernels; default to stable fp32 in grad paths."""
    if not requested_tf32:
        return False
    raw = os.environ.get("ELSA_TRITON_FP32_KERNEL_ALLOW_TF32", "").strip().lower()
    if raw:
        return raw in ("1", "true", "on", "yes", "force")
    # Default safety policy: training/finetune/backward paths keep fp32 kernels.
    return not needs_grad


def _elsa_fp32_candidates(D: int) -> list[tuple[int, int, int, int]]:
    wide = os.environ.get("ELSA_TRITON_FP32_TUNE_WIDE") == "1"
    if D >= 256:
        candidates = [
            (8, 128, 4, 2),
            (8, 256, 4, 2),
            (16, 32, 2, 1),
            (16, 64, 4, 2),
            (16, 128, 4, 2),
            (32, 32, 2, 1),
            (32, 64, 4, 2),
            (32, 128, 4, 2),
            (32, 256, 4, 2),
            (32, 512, 4, 2),
            (64, 32, 4, 2),
            (64, 64, 4, 2),
            (64, 128, 4, 2),
            (64, 256, 4, 2),
            (64, 512, 4, 2),
            (128, 32, 4, 2),
            (128, 64, 4, 2),
            (128, 128, 4, 2),
            (128, 64, 8, 2),
            (128, 128, 8, 2),
            (128, 128, 8, 3),
            (32, 128, 8, 2),
            (64, 128, 8, 2),
            (32, 256, 8, 2),
            (64, 256, 8, 2),
            (32, 128, 8, 3),
            (64, 128, 8, 3),
            (32, 256, 8, 3),
            (64, 256, 8, 3),
        ]
        if wide:
            extra = []
            for block_q in (16, 32, 48, 64):
                for block_n in (64, 96, 128, 160, 192, 256):
                    extra.append((block_q, block_n, 4, 2))
                    extra.append((block_q, block_n, 8, 2))
            candidates.extend(extra)
        return candidates
    candidates = [
        (32, 32, 2, 1),
        (32, 64, 4, 2),
        (32, 128, 4, 2),
        (64, 32, 4, 2),
        (64, 64, 4, 2),
        (64, 64, 8, 3),
        (64, 128, 4, 2),
        (64, 256, 4, 2),
        (128, 32, 4, 2),
        (128, 64, 4, 2),
        (128, 128, 4, 2),
        (128, 128, 8, 3),
    ]
    if wide:
        extra = []
        for block_q in (32, 48, 64):
            for block_n in (64, 96, 128, 160):
                extra.append((block_q, block_n, 4, 2))
                extra.append((block_q, block_n, 8, 2))
        candidates.extend(extra)
    return candidates

def _tune_elsa_fp32_kernel(
    kernel,
    q_,
    k_,
    v_,
    out_s,
    out_z,
    out_m,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    allow_tf32: bool = False,
):
    candidates = _elsa_fp32_candidates(D)
    best = None
    best_ms = None
    for block_q, block_n, num_wp, num_stages in candidates:
        if block_q > N or block_n > N:
            continue
        block_d = 32 * ((D + 31) // 32)
        grid = (triton.cdiv(N, block_q), B * H)
        # Warmup to avoid compile time in timing.
        try:
            kernel[grid](
                q_, k_, v_,
                out_s, out_z, out_m,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out_s.stride(0), out_s.stride(1), out_s.stride(2),
                out_z.stride(1), out_z.stride(0), out_z.stride(1),
                out_m.stride(1), out_m.stride(0), out_m.stride(1),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_,
                out_s, out_z, out_m,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out_s.stride(0), out_s.stride(1), out_s.stride(2),
                out_z.stride(1), out_z.stride(0), out_z.stride(1),
                out_m.stride(1), out_m.stride(0), out_m.stride(1),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_q, block_n, num_wp, num_stages)
    return best


def _tune_elsa_fp32_infer_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    allow_tf32: bool = False,
):
    candidates = _elsa_fp32_candidates(D)
    best = None
    best_ms = None
    for block_q, block_n, num_wp, num_stages in candidates:
        if block_q > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_q), B * H)
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_q, block_n, num_wp, num_stages)
    return best


def _tune_elsa_fp32_splitd_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    allow_tf32: bool = False,
):
    candidates = [
        (16, 64, 64, 4, 2),
        (16, 128, 64, 4, 2),
        (32, 64, 64, 4, 2),
        (32, 128, 64, 4, 2),
        (32, 128, 64, 8, 2),
        (32, 256, 64, 8, 2),
        (32, 128, 128, 4, 2),
        (32, 256, 128, 8, 2),
        (64, 64, 64, 4, 2),
        (64, 128, 64, 8, 2),
        (64, 128, 128, 8, 2),
        (64, 256, 64, 8, 2),
    ]
    if os.environ.get("ELSA_TRITON_FP32_TUNE_WIDE") == "1":
        extra = []
        for block_q in (16, 32, 48, 64):
            for block_n in (64, 96, 128, 160, 192, 256):
                for block_d in (64, 128):
                    extra.append((block_q, block_n, block_d, 4, 2))
                    extra.append((block_q, block_n, block_d, 8, 2))
        candidates.extend(extra)
    best = None
    best_ms = None
    for block_q, block_n, block_d, num_wp, num_stages in candidates:
        if block_q > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_q), B * H)
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                BLOCK_D=block_d,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_, out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                BLOCK_D=block_d,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_q, block_n, block_d, num_wp, num_stages)
    return best


def _tune_elsa_fp32_fast_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
):
    block_d = 32 * ((D + 31) // 32)
    allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    candidates = [
        (16, 32, 2, 1),
        (16, 64, 4, 2),
        (16, 128, 4, 2),
        (32, 32, 2, 1),
        (32, 64, 4, 2),
        (32, 128, 4, 2),
        (32, 256, 4, 2),
        (64, 32, 4, 2),
        (64, 64, 4, 2),
        (64, 128, 4, 2),
        (64, 256, 4, 2),
        (32, 128, 8, 2),
        (64, 128, 8, 2),
        (32, 256, 8, 2),
        (64, 256, 8, 2),
    ]
    best = None
    best_ms = None
    for block_m, block_n, num_wp, num_stages in candidates:
        if block_m > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_m), B * H)
        try:
            kernel[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                num_warps=num_wp,
                num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_m, block_n, num_wp, num_stages)
    return best


def _tune_elsa_fp32_fast_mz_kernel(
    kernel,
    q_,
    k_,
    v_,
    out,
    out_m,
    out_z,
    B: int,
    H: int,
    N: int,
    D: int,
    scale: float,
    *,
    allow_tf32: bool,
):
    candidates = _elsa_fp32_candidates(D)
    best = None
    best_ms = None
    block_d = 32 * ((D + 31) // 32)
    for block_m, block_n, num_wp, num_stages in candidates:
        if block_m > N or block_n > N:
            continue
        grid = (triton.cdiv(N, block_m), B * H)
        try:
            kernel[grid](
                q_,
                k_,
                v_,
                out,
                out_m,
                out_z,
                q_.stride(0),
                q_.stride(1),
                q_.stride(2),
                k_.stride(0),
                k_.stride(1),
                k_.stride(2),
                v_.stride(0),
                v_.stride(1),
                v_.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out_m.stride(0),
                out_m.stride(1),
                out_z.stride(0),
                out_z.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
            torch.cuda.synchronize()
        except Exception:
            continue
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            kernel[grid](
                q_,
                k_,
                v_,
                out,
                out_m,
                out_z,
                q_.stride(0),
                q_.stride(1),
                q_.stride(2),
                k_.stride(0),
                k_.stride(1),
                k_.stride(2),
                v_.stride(0),
                v_.stride(1),
                v_.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                out_m.stride(0),
                out_m.stride(1),
                out_z.stride(0),
                out_z.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
        except Exception:
            continue
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best = (block_m, block_n, num_wp, num_stages)
    return best


@triton.jit
def elsa_swinv2_kernel_short(
    Q, K, V, Out,
    LogitScale, RelBias, Mask,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    B, H, N, D, DV,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    base_q = Q + b * stride_qb + h * stride_qh
    base_k = K + b * stride_kb + h * stride_kh
    base_v = V + b * stride_vb + h * stride_vh
    base_out = Out + b * stride_ob + h * stride_oh

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    scale = tl.exp(tl.load(LogitScale + h)).to(tl.float32)

    acc = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
    for d0 in tl.static_range(0, 64, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        q_ptrs = base_q + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd
        k_ptrs = base_k + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn

        q_chunk = tl.load(q_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        k_chunk = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(q_chunk, k_chunk, allow_tf32=ALLOW_TF32)

    acc = acc * scale

    if HAS_BIAS:
        bias = tl.load(
            RelBias + h * stride_rb_h + offs_n[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += bias

    if HAS_MASK:
        mask_vals = tl.load(
            Mask
            + b * stride_mask_b
            + h * stride_mask_h
            + offs_n[:, None] * stride_mask_n
            + offs_n[None, :] * stride_mask_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += mask_vals

    acc = tl.where(mask_n[None, :], acc, float("-inf"))
    m = tl.max(acc, axis=1)
    acc = acc - m[:, None]
    p = tl.exp(acc)
    l = tl.sum(p, axis=1)
    attn = p / tl.maximum(l[:, None], 1e-6)

    for d0 in tl.static_range(0, 64, BLOCK_D):
        offs_dv = d0 + tl.arange(0, BLOCK_D)
        mask_dv = offs_dv < DV
        v_ptrs = base_v + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd
        v_chunk = tl.load(v_ptrs, mask=mask_n[:, None] & mask_dv[None, :], other=0.0).to(tl.float32)
        out_chunk = tl.dot(attn, v_chunk, allow_tf32=ALLOW_TF32).to(tl.float32)
        tl.store(
            base_out + offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od,
            out_chunk.to(Out.dtype.element_ty),
            mask=mask_n[:, None] & mask_dv[None, :],
        )


@triton.jit
def elsa_swinv2_kernel_short_fused(
    Q, K, V, Out,
    LogitScale, RelBias, Mask,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    B, H, N, D, DV,
    NUM_WINDOWS,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_IS_COMPACT: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    base_q = Q + b * stride_qb + h * stride_qh
    base_k = K + b * stride_kb + h * stride_kh
    base_v = V + b * stride_vb + h * stride_vh
    base_out = Out + b * stride_ob + h * stride_oh

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q = tl.load(
        base_q + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        base_k + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    scale = tl.exp(tl.load(LogitScale + h)).to(tl.float32)
    q = q * scale

    # Triton 3.x: `trans_b` arg was removed, use explicit transpose.
    scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32).to(tl.float32)

    if HAS_BIAS:
        bias = tl.load(
            RelBias + h * stride_rb_h + offs_n[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += bias

    if HAS_MASK:
        mask_b = b
        if MASK_IS_COMPACT:
            mask_b = b % NUM_WINDOWS
        mask_vals = tl.load(
            Mask
            + mask_b * stride_mask_b
            + h * stride_mask_h
            + offs_n[:, None] * stride_mask_n
            + offs_n[None, :] * stride_mask_m,
            mask=mask_n[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += mask_vals

    scores = tl.where(mask_n[None, :], scores, float("-inf"))
    m = tl.max(scores, axis=1)
    scores = scores - m[:, None]
    p = tl.exp(scores)
    l = tl.sum(p, axis=1)
    attn = p / tl.maximum(l[:, None], 1e-6)

    offs_dv = tl.arange(0, BLOCK_DV)
    mask_dv = offs_dv < DV
    v = tl.load(
        base_v + offs_n[:, None] * stride_vn + offs_dv[None, :] * stride_vd,
        mask=mask_n[:, None] & mask_dv[None, :],
        other=0.0,
    ).to(tl.float32)

    out = tl.dot(attn, v, allow_tf32=ALLOW_TF32).to(tl.float32)

    tl.store(
        base_out + offs_n[:, None] * stride_on + offs_dv[None, :] * stride_od,
        out.to(Out.dtype.element_ty),
        mask=mask_n[:, None] & mask_dv[None, :],
    )


@triton.jit
def kernel_elsa_attention_fwd_fixed(
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """
    修正的 ELSA Attention kernel - 兼容 Triton 3.2.0
    - 正確的 dtype 處理以使用 Tensor Core
    - 移除不支援的 acc_dtype 參數
    """
    # Program IDs
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    # 基礎偏移
    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh
    
    # M 維度範圍
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    
    # D 維度範圍
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # ===== 載入 Q block ===== #
    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    # 保持 FP16 以利用 Tensor Core
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    
    # 縮放 Q - 保持 FP16
    q = q * scale
    
    # ===== 初始化累積變量 ===== #
    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    # ===== 主循環 ===== #
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        # Causal mask
        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])
        
        # ===== 載入 K block (保持 FP16) ===== #
        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        
        # ===== 計算 QK^T ===== #
        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        
        # 轉換為 FP32 進行 softmax 計算
        qk = qk.to(tl.float32)
        
        # 應用 mask
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        
        # ===== Online softmax ===== #
        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        
        # exp2 is generally faster than exp on recent Triton/CUDA stacks.
        p = tl.exp2((qk - m_new[:, None]) * 1.4426950408889634)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        
        # 更新累積值
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # ===== 載入 V block (保持 FP16) ===== #
        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # ===== 累積輸出 ===== #
        p_cast = p.to(v.dtype)
        pv = tl.dot(p_cast, v)
        
        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        
        # 更新 m_i
        m_i = m_new
    
    # ===== 最終歸一化 ===== #
    if ACC_IN_FP16:
        acc = acc.to(tl.float32) / tl.maximum(l_i[:, None], 1e-6)
    else:
        acc = acc / tl.maximum(l_i[:, None], 1e-6)
    
    # ===== 寫回結果 ===== #
    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    # 轉回 FP16
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_attention_fwd_fixed_nomask(
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    USE_TF32: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """Fast path for fp16/bf16 inference when N/D are block-aligned.

    Assumptions enforced by caller:
    - non-causal
    - N % BLOCK_M == 0 and N % BLOCK_N == 0
    - D == BLOCK_D (no D-tail masking needed)
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    log2e = 1.4426950408889634
    num_blocks_n = N // BLOCK_N
    for block_id in range(num_blocks_n):
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs)

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs)
        pv = tl.dot(p.to(v.dtype), v)

        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

    if ACC_IN_FP16:
        acc = acc.to(tl.float32) / l_i[:, None]
    else:
        acc = acc / l_i[:, None]

    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype))


@triton.jit
def kernel_elsa_attention_fwd_fixed_nomask_flat(
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    USE_TF32: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """Same math as nomask kernel, but flattened BH launch order."""
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    log2e = 1.4426950408889634
    num_blocks_n = N // BLOCK_N
    for block_id in range(num_blocks_n):
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs)

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs)
        pv = tl.dot(p.to(v.dtype), v)

        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

    if ACC_IN_FP16:
        out = acc.to(tl.float32) / l_i[:, None]
    else:
        out = acc / l_i[:, None]

    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(q.dtype))


@triton.jit
def kernel_elsa_attention_fwd_fixed_nomask_fp16stats(
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
):
    """Aggressive fp16 fast path for aligned non-causal inference.

    This path keeps softmax running statistics and accumulation in fp16 to
    reduce fp32 reduction overhead on long sequences.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs)
    q = q * scale

    # fp16 stats + fp16 accumulator for speed.
    m_i = tl.full((BLOCK_M,), value=-65504.0, dtype=q.dtype)
    l_i = tl.zeros((BLOCK_M,), dtype=q.dtype)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)

    log2e = 1.4426950408889634
    num_blocks_n = N // BLOCK_N
    for block_id in range(num_blocks_n):
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs)

        qk = tl.dot(q, k).to(tl.float32)

        m_i_f32 = m_i.to(tl.float32)
        l_i_f32 = l_i.to(tl.float32)
        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i_f32, m_j)
        p_f32 = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha_f32 = tl.exp2((m_i_f32 - m_new) * log2e)
        l_i_f32 = l_i_f32 * alpha_f32 + tl.sum(p_f32, axis=1)

        p = p_f32.to(q.dtype)
        alpha = alpha_f32.to(q.dtype)
        l_i = l_i_f32.to(q.dtype)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs)
        pv = tl.dot(p.to(v.dtype), v).to(acc.dtype)

        acc = (acc * alpha[:, None] + pv).to(acc.dtype)
        m_i = m_new.to(q.dtype)

    out = acc.to(tl.float32) / tl.maximum(l_i.to(tl.float32)[:, None], 1e-3)
    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out.to(q.dtype))


@triton.jit
def kernel_elsa_attention_fwd_fixed_nomask_kblock(
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """Aligned non-causal path using block pointers and K transpose-view loads."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    bh_q = Q + pid_b * stride_qb + pid_h * stride_qh
    bh_k = K + pid_b * stride_kb + pid_h * stride_kh
    bh_v = V + pid_b * stride_vb + pid_h * stride_vh
    bh_o = Out + pid_b * stride_ob + pid_h * stride_oh

    q_ptr = tl.make_block_ptr(
        base=bh_q,
        shape=(N, D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(q_ptr)
    q = q * scale

    # K uses a transpose-view (D, N) to make tl.dot(q, k_block) direct.
    k_ptr = tl.make_block_ptr(
        base=bh_k,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )
    v_ptr = tl.make_block_ptr(
        base=bh_v,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    log2e = 1.4426950408889634

    num_blocks_n = N // BLOCK_N
    for _ in range(num_blocks_n):
        k = tl.load(k_ptr)
        qk = tl.dot(q, k).to(tl.float32)

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(v_ptr)
        pv = tl.dot(p.to(v.dtype), v)
        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    if ACC_IN_FP16:
        out = acc.to(tl.float32) / l_i[:, None]
    else:
        out = acc / l_i[:, None]
    out_ptr = tl.make_block_ptr(
        base=bh_o,
        shape=(N, D),
        strides=(stride_on, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(out_ptr, out.to(q.dtype))


@triton.jit
def kernel_elsa_attention_fwd_fixed_bias_kblock(
    Q, K, V, LogitScale, RelBias, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """Aligned non-causal bias-only path using block pointers and K transpose-view loads."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    bh_q = Q + pid_b * stride_qb + pid_h * stride_qh
    bh_k = K + pid_b * stride_kb + pid_h * stride_kh
    bh_v = V + pid_b * stride_vb + pid_h * stride_vh
    bh_o = Out + pid_b * stride_ob + pid_h * stride_oh
    bh_bias = RelBias + pid_h * stride_rb_h

    q_ptr = tl.make_block_ptr(
        base=bh_q,
        shape=(N, D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(q_ptr)
    logit_scale_val = tl.load(LogitScale + pid_h)
    scale_tc = tl.exp(logit_scale_val.to(tl.float32))

    k_ptr = tl.make_block_ptr(
        base=bh_k,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )
    v_ptr = tl.make_block_ptr(
        base=bh_v,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    log2e = 1.4426950408889634

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    num_blocks_n = N // BLOCK_N
    for block_id in range(num_blocks_n):
        k = tl.load(k_ptr)
        qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False) * scale_tc

        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptrs = bh_bias + offs_m[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m
        bias = tl.load(bias_ptrs).to(tl.float32)
        qk = qk + bias

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(v_ptr)
        pv = tl.dot(p.to(v.dtype), v)
        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    if ACC_IN_FP16:
        out = acc.to(tl.float32) / l_i[:, None]
    else:
        out = acc / l_i[:, None]
    out_ptr = tl.make_block_ptr(
        base=bh_o,
        shape=(N, D),
        strides=(stride_on, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(out_ptr, out.to(q.dtype))


@triton.jit
def kernel_elsa_attention_fwd_fixed_bias_compactmask_block8(
    Q, K, V, LogitScale, RelBias, Mask, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    ACC_IN_FP16: tl.constexpr,
):
    """Compact-mask path specialized for W16 shifted windows with exact 8x8 block gates."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    bh_q = Q + pid_b * stride_qb + pid_h * stride_qh
    bh_k = K + pid_b * stride_kb + pid_h * stride_kh
    bh_v = V + pid_b * stride_vb + pid_h * stride_vh
    bh_o = Out + pid_b * stride_ob + pid_h * stride_oh
    bh_bias = RelBias + pid_h * stride_rb_h
    mask_b = pid_b % NUM_WINDOWS
    bh_mask = Mask + mask_b * stride_mask_b + pid_h * stride_mask_h

    q_ptr = tl.make_block_ptr(
        base=bh_q,
        shape=(N, D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(q_ptr)
    logit_scale_val = tl.load(LogitScale + pid_h)
    scale_tc = tl.exp(logit_scale_val.to(tl.float32))

    k_ptr = tl.make_block_ptr(
        base=bh_k,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )
    v_ptr = tl.make_block_ptr(
        base=bh_v,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    log2e = 1.4426950408889634

    num_blocks_n = N // BLOCK_N
    gate0 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 0 * stride_mask_m).to(tl.float32) > -1.0
    gate1 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 1 * BLOCK_N * stride_mask_m).to(tl.float32) > -1.0
    gate2 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 2 * BLOCK_N * stride_mask_m).to(tl.float32) > -1.0
    gate3 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 3 * BLOCK_N * stride_mask_m).to(tl.float32) > -1.0
    use_checkerboard = (gate0 != gate1) and (gate0 == gate2) and (gate1 == gate3)

    if use_checkerboard:
        start_block = 0 if gate0 else 1
        k_ptr = tl.advance(k_ptr, (0, start_block * BLOCK_N))
        v_ptr = tl.advance(v_ptr, (start_block * BLOCK_N, 0))
        for half_block_id in range(num_blocks_n // 2):
            block_id = start_block + 2 * half_block_id
            offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)

            k = tl.load(k_ptr)
            qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False) * scale_tc

            bias_ptrs = bh_bias + offs_m[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m
            bias = tl.load(bias_ptrs).to(tl.float32)
            qk = qk + bias

            m_j = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_j)
            p = tl.exp2((qk - m_new[:, None]) * log2e)
            alpha = tl.exp2((m_i - m_new) * log2e)

            l_i = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(v_ptr)
            pv = tl.dot(p.to(v.dtype), v)
            if ACC_IN_FP16:
                acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
            else:
                acc = acc * alpha[:, None] + pv.to(tl.float32)
            m_i = m_new

            k_ptr = tl.advance(k_ptr, (0, 2 * BLOCK_N))
            v_ptr = tl.advance(v_ptr, (2 * BLOCK_N, 0))
    else:
        for block_id in range(num_blocks_n):
            offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
            gate_ptr = bh_mask + offs_m[0] * stride_mask_n + offs_n[0] * stride_mask_m
            mask_gate = tl.load(gate_ptr).to(tl.float32)
            if mask_gate < -1.0:
                k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
                v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))
                continue

            k = tl.load(k_ptr)
            qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False) * scale_tc

            bias_ptrs = bh_bias + offs_m[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m
            bias = tl.load(bias_ptrs).to(tl.float32)
            qk = qk + bias

            m_j = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_j)
            p = tl.exp2((qk - m_new[:, None]) * log2e)
            alpha = tl.exp2((m_i - m_new) * log2e)

            l_i = l_i * alpha + tl.sum(p, axis=1)

            v = tl.load(v_ptr)
            pv = tl.dot(p.to(v.dtype), v)
            if ACC_IN_FP16:
                acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
            else:
                acc = acc * alpha[:, None] + pv.to(tl.float32)
            m_i = m_new

            k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
            v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    if ACC_IN_FP16:
        out = acc.to(tl.float32) / l_i[:, None]
    else:
        out = acc / l_i[:, None]
    out_ptr = tl.make_block_ptr(
        base=bh_o,
        shape=(N, D),
        strides=(stride_on, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(out_ptr, out.to(q.dtype))


@triton.jit
def kernel_elsa_attention_fwd_fixed_bias_compactmask_kblock(
    Q, K, V, LogitScale, RelBias, Mask, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    NUM_WINDOWS: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """Aligned non-causal bias + compact-mask path using block pointers and K transpose-view loads."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    bh_q = Q + pid_b * stride_qb + pid_h * stride_qh
    bh_k = K + pid_b * stride_kb + pid_h * stride_kh
    bh_v = V + pid_b * stride_vb + pid_h * stride_vh
    bh_o = Out + pid_b * stride_ob + pid_h * stride_oh
    bh_bias = RelBias + pid_h * stride_rb_h
    mask_b = pid_b % NUM_WINDOWS
    bh_mask = Mask + mask_b * stride_mask_b + pid_h * stride_mask_h

    q_ptr = tl.make_block_ptr(
        base=bh_q,
        shape=(N, D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(q_ptr)
    logit_scale_val = tl.load(LogitScale + pid_h)
    scale_tc = tl.exp(logit_scale_val.to(tl.float32))

    k_ptr = tl.make_block_ptr(
        base=bh_k,
        shape=(D, N),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_D, BLOCK_N),
        order=(0, 1),
    )
    v_ptr = tl.make_block_ptr(
        base=bh_v,
        shape=(N, D),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_D),
        order=(1, 0),
    )

    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    log2e = 1.4426950408889634

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if N == 256 and BLOCK_N == 64:
        gate0 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 0 * stride_mask_m).to(tl.float32)
        gate1 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 64 * stride_mask_m).to(tl.float32)
        gate2 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 128 * stride_mask_m).to(tl.float32)
        gate3 = tl.load(bh_mask + offs_m[0] * stride_mask_n + 192 * stride_mask_m).to(tl.float32)
        valid0 = gate0 > -1.0
        valid1 = gate1 > -1.0
        valid2 = gate2 > -1.0
        valid3 = gate3 > -1.0
        valid_count = valid0.to(tl.int32) + valid1.to(tl.int32) + valid2.to(tl.int32) + valid3.to(tl.int32)
        if valid_count == 1:
            selected_n = tl.where(
                valid0,
                0,
                tl.where(valid1, 64, tl.where(valid2, 128, 192)),
            )
            offs_n = selected_n + tl.arange(0, BLOCK_N)
            offs_d = tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            k_ptrs = bh_k + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
            v_ptrs = bh_v + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
            k = tl.load(k_ptrs, mask=mask_d[:, None], other=0.0)
            qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False) * scale_tc

            bias_ptrs = bh_bias + offs_m[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m
            bias = tl.load(bias_ptrs).to(tl.float32)
            qk = qk + bias

            m_j = tl.max(qk, axis=1)
            p = tl.exp2((qk - m_j[:, None]) * log2e)
            l_j = tl.sum(p, axis=1)
            v = tl.load(v_ptrs, mask=mask_d[None, :], other=0.0)
            pv = tl.dot(p.to(v.dtype), v)
            if ACC_IN_FP16:
                out = pv.to(tl.float32) / l_j[:, None]
            else:
                out = pv.to(tl.float32) / l_j[:, None]
            out_ptr = tl.make_block_ptr(
                base=bh_o,
                shape=(N, D),
                strides=(stride_on, stride_od),
                offsets=(pid_m * BLOCK_M, 0),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            tl.store(out_ptr, out.to(q.dtype))
            return

    num_blocks_n = N // BLOCK_N
    for block_id in range(num_blocks_n):
        offs_n = block_id * BLOCK_N + tl.arange(0, BLOCK_N)
        bias_ptrs = bh_bias + offs_m[:, None] * stride_rb_n + offs_n[None, :] * stride_rb_m
        if N == 256 and BLOCK_N == 64:
            gate_ptr = bh_mask + offs_m[0] * stride_mask_n + offs_n[0] * stride_mask_m
            mask_gate = tl.load(gate_ptr).to(tl.float32)
            if mask_gate < -1.0:
                k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
                v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))
                continue

        k = tl.load(k_ptr)
        qk = tl.dot(q, k, out_dtype=tl.float32, allow_tf32=False) * scale_tc

        bias = tl.load(bias_ptrs).to(tl.float32)
        if N == 256 and BLOCK_N == 64:
            qk = qk + bias
        else:
            mask_ptrs = bh_mask + offs_m[:, None] * stride_mask_n + offs_n[None, :] * stride_mask_m
            mask_vals = tl.load(mask_ptrs).to(tl.float32)
            qk = qk + bias + mask_vals

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)

        l_i = l_i * alpha + tl.sum(p, axis=1)

        v = tl.load(v_ptr)
        pv = tl.dot(p.to(v.dtype), v)
        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

        k_ptr = tl.advance(k_ptr, (0, BLOCK_N))
        v_ptr = tl.advance(v_ptr, (BLOCK_N, 0))

    if ACC_IN_FP16:
        out = acc.to(tl.float32) / l_i[:, None]
    else:
        out = acc / l_i[:, None]
    out_ptr = tl.make_block_ptr(
        base=bh_o,
        shape=(N, D),
        strides=(stride_on, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(out_ptr, out.to(q.dtype))


@triton.jit
def kernel_elsa_attention_fwd_fp16_tc(
    Q, K, V, Out,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
):
    """Tensor-core oriented fp16 forward kernel for aligned non-causal inference."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    bh_q = Q + pid_b * stride_qb + pid_h * stride_qh
    bh_k = K + pid_b * stride_kb + pid_h * stride_kh
    bh_v = V + pid_b * stride_vb + pid_h * stride_vh
    bh_o = Out + pid_b * stride_ob + pid_h * stride_oh

    q_ptr = tl.make_block_ptr(
        base=bh_q,
        shape=(N, D),
        strides=(stride_qn, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    q = tl.load(q_ptr)
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    log2e = 1.4426950408889634

    for start_n in range(0, N, BLOCK_N):
        k_ptr = tl.make_block_ptr(
            base=bh_k,
            shape=(N, D),
            strides=(stride_kn, stride_kd),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        v_ptr = tl.make_block_ptr(
            base=bh_v,
            shape=(N, D),
            strides=(stride_vn, stride_vd),
            offsets=(start_n, 0),
            block_shape=(BLOCK_N, BLOCK_D),
            order=(1, 0),
        )
        k = tl.load(k_ptr)
        v = tl.load(v_ptr)

        qk = tl.dot(q, tl.trans(k))
        qk = qk.to(tl.float32)

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    out_ptr = tl.make_block_ptr(
        base=bh_o,
        shape=(N, D),
        strides=(stride_on, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_D),
        order=(1, 0),
    )
    tl.store(out_ptr, out.to(tl.float16))


@triton.jit
def kernel_elsa_attention_fwd_fixed_mz(
    Q, K, V, Out, Out_M, Out_Z,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh, stride_mn,
    stride_zb, stride_zh, stride_zn,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """Training variant that also writes per-row max (M) and sum-exp (Z)."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_blocks_n = tl.cdiv(N, BLOCK_N)
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * 1.4426950408889634)
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        pv = tl.dot(p.to(v.dtype), v)
        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)

        m_i = m_new

    if ACC_IN_FP16:
        acc = acc.to(tl.float32) / tl.maximum(l_i[:, None], 1e-6)
    else:
        acc = acc / tl.maximum(l_i[:, None], 1e-6)

    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])

    m_ptrs = Out_M + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mn
    z_ptrs = Out_Z + pid_b * stride_zb + pid_h * stride_zh + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)


@triton.jit
def kernel_elsa_attention_fwd_fixed_mz_split(
    Q, K, V, Out, Out_M, Out_Z,
    # Strides
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh, stride_mn,
    stride_zb, stride_zh, stride_zn,
    # Shape
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    # Config
    scale: tl.constexpr,
    USE_TF32: tl.constexpr,
    ACC_IN_FP16: tl.constexpr,
):
    """Training forward (no causal): split full/tailed KV blocks to reduce mask overhead."""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    if ACC_IN_FP16:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=q.dtype)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    log2e = 1.4426950408889634
    num_full_blocks_n = N // BLOCK_N
    rem_n = N - num_full_blocks_n * BLOCK_N

    # Full KV blocks: avoid per-iteration tail masking.
    for block_id in range(num_full_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None], other=0.0)

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_d[None, :], other=0.0)
        pv = tl.dot(p.to(v.dtype), v)
        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

    # Tail KV block (only when N is not divisible by BLOCK_N).
    if rem_n > 0:
        start_n = num_full_blocks_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp2((qk - m_new[:, None]) * log2e)
        alpha = tl.exp2((m_i - m_new) * log2e)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        pv = tl.dot(p.to(v.dtype), v)
        if ACC_IN_FP16:
            acc = acc * alpha[:, None].to(acc.dtype) + pv.to(acc.dtype)
        else:
            acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

    if ACC_IN_FP16:
        acc = acc.to(tl.float32) / tl.maximum(l_i[:, None], 1e-6)
    else:
        acc = acc / tl.maximum(l_i[:, None], 1e-6)

    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])

    m_ptrs = Out_M + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mn
    z_ptrs = Out_Z + pid_b * stride_zb + pid_h * stride_zh + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)


@triton.jit
def kernel_elsa_attention_fwd_qknorm(
    Q, K, V, Out,
    Q_norm_w, Q_norm_b, K_norm_w, K_norm_b,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    """支援 QK normalization 的版本"""
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)
    
    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N
    
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # 載入 norm weights/bias
    q_w = tl.load(Q_norm_w + offs_d, mask=mask_d, other=1.0)
    q_b = tl.load(Q_norm_b + offs_d, mask=mask_d, other=0.0)
    k_w = tl.load(K_norm_w + offs_d, mask=mask_d, other=1.0)
    k_b = tl.load(K_norm_b + offs_d, mask=mask_d, other=0.0)
    
    # 載入並正規化 Q
    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    
    # 簡化的 LayerNorm (假設已預先計算 mean/std)
    q = q * q_w[None, :] + q_b[None, :]
    q = q * scale
    
    # 初始化
    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    
    num_blocks_n = tl.cdiv(N, BLOCK_N)
    
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        
        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])
        
        # 載入並正規化 K
        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        k = k * k_w[:, None] + k_b[:, None]
        
        # QK^T
        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))
        
        # Softmax
        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # V
        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # 累積
        p_cast = p.to(v.dtype)
        pv = tl.dot(p_cast, v)
        acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new
    
    # 歸一化並存儲
    acc = acc / tl.maximum(l_i[:, None], 1e-6)
    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_attention_fwd_qknorm_mz(
    Q, K, V, Out, Out_M, Out_Z,
    Q_norm_w, Q_norm_b, K_norm_w, K_norm_b,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_mb, stride_mh, stride_mn,
    stride_zb, stride_zh, stride_zn,
    B: tl.constexpr, H: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    scale: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    USE_TF32: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    batch_head_offset = pid_b * stride_qb + pid_h * stride_qh

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offs_m < N

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_w = tl.load(Q_norm_w + offs_d, mask=mask_d, other=1.0)
    q_b = tl.load(Q_norm_b + offs_d, mask=mask_d, other=0.0)
    k_w = tl.load(K_norm_w + offs_d, mask=mask_d, other=1.0)
    k_b = tl.load(K_norm_b + offs_d, mask=mask_d, other=0.0)

    q_ptrs = Q + batch_head_offset + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    q = q * q_w[None, :] + q_b[None, :]
    q = q * scale

    m_i = tl.full((BLOCK_M,), value=-float('inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    num_blocks_n = tl.cdiv(N, BLOCK_N)
    for block_id in range(num_blocks_n):
        start_n = block_id * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        if IS_CAUSAL:
            mask_n = mask_n & (offs_m[:, None] >= offs_n[None, :])

        k_ptrs = K + batch_head_offset + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0)
        k = k * k_w[:, None] + k_b[:, None]

        if USE_TF32:
            qk = tl.dot(q, k, input_precision="tf32")
        else:
            qk = tl.dot(q, k)
        qk = qk.to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, -float('inf'))

        m_j = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_j)
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)

        v_ptrs = V + batch_head_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        pv = tl.dot(p.to(v.dtype), v)
        acc = acc * alpha[:, None] + pv.to(tl.float32)

        m_i = m_new

    acc = acc / tl.maximum(l_i[:, None], 1e-6)

    out_ptrs = Out + batch_head_offset + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(q.dtype), mask=mask_m[:, None] & mask_d[None, :])

    m_ptrs = Out_M + pid_b * stride_mb + pid_h * stride_mh + offs_m * stride_mn
    z_ptrs = Out_Z + pid_b * stride_zb + pid_h * stride_zh + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)

@triton.jit
def kernel_elsa_attention_fp32_fast(
    Q, K, V, OUT,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=ALLOW_TF32)

        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-6)
    out = acc * inv_l[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_attention_fp32_fast_mz(
    Q, K, V, OUT, OUT_M, OUT_Z,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    stride_mb, stride_mn,
    stride_zb, stride_zn,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=ALLOW_TF32)

        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-6)
    out = acc * inv_l[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])

    m_ptrs = OUT_M + pid_bh * stride_mb + offs_m * stride_mn
    z_ptrs = OUT_Z + pid_bh * stride_zb + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)


@triton.jit
def kernel_elsa_attention_fp32_stats(
    Q, K, OUT_M, OUT_Z,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_mb, stride_mn,
    stride_zb, stride_zn,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    m_ptrs = OUT_M + pid_bh * stride_mb + offs_m * stride_mn
    z_ptrs = OUT_Z + pid_bh * stride_zb + offs_m * stride_zn
    tl.store(m_ptrs, m_i, mask=mask_m)
    tl.store(z_ptrs, l_i, mask=mask_m)


@triton.jit
def kernel_elsa_attention_fp32_rowmax(
    Q, K, OUT_M,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_mb, stride_mn,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_i = tl.maximum(m_i, m_curr)

    m_ptrs = OUT_M + pid_bh * stride_mb + offs_m * stride_mn
    tl.store(m_ptrs, m_i, mask=mask_m)


@triton.jit
def kernel_elsa_attention_fp32_rowmax_lse(
    Q, K, Z, OUT_LSE,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_zb, stride_zn,
    stride_lb, stride_ln,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    N_PAD: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_out = offs_m < N_PAD
    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))
        # For padded query rows, force -inf.
        scores = tl.where(mask_m[:, None], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_i = tl.maximum(m_i, m_curr)

    z_ptrs = Z + pid_bh * stride_zb + offs_m * stride_zn
    z = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
    lse = m_i + tl.log(tl.maximum(z, 1e-20))
    lse = tl.where(mask_m, lse, float("inf"))

    l_ptrs = OUT_LSE + pid_bh * stride_lb + offs_m * stride_ln
    tl.store(l_ptrs, lse, mask=mask_out)

def _mem_autotune_configs():
    return [
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 256}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_mem_autotune_configs(), key=["N_CTX", "D_HEAD"])
@triton.jit
def kernel_elsa_attention_fp32_fast_tuned(
    Q, K, V, OUT,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        if IS_CAUSAL:
            causal = offs_m[:, None] >= offs_n_block[None, :]
            scores = tl.where(causal, scores, float("-inf"))
        scores = tl.where(mask_n[None, :], scores, float("-inf"))

        m_curr = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_curr)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(mask_n[None, :], p, 0.0)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v, allow_tf32=ALLOW_TF32)

        m_i = m_new

    inv_l = 1.0 / tl.maximum(l_i, 1e-6)
    out = acc * inv_l[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_d[None, :])
    
@triton.jit
def kernel_integral_mhsa_stable(
    Q, K, V,
    OUT_S, OUT_Z, OUT_M,
    BH, N_CTX: tl.constexpr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_sbh, stride_sn, stride_sd,
    stride_zh, stride_z0, stride_z1,
    stride_mh, stride_m0, stride_m1,
    BLOCK_Q : tl.constexpr,   
    BLOCK_N : tl.constexpr,   
    D_HEAD  : tl.constexpr,   
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_q  = tl.program_id(0)               
    pid_bh = tl.program_id(1)               

    offs_q  = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)      
    offs_n  = tl.arange(0, BLOCK_N)                        
    offs_d  = tl.arange(0, D_HEAD)                         
    mask_q  = offs_q < N_CTX

    # ---- load Q (保持原始邏輯) ----------------------------------- #
    q_ptrs = Q + pid_bh*stride_qbh + offs_q[:,None]*stride_qn + offs_d[None,:]*stride_qd
    q      = tl.load(q_ptrs, mask=mask_q[:,None]).to(tl.float32)

    # ---- running stats (使用 D_HEAD 而非 PAD_D) ------------------------------------ #
    m_q = tl.full((BLOCK_Q,), -1e9, tl.float32)
    z_q = tl.zeros((BLOCK_Q,), tl.float32)
    s_q = tl.zeros((BLOCK_Q, D_HEAD), tl.float32)

    # ---- sweep over sequence (保持原始邏輯) ------------------------------ #
    for start_n in range(0, N_CTX, BLOCK_N):
        mask_n = (start_n + offs_n) < N_CTX
        
        k_ptrs = K + pid_bh * stride_kbh + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
        v_ptrs = V + pid_bh*stride_vbh + (start_n+offs_n)[:,None]*stride_vn + offs_d[None,:]*stride_vd
        
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:,None], other=0.).to(tl.float32)
        
        m_prev = m_q
        scores = tl.dot(q, k, allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, -1e30)  # 使用更小的值
        
        cur_m = tl.max(scores, 1)
        new_m = tl.maximum(m_prev, cur_m)
        
        alpha = tl.exp(m_prev - new_m)
        beta = tl.exp(scores - new_m[:,None])
        beta = tl.where(mask_n[None,:], beta, 0.0)  # 確保masked位置為0
        
        z_q = z_q * alpha + tl.sum(beta, 1)
        s_q = s_q * alpha[:,None] + tl.dot(beta, v, allow_tf32=ALLOW_TF32)
        
        m_q = new_m
    # ---- 關鍵修復：添加數值穩定性保護 ------- #
    # z_q_safe = tl.maximum(z_q, 1e-8)
    # result = s_q / z_q_safe[:,None]
    
    s_ptrs = OUT_S + pid_bh*stride_sbh + offs_q[:,None]*stride_sn + offs_d[None,:]*stride_sd
    z_ptrs = OUT_Z + pid_bh*stride_z0 + offs_q*stride_z1
    m_ptrs = OUT_M + pid_bh*stride_m0 + offs_q*stride_m1
    
    mask_sd = mask_q[:,None]
    tl.store(s_ptrs, s_q, mask=mask_sd)  # 存儲 s_q，不是 z_q！
    tl.store(z_ptrs, z_q, mask=mask_q)   # 存儲 z_q
    tl.store(m_ptrs, m_q, mask=mask_q)   # 存儲 m_q


@triton.jit
def kernel_integral_mhsa_stable_infer(
    Q, K, V,
    OUT,
    BH, N_CTX: tl.constexpr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)
    mask_q = offs_q < N_CTX

    q_ptrs = Q + pid_bh * stride_qbh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_q[:, None]).to(tl.float32)

    m_q = tl.full((BLOCK_Q,), -1e9, tl.float32)
    z_q = tl.zeros((BLOCK_Q,), tl.float32)
    s_q = tl.zeros((BLOCK_Q, D_HEAD), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        mask_n = (start_n + offs_n) < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
        v_ptrs = V + pid_bh * stride_vbh + (start_n + offs_n)[:, None] * stride_vn + offs_d[None, :] * stride_vd

        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        m_prev = m_q
        scores = tl.dot(q, k, allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, -1e30)

        cur_m = tl.max(scores, 1)
        new_m = tl.maximum(m_prev, cur_m)

        alpha = tl.exp(m_prev - new_m)
        beta = tl.exp(scores - new_m[:, None])
        beta = tl.where(mask_n[None, :], beta, 0.0)

        z_q = z_q * alpha + tl.sum(beta, 1)
        s_q = s_q * alpha[:, None] + tl.dot(beta, v, allow_tf32=ALLOW_TF32)

        m_q = new_m

    inv_z = 1.0 / tl.maximum(z_q, 1e-6)
    out = s_q * inv_z[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_q[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_q[:, None])


@triton.jit
def kernel_integral_mhsa_splitd_infer(
    Q, K, V,
    OUT,
    BH, N_CTX: tl.constexpr,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_obh, stride_on, stride_od,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    D_HEAD: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = tl.arange(0, BLOCK_N)
    mask_q = offs_q < N_CTX

    m_q = tl.full((BLOCK_Q,), -1e9, tl.float32)
    z_q = tl.zeros((BLOCK_Q,), tl.float32)
    s_q = tl.zeros((BLOCK_Q, D_HEAD), tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        mask_n = (start_n + offs_n) < N_CTX
        scores = tl.zeros((BLOCK_Q, BLOCK_N), tl.float32)
        for start_d in range(0, D_HEAD, BLOCK_D):
            offs_d = start_d + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D_HEAD
            q_ptrs = Q + pid_bh * stride_qbh + offs_q[:, None] * stride_qn + offs_d[None, :] * stride_qd
            k_ptrs = K + pid_bh * stride_kbh + offs_d[:, None] * stride_kd + (start_n + offs_n)[None, :] * stride_kn
            q = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            k = tl.load(k_ptrs, mask=mask_d[:, None] & mask_n[None, :], other=0.0).to(tl.float32)
            scores += tl.dot(q, k, allow_tf32=ALLOW_TF32)

        scores = scores * SCALE
        scores = tl.where(mask_n[None, :], scores, -1e30)

        cur_m = tl.max(scores, 1)
        new_m = tl.maximum(m_q, cur_m)

        alpha = tl.exp(m_q - new_m)
        beta = tl.exp(scores - new_m[:, None])
        beta = tl.where(mask_n[None, :], beta, 0.0)

        z_q = z_q * alpha + tl.sum(beta, 1)

        v_ptrs = V + pid_bh * stride_vbh + (start_n + offs_n)[:, None] * stride_vn + tl.arange(0, D_HEAD)[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)
        s_q = s_q * alpha[:, None] + tl.dot(beta, v, allow_tf32=ALLOW_TF32)

        m_q = new_m

    inv_z = 1.0 / tl.maximum(z_q, 1e-6)
    out = s_q * inv_z[:, None]

    out_ptrs = OUT + pid_bh * stride_obh + offs_q[:, None] * stride_on + tl.arange(0, D_HEAD)[None, :] * stride_od
    tl.store(out_ptrs, out, mask=mask_q[:, None])


@triton.jit
def kernel_elsa_bwd_delta(
    Q, K, V, DO, M, Z, DELTA,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_dobh, stride_don, stride_dod,
    stride_mbh, stride_mn,
    stride_zbh, stride_zn,
    stride_dbh, stride_dn,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    do_ptrs = DO + pid_bh * stride_dobh + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_ptrs = M + pid_bh * stride_mbh + offs_m * stride_mn
    z_ptrs = Z + pid_bh * stride_zbh + offs_m * stride_zn
    m = tl.load(m_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
    z = tl.maximum(z, 1e-6)

    delta = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, float("-inf"))
        p = tl.exp(scores - m[:, None]) / z[:, None]
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
        delta += tl.sum(dp * p, axis=1)

    delta_ptrs = DELTA + pid_bh * stride_dbh + offs_m * stride_dn
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def kernel_elsa_bwd_dq(
    Q, K, V, DO, M, Z, DELTA, DQ,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_dobh, stride_don, stride_dod,
    stride_mbh, stride_mn,
    stride_zbh, stride_zn,
    stride_dbh, stride_dn,
    stride_dqbh, stride_dqn, stride_dqd,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < N_CTX
    mask_d = offs_d < D_HEAD

    q_ptrs = Q + pid_bh * stride_qbh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    do_ptrs = DO + pid_bh * stride_dobh + offs_m[:, None] * stride_don + offs_d[None, :] * stride_dod
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    m_ptrs = M + pid_bh * stride_mbh + offs_m * stride_mn
    z_ptrs = Z + pid_bh * stride_zbh + offs_m * stride_zn
    d_ptrs = DELTA + pid_bh * stride_dbh + offs_m * stride_dn
    m = tl.load(m_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
    d = tl.load(d_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    z = tl.maximum(z, 1e-6)

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_block = start_n + offs_n
        mask_n = offs_n_block < N_CTX

        k_ptrs = K + pid_bh * stride_kbh + offs_n_block[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = V + pid_bh * stride_vbh + offs_n_block[:, None] * stride_vn + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_n[None, :], scores, float("-inf"))
        p = tl.exp(scores - m[:, None]) / z[:, None]
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
        ds = (dp - d[:, None]) * p
        acc += tl.dot(ds, k, allow_tf32=ALLOW_TF32) * SCALE

    dq_ptrs = DQ + pid_bh * stride_dqbh + offs_m[:, None] * stride_dqn + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def kernel_elsa_bwd_dkv(
    Q, K, V, DO, M, Z, DELTA, DK, DV,
    stride_qbh, stride_qn, stride_qd,
    stride_kbh, stride_kn, stride_kd,
    stride_vbh, stride_vn, stride_vd,
    stride_dobh, stride_don, stride_dod,
    stride_mbh, stride_mn,
    stride_zbh, stride_zn,
    stride_dbh, stride_dn,
    stride_dkbh, stride_dkn, stride_dkd,
    stride_dvbh, stride_dvn, stride_dvd,
    BH: tl.constexpr,
    N_CTX: tl.constexpr,
    D_HEAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SCALE: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_n = offs_n < N_CTX
    mask_d = offs_d < D_HEAD

    k_ptrs = K + pid_bh * stride_kbh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    v_ptrs = V + pid_bh * stride_vbh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    acc_k = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)
    acc_v = tl.zeros((BLOCK_N, BLOCK_D), dtype=tl.float32)

    for start_m in range(0, N_CTX, BLOCK_M):
        offs_m_block = start_m + offs_m
        mask_m = offs_m_block < N_CTX

        q_ptrs = Q + pid_bh * stride_qbh + offs_m_block[:, None] * stride_qn + offs_d[None, :] * stride_qd
        do_ptrs = DO + pid_bh * stride_dobh + offs_m_block[:, None] * stride_don + offs_d[None, :] * stride_dod
        q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        m_ptrs = M + pid_bh * stride_mbh + offs_m_block * stride_mn
        z_ptrs = Z + pid_bh * stride_zbh + offs_m_block * stride_zn
        d_ptrs = DELTA + pid_bh * stride_dbh + offs_m_block * stride_dn
        m = tl.load(m_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        z = tl.load(z_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        d = tl.load(d_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        z = tl.maximum(z, 1e-6)

        scores = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * SCALE
        scores = tl.where(mask_m[:, None] & mask_n[None, :], scores, float("-inf"))
        p = tl.exp(scores - m[:, None]) / z[:, None]
        p = tl.where(mask_m[:, None] & mask_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
        ds = (dp - d[:, None]) * p

        acc_k += tl.dot(tl.trans(ds), q, allow_tf32=ALLOW_TF32) * SCALE
        acc_v += tl.dot(tl.trans(p), do, allow_tf32=ALLOW_TF32)

    dk_ptrs = DK + pid_bh * stride_dkbh + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + pid_bh * stride_dvbh + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd
    tl.store(dk_ptrs, acc_k, mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dv_ptrs, acc_v, mask=mask_n[:, None] & mask_d[None, :])

class ELSA_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, qk_norm_weights=None, is_causal=False):
        B, H, N, D = q.shape

        if not q.is_cuda:
            from elsa_cpu import elsa_forward
            return elsa_forward(q, k, v, float(scale), bool(is_causal))

        needs_grad = q.requires_grad or k.requires_grad or v.requires_grad
        if needs_grad and qk_norm_weights is not None:
            raise RuntimeError("ELSA_triton backward does not support qk_norm.")
        
        # Keep legacy default as contiguous copies for stability/perf parity.
        contig_mode = os.environ.get("ELSA_TRITON_TRAIN_CONTIG", "1").strip().lower()
        if contig_mode in ("1", "true", "on", "yes", "force"):
            force_contig = True
        elif contig_mode in ("0", "false", "off", "no"):
            force_contig = False
        else:
            # Auto policy: avoid q/k/v copy on fp16/bf16 training tensors when
            # last-dim is already contiguous (common ViT/Swin layout).
            force_contig = not (
                needs_grad
                and qk_norm_weights is None
                and q.dtype in (torch.float16, torch.bfloat16)
                and q.stride(-1) == 1
                and k.stride(-1) == 1
                and v.stride(-1) == 1
            )

        if force_contig:
            q = q if q.is_contiguous() else q.contiguous()
            k = k if k.is_contiguous() else k.contiguous()
            v = v if v.is_contiguous() else v.contiguous()

        out = torch.empty((B, H, N, D), device=q.device, dtype=q.dtype)
        use_tf32 = bool(
            q.dtype == torch.float32
            and k.dtype == torch.float32
            and v.dtype == torch.float32
            and torch.backends.cuda.matmul.allow_tf32
        )
        
        # Block 大小
        BLOCK_D = 16 * ((D + 15) // 16)
        
        # 1. 取得序列長度與 GPU 性能
        dev_prop = torch.cuda.get_device_properties(q.device)
        blk = _choose_tile(N, dev_prop, prefer_large=True)

        stream_env = os.environ.get("ELSA_TRITON_STREAM", "0") == "1"
        if stream_env:
            try:
                stream_q = int(os.environ.get("ELSA_STREAM_Q_BLOCK", "0"))
            except ValueError:
                stream_q = 0
            try:
                stream_kv = int(os.environ.get("ELSA_STREAM_KV_BLOCK", "0"))
            except ValueError:
                stream_kv = 0
        
        # 2. 根據 blk 選擇對應 warp/stage（保持簡易）
        if blk == 128:
            BLOCK_M = BLOCK_N = 128 if D <= 64 else 64
            num_warps = 4 if D <= 64 else 8
        elif blk == 96:
            BLOCK_M = BLOCK_N = 96
            num_warps = 4
        else:  # 64
            BLOCK_M = BLOCK_N = 64
            num_warps = 4
        num_stages = 2

        # fp16/bf16 tuned defaults (A100 + CUDA 12.6 + Triton 3.3.1).
        # Training route (needs_grad) is tuned separately from inference route.
        if q.dtype in (torch.float16, torch.bfloat16):
            if needs_grad and D <= 128 and (not is_causal):
                if N >= 1024:
                    # Long fp16 train/ft (ViT): larger Q tile improves full-step throughput.
                    BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 4, 2
                elif N >= 512:
                    # Mid-range fp16 train/ft keeps better balance with wider K/V tile.
                    BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 128, 4, 2
            else:
                if N >= 4096 and D <= 64:
                    BLOCK_M, BLOCK_N, num_warps, num_stages = 128, 64, 4, 2
                elif N >= 4096:
                    BLOCK_M, BLOCK_N, num_warps, num_stages = 64, 64, 4, 2

            def _read_env_int(name: str, default: int) -> int:
                try:
                    return int(os.environ.get(name, str(default)))
                except ValueError:
                    return default

            bm_env = _read_env_int("ELSA_TRITON_FP16_FWD_BLOCK_M", 0)
            bn_env = _read_env_int("ELSA_TRITON_FP16_FWD_BLOCK_N", 0)
            wp_env = _read_env_int("ELSA_TRITON_FP16_FWD_WARPS", 0)
            st_env = _read_env_int("ELSA_TRITON_FP16_FWD_STAGES", 0)
            if bm_env > 0:
                BLOCK_M = max(16, (bm_env // 16) * 16)
            if bn_env > 0:
                BLOCK_N = max(16, (bn_env // 16) * 16)
            if wp_env > 0:
                num_warps = max(1, wp_env)
            if st_env > 0:
                num_stages = max(1, st_env)

        if stream_env:
            if stream_q > 0:
                BLOCK_M = max(16, (stream_q // 16) * 16)
            if stream_kv > 0:
                BLOCK_N = max(16, (stream_kv // 16) * 16)
        if q.dtype in (torch.float16, torch.bfloat16):
            BLOCK_M = _sanitize_fp16_fwd_block(BLOCK_M, name="ELSA_TRITON_FP16_FWD_BLOCK_M")
            BLOCK_N = _sanitize_fp16_fwd_block(BLOCK_N, name="ELSA_TRITON_FP16_FWD_BLOCK_N")
        auto_infer_kblock = bool(
            q.dtype in (torch.float16, torch.bfloat16)
            and (not needs_grad)
            and _fp16_kblock_auto_enabled(n_ctx=N, d_head=D, is_causal=is_causal)
        )
        fp16_fast_acc = _resolve_fp16_fast_accum(
            q=q,
            n_ctx=N,
            d_head=D,
            needs_grad=needs_grad,
            is_causal=is_causal,
            prefer_infer_fast=auto_infer_kblock,
        )

        k_fwd = k
        k_stride_kn = k.stride(2)
        k_stride_kd = k.stride(3)
        # For long fp16/bf16 sequences, pre-transposing K improves global-memory
        # access locality for the QK^T path.
        if qk_norm_weights is None and q.dtype in (torch.float16, torch.bfloat16):
            use_k_transpose = os.environ.get("ELSA_TRITON_FP16_FWD_TRANSPOSE_K", "0") != "0"
            try:
                k_transpose_min_n = int(os.environ.get("ELSA_TRITON_FP16_FWD_TRANSPOSE_K_MIN_N", "8192"))
            except ValueError:
                k_transpose_min_n = 8192
            if use_k_transpose and N >= max(1, k_transpose_min_n):
                k_t = k.transpose(-1, -2).contiguous()
                k_fwd = k_t
                k_stride_kn = k_t.stride(3)
                k_stride_kd = k_t.stride(2)

        
        grid = (B, H, triton.cdiv(N, BLOCK_M))
        
        fp16_grad_kind = ""
        use_stateless_fp16_train_fwd = False
        if (
            needs_grad
            and qk_norm_weights is None
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            fp16_grad_kind = _resolve_fp16_bwd_kind(
                q=q,
                n_ctx=N,
                impl=os.environ.get("ELSA_TRITON_FP16_BWD", "auto"),
            )
            if fp16_grad_kind in ("flash", "math", "mem"):
                stateless_mode = os.environ.get("ELSA_TRITON_FP16_TRAIN_STATELESS_FWD", "auto").strip().lower()
                if stateless_mode in ("1", "true", "on", "yes", "force"):
                    use_stateless_fp16_train_fwd = True
                elif stateless_mode in ("0", "false", "off", "no"):
                    use_stateless_fp16_train_fwd = False
                else:
                    try:
                        stateless_min_n = int(os.environ.get("ELSA_TRITON_FP16_TRAIN_STATELESS_MIN_N", "512"))
                    except ValueError:
                        stateless_min_n = 512
                    try:
                        stateless_max_n = int(os.environ.get("ELSA_TRITON_FP16_TRAIN_STATELESS_MAX_N", "8192"))
                    except ValueError:
                        stateless_max_n = 8192
                    try:
                        # Default to batch<=2 so common ViT train runs can reuse
                        # the stateless bridge path without manual overrides.
                        stateless_max_batch = int(os.environ.get("ELSA_TRITON_FP16_TRAIN_STATELESS_MAX_BATCH", "2"))
                    except ValueError:
                        stateless_max_batch = 2
                    use_stateless_fp16_train_fwd = (
                        N >= max(64, stateless_min_n)
                        and N <= max(64, stateless_max_n)
                        and B <= max(1, stateless_max_batch)
                    )

        # 選擇 kernel
        if qk_norm_weights is not None:
            q_norm_w, q_norm_b, k_norm_w, k_norm_b = qk_norm_weights
            if needs_grad:
                out_m = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
                out_z = torch.empty_like(out_m)
                kernel_elsa_attention_fwd_qknorm_mz[grid](
                    q, k, v, out, out_m, out_z,
                    q_norm_w, q_norm_b, k_norm_w, k_norm_b,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    out_m.stride(0), out_m.stride(1), out_m.stride(2),
                    out_z.stride(0), out_z.stride(1), out_z.stride(2),
                    B, H, N, D,
                    BLOCK_M, BLOCK_N, BLOCK_D,
                    scale=scale,
                    IS_CAUSAL=is_causal,
                    USE_TF32=use_tf32,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            else:
                kernel_elsa_attention_fwd_qknorm[grid](
                    q, k, v, out,
                    q_norm_w, q_norm_b, k_norm_w, k_norm_b,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    B, H, N, D,
                    BLOCK_M, BLOCK_N, BLOCK_D,
                    scale=scale,
                    IS_CAUSAL=is_causal,
                    USE_TF32=use_tf32,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
        else:
            if needs_grad:
                if use_stateless_fp16_train_fwd:
                    use_nomask = bool(
                        q.dtype in (torch.float16, torch.bfloat16)
                        and os.environ.get("ELSA_TRITON_FP16_NOMASK", "1") != "0"
                        and (not is_causal)
                        and D == BLOCK_D
                        and N % BLOCK_M == 0
                        and N % BLOCK_N == 0
                    )
                    train_kblock_mode = os.environ.get("ELSA_TRITON_FP16_TRAIN_KBLOCK", "0").strip().lower()
                    if train_kblock_mode in ("1", "true", "on", "yes", "force"):
                        use_train_kblock = use_nomask
                    elif train_kblock_mode in ("0", "false", "off", "no"):
                        use_train_kblock = False
                    else:
                        try:
                            train_kblock_min_n = int(
                                os.environ.get("ELSA_TRITON_FP16_TRAIN_KBLOCK_MIN_N", "2048")
                            )
                        except ValueError:
                            train_kblock_min_n = 2048
                        use_train_kblock = use_nomask and N >= max(64, train_kblock_min_n)

                    train_tc_mode = os.environ.get("ELSA_TRITON_FP16_TRAIN_TC", "0").strip().lower()
                    use_train_tc = bool(
                        use_nomask
                        and train_tc_mode in ("1", "true", "on", "yes", "force")
                    )

                    if use_train_kblock:
                        kernel_elsa_attention_fwd_fixed_nomask_kblock[grid](
                            q, k_fwd, v, out,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            B, H, N, D,
                            BLOCK_M, BLOCK_N, BLOCK_D,
                            scale=scale,
                            ACC_IN_FP16=fp16_fast_acc,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    elif use_train_tc:
                        kernel_elsa_attention_fwd_fp16_tc[grid](
                            q, k_fwd, v, out,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            B, H, N, D,
                            BLOCK_M, BLOCK_N, BLOCK_D,
                            scale=scale,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    elif use_nomask:
                        kernel_elsa_attention_fwd_fixed_nomask[grid](
                            q, k_fwd, v, out,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            B, H, N, D,
                            BLOCK_M, BLOCK_N, BLOCK_D,
                            scale=scale,
                            USE_TF32=use_tf32,
                            ACC_IN_FP16=fp16_fast_acc,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    else:
                        kernel_elsa_attention_fwd_fixed[grid](
                            q, k_fwd, v, out,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            B, H, N, D,
                            BLOCK_M, BLOCK_N, BLOCK_D,
                            scale=scale,
                            IS_CAUSAL=is_causal,
                            USE_TF32=use_tf32,
                            ACC_IN_FP16=fp16_fast_acc,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                else:
                    out_m = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
                    out_z = torch.empty_like(out_m)
                    split_mode = os.environ.get("ELSA_TRITON_FP16_TRAIN_SPLIT_MZ", "auto").strip().lower()
                    if split_mode in ("1", "true", "on", "yes", "force"):
                        use_split_mz = True
                    elif split_mode in ("0", "false", "off", "no"):
                        use_split_mz = False
                    else:
                        try:
                            # Default lowered to 1024 so common ViT-512 training
                            # shapes can use the split M/Z path without env overrides.
                            split_min_n = int(os.environ.get("ELSA_TRITON_FP16_TRAIN_SPLIT_MZ_MIN_N", "1024"))
                        except ValueError:
                            split_min_n = 1280
                        use_split_mz = N >= max(64, split_min_n)
                    use_split_mz = bool(
                        use_split_mz
                        and q.dtype in (torch.float16, torch.bfloat16)
                        and (not is_causal)
                    )
                    if use_split_mz:
                        kernel_elsa_attention_fwd_fixed_mz_split[grid](
                            q, k_fwd, v, out, out_m, out_z,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            out_m.stride(0), out_m.stride(1), out_m.stride(2),
                            out_z.stride(0), out_z.stride(1), out_z.stride(2),
                            B, H, N, D,
                            BLOCK_M, BLOCK_N, BLOCK_D,
                            scale=scale,
                            USE_TF32=use_tf32,
                            ACC_IN_FP16=fp16_fast_acc,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
                    else:
                        kernel_elsa_attention_fwd_fixed_mz[grid](
                            q, k_fwd, v, out, out_m, out_z,
                            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                            k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                            out_m.stride(0), out_m.stride(1), out_m.stride(2),
                            out_z.stride(0), out_z.stride(1), out_z.stride(2),
                            B, H, N, D,
                            BLOCK_M, BLOCK_N, BLOCK_D,
                            scale=scale,
                            IS_CAUSAL=is_causal,
                            USE_TF32=use_tf32,
                            ACC_IN_FP16=fp16_fast_acc,
                            num_warps=num_warps,
                            num_stages=num_stages,
                        )
            else:
                use_ultra_fp16 = bool(
                    q.dtype == torch.float16
                    and os.environ.get("ELSA_TRITON_FP16_ULTRA_FAST", "0") != "0"
                    and (not is_causal)
                    and D == BLOCK_D
                    and N % BLOCK_M == 0
                    and N % BLOCK_N == 0
                )
                flat_mode = os.environ.get("ELSA_TRITON_FP16_FLAT", "auto").strip().lower()
                if flat_mode in ("1", "true", "force", "on"):
                    flat_enabled = True
                elif flat_mode in ("auto", ""):
                    flat_enabled = bool(
                        q.dtype in (torch.float16, torch.bfloat16)
                        and _fp16_flat_auto_enabled(n_ctx=N, d_head=D, is_causal=is_causal)
                    )
                else:
                    flat_enabled = False
                use_flat_nomask = bool(
                    q.dtype in (torch.float16, torch.bfloat16)
                    and flat_enabled
                    and (not is_causal)
                    and D == BLOCK_D
                    and N % BLOCK_M == 0
                    and N % BLOCK_N == 0
                )
                use_tc = bool(
                    q.dtype in (torch.float16, torch.bfloat16)
                    and os.environ.get("ELSA_TRITON_FP16_TC", "0") != "0"
                    and (not is_causal)
                    and D == BLOCK_D
                    and N % BLOCK_M == 0
                    and N % BLOCK_N == 0
                )
                kblock_mode = os.environ.get("ELSA_TRITON_FP16_KBLOCK", "auto").strip().lower()
                if kblock_mode in ("1", "true", "force", "on", "yes"):
                    kblock_enabled = True
                elif kblock_mode in ("0", "false", "off", "no"):
                    kblock_enabled = False
                else:
                    kblock_enabled = _fp16_kblock_auto_enabled(n_ctx=N, d_head=D, is_causal=is_causal)
                use_kblock = bool(
                    q.dtype in (torch.float16, torch.bfloat16)
                    and kblock_enabled
                    and (not is_causal)
                    and D == BLOCK_D
                    and N % BLOCK_M == 0
                    and N % BLOCK_N == 0
                )
                use_nomask = bool(
                    q.dtype in (torch.float16, torch.bfloat16)
                    and os.environ.get("ELSA_TRITON_FP16_NOMASK", "1") != "0"
                    and (not is_causal)
                    and D == BLOCK_D
                    and N % BLOCK_M == 0
                    and N % BLOCK_N == 0
                )
                if use_ultra_fp16:
                    kernel_elsa_attention_fwd_fixed_nomask_fp16stats[grid](
                        q, k_fwd, v, out,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        B, H, N, D,
                        BLOCK_M, BLOCK_N, BLOCK_D,
                        scale=scale,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                elif use_flat_nomask:
                    grid_flat = (triton.cdiv(N, BLOCK_M), B * H)
                    kernel_elsa_attention_fwd_fixed_nomask_flat[grid_flat](
                        q, k_fwd, v, out,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        B, H, N, D,
                        BLOCK_M, BLOCK_N, BLOCK_D,
                        scale=scale,
                        USE_TF32=use_tf32,
                        ACC_IN_FP16=fp16_fast_acc,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                elif use_kblock:
                    kernel_elsa_attention_fwd_fixed_nomask_kblock[grid](
                        q, k_fwd, v, out,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        B, H, N, D,
                        BLOCK_M, BLOCK_N, BLOCK_D,
                        scale=scale,
                        ACC_IN_FP16=fp16_fast_acc,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                elif use_tc:
                    kernel_elsa_attention_fwd_fp16_tc[grid](
                        q, k_fwd, v, out,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        B, H, N, D,
                        BLOCK_M, BLOCK_N, BLOCK_D,
                        scale=scale,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                elif use_nomask:
                    kernel_elsa_attention_fwd_fixed_nomask[grid](
                        q, k_fwd, v, out,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        B, H, N, D,
                        BLOCK_M, BLOCK_N, BLOCK_D,
                        scale=scale,
                        USE_TF32=use_tf32,
                        ACC_IN_FP16=fp16_fast_acc,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                else:
                    kernel_elsa_attention_fwd_fixed[grid](
                        q, k_fwd, v, out,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k_fwd.stride(0), k_fwd.stride(1), k_stride_kn, k_stride_kd,
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        B, H, N, D,
                        BLOCK_M, BLOCK_N, BLOCK_D,
                        scale=scale,
                        IS_CAUSAL=is_causal,
                        USE_TF32=use_tf32,
                        ACC_IN_FP16=fp16_fast_acc,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )

        ctx.fp16_bridge_kind = ""
        if needs_grad:
            fp16_kind = fp16_grad_kind
            if (not fp16_kind) and q.dtype in (torch.float16, torch.bfloat16):
                fp16_kind = _resolve_fp16_bwd_kind(
                    q=q,
                    n_ctx=N,
                    impl=os.environ.get("ELSA_TRITON_FP16_BWD", "auto"),
                )
            if use_stateless_fp16_train_fwd and fp16_kind in ("flash", "math", "mem"):
                ctx.save_for_backward(q, k, v)
                ctx.fp16_bridge_kind = f"fp16_vjp_{fp16_kind}"
            elif fp16_kind == "mem_saved_lse":
                rounded_q = ((N + 31) // 32) * 32
                lse = out_m.float() + out_z.float().clamp_min(1e-20).log()
                if lse.shape[-1] != rounded_q:
                    lse_pad = torch.full(
                        (B, H, rounded_q),
                        float("inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    lse_pad[..., :N] = lse
                    lse = lse_pad
                out_saved = out if out.is_contiguous() else out.contiguous()
                ctx.save_for_backward(q, k, v, out_saved, lse)
                ctx.fp16_bridge_kind = "mem_saved_lse"
            elif (
                fp16_kind == "flash"
                and q.dtype in (torch.float16, torch.bfloat16)
                and q.is_cuda
                and os.environ.get("ELSA_TRITON_FP16_BWD_FLASH_SAVE", "1") != "0"
            ):
                # Save exact forward output + row-wise logsumexp so backward can
                # call flash-attn backward directly without replaying flash forward.
                lse = out_m.float() + out_z.float().clamp_min(1e-20).log()
                out_saved = out if out.is_contiguous() else out.contiguous()
                lse_saved = lse if lse.is_contiguous() else lse.contiguous()
                ctx.save_for_backward(q, k, v, out_saved, lse_saved)
                ctx.fp16_bridge_kind = "flash_saved"
            else:
                ctx.save_for_backward(q, k, v, out_m, out_z)

        ctx.scale = scale
        ctx.use_tf32 = use_tf32
        ctx.needs_grad = needs_grad
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        if not getattr(ctx, "needs_grad", False):
            return None, None, None, None, None, None
        fp16_bridge_kind = getattr(ctx, "fp16_bridge_kind", "")
        if fp16_bridge_kind.startswith("fp16_vjp_"):
            q, k, v = ctx.saved_tensors
            do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            kind = fp16_bridge_kind[len("fp16_vjp_") :]
            if kind not in ("flash", "math", "mem"):
                kind = "flash" if (q.dtype == torch.float16 and q.is_cuda) else "mem"
            dq_, dk_, dv_ = _sdpa_vjp(q, k, v, do, scale=float(ctx.scale), kind=kind)
            return dq_, dk_, dv_, None, None, None
        if fp16_bridge_kind == "mem_saved_lse":
            q, k, v, out, lse = ctx.saved_tensors
            do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            try:
                dq_, dk_, dv_, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                    do,
                    q,
                    k,
                    v,
                    None,
                    out,
                    lse,
                    _SDPA_ZERO_SEED,
                    _SDPA_ZERO_OFFSET,
                    0.0,
                    [True, True, True, False],
                    False,
                    scale=float(ctx.scale),
                )
            except Exception:
                q_len = q.shape[-2]
                k_len = k.shape[-2]
                attn_bias = _get_sdpa_zero_bias(q, q_len=q_len, k_len=k_len)
                dq_, dk_, dv_, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                    do,
                    q,
                    k,
                    v,
                    attn_bias,
                    out,
                    lse,
                    _SDPA_ZERO_SEED,
                    _SDPA_ZERO_OFFSET,
                    0.0,
                    [True, True, True, False],
                    False,
                    scale=float(ctx.scale),
                )
            return dq_, dk_, dv_, None, None, None
        if fp16_bridge_kind == "flash_saved":
            q, k, v, out, lse = ctx.saved_tensors
            do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            try:
                flash_bwd_impl = _resolve_fp16_flash_bwd_impl(q=q, n_ctx=q.shape[-2])
                use_fa2_op = flash_bwd_impl == "fa2"
                if use_fa2_op:
                    # flash-attn op expects (B, N, H, D) layout.
                    q_bnhd = q.permute(0, 2, 1, 3).contiguous()
                    k_bnhd = k.permute(0, 2, 1, 3).contiguous()
                    v_bnhd = v.permute(0, 2, 1, 3).contiguous()
                    o_bnhd = out.permute(0, 2, 1, 3).contiguous()
                    do_bnhd = do.permute(0, 2, 1, 3).contiguous()
                    dq_bnhd = torch.empty_like(q_bnhd)
                    dk_bnhd = torch.empty_like(k_bnhd)
                    dv_bnhd = torch.empty_like(v_bnhd)
                    rng_state = _get_sdpa_zero_rng_state(q.device)
                    _ = torch.ops.flash_attn._flash_attn_backward(
                        do_bnhd,
                        q_bnhd,
                        k_bnhd,
                        v_bnhd,
                        o_bnhd,
                        lse,
                        dq_bnhd,
                        dk_bnhd,
                        dv_bnhd,
                        0.0,
                        float(ctx.scale),
                        False,
                        -1,
                        -1,
                        0.0,
                        None,
                        False,
                        rng_state,
                    )
                    dq_ = dq_bnhd.permute(0, 2, 1, 3)
                    dk_ = dk_bnhd.permute(0, 2, 1, 3)
                    dv_ = dv_bnhd.permute(0, 2, 1, 3)
                else:
                    philox_seed, philox_offset = _get_sdpa_zero_philox(q.device)
                    dq_, dk_, dv_ = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                        do,
                        q,
                        k,
                        v,
                        out,
                        lse,
                        None,
                        None,
                        q.shape[-2],
                        k.shape[-2],
                        0.0,
                        False,
                        philox_seed,
                        philox_offset,
                        scale=float(ctx.scale),
                    )
                return dq_, dk_, dv_, None, None, None
            except Exception:
                # Fallback to replay-based bridge on stacks where this direct path
                # has stricter metadata checks.
                dq_, dk_, dv_ = _sdpa_vjp(q, k, v, do, scale=float(ctx.scale), kind="flash")
                return dq_, dk_, dv_, None, None, None

        q, k, v, out_m, out_z = ctx.saved_tensors
        scale = ctx.scale
        allow_tf32 = bool(getattr(ctx, "use_tf32", False))

        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()
        do = grad_out.contiguous()

        B, H, N, D = q_.shape
        fp16_bwd_impl = os.environ.get("ELSA_TRITON_FP16_BWD", "auto").strip().lower()
        if q_.dtype in (torch.float16, torch.bfloat16):
            kind = _resolve_fp16_bwd_kind(q=q_, n_ctx=N, impl=fp16_bwd_impl)
            if kind:
                if kind not in ("flash", "math", "mem"):
                    kind = "flash" if (q_.dtype == torch.float16 and q_.is_cuda) else "mem"
                dq_, dk_, dv_ = _sdpa_vjp(q_, k_, v_, do, scale=float(scale), kind=kind)
                return dq_, dk_, dv_, None, None, None

        qh = q_.view(B * H, N, D)
        kh = k_.view(B * H, N, D)
        vh = v_.view(B * H, N, D)
        doh = do.view(B * H, N, D)
        mh = out_m.view(B * H, N)
        zh = out_z.view(B * H, N)

        block_m, block_n, block_d, num_warps, num_stages = _get_bwd_launch_params(N, D)

        delta = torch.empty_like(zh)
        dq = torch.empty_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)

        grid_q = (triton.cdiv(N, block_m), B * H)
        try:
            kernel_elsa_bwd_delta[grid_q](
                qh, kh, vh, doh, mh, zh, delta,
                qh.stride(0), qh.stride(1), qh.stride(2),
                kh.stride(0), kh.stride(1), kh.stride(2),
                vh.stride(0), vh.stride(1), vh.stride(2),
                doh.stride(0), doh.stride(1), doh.stride(2),
                mh.stride(0), mh.stride(1),
                zh.stride(0), zh.stride(1),
                delta.stride(0), delta.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            kernel_elsa_bwd_dq[grid_q](
                qh, kh, vh, doh, mh, zh, delta, dq,
                qh.stride(0), qh.stride(1), qh.stride(2),
                kh.stride(0), kh.stride(1), kh.stride(2),
                vh.stride(0), vh.stride(1), vh.stride(2),
                doh.stride(0), doh.stride(1), doh.stride(2),
                mh.stride(0), mh.stride(1),
                zh.stride(0), zh.stride(1),
                delta.stride(0), delta.stride(1),
                dq.stride(0), dq.stride(1), dq.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            grid_k = (triton.cdiv(N, block_n), B * H)
            kernel_elsa_bwd_dkv[grid_k](
                qh, kh, vh, doh, mh, zh, delta, dk, dv,
                qh.stride(0), qh.stride(1), qh.stride(2),
                kh.stride(0), kh.stride(1), kh.stride(2),
                vh.stride(0), vh.stride(1), vh.stride(2),
                doh.stride(0), doh.stride(1), doh.stride(2),
                mh.stride(0), mh.stride(1),
                zh.stride(0), zh.stride(1),
                delta.stride(0), delta.stride(1),
                dk.stride(0), dk.stride(1), dk.stride(2),
                dv.stride(0), dv.stride(1), dv.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=num_stages,
            )
        except Exception as exc:
            # Graceful fallback for long-sequence fp16 when Triton launch exceeds
            # shared-memory/resource limits on current stack.
            fallback_ok = os.environ.get("ELSA_TRITON_BWD_FALLBACK_SDPA", "1") != "0"
            is_resource_err = "OutOfResources" in exc.__class__.__name__
            if not (fallback_ok and is_resource_err):
                raise
            if q_.dtype == torch.float16:
                kind = os.environ.get("ELSA_TRITON_BWD_FALLBACK_KIND_FP16", "flash").lower()
            else:
                kind = os.environ.get("ELSA_TRITON_BWD_FALLBACK_KIND_FP32", "mem").lower()
            if kind not in ("math", "mem", "flash"):
                kind = "flash" if q_.dtype == torch.float16 else "mem"
            dq_, dk_, dv_ = _sdpa_vjp(q_, k_, v_, do, scale=float(scale), kind=kind)
            return dq_, dk_, dv_, None, None, None

        return dq.view(B, H, N, D), dk.view(B, H, N, D), dv.view(B, H, N, D), None, None, None

class ELSA_triton_fp32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale):

        if not q.is_cuda:
            from elsa_cpu import elsa_forward
            return elsa_forward(q, k, v, float(scale), False)

        use_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        kernel = kernel_integral_mhsa_stable
        block_n = int(os.environ.get("ELSA_TRITON_FWD_BLOCK_N", "64"))
        block_q = int(os.environ.get("ELSA_TRITON_FWD_BLOCK_Q", "64"))
        num_wp = int(os.environ.get("ELSA_TRITON_FWD_WARPS", "4"))
        num_stages = int(os.environ.get("ELSA_TRITON_FWD_STAGES", "2"))
        auto_tune_env = os.environ.get("ELSA_TRITON_FWD_AUTOTUNE")
        auto_tune = bool(int(auto_tune_env)) if auto_tune_env is not None else False
        manual_override = any(
            key in os.environ
            for key in (
                "ELSA_TRITON_FWD_BLOCK_N",
                "ELSA_TRITON_FWD_BLOCK_Q",
                "ELSA_TRITON_FWD_WARPS",
                "ELSA_TRITON_FWD_STAGES",
            )
        )
        stream_env = os.environ.get("ELSA_TRITON_FP32_STREAM", "0") == "1"
        if stream_env:
            try:
                stream_q = int(os.environ.get("ELSA_STREAM_Q_BLOCK", "0"))
            except ValueError:
                stream_q = 0
            try:
                stream_kv = int(os.environ.get("ELSA_STREAM_KV_BLOCK", "0"))
            except ValueError:
                stream_kv = 0
            if stream_q > 0:
                block_q = max(16, (stream_q // 16) * 16)
            if stream_kv > 0:
                block_n = max(16, (stream_kv // 16) * 16)
            auto_tune = False
            manual_override = True

        B, H, N, D = q.shape
        needs_grad = q.requires_grad or k.requires_grad or v.requires_grad
        bwd_impl = os.environ.get("ELSA_TRITON_FP32_BWD", "auto").lower()
        bridge_bwd_kind = ""
        if bwd_impl in ("math", "mem", "flash"):
            bridge_bwd_kind = bwd_impl
        elif bwd_impl == "auto" and q.is_cuda:
            bridge_bwd_kind = _resolve_fp32_auto_bwd_kind(int(N))
        use_bridge_infer = (
            needs_grad
            and bridge_bwd_kind != ""
            and bridge_bwd_kind != "mem"
            and os.environ.get("ELSA_TRITON_FP32_BRIDGE_INFER", "1") != "0"
        )
        # Keep train defaults deterministic; enable autotune only via explicit env.
        prefer_train_fast = True
        if needs_grad and not manual_override:
            # Training-oriented defaults tuned on A100:
            # N~577 (384 input) prefers 32x64; N~1025 (512 input) prefers 64x32.
            if N >= 896:
                block_q, block_n, num_wp, num_stages = 64, 32, 8, 2
            elif N >= 384:
                block_q, block_n, num_wp, num_stages = 32, 64, 8, 2

        kernel_allow_tf32 = _resolve_fp32_kernel_allow_tf32(
            requested_tf32=use_tf32,
            needs_grad=needs_grad,
        )
        use_stable_train_fwd = _should_use_stable_train_fwd(
            needs_grad=needs_grad,
            use_tf32=kernel_allow_tf32,
            n_ctx=N,
            d_head=D,
            bwd_impl=bwd_impl,
        )
        if use_stable_train_fwd:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            # Optional hybrid train bridge:
            # compute row-max stats in a lightweight Triton pass so backward can
            # consume saved LSE directly (avoid SDPA forward-recompute in VJP).
            use_stable_train_stats = (
                os.environ.get("ELSA_TRITON_FP32_STABLE_TRAIN_STATS", "0") != "0"
            )
            if use_stable_train_stats and bridge_bwd_kind == "mem":
                try:
                    stats_source = os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_SOURCE", "auto").strip().lower()
                    if stats_source in ("", "auto"):
                        # Default to local m/z path for fp32 train bridge; it avoids
                        # an extra O(N^2) rowmax pass and is faster on current stack.
                        stats_source = "local"
                    rounded_q = ((N + 31) // 32) * 32
                    lse = torch.empty((B, H, rounded_q), device=q.device, dtype=torch.float32)

                    if stats_source in ("local", "stable_local", "local_mz"):
                        local_tf32_env = os.environ.get("ELSA_TRITON_FP32_STABLE_LOCAL_ALLOW_TF32", "1").strip().lower()
                        local_allow_tf32 = local_tf32_env not in ("0", "off", "false", "disable", "disabled")
                        out, out_m_bhn, out_z_bhn = _stable_local_fp32_forward_with_mz(
                            q,
                            k,
                            v,
                            scale=float(scale),
                            allow_tf32=local_allow_tf32,
                        )
                        lse.fill_(float("inf"))
                        lse[..., :N] = out_m_bhn + out_z_bhn.clamp_min(1e-20).log()
                    else:
                        out, out_z_bhn = _stable_can_fp32_forward_with_z(
                            q,
                            k,
                            v,
                            scale=float(scale),
                        )
                        q_ = q.view(B * H, N, D)
                        k_ = k.view(B * H, N, D)
                        try:
                            stats_block_q = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_BLOCK_Q", "64"))
                        except ValueError:
                            stats_block_q = 64
                        try:
                            stats_block_n = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_BLOCK_N", "64"))
                        except ValueError:
                            stats_block_n = 64
                        try:
                            stats_num_warps = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_WARPS", "4"))
                        except ValueError:
                            stats_num_warps = 4
                        try:
                            stats_num_stages = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_STAGES", "2"))
                        except ValueError:
                            stats_num_stages = 2

                        stats_block_q = max(16, (stats_block_q // 16) * 16)
                        stats_block_n = max(16, (stats_block_n // 16) * 16)
                        stats_num_warps = max(1, stats_num_warps)
                        stats_num_stages = max(1, stats_num_stages)
                        block_d = 32 * ((D + 31) // 32)

                        async_stats = os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_ASYNC", "0") != "0"
                        out_z = out_z_bhn.view(B * H, N)

                        def _compute_lse():
                            # Fused rowmax + log(z) + padded-LSE write for mem VJP bridge.
                            lse_bh = lse.view(B * H, rounded_q)
                            grid = (triton.cdiv(rounded_q, stats_block_q), B * H)
                            kernel_elsa_attention_fp32_rowmax_lse[grid](
                                q_,
                                k_,
                                out_z,
                                lse_bh,
                                q_.stride(0),
                                q_.stride(1),
                                q_.stride(2),
                                k_.stride(0),
                                k_.stride(1),
                                k_.stride(2),
                                out_z.stride(0),
                                out_z.stride(1),
                                lse_bh.stride(0),
                                lse_bh.stride(1),
                                BH=B * H,
                                N_CTX=N,
                                N_PAD=rounded_q,
                                D_HEAD=D,
                                BLOCK_M=stats_block_q,
                                BLOCK_N=stats_block_n,
                                BLOCK_D=block_d,
                                SCALE=scale,
                                IS_CAUSAL=False,
                                ALLOW_TF32=kernel_allow_tf32,
                                num_warps=stats_num_warps,
                                num_stages=stats_num_stages,
                            )

                        if async_stats:
                            stats_stream = torch.cuda.Stream(device=q.device)
                            current_stream = torch.cuda.current_stream(device=q.device)
                            stats_stream.wait_stream(current_stream)
                            with torch.cuda.stream(stats_stream):
                                _compute_lse()
                            stats_event = torch.cuda.Event()
                            stats_event.record(stats_stream)
                            ctx.stats_ready_event = stats_event
                        else:
                            _compute_lse()
                    out_saved = out if out.is_contiguous() else out.contiguous()
                    lse_saved = lse if lse.is_contiguous() else lse.contiguous()
                    ctx.save_for_backward(q, k, v, out_saved, lse_saved)
                    ctx.bridge_bwd_kind = "mem_saved_lse"
                except Exception:
                    stable_mod = _load_elsa_fp32_stable_module()
                    if hasattr(stable_mod, "can_triton_baseline_fp32"):
                        out = stable_mod.can_triton_baseline_fp32(q, k, v, is_causal=False, bias=None)
                    else:
                        out = stable_mod.can_triton_new_fp32(q, k, v, is_causal=False, bias=None)
                    ctx.save_for_backward(q, k, v)
                    ctx.bridge_bwd_kind = "mem"
            else:
                stable_mod = _load_elsa_fp32_stable_module()
                if hasattr(stable_mod, "can_triton_baseline_fp32"):
                    out = stable_mod.can_triton_baseline_fp32(q, k, v, is_causal=False, bias=None)
                else:
                    out = stable_mod.can_triton_new_fp32(q, k, v, is_causal=False, bias=None)
                # Stable forward does not expose (m, z); use ATen mem VJP in backward.
                ctx.save_for_backward(q, k, v)
                ctx.bridge_bwd_kind = "mem"
            ctx.scale = scale
            ctx.use_tf32 = kernel_allow_tf32
            return out

        fast_env = os.environ.get("ELSA_TRITON_FP32_FAST")
        fast_autotune = os.environ.get("ELSA_TRITON_FP32_FAST_AUTOTUNE", "1") == "1"
        infer_env = os.environ.get("ELSA_TRITON_FP32_INFER", "1")
        splitd_env = os.environ.get("ELSA_TRITON_FP32_SPLITD", "0")
        if fast_env is None:
            use_fast = (not needs_grad) and D >= 256
        else:
            use_fast = (not needs_grad) and fast_env == "1"
        use_infer_kernel = (not needs_grad) and infer_env != "0"
        use_splitd_kernel = use_infer_kernel and splitd_env == "1" and D == 256
        q = q if q.is_contiguous() else q.contiguous()
        k = k if k.is_contiguous() else k.contiguous()
        v = v if v.is_contiguous() else v.contiguous()
        q_ = q.view(B * H, N, D)
        k_ = k.view(B * H, N, D)
        v_ = v.view(B * H, N, D)

        if use_fast:
            out = torch.empty_like(q_, dtype=q.dtype)
            block_d = 32 * ((D + 31) // 32)
            cfg = None
            if fast_autotune:
                tune_key = (q.device.index or -1, N, D)
                cfg = _ELSA_FP32_FAST_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_fast_kernel(
                        kernel_elsa_attention_fp32_fast,
                        q_,
                        k_,
                        v_,
                        out,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                    )
                    if cfg:
                        _ELSA_FP32_FAST_TUNE_CACHE[tune_key] = cfg
            if cfg:
                block_m, block_n, num_wp, num_stages = cfg
                grid = (triton.cdiv(N, block_m), B * H)
                kernel_elsa_attention_fp32_fast[grid](
                    q_, k_, v_, out,
                    q_.stride(0), q_.stride(1), q_.stride(2),
                    k_.stride(0), k_.stride(1), k_.stride(2),
                    v_.stride(0), v_.stride(1), v_.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BH=B * H,
                    N_CTX=N,
                    D_HEAD=D,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    BLOCK_D=block_d,
                    SCALE=scale,
                    IS_CAUSAL=False,
                    ALLOW_TF32=kernel_allow_tf32,
                    num_warps=num_wp,
                    num_stages=num_stages,
                )
            else:
                block_m = 64 if D <= 128 else 32
                block_n_fast = 64 if N < 256 else 128
                grid = (triton.cdiv(N, block_m), B * H)
                kernel_elsa_attention_fp32_fast[grid](
                    q_, k_, v_, out,
                    q_.stride(0), q_.stride(1), q_.stride(2),
                    k_.stride(0), k_.stride(1), k_.stride(2),
                    v_.stride(0), v_.stride(1), v_.stride(2),
                    out.stride(0), out.stride(1), out.stride(2),
                    BH=B * H,
                    N_CTX=N,
                    D_HEAD=D,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n_fast,
                    BLOCK_D=block_d,
                    SCALE=scale,
                    IS_CAUSAL=False,
                    ALLOW_TF32=kernel_allow_tf32,
                    num_warps=4,
                    num_stages=2,
                )
            ctx.scale = scale
            return out.view(B, H, N, D).to(q.dtype)

        if use_splitd_kernel:
            out = torch.empty_like(q_, dtype=q.dtype)
            if auto_tune and not manual_override:
                tune_key = (q.device.index or -1, N, D, "splitd")
                cfg = _ELSA_FP32_SPLITD_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_splitd_kernel(
                        kernel_integral_mhsa_splitd_infer,
                        q_,
                        k_,
                        v_,
                        out,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=kernel_allow_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_SPLITD_TUNE_CACHE[tune_key] = cfg
                if cfg:
                    block_q, block_n, block_d, num_wp, num_stages = cfg

            grid = (triton.cdiv(N, block_q), B * H)
            kernel_integral_mhsa_splitd_infer[grid](
                q_, k_, v_,
                out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                BLOCK_D=block_d,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=kernel_allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            ctx.scale = scale
            return out.view(B, H, N, D).to(q.dtype)

        if use_infer_kernel:
            out = torch.empty_like(q_, dtype=q.dtype)
            if auto_tune and not manual_override:
                tune_key = (q.device.index or -1, N, D, "infer")
                cfg = _ELSA_FP32_INFER_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_infer_kernel(
                        kernel_integral_mhsa_stable_infer,
                        q_,
                        k_,
                        v_,
                        out,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=kernel_allow_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_INFER_TUNE_CACHE[tune_key] = cfg
                if cfg:
                    block_q, block_n, num_wp, num_stages = cfg

            grid = (triton.cdiv(N, block_q), B * H)
            kernel_integral_mhsa_stable_infer[grid](
                q_, k_, v_,
                out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=kernel_allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            ctx.scale = scale
            return out.view(B, H, N, D).to(q.dtype)

        if use_bridge_infer:
            out = torch.empty_like(q_, dtype=q.dtype)
            if auto_tune and not manual_override:
                tune_key = (q.device.index or -1, N, D, "infer")
                cfg = _ELSA_FP32_INFER_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_infer_kernel(
                        kernel_integral_mhsa_stable_infer,
                        q_,
                        k_,
                        v_,
                        out,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=kernel_allow_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_INFER_TUNE_CACHE[tune_key] = cfg
                if cfg:
                    block_q, block_n, num_wp, num_stages = cfg

            grid = (triton.cdiv(N, block_q), B * H)
            kernel_integral_mhsa_stable_infer[grid](
                q_, k_, v_,
                out,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=kernel_allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )
            ctx.save_for_backward(q, k, v)
            ctx.scale = scale
            ctx.use_tf32 = kernel_allow_tf32
            ctx.bridge_bwd_kind = bridge_bwd_kind
            return out.view(B, H, N, D).to(q.dtype)

        out = torch.empty_like(q_, dtype=q.dtype)
        out_z = torch.empty(B * H, N, dtype=q.dtype, device=q.device)
        out_m = torch.empty(B * H, N, dtype=q.dtype, device=q.device)

        train_fast_env = os.environ.get("ELSA_TRITON_FP32_TRAIN_FAST")
        if train_fast_env is None:
            train_fast = prefer_train_fast
        else:
            train_fast = train_fast_env == "1"

        if auto_tune and not manual_override:
            tune_key = (q.device.index or -1, N, D)
            if train_fast:
                cfg = _ELSA_FP32_TRAIN_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_fast_mz_kernel(
                        kernel_elsa_attention_fp32_fast_mz,
                        q_,
                        k_,
                        v_,
                        out,
                        out_m,
                        out_z,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=kernel_allow_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_TRAIN_TUNE_CACHE[tune_key] = cfg
            else:
                cfg = _ELSA_FP32_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_kernel(
                        kernel,
                        q_,
                        k_,
                        v_,
                        out,
                        out_z,
                        out_m,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=kernel_allow_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_TUNE_CACHE[tune_key] = cfg
            if cfg:
                block_q, block_n, num_wp, num_stages = cfg
        # BLOCK_N = 64

        block_d = 32 * ((D + 31) // 32)
        grid = (triton.cdiv(N, block_q), B * H)

        if train_fast:
            kernel_elsa_attention_fp32_fast_mz[grid](
                q_, k_, v_, out, out_m, out_z,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                out_m.stride(0), out_m.stride(1),
                out_z.stride(0), out_z.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_q,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=kernel_allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
        else:
            kernel[grid](
                q_, k_, v_,
                out, out_z, out_m,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                out_z.stride(1), out_z.stride(0), out_z.stride(1),
                out_m.stride(1), out_m.stride(0), out_m.stride(1),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=kernel_allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )

        # Triton scan kernels above produce numerator accumulator (S) plus Z.
        # Convert to final attention output before any save/return.
        _normalize_scan_accumulator_(out, out_z)

        if needs_grad and bridge_bwd_kind == "mem":
            # Save ELSA forward outputs + padded LSE so mem VJP can skip SDPA forward
            # recompute and avoid per-step (m,z)->LSE reconstruction in backward.
            out_saved = out.view(B, H, N, D)
            rounded_q = ((N + 31) // 32) * 32
            lse = out_m.float() + out_z.float().clamp_min(1e-20).log()
            if lse.dim() == 2:
                lse = lse.view(B, H, N)
            if lse.shape[-1] != rounded_q:
                lse_pad = torch.full((B, H, rounded_q), float("inf"), device=q.device, dtype=torch.float32)
                lse_pad[..., :N] = lse
                lse = lse_pad
            ctx.save_for_backward(q, k, v, out_saved, lse)
            ctx.bridge_bwd_kind = "mem_saved_lse"
        else:
            ctx.save_for_backward(q, k, v, out_m, out_z)
            ctx.bridge_bwd_kind = ""
        ctx.scale = scale
        ctx.use_tf32 = kernel_allow_tf32
        return out.view(B, H, N, D).to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        bridge_bwd_kind = getattr(ctx, "bridge_bwd_kind", "")
        if bridge_bwd_kind == "mem_saved_lse":
            q, k, v, out, lse = ctx.saved_tensors
            ready_event = getattr(ctx, "stats_ready_event", None)
            if ready_event is not None:
                torch.cuda.current_stream(device=q.device).wait_event(ready_event)
            do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            use_fast_mem_saved = os.environ.get("ELSA_TRITON_FP32_MEM_SAVED_FAST", "1") != "0"
            if use_fast_mem_saved:
                dq, dk, dv, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                    do,
                    q,
                    k,
                    v,
                    None,
                    out,
                    lse,
                    _SDPA_ZERO_SEED,
                    _SDPA_ZERO_OFFSET,
                    0.0,
                    [True, True, True, False],
                    False,
                    scale=float(ctx.scale),
                )
                return dq, dk, dv, None
            dq, dk, dv = _sdpa_mem_vjp_from_saved(
                q,
                k,
                v,
                do,
                out=out,
                lse=lse,
                scale=float(ctx.scale),
            )
            return dq, dk, dv, None
        if bridge_bwd_kind == "mem_saved":
            q, k, v, out, out_m, out_z = ctx.saved_tensors
            do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            use_fast_mem_saved = os.environ.get("ELSA_TRITON_FP32_MEM_SAVED_FAST", "1") != "0"
            if use_fast_mem_saved:
                bsz, nheads, q_len, _ = q.shape
                rounded_q = ((q_len + 31) // 32) * 32
                out_m_bh = out_m.view(bsz, nheads, q_len) if out_m.dim() == 2 else out_m
                out_z_bh = out_z.view(bsz, nheads, q_len) if out_z.dim() == 2 else out_z
                lse = out_m_bh.float() + out_z_bh.float().clamp_min(1e-20).log()
                if lse.shape[-1] != rounded_q:
                    lse_pad = torch.full(
                        (bsz, nheads, rounded_q),
                        float("inf"),
                        device=q.device,
                        dtype=torch.float32,
                    )
                    lse_pad[..., :q_len] = lse
                    lse = lse_pad
                out_saved = out.view(bsz, nheads, q_len, out.shape[-1]) if out.dim() == 3 else out
                dq, dk, dv, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                    do,
                    q,
                    k,
                    v,
                    None,
                    out_saved,
                    lse,
                    _SDPA_ZERO_SEED,
                    _SDPA_ZERO_OFFSET,
                    0.0,
                    [True, True, True, False],
                    False,
                    scale=float(ctx.scale),
                )
                return dq, dk, dv, None
            dq, dk, dv = _sdpa_mem_vjp_from_saved(
                q,
                k,
                v,
                do,
                out=out,
                out_m=out_m,
                out_z=out_z,
                scale=float(ctx.scale),
            )
            return dq, dk, dv, None
        if bridge_bwd_kind:
            q, k, v = ctx.saved_tensors
            q_ = q.contiguous()
            k_ = k.contiguous()
            v_ = v.contiguous()
            do = grad_out.contiguous()
            dq, dk, dv = _sdpa_vjp(q_, k_, v_, do, scale=float(ctx.scale), kind=bridge_bwd_kind)
            return dq, dk, dv, None

        q, k, v, out_m, out_z = ctx.saved_tensors
        scale = ctx.scale
        allow_tf32 = bool(getattr(ctx, "use_tf32", False))
        q_ = q if q.is_contiguous() else q.contiguous()
        k_ = k if k.is_contiguous() else k.contiguous()
        v_ = v if v.is_contiguous() else v.contiguous()
        do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

        B, H, N, D = q_.shape
        bwd_impl = os.environ.get("ELSA_TRITON_FP32_BWD", "auto").lower()
        if bwd_impl in ("math", "mem", "flash"):
            dq, dk, dv = _sdpa_vjp(q_, k_, v_, do, scale=float(scale), kind=bwd_impl)
            return dq, dk, dv, None
        if bwd_impl == "auto" and q_.is_cuda:
            auto_kind = _resolve_fp32_auto_bwd_kind(int(N))
            dq, dk, dv = _sdpa_vjp(q_, k_, v_, do, scale=float(scale), kind=auto_kind)
            return dq, dk, dv, None

        qh = q_.view(B * H, N, D)
        kh = k_.view(B * H, N, D)
        vh = v_.view(B * H, N, D)
        doh = do.view(B * H, N, D)
        mh = out_m.view(B * H, N)
        zh = out_z.view(B * H, N)

        block_m, block_n, block_d, num_warps, num_stages = _get_bwd_launch_params(N, D)

        delta = torch.empty_like(zh)
        dq = torch.empty_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)

        grid_q = (triton.cdiv(N, block_m), B * H)
        kernel_elsa_bwd_delta[grid_q](
            qh, kh, vh, doh, mh, zh, delta,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        kernel_elsa_bwd_dq[grid_q](
            qh, kh, vh, doh, mh, zh, delta, dq,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dq.stride(0), dq.stride(1), dq.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid_k = (triton.cdiv(N, block_n), B * H)
        kernel_elsa_bwd_dkv[grid_k](
            qh, kh, vh, doh, mh, zh, delta, dk, dv,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return dq.view(B, H, N, D), dk.view(B, H, N, D), dv.view(B, H, N, D), None


class ELSA_triton_fp32_train(torch.autograd.Function):
    """Training-only FP32 path: keep inference path untouched, force train-safe bwd route."""

    @staticmethod
    def _resolve_train_bwd() -> str:
        # Backward-compatible with older benchmark scripts that only set
        # ELSA_TRITON_FP32_BWD.
        bwd_impl = os.environ.get("ELSA_TRITON_FP32_TRAIN_BWD", "").strip().lower()
        if not bwd_impl:
            bwd_impl = os.environ.get("ELSA_TRITON_FP32_BWD", "auto").strip().lower()
        if bwd_impl not in ("auto", "mem", "math", "flash", "triton"):
            bwd_impl = "auto"
        # Full-model fp32 training with triton backward is currently unstable on
        # this stack (can be orders-of-magnitude slower). Keep it opt-in only.
        if bwd_impl == "triton" and not _allow_unstable_paths():
            _warn_once(
                "fp32_train_bwd_triton_disabled",
                (
                    "ELSA_TRITON_FP32_TRAIN_BWD=triton is disabled by default due severe "
                    "full-model regressions; falling back to auto. Set "
                    "ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1 to force."
                ),
            )
            bwd_impl = "auto"
        return bwd_impl

    @staticmethod
    def _resolve_train_fwd(
        fwd_mode: str,
        seq_len: int,
    ) -> Tuple[str, str]:
        mode = (fwd_mode or "hybrid").lower()
        if mode in ("legacy", "off", "fast"):
            return "off", "0"
        if mode in ("hybrid_nostats", "nostats", "hybrid0", "hybrid_offstats"):
            return "auto", "0"
        if mode in ("adaptive", "hybrid_adaptive"):
            try:
                stats_max_n = int(os.environ.get("ELSA_TRITON_FP32_TRAIN_STATS_MAX_N", "896"))
            except ValueError:
                stats_max_n = 896
            use_stats = seq_len <= max(64, stats_max_n)
            return "auto", "1" if use_stats else "0"
        # Default: hybrid with saved-LSE stats bridge.
        return "auto", "1"

    @staticmethod
    def forward(ctx, q, k, v, scale):
        if not q.is_cuda:
            return ELSA_triton_fp32.forward(ctx, q, k, v, scale)

        save_mem_out = os.environ.get("ELSA_TRITON_FP32_MEM_SAVE_OUT", "1").strip().lower() not in (
            "0",
            "off",
            "false",
            "no",
            "disable",
            "disabled",
        )
        fwd_mode = os.environ.get("ELSA_TRITON_FP32_TRAIN_FWD_MODE", "hybrid").lower()
        bwd_impl = ELSA_triton_fp32_train._resolve_train_bwd()
        ctx._train_bwd_impl = bwd_impl
        stable_mode, stable_stats = ELSA_triton_fp32_train._resolve_train_fwd(fwd_mode, int(q.shape[2]))

        use_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        B, H, N, D = q.shape
        needs_grad = q.requires_grad or k.requires_grad or v.requires_grad
        kernel_allow_tf32 = _resolve_fp32_kernel_allow_tf32(
            requested_tf32=use_tf32,
            needs_grad=needs_grad,
        )

        bridge_bwd_kind = ""
        if bwd_impl in ("math", "mem", "flash"):
            bridge_bwd_kind = bwd_impl
        elif bwd_impl == "auto" and q.is_cuda:
            bridge_bwd_kind = _resolve_fp32_auto_bwd_kind(int(N))

        # Train-specialized stable forward switch without env mutation.
        use_stable_train_fwd = False
        if needs_grad and (not kernel_allow_tf32) and bwd_impl in ("auto", "mem"):
            if stable_mode in ("1", "on", "true", "force"):
                use_stable_train_fwd = True
            elif stable_mode not in ("0", "off", "false", "disable", "disabled"):
                try:
                    min_n = int(os.environ.get("ELSA_TRITON_FP32_STABLE_MIN_N", "896"))
                except ValueError:
                    min_n = 896
                try:
                    max_d = int(os.environ.get("ELSA_TRITON_FP32_STABLE_MAX_D", "128"))
                except ValueError:
                    max_d = 128
                use_stable_train_fwd = N >= max(64, min_n) and D <= max(16, max_d)

        if use_stable_train_fwd:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            use_stats = stable_stats == "1"
            if use_stats and bridge_bwd_kind == "mem":
                try:
                    stats_source = os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_SOURCE", "auto").strip().lower()
                    if stats_source in ("", "auto"):
                        stats_source = "local"

                    rounded_q = ((N + 31) // 32) * 32
                    lse = torch.empty((B, H, rounded_q), device=q.device, dtype=torch.float32)

                    if stats_source in ("local", "stable_local", "local_mz"):
                        local_tf32_env = os.environ.get(
                            "ELSA_TRITON_FP32_STABLE_LOCAL_ALLOW_TF32", "1"
                        ).strip().lower()
                        local_allow_tf32 = local_tf32_env not in (
                            "0",
                            "off",
                            "false",
                            "disable",
                            "disabled",
                        )
                        out, out_m_bhn, out_z_bhn = _stable_local_fp32_forward_with_mz(
                            q,
                            k,
                            v,
                            scale=float(scale),
                            allow_tf32=local_allow_tf32,
                        )
                        lse.fill_(float("inf"))
                        lse[..., :N] = out_m_bhn + out_z_bhn.clamp_min(1e-20).log()
                    else:
                        out, out_z_bhn = _stable_can_fp32_forward_with_z(
                            q,
                            k,
                            v,
                            scale=float(scale),
                        )
                        q_ = q.view(B * H, N, D)
                        k_ = k.view(B * H, N, D)
                        out_z = out_z_bhn.view(B * H, N)

                        try:
                            stats_block_q = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_BLOCK_Q", "64"))
                        except ValueError:
                            stats_block_q = 64
                        try:
                            stats_block_n = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_BLOCK_N", "64"))
                        except ValueError:
                            stats_block_n = 64
                        try:
                            stats_num_warps = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_WARPS", "4"))
                        except ValueError:
                            stats_num_warps = 4
                        try:
                            stats_num_stages = int(os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_STAGES", "2"))
                        except ValueError:
                            stats_num_stages = 2

                        stats_block_q = max(16, (stats_block_q // 16) * 16)
                        stats_block_n = max(16, (stats_block_n // 16) * 16)
                        stats_num_warps = max(1, stats_num_warps)
                        stats_num_stages = max(1, stats_num_stages)
                        block_d = 32 * ((D + 31) // 32)

                        async_stats = os.environ.get("ELSA_TRITON_FP32_STABLE_STATS_ASYNC", "0") != "0"

                        def _compute_lse():
                            lse_bh = lse.view(B * H, rounded_q)
                            grid = (triton.cdiv(rounded_q, stats_block_q), B * H)
                            kernel_elsa_attention_fp32_rowmax_lse[grid](
                                q_,
                                k_,
                                out_z,
                                lse_bh,
                                q_.stride(0),
                                q_.stride(1),
                                q_.stride(2),
                                k_.stride(0),
                                k_.stride(1),
                                k_.stride(2),
                                out_z.stride(0),
                                out_z.stride(1),
                                lse_bh.stride(0),
                                lse_bh.stride(1),
                                BH=B * H,
                                N_CTX=N,
                                N_PAD=rounded_q,
                                D_HEAD=D,
                                BLOCK_M=stats_block_q,
                                BLOCK_N=stats_block_n,
                                BLOCK_D=block_d,
                                SCALE=scale,
                                IS_CAUSAL=False,
                                ALLOW_TF32=kernel_allow_tf32,
                                num_warps=stats_num_warps,
                                num_stages=stats_num_stages,
                            )

                        if async_stats:
                            stats_stream = torch.cuda.Stream(device=q.device)
                            current_stream = torch.cuda.current_stream(device=q.device)
                            stats_stream.wait_stream(current_stream)
                            with torch.cuda.stream(stats_stream):
                                _compute_lse()
                            stats_event = torch.cuda.Event()
                            stats_event.record(stats_stream)
                            ctx.stats_ready_event = stats_event
                        else:
                            _compute_lse()

                    lse_saved = lse if lse.is_contiguous() else lse.contiguous()
                    if save_mem_out:
                        out_saved = out if out.is_contiguous() else out.contiguous()
                        ctx.save_for_backward(q, k, v, out_saved, lse_saved)
                        ctx.mem_saved_has_out = True
                    else:
                        ctx.save_for_backward(q, k, v, lse_saved)
                        ctx.mem_saved_has_out = False
                    ctx.bridge_bwd_kind = "mem_saved_lse"
                except Exception:
                    stable_mod = _load_elsa_fp32_stable_module()
                    if hasattr(stable_mod, "can_triton_baseline_fp32"):
                        out = stable_mod.can_triton_baseline_fp32(q, k, v, is_causal=False, bias=None)
                    else:
                        out = stable_mod.can_triton_new_fp32(q, k, v, is_causal=False, bias=None)
                    ctx.save_for_backward(q, k, v)
                    ctx.bridge_bwd_kind = "mem"
            else:
                stable_mod = _load_elsa_fp32_stable_module()
                if hasattr(stable_mod, "can_triton_baseline_fp32"):
                    out = stable_mod.can_triton_baseline_fp32(q, k, v, is_causal=False, bias=None)
                else:
                    out = stable_mod.can_triton_new_fp32(q, k, v, is_causal=False, bias=None)
                ctx.save_for_backward(q, k, v)
                ctx.bridge_bwd_kind = bridge_bwd_kind if bridge_bwd_kind else "mem"

            ctx.scale = scale
            ctx.use_tf32 = kernel_allow_tf32
            return out

        # Train-specialized native path (no _temp_env + no generic dispatch).
        block_n = int(os.environ.get("ELSA_TRITON_FWD_BLOCK_N", "64"))
        block_q = int(os.environ.get("ELSA_TRITON_FWD_BLOCK_Q", "64"))
        num_wp = int(os.environ.get("ELSA_TRITON_FWD_WARPS", "4"))
        num_stages = int(os.environ.get("ELSA_TRITON_FWD_STAGES", "2"))
        auto_tune_env = os.environ.get("ELSA_TRITON_FWD_AUTOTUNE")
        auto_tune = bool(int(auto_tune_env)) if auto_tune_env is not None else False
        manual_override = any(
            key in os.environ
            for key in (
                "ELSA_TRITON_FWD_BLOCK_N",
                "ELSA_TRITON_FWD_BLOCK_Q",
                "ELSA_TRITON_FWD_WARPS",
                "ELSA_TRITON_FWD_STAGES",
            )
        )

        # Keep train defaults deterministic; enable autotune only via explicit env.
        if needs_grad and not manual_override:
            if N >= 896:
                block_q, block_n, num_wp, num_stages = 64, 32, 8, 2
            elif N >= 384:
                block_q, block_n, num_wp, num_stages = 32, 64, 8, 2

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        q_ = q.view(B * H, N, D)
        k_ = k.view(B * H, N, D)
        v_ = v.view(B * H, N, D)

        out = torch.empty_like(q_, dtype=q.dtype)
        out_z = torch.empty(B * H, N, dtype=q.dtype, device=q.device)
        out_m = torch.empty(B * H, N, dtype=q.dtype, device=q.device)

        train_fast_env = os.environ.get("ELSA_TRITON_FP32_TRAIN_FAST")
        if train_fast_env is None:
            # Sequence-aware default: short ViT train/ft specs are often better on
            # the non-fast route, while longer sequences benefit from fast M/Z path.
            try:
                auto_min_n = int(os.environ.get("ELSA_TRITON_FP32_TRAIN_FAST_AUTO_MIN_N", "1280"))
            except ValueError:
                auto_min_n = 1280
            train_fast = N >= max(256, auto_min_n)
        else:
            train_fast = train_fast_env == "1"

        if auto_tune and not manual_override:
            tune_key = (q.device.index or -1, N, D)
            cfg = None
            if train_fast:
                cfg = _ELSA_FP32_TRAIN_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_fast_mz_kernel(
                        kernel_elsa_attention_fp32_fast_mz,
                        q_,
                        k_,
                        v_,
                        out,
                        out_m,
                        out_z,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=kernel_allow_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_TRAIN_TUNE_CACHE[tune_key] = cfg
            else:
                cfg = _ELSA_FP32_TUNE_CACHE.get(tune_key)
                if cfg is None:
                    cfg = _tune_elsa_fp32_kernel(
                        kernel_integral_mhsa_stable,
                        q_,
                        k_,
                        v_,
                        out,
                        out_z,
                        out_m,
                        B,
                        H,
                        N,
                        D,
                        float(scale),
                        allow_tf32=kernel_allow_tf32,
                    )
                    if cfg:
                        _ELSA_FP32_TUNE_CACHE[tune_key] = cfg
            if cfg:
                block_q, block_n, num_wp, num_stages = cfg

        block_d = 32 * ((D + 31) // 32)
        grid = (triton.cdiv(N, block_q), B * H)
        if train_fast:
            kernel_elsa_attention_fp32_fast_mz[grid](
                q_, k_, v_, out, out_m, out_z,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                out_m.stride(0), out_m.stride(1),
                out_z.stride(0), out_z.stride(1),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=block_q,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                SCALE=scale,
                IS_CAUSAL=False,
                ALLOW_TF32=kernel_allow_tf32,
                num_warps=num_wp,
                num_stages=num_stages,
            )
        else:
            kernel_integral_mhsa_stable[grid](
                q_, k_, v_,
                out, out_z, out_m,
                B * H, N,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                out_z.stride(1), out_z.stride(0), out_z.stride(1),
                out_m.stride(1), out_m.stride(0), out_m.stride(1),
                BLOCK_Q=block_q, BLOCK_N=block_n,
                D_HEAD=D,
                SCALE=scale,
                ALLOW_TF32=kernel_allow_tf32,
                num_warps=num_wp, num_stages=num_stages,
            )

        # Triton scan kernels above produce numerator accumulator (S) plus Z.
        # Convert to final attention output before any save/return.
        _normalize_scan_accumulator_(out, out_z)

        if needs_grad and bridge_bwd_kind == "mem":
            out_saved = out.view(B, H, N, D)
            rounded_q = ((N + 31) // 32) * 32
            lse = out_m.float() + out_z.float().clamp_min(1e-20).log()
            if lse.dim() == 2:
                lse = lse.view(B, H, N)
            if lse.shape[-1] != rounded_q:
                lse_pad = torch.full((B, H, rounded_q), float("inf"), device=q.device, dtype=torch.float32)
                lse_pad[..., :N] = lse
                lse = lse_pad
            if save_mem_out:
                ctx.save_for_backward(q, k, v, out_saved, lse)
                ctx.mem_saved_has_out = True
            else:
                ctx.save_for_backward(q, k, v, lse)
                ctx.mem_saved_has_out = False
            ctx.bridge_bwd_kind = "mem_saved_lse"
        elif needs_grad and bridge_bwd_kind in ("math", "flash"):
            ctx.save_for_backward(q, k, v)
            ctx.bridge_bwd_kind = bridge_bwd_kind
        else:
            ctx.save_for_backward(q, k, v, out_m, out_z)
            ctx.bridge_bwd_kind = ""

        ctx.scale = scale
        ctx.use_tf32 = kernel_allow_tf32
        return out.view(B, H, N, D).to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        # Training fast path redesign:
        # handle mem_saved bridge directly here to avoid extra dispatch/env churn.
        bridge_bwd_kind = getattr(ctx, "bridge_bwd_kind", "")
        if bridge_bwd_kind == "mem_saved_lse":
            has_out = bool(getattr(ctx, "mem_saved_has_out", True))
            if has_out:
                q, k, v, out, lse = ctx.saved_tensors
            else:
                q, k, v, lse = ctx.saved_tensors
                # Recompute output to avoid per-layer persistent out tensor in ctx.
                with _tf32_context(bool(getattr(ctx, "use_tf32", False))), torch.backends.cuda.sdp_kernel(
                    enable_math=False,
                    enable_mem_efficient=True,
                    enable_flash=False,
                ):
                    out = F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=None,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=float(ctx.scale),
                    )
            ready_event = getattr(ctx, "stats_ready_event", None)
            if ready_event is not None:
                torch.cuda.current_stream(device=q.device).wait_event(ready_event)
            do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            dq, dk, dv, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                do,
                q,
                k,
                v,
                None,
                out,
                lse,
                _SDPA_ZERO_SEED,
                _SDPA_ZERO_OFFSET,
                0.0,
                [True, True, True, False],
                False,
                scale=float(ctx.scale),
            )
            return dq, dk, dv, None
        if bridge_bwd_kind == "mem_saved":
            q, k, v, out, out_m, out_z = ctx.saved_tensors
            do = grad_out if grad_out.is_contiguous() else grad_out.contiguous()
            bsz, nheads, q_len, _ = q.shape
            rounded_q = ((q_len + 31) // 32) * 32
            out_m_bh = out_m.view(bsz, nheads, q_len) if out_m.dim() == 2 else out_m
            out_z_bh = out_z.view(bsz, nheads, q_len) if out_z.dim() == 2 else out_z
            lse = out_m_bh.float() + out_z_bh.float().clamp_min(1e-20).log()
            if lse.shape[-1] != rounded_q:
                lse_pad = torch.full(
                    (bsz, nheads, rounded_q),
                    float("inf"),
                    device=q.device,
                    dtype=torch.float32,
                )
                lse_pad[..., :q_len] = lse
                lse = lse_pad
            out_saved = out.view(bsz, nheads, q_len, out.shape[-1]) if out.dim() == 3 else out
            dq, dk, dv, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                do,
                q,
                k,
                v,
                None,
                out_saved,
                lse,
                _SDPA_ZERO_SEED,
                _SDPA_ZERO_OFFSET,
                0.0,
                [True, True, True, False],
                False,
                scale=float(ctx.scale),
            )
            return dq, dk, dv, None
        if bridge_bwd_kind:
            # Keep rare bridge kinds routed through the generic implementation.
            return ELSA_triton_fp32.backward(ctx, grad_out)

        bwd_impl = getattr(ctx, "_train_bwd_impl", ELSA_triton_fp32_train._resolve_train_bwd())

        q, k, v, out_m, out_z = ctx.saved_tensors
        scale = ctx.scale
        allow_tf32 = bool(getattr(ctx, "use_tf32", False))
        q_ = q.contiguous()
        k_ = k.contiguous()
        v_ = v.contiguous()
        do = grad_out.contiguous()

        B, H, N, D = q_.shape
        if bwd_impl in ("math", "mem", "flash"):
            dq, dk, dv = _sdpa_vjp(q_, k_, v_, do, scale=float(scale), kind=bwd_impl)
            return dq, dk, dv, None
        if bwd_impl == "auto" and q_.is_cuda:
            auto_kind = _resolve_fp32_auto_bwd_kind(int(N))
            dq, dk, dv = _sdpa_vjp(q_, k_, v_, do, scale=float(scale), kind=auto_kind)
            return dq, dk, dv, None

        qh = q_.view(B * H, N, D)
        kh = k_.view(B * H, N, D)
        vh = v_.view(B * H, N, D)
        doh = do.view(B * H, N, D)
        mh = out_m.view(B * H, N)
        zh = out_z.view(B * H, N)

        block_m, block_n, block_d, num_warps, num_stages = _get_bwd_launch_params(N, D)

        delta = torch.empty_like(zh)
        dq = torch.empty_like(qh)
        dk = torch.zeros_like(kh)
        dv = torch.zeros_like(vh)

        grid_q = (triton.cdiv(N, block_m), B * H)
        kernel_elsa_bwd_delta[grid_q](
            qh, kh, vh, doh, mh, zh, delta,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        kernel_elsa_bwd_dq[grid_q](
            qh, kh, vh, doh, mh, zh, delta, dq,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dq.stride(0), dq.stride(1), dq.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        grid_k = (triton.cdiv(N, block_n), B * H)
        kernel_elsa_bwd_dkv[grid_k](
            qh, kh, vh, doh, mh, zh, delta, dk, dv,
            qh.stride(0), qh.stride(1), qh.stride(2),
            kh.stride(0), kh.stride(1), kh.stride(2),
            vh.stride(0), vh.stride(1), vh.stride(2),
            doh.stride(0), doh.stride(1), doh.stride(2),
            mh.stride(0), mh.stride(1),
            zh.stride(0), zh.stride(1),
            delta.stride(0), delta.stride(1),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            BH=B * H,
            N_CTX=N,
            D_HEAD=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            SCALE=scale,
            ALLOW_TF32=allow_tf32,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        return dq.view(B, H, N, D), dk.view(B, H, N, D), dv.view(B, H, N, D), None


class ELSA_triton_mem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale, is_causal=False):
        B, H, N, D = q.shape

        if not q.is_cuda:
            from elsa_cpu import elsa_forward
            return elsa_forward(q, k, v, float(scale), bool(is_causal))

        q_ = q.contiguous().view(B * H, N, D)
        k_ = k.contiguous().view(B * H, N, D)
        v_ = v.contiguous().view(B * H, N, D)

        BLOCK_D = 32 * ((D + 31) // 32)
        allow_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
        use_autotune = bool(int(os.environ.get("ELSA_MEM_AUTOTUNE", "0")))
        if use_autotune:
            grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
        else:
            BLOCK_M = 64 if D <= 128 else 32
            base_block_n = 128 if N >= 128 else 64
            try:
                env_eta = float(os.environ.get("ELSA_MEM_ETA", "1.0"))
            except ValueError:
                env_eta = 1.0
            env_eta = max(0.125, min(1.0, env_eta))
            scaled_block_n = int(base_block_n * env_eta)
            granularity = 32 if base_block_n >= 64 else 16
            BLOCK_N = granularity * max(1, math.ceil(scaled_block_n / granularity))
            num_warps = 4 if D <= 128 else 8
            grid = (triton.cdiv(N, BLOCK_M), B * H)

        out = torch.empty_like(q_)

        if use_autotune:
            kernel_elsa_attention_fp32_fast_tuned[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_D=BLOCK_D,
                SCALE=scale,
                IS_CAUSAL=is_causal,
                ALLOW_TF32=allow_tf32,
            )
        else:
            kernel_elsa_attention_fp32_fast[grid](
                q_, k_, v_, out,
                q_.stride(0), q_.stride(1), q_.stride(2),
                k_.stride(0), k_.stride(1), k_.stride(2),
                v_.stride(0), v_.stride(1), v_.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                BH=B * H,
                N_CTX=N,
                D_HEAD=D,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_D=BLOCK_D,
                SCALE=scale,
                IS_CAUSAL=is_causal,
                ALLOW_TF32=allow_tf32,
                num_warps=num_warps,
                num_stages=2,
            )

        return out.view(B, H, N, D).to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        return None, None, None, None, None
        
# 優化的 PyTorch 實作
def ELSA_pytorch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    dropout_p: float = 0.0,
    is_causal: bool = False
) -> torch.Tensor:
    """優化的 PyTorch 實作 - 正確處理混合精度以使用 Tensor Core"""
    B, H, N, D = q.shape
    
    # 保持原始 dtype (FP16)
    dtype = q.dtype
    device = q.device
    

    # 初始化 (FP32 累積)
    m = torch.full((B, H, N, 1), -torch.inf, dtype=torch.float32, device=device)
    l = torch.zeros((B, H, N, 1), dtype=torch.float32, device=device)
    o = torch.zeros((B, H, N, D), dtype=torch.float32, device=device)
    
    # 縮放 Q 但保持 FP16
    q_scaled = q * scale
    
    # 優化的 chunk 大小
    chunk_size = 256 if N > 1024 else 128
    
    for i in range(0, N, chunk_size):
        j = min(i + chunk_size, N)
        
        # 使用自動混合精度來最大化 Tensor Core 使用
        with torch.amp.autocast(enabled=True, dtype=dtype, device_type=q.device.type):
            # 這會使用 Tensor Core (FP16 × FP16 → FP32)
            scores = torch.matmul(q_scaled, k[:, :, i:j].transpose(-2, -1))
            scores = scores.float()  # 確保是 FP32 for softmax
        
        # Causal mask
        if is_causal:
            mask = torch.triu(
                torch.ones(N, j-i, dtype=torch.bool, device=device),
                diagonal=i+1-torch.arange(N, device=device).unsqueeze(1)
            )
            scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -torch.inf)
        
        # Online softmax (FP32)
        m_curr = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, m_curr)
        
        p = torch.exp(scores - m_new)
        alpha = torch.exp(m - m_new)
        
        # Dropout
        if dropout_p > 0 and q.requires_grad:
            p = F.dropout(p, p=dropout_p, training=True)
        
        # 更新 l
        l = l * alpha + p.sum(dim=-1, keepdim=True)
        
        # 對於 o 的更新，使用混合精度以利用 Tensor Core
        with torch.amp.autocast(enabled=True, dtype=dtype, device_type=q.device.type):
            # p 會自動轉為 FP16，v 保持 FP16
            # 內部使用 Tensor Core: FP16 × FP16 → FP32 累積
            o_update = torch.matmul(p.to(dtype), v[:, :, i:j])
        
        # FP32 累積
        o = o * alpha + o_update.float()
        m = m_new
    
    # 歸一化並轉回原始 dtype
    return (o / l.clamp(min=1e-6)).to(dtype)



    
# ========= 0-A  PyTorch 版 ELSA -– 帶 bias / mask =========
@torch.jit.script
def _softmax_row(scores: torch.Tensor) -> torch.Tensor:  # (B,H,N,N)
    max_v, _ = scores.max(-1, keepdim=True)
    exp_v = (scores - max_v).exp()
    return exp_v / exp_v.sum(-1, keepdim=True)
    
def ELSA_swin_pytorch(q, k, v,
                           log_scale,         # (H,1,1) fp32
                           rel_bias=None,     # (H,N,N) fp32 / None
                           attn_mask=None):   # (B_,1,N,N) fp32 / None
    # q = F.normalize(q.to(torch.float32), dim=-1)
    # k = F.normalize(k.to(torch.float32), dim=-1)
    scores = torch.matmul(q, k.transpose(-1, -2))        # (B,H,N,N) fp32
    scores *= log_scale                                  # broadcast

    if rel_bias is not None:
        scores += rel_bias.unsqueeze(0)                  # (1,H,N,N)
    if attn_mask is not None:
        scores += attn_mask                              # (B_,1,N,N)

    attn = _softmax_row(scores)                          # bit-wise = baseline
    out  = torch.matmul(attn, v.to(torch.float32))
    return out.to(v.dtype)
             # (B,H,N,D)


import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math
from typing import Optional, Tuple


def ELSA_swinv2_pytorch(q, k, v, logit_scale, relative_position_bias=None, mask=None, chunk_size=128):
    """
    ELSA attention PyTorch implementation for Swin Transformer v2
    
    Args:
        q, k, v: (B, H, N, D) - already normalized for cosine attention
        logit_scale: (H, 1, 1) - learnable temperature parameter
        relative_position_bias: (H, N, N) - relative position bias
        mask: (B, 1, N, N) - attention mask, 0 or -inf
        chunk_size: chunk size for memory-efficient computation
    
    Returns:
        output: (B, H, N, D)
    """
    B, H, N, D = q.shape
    
    # Initialize running statistics
    m = torch.full((B, H, N, 1), -1e10, dtype=torch.float32, device=q.device)
    l = torch.zeros_like(m)
    acc = torch.zeros(B, H, N, D, dtype=torch.float32, device=q.device)
    
    # Apply logit scale
    scale = logit_scale.exp()
    
    # Process in chunks
    for start_idx in range(0, N, chunk_size):
        end_idx = min(start_idx + chunk_size, N)
        
        # Get key and value chunks
        k_chunk = k[:, :, start_idx:end_idx]
        v_chunk = v[:, :, start_idx:end_idx]
        
        # Compute cosine similarity scores
        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        
        # Add relative position bias if provided
        if relative_position_bias is not None:
            scores = scores + relative_position_bias[:, :, start_idx:end_idx].unsqueeze(0)
        
        # Add mask if provided
        if mask is not None:
            scores = scores + mask[:, :, :, start_idx:end_idx]
        
        # Stable softmax update
        scores_max = scores.amax(dim=-1, keepdim=True)
        m_new = torch.maximum(m, scores_max)
        
        # Update accumulator
        correction = torch.exp(m - m_new)
        scores_exp = torch.exp(scores - m_new)
        
        l = l * correction + scores_exp.sum(dim=-1, keepdim=True)
        acc = acc * correction + torch.matmul(scores_exp, v_chunk.to(torch.float32))
        
        m = m_new
    
    # Normalize
    output = acc / l
    
    return output.to(q.dtype)


def ELSA_swinv2_pytorch_short(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Specialized fast-path for short Swin windows (N ≤ 256).
    Windows are batched across heads to maximise GEMM throughput.
    """
    B, H, N, D = q.shape
    dv = v.shape[-1]

    triton_enabled = (
        q.is_cuda
        and bool(int(os.environ.get("ELSA_SWIN_SHORT_TRITON", "0")))
        and elsa_swinv2_triton is not None
    )
    if triton_enabled:
        try:
            allow_tf32 = bool(int(os.environ.get("ELSA_SWIN_SHORT_TF32", "1")))
            if N <= 64 and D <= 64:
                out_short = elsa_swinv2_triton_short_kernel(
                    q,
                    k,
                    v,
                    logit_scale=logit_scale,
                    relative_position_bias=relative_position_bias,
                    mask=mask,
                    allow_tf32=allow_tf32,
                )
                return out_short
            q_contig = q.to(torch.float32).contiguous()
            k_contig = k.to(torch.float32).contiguous()
            v_contig = v.to(torch.float32).contiguous()

            rel_bias_contig = relative_position_bias
            if relative_position_bias is not None:
                rel_bias_contig = relative_position_bias.to(torch.float32).contiguous()
            mask_contig = mask
            if mask is not None:
                mask_contig = mask.to(torch.float32).contiguous()

            out_triton = elsa_swinv2_triton(
                q_contig,
                k_contig,
                v_contig,
                logit_scale=logit_scale.to(torch.float32).contiguous(),
                relative_position_bias=rel_bias_contig,
                mask=mask_contig,
                use_half_qk=allow_tf32,
            )
            return out_triton.to(q.dtype)
        except Exception:
            triton_enabled = False

    sdpa_enabled = (
        hasattr(F, "scaled_dot_product_attention")
        and q.is_cuda
        and bool(int(os.environ.get("ELSA_SWIN_USE_SDPA_SHORT", "0")))
    )
    if sdpa_enabled:
        try:
            q_fp32 = _as_fp32_contig(q)
            k_fp32 = _as_fp32_contig(k)
            v_fp32 = _as_fp32_contig(v)
            q_flat = q_fp32.view(B * H, N, D)
            k_flat = k_fp32.view(B * H, N, D)
            v_flat = v_fp32.view(B * H, N, dv)

            scale = logit_scale.exp().clamp_min(1e-6)
            scale_flat = scale.view(1, H, 1, 1).expand(B, -1, -1, -1).reshape(B * H, 1, 1)
            sqrt_scale = torch.sqrt(scale_flat)
            q_scaled = q_flat * sqrt_scale
            k_scaled = k_flat * sqrt_scale

            attn_bias = None
            if relative_position_bias is not None:
                bias = _as_fp32_contig(relative_position_bias)
                bias = bias.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * H, N, N)
                attn_bias = bias
            if mask is not None:
                mask_bias = _as_fp32_contig(mask)
                if mask_bias.dim() == 4 and mask_bias.size(1) == 1:
                    mask_bias = mask_bias.view(B, 1, N, N).expand(-1, H, -1, -1)
                mask_bias = mask_bias.reshape(B * H, N, N)
                attn_bias = mask_bias if attn_bias is None else attn_bias + mask_bias

            sdp_ctx = torch.backends.cuda.sdp_kernel
            with sdp_ctx(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(
                    q_scaled,
                    k_scaled,
                    v_flat,
                    attn_mask=attn_bias,
                    dropout_p=0.0,
                    is_causal=False,
                )
            return out.view(B, H, N, dv).to(q.dtype)
        except RuntimeError:
            sdpa_enabled = False

    q_fp32 = _as_fp32_contig(q)
    k_fp32 = _as_fp32_contig(k)
    v_fp32 = _as_fp32_contig(v)
    rel_bias_fp32 = _as_fp32_contig(relative_position_bias) if relative_position_bias is not None else None
    mask_fp32 = _as_fp32_contig(mask) if mask is not None else None
    logit_scale_fp32 = _as_fp32_contig(logit_scale)

    compiled_fn = None
    # Auto-enable the compiled short path for fp16 short-window training, where
    # Swin full-model train throughput is sensitive to Python overhead.
    compile_train_raw = os.environ.get("ELSA_SWIN_SHORT_COMPILE_TRAIN", "auto").strip().lower()
    if compile_train_raw in ("1", "true", "on", "yes", "force"):
        compile_in_train = True
    elif compile_train_raw in ("0", "false", "off", "no"):
        compile_in_train = False
    else:
        compile_in_train = bool(
            q.is_cuda
            and torch.is_grad_enabled()
            and q.dtype == torch.float16
            and N <= 64
            and D <= 64
        )
    should_use_compile = q_fp32.is_cuda and (compile_in_train or (not torch.is_grad_enabled()))
    if should_use_compile:
        compiled_fn = _short_attention_compiled()
    if compiled_fn is not None:
        try:
            out_compiled = compiled_fn(q_fp32, k_fp32, v_fp32, logit_scale_fp32, rel_bias_fp32, mask_fp32)
            return out_compiled.to(q.dtype)
        except Exception:
            pass

    half_qk_raw = os.environ.get("ELSA_SWIN_SHORT_HALF_QK", "auto").strip().lower()
    if half_qk_raw in ("1", "true", "on", "yes", "force"):
        use_half_qk = True
    elif half_qk_raw in ("0", "false", "off", "no"):
        use_half_qk = False
    else:
        use_half_qk = bool(
            q_fp32.is_cuda
            and torch.is_grad_enabled()
            and q.dtype == torch.float16
            and N <= 64
            and D <= 64
        )
    if not use_half_qk:
        out = _short_attention_base(q_fp32, k_fp32, v_fp32, logit_scale_fp32, rel_bias_fp32, mask_fp32)
        return out.to(q.dtype)

    q_half = q_fp32.to(torch.float16)
    k_half = k_fp32.to(torch.float16)
    q_half_flat = q_half.view(B * H, N, D)
    k_half_flat = k_half.view(B * H, N, D)
    q_flat = q_fp32.view(B * H, N, D)
    k_flat = k_fp32.view(B * H, N, D)

    scores_main = torch.bmm(q_half_flat, k_half_flat.transpose(1, 2)).to(torch.float32)
    dq = (q_fp32 - q_half.to(torch.float32)).view(B * H, N, D)
    dk = (k_fp32 - k_half.to(torch.float32)).view(B * H, N, D)
    corr1 = torch.bmm(dq, k_flat.transpose(1, 2))
    corr2 = torch.bmm(q_flat, dk.transpose(1, 2))
    scores = scores_main + corr1 + corr2

    scale = logit_scale_fp32.exp().view(1, H, 1, 1).expand(B, -1, -1, -1)
    scores = scores.view(B, H, N, N) * scale
    if rel_bias_fp32 is not None:
        scores = scores + rel_bias_fp32.unsqueeze(0)
    if mask_fp32 is not None:
        temp_mask = mask_fp32
        if temp_mask.dim() == 4 and temp_mask.size(1) == 1:
            temp_mask = temp_mask.view(B, 1, N, N)
        scores = scores + temp_mask

    scores = scores.reshape(B * H, N, N)
    max_scores = scores.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores - max_scores)
    denom = weights.sum(dim=-1, keepdim=True)
    attn = weights / denom

    out_flat = torch.bmm(attn, v_fp32.view(B * H, N, dv))
    out = out_flat.view(B, H, N, dv)
    return out.to(q.dtype)


import triton
import triton.language as tl
import torch

# ... (你檔案中的其他程式碼) ...

@triton.jit
def elsa_swinv2_kernel(
    Q, K, V, Out,
    LogitScale, RelBias, Mask,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    stride_rb_h, stride_rb_n, stride_rb_m,
    stride_mask_b, stride_mask_h, stride_mask_n, stride_mask_m,
    B, H, N, D,
    NUM_WINDOWS,
    HAS_BIAS: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HALF_QK: tl.constexpr,
    MASK_IS_COMPACT: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_CORRECTION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    mask_m_compute = offs_m < N
    mask_d_compute = offs_d < D
    
    logit_scale_val = tl.load(LogitScale + pid_h)
    scale = tl.exp(logit_scale_val.to(tl.float32))
    
    m_i = tl.full([BLOCK_M], -1e10, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_m[:, None] * stride_qn + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=mask_m_compute[:, None] & mask_d_compute[None, :], other=0.0)
    if HALF_QK:
        q_tc = q.to(tl.float16)
        if USE_CORRECTION:
            q_fp32 = q.to(tl.float32)
    else:
        q = q.to(tl.float32)  # Cast Q to float32 for high-precision scores

    for start_n in range(0, N, BLOCK_N):
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n_curr < N
        
        k_ptrs = K + pid_b * stride_kb + pid_h * stride_kh + offs_n_curr[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d_compute[:, None], other=0.0)
        if HALF_QK:
            k_tc = k.to(tl.float16)
            if USE_CORRECTION:
                k_fp32 = k.to(tl.float32)
                main_scores = tl.dot(q_tc, k_tc, out_dtype=tl.float32, allow_tf32=False)
                dq = q_fp32 - q_tc.to(tl.float32)
                dk = k_fp32 - k_tc.to(tl.float32)
                corr1 = tl.dot(dq, k_fp32, allow_tf32=False)
                corr2 = tl.dot(q_tc.to(tl.float32), dk, allow_tf32=False)
                scores = (main_scores + corr1 + corr2) * scale
            else:
                scores = tl.dot(q_tc, k_tc, out_dtype=tl.float32, allow_tf32=False) * scale
        else:
            scores = tl.dot(q, k.to(tl.float32), allow_tf32=ALLOW_TF32) * scale
        
        if HAS_BIAS:
            bias_ptrs = RelBias + pid_h * stride_rb_h + offs_m[:, None] * stride_rb_n + offs_n_curr[None, :] * stride_rb_m
            bias = tl.load(bias_ptrs, mask=mask_m_compute[:, None] & mask_n[None, :], other=0.0)
            scores += bias.to(tl.float32)
            
        if HAS_MASK:
            mask_b = pid_b
            if MASK_IS_COMPACT:
                mask_b = pid_b % NUM_WINDOWS
            mask_ptrs = Mask + mask_b * stride_mask_b + pid_h * stride_mask_h + offs_m[:, None] * stride_mask_n + offs_n_curr[None, :] * stride_mask_m
            mask_vals = tl.load(mask_ptrs, mask=mask_m_compute[:, None] & mask_n[None, :], other=-1e10)
            scores += mask_vals.to(tl.float32)
            
        scores = tl.where(mask_n[None, :], scores, -1e10)
        
        m_ij = tl.max(scores, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        correction = tl.exp(m_i - m_i_new)
        scores_exp = tl.exp(scores - m_i_new[:, None])
        
        l_i = l_i * correction + tl.sum(scores_exp, axis=1)
        
        v_ptrs = V + pid_b * stride_vb + pid_h * stride_vh + offs_n_curr[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d_compute[None, :], other=0.0)
        
        # --- THIS IS THE FIX ---
        # Ensure accumulation happens in high precision by up-casting v
        # Accumulate via Tensor Core path with FP16 inputs and FP32 outputs
        scores_comp = scores_exp.to(v.dtype)
        acc = acc * correction[:, None] + tl.dot(scores_comp, v, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)
        m_i = m_i_new

    output = acc / (l_i[:, None] + 1e-6)
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od
    tl.store(out_ptrs, output.to(Out.dtype.element_ty), mask=mask_m_compute[:, None] & mask_d_compute[None, :])


def _maybe_contig_last(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


_SDPA_ZERO_BIAS_CACHE = {}
_SDPA_ZERO_SEED = torch.zeros((), dtype=torch.int64)
_SDPA_ZERO_OFFSET = torch.zeros((), dtype=torch.int64)
_SDPA_ZERO_PHILOX_CACHE: dict[tuple[str, int | None], tuple[torch.Tensor, torch.Tensor]] = {}
_SDPA_ZERO_RNG_STATE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _get_sdpa_zero_philox(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (device.type, device.index)
    cached = _SDPA_ZERO_PHILOX_CACHE.get(key)
    if cached is not None:
        return cached
    if device.type == "cuda":
        seed = torch.zeros((1,), device=device, dtype=torch.uint64)
        offset = torch.zeros((1,), device=device, dtype=torch.uint64)
    else:
        seed = torch.zeros((1,), dtype=torch.uint64)
        offset = torch.zeros((1,), dtype=torch.uint64)
    _SDPA_ZERO_PHILOX_CACHE[key] = (seed, offset)
    return seed, offset


def _get_sdpa_zero_rng_state(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    cached = _SDPA_ZERO_RNG_STATE_CACHE.get(key)
    if cached is not None:
        return cached
    if device.type == "cuda":
        rng_state = torch.zeros((2,), device=device, dtype=torch.int64)
    else:
        rng_state = torch.zeros((2,), dtype=torch.int64)
    _SDPA_ZERO_RNG_STATE_CACHE[key] = rng_state
    return rng_state

def _get_sdpa_zero_bias(
    q: torch.Tensor,
    *,
    q_len: int,
    k_len: int,
) -> torch.Tensor:
    """Reuse a stride-0 zero bias view to avoid per-call allocation in bridge VJP."""
    key = (q.device.type, q.device.index or -1, str(q.dtype), q.shape[0], q.shape[1], q_len, k_len)
    cached = _SDPA_ZERO_BIAS_CACHE.get(key)
    if cached is not None and cached.device == q.device:
        return cached
    bias_line = q.new_zeros((k_len,), dtype=q.dtype)
    attn_bias = torch.as_strided(
        bias_line,
        size=(q.shape[0], q.shape[1], q_len, k_len),
        stride=(0, 0, 0, 1),
    )
    _SDPA_ZERO_BIAS_CACHE[key] = attn_bias
    return attn_bias


def _sdpa_mem_vjp_from_saved(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    out: torch.Tensor,
    out_m: Optional[torch.Tensor] = None,
    out_z: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fast bridge VJP using saved ELSA forward statistics (no SDPA forward recompute)."""
    if not q.is_cuda:
        return _sdpa_vjp(q, k, v, grad_out, scale=scale, kind="mem")
    bsz, nheads, q_len, _ = q.shape
    k_len = k.shape[-2]
    if out.dim() == 3:
        out = out.view(bsz, nheads, q_len, out.shape[-1])
    if lse is None:
        if out_m is None or out_z is None:
            raise ValueError("Either lse or (out_m, out_z) must be provided for mem_saved VJP.")
        if out_m.dim() == 2:
            out_m = out_m.view(bsz, nheads, out_m.shape[-1])
        if out_z.dim() == 2:
            out_z = out_z.view(bsz, nheads, out_z.shape[-1])
        rounded_q = ((q_len + 31) // 32) * 32
        lse = out_m.float() + out_z.float().clamp_min(1e-20).log()
        if lse.shape[-1] != rounded_q:
            lse_pad = torch.full((bsz, nheads, rounded_q), float("inf"), device=q.device, dtype=torch.float32)
            lse_pad[..., :q_len] = lse
            lse = lse_pad

    def _call_mem_bwd(attn_bias):
        dq_, dk_, dv_, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
            grad_out,
            q,
            k,
            v,
            attn_bias,
            out,
            lse,
            _SDPA_ZERO_SEED,
            _SDPA_ZERO_OFFSET,
            0.0,
            [True, True, True, False],
            False,
            scale=scale,
        )
        return dq_, dk_, dv_

    prefer_none_bias = os.environ.get("ELSA_SDPA_MEM_BWD_USE_NONE_BIAS", "1") != "0"
    if prefer_none_bias:
        try:
            return _call_mem_bwd(None)
        except Exception:
            pass
    attn_bias = _get_sdpa_zero_bias(q, q_len=q_len, k_len=k_len)
    dq, dk, dv = _call_mem_bwd(attn_bias)
    return dq, dk, dv


def _sdpa_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    *,
    scale: float,
    kind: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute dQ/dK/dV via SDPA VJP; used as a fast fallback bridge in training."""
    kind = kind.lower()
    if kind not in ("math", "mem", "flash"):
        raise ValueError(f"Unsupported SDPA VJP kind '{kind}'")

    if kind == "flash" and q.is_cuda:
        # Keep ATen flash-VJP as the stable default on this stack; FA2 path stays opt-in.
        flash_vjp_impl = os.environ.get("ELSA_SDPA_FLASH_VJP_IMPL", "aten").strip().lower()
        if flash_vjp_impl not in ("auto", "fa2", "aten", ""):
            flash_vjp_impl = "auto"
        try:
            fa2_min_n = int(os.environ.get("ELSA_SDPA_FLASH_VJP_FA2_MIN_N", "512"))
        except ValueError:
            fa2_min_n = 512
        prefer_fa2 = (
            flash_vjp_impl == "fa2"
            and q.dtype == torch.float16
            and q.shape[-2] >= max(64, fa2_min_n)
            and hasattr(torch.ops, "flash_attn")
            and hasattr(torch.ops.flash_attn, "_flash_attn_forward")
            and hasattr(torch.ops.flash_attn, "_flash_attn_backward")
        )
        if prefer_fa2:
            try:
                # flash-attn op expects BNHD layout.
                q_bnhd = q.permute(0, 2, 1, 3).contiguous()
                k_bnhd = k.permute(0, 2, 1, 3).contiguous()
                v_bnhd = v.permute(0, 2, 1, 3).contiguous()
                do_bnhd = grad_out.permute(0, 2, 1, 3).contiguous()
                out_bnhd, lse, _softmax, rng_state = torch.ops.flash_attn._flash_attn_forward(
                    q_bnhd,
                    k_bnhd,
                    v_bnhd,
                    0.0,
                    float(scale),
                    False,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                )
                dq_bnhd = torch.empty_like(q_bnhd)
                dk_bnhd = torch.empty_like(k_bnhd)
                dv_bnhd = torch.empty_like(v_bnhd)
                _ = torch.ops.flash_attn._flash_attn_backward(
                    do_bnhd,
                    q_bnhd,
                    k_bnhd,
                    v_bnhd,
                    out_bnhd,
                    lse,
                    dq_bnhd,
                    dk_bnhd,
                    dv_bnhd,
                    0.0,
                    float(scale),
                    False,
                    -1,
                    -1,
                    0.0,
                    None,
                    False,
                    rng_state,
                )
                return (
                    dq_bnhd.permute(0, 2, 1, 3),
                    dk_bnhd.permute(0, 2, 1, 3),
                    dv_bnhd.permute(0, 2, 1, 3),
                )
            except Exception:
                # Fall back to ATen/autograd bridge below.
                pass

    if (
        kind == "flash"
        and q.is_cuda
        and os.environ.get("ELSA_SDPA_ATEN_FLASH_VJP", "1") != "0"
    ):
        try:
            out, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k, rng_state, _unused, _dbg = (
                torch.ops.aten._scaled_dot_product_flash_attention(
                    q,
                    k,
                    v,
                    0.0,
                    False,
                    False,
                    scale=scale,
                )
            )
            if isinstance(rng_state, torch.Tensor) and rng_state.numel() >= 2:
                philox_seed = rng_state[0].view(1)
                philox_offset = rng_state[1].view(1)
            else:
                philox_seed, philox_offset = _get_sdpa_zero_philox(q.device)
            dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
                grad_out,
                q,
                k,
                v,
                out,
                logsumexp,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                0.0,
                False,
                philox_seed,
                philox_offset,
                scale=scale,
            )
            return dq, dk, dv
        except Exception:
            pass

    if (
        kind == "mem"
        and q.is_cuda
        and os.environ.get("ELSA_SDPA_ATEN_VJP", "1") != "0"
    ):
        try:
            out, logsumexp, philox_seed, philox_offset = torch.ops.aten._scaled_dot_product_efficient_attention(
                q, k, v, None, True, 0.0, False, scale=scale
            )
            bsz, nheads, q_len = q.shape[0], q.shape[1], q.shape[-2]
            k_len = k.shape[-2]
            attn_bias = _get_sdpa_zero_bias(q, q_len=q_len, k_len=k_len)
            dq, dk, dv, _dbias = torch.ops.aten._scaled_dot_product_efficient_attention_backward(
                grad_out,
                q,
                k,
                v,
                attn_bias,
                out,
                logsumexp,
                philox_seed,
                philox_offset,
                0.0,
                [True, True, True, False],
                False,
                scale=scale,
            )
            return dq, dk, dv
        except Exception:
            # Fall back to autograd path on stacks where this aten op signature differs.
            pass

    create_graph = torch.is_grad_enabled()
    with torch.enable_grad():
        q_req = q.detach().requires_grad_(True)
        k_req = k.detach().requires_grad_(True)
        v_req = v.detach().requires_grad_(True)

        def _run():
            try:
                return F.scaled_dot_product_attention(
                    q_req, k_req, v_req, dropout_p=0.0, is_causal=False, scale=scale
                )
            except TypeError:
                return F.scaled_dot_product_attention(
                    q_req, k_req, v_req, dropout_p=0.0, is_causal=False
                )

        if q_req.is_cuda:
            with torch.backends.cuda.sdp_kernel(
                enable_math=kind == "math",
                enable_mem_efficient=kind == "mem",
                enable_flash=kind == "flash",
            ):
                out = _run()
        else:
            out = _run()

        dq, dk, dv = torch.autograd.grad(
            out,
            (q_req, k_req, v_req),
            grad_out,
            retain_graph=False,
            create_graph=create_graph,
            allow_unused=False,
        )
    return dq, dk, dv


def _resolve_fp16_bwd_kind(
    q: torch.Tensor,
    n_ctx: int,
    impl: str,
) -> str:
    """Resolve fp16/bf16 backward bridge kind from env policy."""
    def _sanitize(kind: str) -> str:
        # Keep fp16 CUDA backward bridge on stable/fast routes by default.
        if (
            q.dtype == torch.float16
            and q.is_cuda
            and kind in ("math", "mem")
            and not _allow_unstable_paths()
        ):
            _warn_once(
                f"fp16_bwd_{kind}",
                (
                    f"ELSA_TRITON_FP16_BWD={kind} is disabled by default due severe perf regression; "
                    "falling back to flash. Set ELSA_TRITON_ALLOW_UNSTABLE_PATHS=1 to force."
                ),
            )
            return "flash"
        return kind

    impl = (impl or "auto").strip().lower()
    if impl == "triton":
        # Triton-native fp16 backward path is currently unstable/perf-regressed
        # on this stack; force a stable bridge path instead of allowing the
        # catastrophic slow fallback.
        warnings.warn(
            "ELSA_TRITON_FP16_BWD=triton is disabled due severe perf regression; "
            "falling back to flash bridge.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "flash" if (q.dtype == torch.float16 and q.is_cuda) else "mem"
    if impl in ("flash", "math", "mem", "mem_saved", "mem_saved_lse"):
        # On current stack, fp16 mem_saved(_lse) can trigger illegal-memory-access
        # in long-sequence full-model benchmarks. Keep it opt-in only.
        if (
            impl in ("mem_saved", "mem_saved_lse")
            and q.dtype == torch.float16
            and q.is_cuda
            and os.environ.get("ELSA_TRITON_FP16_ALLOW_MEM_SAVED", "0") == "0"
        ):
            warnings.warn(
                f"ELSA_TRITON_FP16_BWD={impl} is disabled by default for fp16 "
                "due observed illegal-memory-access on long sequences; falling back to flash.",
                RuntimeWarning,
                stacklevel=2,
            )
            return "flash"
        return _sanitize(impl)
    if impl == "auto":
        try:
            min_n = int(os.environ.get("ELSA_TRITON_FP16_BWD_AUTO_MIN_N", "192"))
        except ValueError:
            min_n = 192
        if n_ctx < max(1, min_n):
            return ""

        if q.dtype == torch.float16 and q.is_cuda:
            try:
                long_min_n = int(os.environ.get("ELSA_TRITON_FP16_BWD_LONG_MIN_N", "2048"))
            except ValueError:
                long_min_n = 2048
            if n_ctx >= max(1, long_min_n):
                long_kind = os.environ.get(
                    "ELSA_TRITON_FP16_BWD_LONG_KIND_FP16",
                    "flash",
                ).strip().lower()
                if long_kind in ("flash", "math", "mem", "mem_saved", "mem_saved_lse"):
                    return _sanitize(long_kind)
            return _sanitize(os.environ.get(
                "ELSA_TRITON_FP16_BWD_AUTO_KIND_FP16",
                "flash",
            ).strip().lower())

        return os.environ.get(
            "ELSA_TRITON_FP16_BWD_AUTO_KIND_BF16",
            "mem",
        ).strip().lower()

    raise ValueError(
        "ELSA_TRITON_FP16_BWD must be one of auto|flash|math|mem|mem_saved|mem_saved_lse "
        "(triton is disabled)."
    )


def _resolve_fp16_flash_bwd_impl(
    q: torch.Tensor,
    n_ctx: int,
) -> str:
    """Resolve fp16 flash-bridge backward implementation.

    - `fa2`: usually better for long sequences.
    - `aten`: avoids extra layout conversion overhead on shorter sequences.
    """
    impl = os.environ.get("ELSA_TRITON_FP16_FLASH_BWD_IMPL", "auto").strip().lower()
    if impl in ("fa2", "aten"):
        return impl
    if impl not in ("", "auto"):
        return "aten"

    try:
        # Current A100 stack regresses on FA2 backward bridge around the common
        # ViT-512 token shape (~1025). Prefer the leaner ATen path until 2K+.
        min_n = int(os.environ.get("ELSA_TRITON_FP16_FLASH_BWD_FA2_MIN_N", "2048"))
    except ValueError:
        min_n = 1024
    can_fa2 = (
        q.dtype == torch.float16
        and q.is_cuda
        and n_ctx >= max(64, min_n)
        and hasattr(torch.ops, "flash_attn")
        and hasattr(torch.ops.flash_attn, "_flash_attn_backward")
    )
    return "fa2" if can_fa2 else "aten"


def _resolve_fp32_auto_bwd_kind(n_ctx: int) -> str:
    """Resolve fp32 auto backward bridge kind by sequence length.

    Small-token training/FT shapes are more stable with math VJP; longer shapes
    keep mem VJP for throughput.
    """
    try:
        math_max_n = int(os.environ.get("ELSA_TRITON_FP32_BWD_AUTO_MATH_MAX_N", "512"))
    except ValueError:
        math_max_n = 512
    math_max_n = max(64, math_max_n)
    if n_ctx <= math_max_n:
        return "math"

    long_kind = os.environ.get("ELSA_TRITON_FP32_BWD_AUTO_LONG_KIND", "mem").strip().lower()
    if long_kind not in ("mem", "math", "flash"):
        long_kind = "mem"
    return long_kind


def _get_bwd_launch_params(n_ctx: int, d_head: int):
    block_m_env = int(os.environ.get("ELSA_TRITON_BWD_BLOCK_M", "0"))
    block_n_env = int(os.environ.get("ELSA_TRITON_BWD_BLOCK_N", "0"))
    warps_env = int(os.environ.get("ELSA_TRITON_BWD_WARPS", "0"))
    stages_env = int(os.environ.get("ELSA_TRITON_BWD_STAGES", "0"))

    # Training-tuned defaults by sequence length; keep kernel-compatible tile set.
    # NOTE:
    # N~577 (ViT-384) and N~1025 (ViT-512) regressed badly with 128x64x(8w,2s)
    # on current A100+driver/triton stack. 32x64x(8w,1s) is consistently faster
    # for those common training shapes while preserving exact gradients.
    if n_ctx >= 1536:
        # Long strict two-scan fp16 bwd is faster with narrower query tiles on
        # the current A100 stack, and avoids the old 128x64 shared-memory OOR.
        block_m = 32
        block_n = 64
        num_warps = 4
        num_stages = 1
    elif n_ctx >= 512:
        # Tuned on current A100 stack: 64x32 is faster than 32x64 for N~1k.
        block_m = 64
        block_n = 32
        num_warps = 8
        num_stages = 1
    else:
        block_m = 32
        block_n = 64
        num_warps = 8
        num_stages = 1

    if block_m_env > 0:
        if block_m_env >= 128:
            block_m = 128
        elif block_m_env >= 64:
            block_m = 64
        else:
            block_m = 32
    if block_n_env > 0:
        if block_n_env >= 128:
            block_n = 128
        elif block_n_env >= 64:
            block_n = 64
        else:
            block_n = 32
    if warps_env > 0:
        num_warps = warps_env
    if stages_env > 0:
        num_stages = stages_env

    block_d = 16 * ((d_head + 15) // 16)
    return block_m, block_n, block_d, num_warps, num_stages

def elsa_swinv2_triton(
    q,
    k,
    v,
    logit_scale,
    relative_position_bias=None,
    mask=None,
    use_half_qk: bool = False,
    launch_cfg: Optional[dict] = None,
    out_layout: str = "HND",
):
    B, H, N, D = q.shape

    q = _maybe_contig_last(q)
    k = _maybe_contig_last(k)
    v = _maybe_contig_last(v)
    dv = v.shape[-1]
    out_layout = str(out_layout).strip().upper()
    if out_layout == "NH":
        out = torch.empty((B, N, H, dv), device=q.device, dtype=q.dtype)
        out_strides = (out.stride(0), out.stride(2), out.stride(1), out.stride(3))
    else:
        out = torch.empty_like(q, memory_format=torch.contiguous_format)
        out_strides = out.stride()
    
    has_bias = relative_position_bias is not None
    if has_bias:
        relative_position_bias = _maybe_contig_last(relative_position_bias)
        rel_bias_strides = relative_position_bias.stride()
    else:
        # Use dummy tensor and strides if not provided
        relative_position_bias = torch.empty(0, device=q.device)
        rel_bias_strides = (0, 0, 0)

    has_mask = mask is not None
    mask_is_compact = False
    num_windows = 0
    if has_mask:
        # Robustly handle mask broadcasting and get strides (avoid copies when possible)
        while mask.ndim < 4:
            mask = mask.unsqueeze(0)
        mask_is_compact = mask.size(0) != B
        num_windows = int(mask.size(0))
        if mask.size(1) != H or (not mask_is_compact and mask.size(0) != B):
            mask = mask.expand(B, H, N, N)
        mask = _maybe_contig_last(mask)
        mask_strides = mask.stride()
    else:
        mask = torch.empty(0, device=q.device)
        mask_strides = (0, 0, 0, 0)
    
    if N <= 64:
        BLOCK_M = BLOCK_N = 32
        num_warps = 4
    elif N <= 128:
        BLOCK_M = BLOCK_N = 64
        num_warps = 4
    elif N <= 256:
        BLOCK_M = BLOCK_N = 128
        num_warps = 8
    else:
        BLOCK_M = BLOCK_N = 64
        num_warps = 4

    # Strict/full-model fp16 window-attention benefits from a smaller Q tile
    # only for the short-window W8 case (N=64) on the current A100 stack.
    # W16 did not show a robust win from a single global launch override.
    if q.dtype in (torch.float16, torch.bfloat16) and (not use_half_qk):
        if N <= 64:
            BLOCK_M, BLOCK_N, num_warps = 16, 64, 4

    # Allow targeted launch tuning for strict/full-model window-attention cases
    # without changing the default clean routing.
    launch_cfg = dict(launch_cfg) if launch_cfg else {}
    if "block_m" in launch_cfg:
        try:
            BLOCK_M = max(16, int(launch_cfg["block_m"]))
        except (TypeError, ValueError):
            pass
    if "block_n" in launch_cfg:
        try:
            BLOCK_N = max(16, int(launch_cfg["block_n"]))
        except (TypeError, ValueError):
            pass
    if "num_warps" in launch_cfg:
        try:
            num_warps = max(1, int(launch_cfg["num_warps"]))
        except (TypeError, ValueError):
            pass
    if "num_stages" in launch_cfg:
        try:
            num_stages = max(1, int(launch_cfg["num_stages"]))
        except (TypeError, ValueError):
            num_stages = 1 if N <= 64 else 2

    override_block_m = os.environ.get("ELSA_SWIN_TRITON_BLOCK_M")
    override_block_n = os.environ.get("ELSA_SWIN_TRITON_BLOCK_N")
    override_num_warps = os.environ.get("ELSA_SWIN_TRITON_NUM_WARPS")
    override_num_stages = os.environ.get("ELSA_SWIN_TRITON_NUM_STAGES")
    if override_block_m is not None:
        try:
            BLOCK_M = max(16, int(override_block_m))
        except ValueError:
            pass
    if override_block_n is not None:
        try:
            BLOCK_N = max(16, int(override_block_n))
        except ValueError:
            pass
    if override_num_warps is not None:
        try:
            num_warps = max(1, int(override_num_warps))
        except ValueError:
            pass
    if override_num_stages is not None:
        try:
            num_stages = max(1, int(override_num_stages))
        except ValueError:
            num_stages = 1 if N <= 64 else 2
    else:
        if q.dtype in (torch.float16, torch.bfloat16) and (not use_half_qk) and N <= 64:
            num_stages = 1
        else:
            num_stages = 1 if N <= 64 else 2

    BLOCK_D = min(128, triton.next_power_of_2(max(D, 16)))

    use_fp32_fused = (
        q.dtype == torch.float32
        and not use_half_qk
        and not torch.backends.cuda.matmul.allow_tf32
        and N <= 64
        and D <= 64
        and bool(int(os.environ.get("ELSA_SWIN_FP32_FUSED", "1")))
    )
    if use_fp32_fused:
        return elsa_swinv2_triton_short_kernel(
            q,
            k,
            v,
            logit_scale=logit_scale,
            relative_position_bias=relative_position_bias if has_bias else None,
            mask=mask if has_mask else None,
            allow_tf32=False,
        )

    use_fixed_bias_kblock = (
        q.dtype in (torch.float16, torch.bfloat16)
        and not use_half_qk
        and has_bias
        and not has_mask
        and N <= 256
        and (N % BLOCK_M == 0)
        and (N % BLOCK_N == 0)
        and D <= BLOCK_D
        and bool(int(os.environ.get("ELSA_SWIN_FP16_FIXED_BIAS_KBLOCK", "1")))
    )
    use_fixed_bias_compactmask_kblock = (
        q.dtype in (torch.float16, torch.bfloat16)
        and not use_half_qk
        and has_bias
        and has_mask
        and mask_is_compact
        and N <= 256
        and (N % BLOCK_M == 0)
        and (N % BLOCK_N == 0)
        and D <= BLOCK_D
        and bool(int(os.environ.get("ELSA_SWIN_FP16_FIXED_BIAS_COMPACTMASK_KBLOCK", "1")))
    )
    use_fixed_bias_compactmask_block8 = (
        q.dtype in (torch.float16, torch.bfloat16)
        and not use_half_qk
        and has_bias
        and has_mask
        and mask_is_compact
        and N == 256
        and D <= BLOCK_D
        and bool(int(os.environ.get("ELSA_SWIN_FP16_FIXED_BIAS_COMPACTMASK_BLOCK8", "1")))
    )
    fixed_kblock_acc16 = bool(int(os.environ.get("ELSA_SWIN_FP16_FIXED_KBLOCK_ACC16", "1")))
    if use_fixed_bias_compactmask_block8:
        block8_m = 8
        block8_n = 8
        grid = (B, H, triton.cdiv(N, block8_m))
        kernel_elsa_attention_fwd_fixed_bias_compactmask_block8[grid](
            q, k, v, logit_scale.contiguous(), relative_position_bias, mask, out,
            *q.stride(), *k.stride(), *v.stride(),
            *rel_bias_strides,
            *mask_strides,
            *out_strides,
            B, H, N, D, num_windows,
            BLOCK_M=block8_m, BLOCK_N=block8_n, BLOCK_D=BLOCK_D,
            ACC_IN_FP16=fixed_kblock_acc16,
            num_warps=4, num_stages=1,
        )
        return out
    if use_fixed_bias_kblock:
        grid = (B, H, triton.cdiv(N, BLOCK_M))
        kernel_elsa_attention_fwd_fixed_bias_kblock[grid](
            q, k, v, logit_scale.contiguous(), relative_position_bias, out,
            *q.stride(), *k.stride(), *v.stride(),
            *rel_bias_strides,
            *out_strides,
            B, H, N, D,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            scale=1.0,
            ACC_IN_FP16=fixed_kblock_acc16,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out
    if use_fixed_bias_compactmask_kblock:
        grid = (B, H, triton.cdiv(N, BLOCK_M))
        kernel_elsa_attention_fwd_fixed_bias_compactmask_kblock[grid](
            q, k, v, logit_scale.contiguous(), relative_position_bias, mask, out,
            *q.stride(), *k.stride(), *v.stride(),
            *rel_bias_strides,
            *mask_strides,
            *out_strides,
            B, H, N, D, num_windows,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            scale=1.0,
            ACC_IN_FP16=fixed_kblock_acc16,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out

    grid = (B, H, triton.cdiv(N, BLOCK_M))

    allow_tf32 = (
        q.dtype == torch.float32
        and torch.backends.cuda.matmul.allow_tf32
        and not use_half_qk
        and bool(int(os.environ.get("ELSA_SWIN_USE_TF32", "1")))
    )
    use_correction = use_half_qk and q.dtype == torch.float32
    half_qk = use_half_qk or q.dtype == torch.float16

    elsa_swinv2_kernel[grid](
        q, k, v, out,
        logit_scale.contiguous(), relative_position_bias, mask,
        *q.stride(), *k.stride(), *v.stride(), *out_strides,
        *rel_bias_strides,
        *mask_strides,
        B, H, N, D,
        num_windows,
        HAS_BIAS=has_bias,
        HAS_MASK=has_mask,
        HALF_QK=half_qk,
        MASK_IS_COMPACT=mask_is_compact,
        ALLOW_TF32=allow_tf32,
        USE_CORRECTION=use_correction,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        num_warps=num_warps, num_stages=num_stages,
    )

    return out


def elsa_swinv2_triton_short_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    logit_scale: torch.Tensor,
    relative_position_bias: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    allow_tf32: bool = True,
) -> torch.Tensor:
    B, H, N, D = q.shape
    dv = v.shape[-1]
    out = torch.empty_like(q, memory_format=torch.contiguous_format)

    q32 = q if q.dtype == torch.float32 else q.to(torch.float32)
    k32 = k if k.dtype == torch.float32 else k.to(torch.float32)
    v32 = v if v.dtype == torch.float32 else v.to(torch.float32)
    logit_scale32 = logit_scale if logit_scale.dtype == torch.float32 else logit_scale.to(torch.float32)
    q32 = _maybe_contig_last(q32)
    k32 = _maybe_contig_last(k32)
    v32 = _maybe_contig_last(v32)
    logit_scale32 = _maybe_contig_last(logit_scale32)

    if relative_position_bias is not None:
        rel_bias = relative_position_bias if relative_position_bias.dtype == torch.float32 else relative_position_bias.to(torch.float32)
        rel_bias = _maybe_contig_last(rel_bias)
        rel_bias_strides = rel_bias.stride()
    else:
        rel_bias = torch.empty(0, device=q.device)
        rel_bias_strides = (0, 0, 0)

    mask_is_compact = False
    num_windows = 0
    if mask is not None:
        mask_t = mask if mask.dtype == torch.float32 else mask.to(torch.float32)
        while mask_t.ndim < 4:
            mask_t = mask_t.unsqueeze(0)
        mask_is_compact = mask_t.size(0) != B
        num_windows = int(mask_t.size(0))
        if mask_t.size(1) != H or (not mask_is_compact and mask_t.size(0) != B):
            mask_t = mask_t.expand(B, H, N, N)
        mask_t = _maybe_contig_last(mask_t)
        mask_strides = mask_t.stride()
    else:
        mask_t = torch.empty(0, device=q.device)
        mask_strides = (0, 0, 0, 0)

    grid = (B * H,)
    BLOCK_N = 64 if N > 32 else 32
    BLOCK_D = 64 if D > 32 else 32
    BLOCK_DV = 64 if dv > 32 else 32
    elsa_swinv2_kernel_short_fused[grid](
        q32,
        k32,
        v32,
        out,
        logit_scale32,
        rel_bias,
        mask_t,
        *q32.stride(),
        *k32.stride(),
        *v32.stride(),
        *out.stride(),
        *rel_bias_strides,
        *mask_strides,
        B,
        H,
        N,
        D,
        dv,
        num_windows,
        HAS_BIAS=relative_position_bias is not None,
        HAS_MASK=mask is not None,
        MASK_IS_COMPACT=mask_is_compact,
        ALLOW_TF32=allow_tf32,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        BLOCK_DV=BLOCK_DV,
        num_warps=4,
        num_stages=1,
    )
    return out.to(q.dtype)


def elsa_triton_vit_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        )
    dtype = q.dtype
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton_mem.apply(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), scale, is_causal)
    return out.to(dtype)


def elsa_triton_mem_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = q.dtype
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        ).to(dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton_fp32.apply(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), scale)
    return out.to(dtype)


def elsa_triton_baseline_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Expose baseline Triton implementation for benchmarks (FP32 path)."""
    dtype = q.dtype
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        ).to(dtype)
    return elsa_triton_vit_fp32(q, k, v, is_causal=is_causal, bias=bias)


def elsa_triton_tensor_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = q.dtype
    if bias is not None or is_causal:
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=bias,
            is_causal=is_causal,
            dropout_p=0.0,
        ).to(dtype)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton_fp32.apply(q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), scale)
    return out.to(dtype)


def elsa_triton_new_fp32_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    dtype = q.dtype
    attn_mask = bias
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = ELSA_triton.apply(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        scale,
        None,
        is_causal,
    )
    if attn_mask is not None or is_causal:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=0.0,
        )
    return out.to(dtype)


def elsa_triton_new_fp32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Default FP32 path uses the faster streaming kernel."""
    return elsa_triton_mem_fp32(q, k, v, is_causal=is_causal, bias=bias)


def elsa_triton_new_fp32_fast(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Alias of the default FP32 path."""
    return elsa_triton_new_fp32(q, k, v, is_causal=is_causal, bias=bias)


def elsa_triton_fp16_infer(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
) -> torch.Tensor:
    """Inference-only fast path for fp16/bf16 ELSA Triton forward.

    This bypasses autograd context handling and is intended for benchmark /
    eval loops where gradients are not required.
    """
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("elsa_triton_fp16_infer expects fp16 or bf16 inputs.")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise RuntimeError("elsa_triton_fp16_infer requires CUDA tensors.")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    B, H, N, D = q.shape
    if k.shape != q.shape or v.shape != q.shape:
        raise RuntimeError("q/k/v must have identical shape [B, H, N, D].")

    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(D)
    block_d = 16 * ((D + 15) // 16)

    dev_prop = torch.cuda.get_device_properties(q.device)
    blk = _choose_tile(N, dev_prop, prefer_large=True)
    if blk == 128:
        block_m = block_n = 128 if D <= 64 else 64
        num_warps = 4 if D <= 64 else 8
    elif blk == 96:
        block_m = block_n = 96
        num_warps = 4
    else:
        block_m = block_n = 64
        num_warps = 4
    num_stages = 2

    if N >= 4096 and D <= 64:
        block_m, block_n, num_warps, num_stages = 128, 64, 4, 2
    elif N >= 4096:
        block_m, block_n, num_warps, num_stages = 64, 64, 4, 2

    def _read_env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except ValueError:
            return default

    bm_env = _read_env_int("ELSA_TRITON_FP16_FWD_BLOCK_M", 0)
    bn_env = _read_env_int("ELSA_TRITON_FP16_FWD_BLOCK_N", 0)
    wp_env = _read_env_int("ELSA_TRITON_FP16_FWD_WARPS", 0)
    st_env = _read_env_int("ELSA_TRITON_FP16_FWD_STAGES", 0)
    if bm_env > 0:
        block_m = max(16, (bm_env // 16) * 16)
    if bn_env > 0:
        block_n = max(16, (bn_env // 16) * 16)
    if wp_env > 0:
        num_warps = max(1, wp_env)
    if st_env > 0:
        num_stages = max(1, st_env)
    if q.dtype in (torch.float16, torch.bfloat16):
        block_m = _sanitize_fp16_fwd_block(block_m, name="ELSA_TRITON_FP16_FWD_BLOCK_M")
        block_n = _sanitize_fp16_fwd_block(block_n, name="ELSA_TRITON_FP16_FWD_BLOCK_N")

    # Keep parity with main forward behavior: optional in-function K transpose.
    k_fwd = k
    k_stride_kn = k.stride(2)
    k_stride_kd = k.stride(3)
    use_k_transpose = os.environ.get("ELSA_TRITON_FP16_FWD_TRANSPOSE_K", "0") != "0"
    if use_k_transpose:
        try:
            k_transpose_min_n = int(os.environ.get("ELSA_TRITON_FP16_FWD_TRANSPOSE_K_MIN_N", "8192"))
        except ValueError:
            k_transpose_min_n = 8192
        if N >= max(1, k_transpose_min_n):
            k_t = k.transpose(-1, -2).contiguous()
            k_fwd = k_t
            k_stride_kn = k_t.stride(3)
            k_stride_kd = k_t.stride(2)

    auto_infer_kblock = _fp16_kblock_auto_enabled(n_ctx=N, d_head=D, is_causal=is_causal)
    fp16_fast_acc = _resolve_fp16_fast_accum(
        q=q,
        n_ctx=N,
        d_head=D,
        needs_grad=False,
        is_causal=is_causal,
        prefer_infer_fast=auto_infer_kblock,
    )
    grid = (B, H, triton.cdiv(N, block_m))
    use_ultra_fp16 = bool(
        q.dtype == torch.float16
        and os.environ.get("ELSA_TRITON_FP16_ULTRA_FAST", "0") != "0"
        and (not is_causal)
        and D == block_d
        and N % block_m == 0
        and N % block_n == 0
    )
    flat_mode = os.environ.get("ELSA_TRITON_FP16_FLAT", "auto").strip().lower()
    if flat_mode in ("1", "true", "force", "on"):
        flat_enabled = True
    elif flat_mode in ("auto", ""):
        flat_enabled = _fp16_flat_auto_enabled(n_ctx=N, d_head=D, is_causal=is_causal)
    else:
        flat_enabled = False
    use_flat_nomask = bool(
        flat_enabled
        and (not is_causal)
        and D == block_d
        and N % block_m == 0
        and N % block_n == 0
    )
    use_tc = bool(
        os.environ.get("ELSA_TRITON_FP16_TC", "0") != "0"
        and (not is_causal)
        and D == block_d
        and N % block_m == 0
        and N % block_n == 0
    )
    kblock_mode = os.environ.get("ELSA_TRITON_FP16_KBLOCK", "auto").strip().lower()
    if kblock_mode in ("1", "true", "force", "on", "yes"):
        kblock_enabled = True
    elif kblock_mode in ("0", "false", "off", "no"):
        kblock_enabled = False
    else:
        kblock_enabled = auto_infer_kblock
    use_kblock = bool(
        kblock_enabled
        and (not is_causal)
        and D == block_d
        and N % block_m == 0
        and N % block_n == 0
    )
    use_nomask = bool(
        os.environ.get("ELSA_TRITON_FP16_NOMASK", "1") != "0"
        and (not is_causal)
        and D == block_d
        and N % block_m == 0
        and N % block_n == 0
    )
    if use_ultra_fp16:
        kernel_elsa_attention_fwd_fixed_nomask_fp16stats[grid](
            q,
            k_fwd,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_fwd.stride(0),
            k_fwd.stride(1),
            k_stride_kn,
            k_stride_kd,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            N,
            D,
            block_m,
            block_n,
            block_d,
            scale=scale,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif use_flat_nomask:
        grid_flat = (triton.cdiv(N, block_m), B * H)
        kernel_elsa_attention_fwd_fixed_nomask_flat[grid_flat](
            q,
            k_fwd,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_fwd.stride(0),
            k_fwd.stride(1),
            k_stride_kn,
            k_stride_kd,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            N,
            D,
            block_m,
            block_n,
            block_d,
            scale=scale,
            USE_TF32=False,
            ACC_IN_FP16=fp16_fast_acc,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif use_kblock:
        kernel_elsa_attention_fwd_fixed_nomask_kblock[grid](
            q,
            k_fwd,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_fwd.stride(0),
            k_fwd.stride(1),
            k_stride_kn,
            k_stride_kd,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            N,
            D,
            block_m,
            block_n,
            block_d,
            scale=scale,
            ACC_IN_FP16=fp16_fast_acc,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif use_tc:
        kernel_elsa_attention_fwd_fp16_tc[grid](
            q,
            k_fwd,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_fwd.stride(0),
            k_fwd.stride(1),
            k_stride_kn,
            k_stride_kd,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            N,
            D,
            block_m,
            block_n,
            block_d,
            scale=scale,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    elif use_nomask:
        kernel_elsa_attention_fwd_fixed_nomask[grid](
            q,
            k_fwd,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_fwd.stride(0),
            k_fwd.stride(1),
            k_stride_kn,
            k_stride_kd,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            N,
            D,
            block_m,
            block_n,
            block_d,
            scale=scale,
            USE_TF32=False,
            ACC_IN_FP16=fp16_fast_acc,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        kernel_elsa_attention_fwd_fixed[grid](
            q,
            k_fwd,
            v,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_fwd.stride(0),
            k_fwd.stride(1),
            k_stride_kn,
            k_stride_kd,
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            out.stride(3),
            B,
            H,
            N,
            D,
            block_m,
            block_n,
            block_d,
            scale=scale,
            IS_CAUSAL=is_causal,
            USE_TF32=False,
            ACC_IN_FP16=fp16_fast_acc,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out


def elsa_triton_new(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
    precision: str = "auto",
) -> torch.Tensor:
    """Precision-aware wrapper for ELSA Triton kernels.

    precision: auto|fp32|tf32|fp16|bf16
    """
    precision = precision.lower()
    orig_dtype = q.dtype
    if precision == "auto":
        target_dtype = orig_dtype
        tf32_override = None
    elif precision == "fp32":
        target_dtype = torch.float32
        tf32_override = False
    elif precision == "tf32":
        target_dtype = torch.float32
        tf32_override = True
    elif precision == "fp16":
        target_dtype = torch.float16
        tf32_override = None
    elif precision == "bf16":
        target_dtype = torch.bfloat16
        tf32_override = None
    else:
        raise ValueError(f"Unsupported precision '{precision}'.")

    if target_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError("ELSA_triton requires fp16/bf16/fp32 inputs.")

    q_t = q.to(target_dtype)
    k_t = k.to(target_dtype)
    v_t = v.to(target_dtype)
    attn_mask = bias
    scale = 1.0 / math.sqrt(q_t.shape[-1])

    out_dtype = orig_dtype if precision == "auto" else target_dtype

    with _tf32_context(tf32_override):
        if attn_mask is not None or is_causal:
            out = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=attn_mask,
                is_causal=is_causal,
                dropout_p=0.0,
            )
        else:
            if target_dtype == torch.float32:
                out = elsa_triton_mem_fp32(q_t, k_t, v_t, is_causal=is_causal, bias=None)
            else:
                out = ELSA_triton.apply(q_t, k_t, v_t, scale, None, is_causal)
    return out.to(out_dtype)


def elsa_triton_new_fp16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="fp16")


def elsa_triton_new_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="bf16")


def elsa_triton_new_tf32(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool = False,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return elsa_triton_new(q, k, v, is_causal=is_causal, bias=bias, precision="tf32")
