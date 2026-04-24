import math
import os
import warnings
from typing import List, Optional

import torch
import torch.nn.functional as F

from . import elsa_swin as base
from .elsa_triton_swin_fused import elsa_swinv2_triton_fused, elsa_swinv2_triton_proj


class WindowAttentionFused(base.WindowAttention):
    def _cache_enabled(self) -> bool:
        return (
            (not self.training)
            and (not torch.is_grad_enabled())
            and bool(int(os.environ.get("ELSA_SWIN_FUSED_CACHE", "1")))
        )

    def _bias_dtype(self, q_dtype: torch.dtype) -> torch.dtype:
        override = os.environ.get("ELSA_SWIN_FUSED_BIAS_DTYPE", "auto").lower()
        if override in ("fp16", "float16"):
            return torch.float16
        if override in ("bf16", "bfloat16"):
            return torch.bfloat16
        if override in ("fp32", "float32"):
            return torch.float32
        if q_dtype in (torch.float16, torch.bfloat16):
            return q_dtype
        return torch.float32

    def _run_backend(
        self,
        backend: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        logit_scale: torch.Tensor,
        relative_position_bias: Optional[torch.Tensor],
        relative_position_bias_table: Optional[torch.Tensor],
        relative_position_index: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
        normalize_qk: bool = False,
        use_bias_table: bool = False,
        out_layout: str = "HND",
    ) -> torch.Tensor:
        if backend == "triton":
            if not (base._ELSATRITON_AVAILABLE and elsa_swinv2_triton_fused is not None):
                raise RuntimeError("ELSA Triton kernels unavailable.")
            tf32_mode = os.environ.get("ELSA_SWIN_TF32_MODE", "half").lower()
            use_half_qk = (
                q.dtype == torch.float32
                and torch.backends.cuda.matmul.allow_tf32
                and not self.training
                and bool(int(os.environ.get("ELSA_SWIN_FP32_TURBO", "1")))
            )
            if q.dtype == torch.float32 and torch.backends.cuda.matmul.allow_tf32:
                if tf32_mode in ("full", "tf32", "native"):
                    use_half_qk = False
            return elsa_swinv2_triton_fused(
                q,
                k,
                v,
                logit_scale=logit_scale,
                relative_position_bias=relative_position_bias,
                relative_position_bias_table=relative_position_bias_table,
                relative_position_index=relative_position_index,
                mask=mask,
                use_half_qk=use_half_qk,
                normalize_qk=normalize_qk,
                use_bias_table=use_bias_table,
                out_layout=out_layout,
            )
        return super()._run_backend(backend, q, k, v, logit_scale, relative_position_bias, mask)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias))

        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        fused_qknorm_env = os.environ.get("ELSA_SWIN_FUSED_QKNORM")
        use_fused_qknorm = bool(int(fused_qknorm_env)) if fused_qknorm_env is not None else True
        fused_bias_env = os.environ.get("ELSA_SWIN_FUSED_RELBIAS")
        use_fused_bias = bool(int(fused_bias_env)) if fused_bias_env is not None else True
        q_norm = None
        k_norm = None

        cache_ok = self._cache_enabled()
        rel_bias_key = (q.device, q.dtype)
        relative_position_bias = None
        relative_position_bias_table = None
        relative_position_index = None
        if cache_ok:
            relative_position_bias = self._rel_bias_cache.get(rel_bias_key)
        if use_fused_bias:
            bias_table_key = (q.device, q.dtype, "table")
            if cache_ok:
                relative_position_bias_table = self._rel_bias_cache.get(bias_table_key)
            if relative_position_bias_table is None:
                relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
                relative_position_bias_table = relative_position_bias_table.transpose(0, 1).contiguous()
                relative_position_bias_table = 16 * torch.sigmoid(relative_position_bias_table)
                bias_dtype = self._bias_dtype(q.dtype)
                if relative_position_bias_table.dtype != bias_dtype:
                    relative_position_bias_table = relative_position_bias_table.to(bias_dtype)
                if cache_ok:
                    self._rel_bias_cache[bias_table_key] = relative_position_bias_table
            relative_position_index = getattr(self, "_rel_bias_index_cache", None)
            if relative_position_index is None or relative_position_index.device != q.device:
                relative_position_index = self.relative_position_index.to(device=q.device, dtype=torch.int32)
                self._rel_bias_index_cache = relative_position_index
        if relative_position_bias is None and not use_fused_bias:
            bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
            bias_table = 16 * torch.sigmoid(bias_table)
            bias_dtype = self._bias_dtype(q.dtype)
            if bias_table.dtype != bias_dtype:
                bias_table = bias_table.to(bias_dtype)
            relative_position_bias = bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            if cache_ok:
                self._rel_bias_cache[rel_bias_key] = relative_position_bias

        logit_scale = None
        if cache_ok:
            logit_scale = self._logit_scale_cache.get(rel_bias_key)
        if logit_scale is None:
            logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01))
            if cache_ok:
                self._logit_scale_cache[rel_bias_key] = logit_scale

        mask_full = None
        mask_compact = None
        num_win = None
        if mask is not None:
            num_win = mask.shape[0]
            mask_cache_ok = cache_ok or bool(int(os.environ.get("ELSA_SWIN_FUSED_MASK_CACHE_TRAIN", "1")))
            use_compact = (
                self.enable_triton
                and self.backend_preference.startswith("triton")
                and bool(int(os.environ.get("ELSA_SWIN_FUSED_COMPACT_MASK", "1")))
            )
            if use_compact:
                mask_key = (mask.data_ptr(), q.device, q.dtype)
                if mask_cache_ok:
                    mask_compact = self._mask_cache.get(mask_key)
                if mask_compact is None:
                    mask_base = mask.view(num_win, N, N)
                    if mask_base.dtype != q.dtype:
                        mask_base = mask_base.to(q.dtype)
                    mask_compact = mask_base
                    if mask_cache_ok:
                        self._mask_cache[mask_key] = mask_compact

        attn_output = None
        errors: List[RuntimeError] = []
        used_backend = None
        used_out_layout = "HND"
        use_out_nh = bool(int(os.environ.get("ELSA_SWIN_FUSED_OUT_NH", "1")))
        for backend in self._candidate_backends(q):
            try:
                normalize_qk = backend.startswith("triton") and use_fused_qknorm
                q_in = q
                k_in = k
                if not normalize_qk:
                    if q_norm is None:
                        q_norm = F.normalize(q, dim=-1)
                        k_norm = F.normalize(k, dim=-1)
                    q_in = q_norm
                    k_in = k_norm
                use_bias_table = backend.startswith("triton") and use_fused_bias
                rel_bias = None if use_bias_table else relative_position_bias
                rel_bias_table = relative_position_bias_table if use_bias_table else None
                rel_bias_index = relative_position_index if use_bias_table else None
                if not backend.startswith("triton") and rel_bias is None:
                    if relative_position_bias_table is None:
                        bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
                        bias_table = 16 * torch.sigmoid(bias_table)
                        bias_dtype = self._bias_dtype(q.dtype)
                        if bias_table.dtype != bias_dtype:
                            bias_table = bias_table.to(bias_dtype)
                    else:
                        bias_table = relative_position_bias_table.transpose(0, 1).contiguous()
                    rel_bias = bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1],
                        self.window_size[0] * self.window_size[1],
                        -1,
                    )
                    rel_bias = rel_bias.permute(2, 0, 1).contiguous()
                out_layout = "NH" if backend.startswith("triton") and use_out_nh else "HND"
                mask_in = mask
                if mask is not None:
                    if backend.startswith("triton") and mask_compact is not None:
                        mask_in = mask_compact
                    else:
                        if mask_full is None:
                            mask_full = mask.view(1, num_win, 1, N, N)
                            mask_full = mask_full.expand(B_ // num_win, num_win, 1, N, N)
                            mask_full = mask_full.reshape(B_, 1, N, N)
                        mask_in = mask_full
                attn_output = self._run_backend(
                    backend,
                    q_in,
                    k_in,
                    v,
                    logit_scale,
                    rel_bias,
                    rel_bias_table,
                    rel_bias_index,
                    mask_in,
                    normalize_qk=normalize_qk,
                    use_bias_table=use_bias_table,
                    out_layout=out_layout,
                )
                used_backend = backend
                used_out_layout = out_layout
                break
            except RuntimeError as err:
                errors.append(err)
                self._warn_backend_failure(backend, err)

        if attn_output is None:
            raise RuntimeError("ELSA Swin attention failed.") from (errors[-1] if errors else None)

        attn_output = self.attn_drop(attn_output)
        # Proj strategy:
        # - off: always use nn.Linear
        # - triton: always use Triton proj
        # - auto: use Triton only where it empirically helps on Swin shapes
        proj_mode = os.environ.get("ELSA_SWIN_FUSED_PROJ", "off").lower()
        use_fused_proj = proj_mode in ("1", "true", "triton", "auto")
        if proj_mode in ("1", "true"):
            proj_mode = "triton"
        if (
            use_fused_proj
            and used_backend is not None
            and used_backend.startswith("triton")
            and used_out_layout == "NH"
        ):
            run_triton_proj = True
            if proj_mode == "auto":
                # microbench-derived heuristic:
                # triton proj tends to help only for fp32 and small channel (C<=128)
                # with large M=B*N windows. Otherwise cuBLAS linear is better.
                c = attn_output.shape[2] * attn_output.shape[3]
                m = attn_output.shape[0] * attn_output.shape[1]
                run_triton_proj = (
                    attn_output.dtype == torch.float32
                    and c <= 128
                    and m >= 8192
                )
            if not run_triton_proj:
                x = attn_output.reshape(B_, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)
                return x
            try:
                x = elsa_swinv2_triton_proj(attn_output, self.proj.weight, self.proj.bias)
            except RuntimeError as err:
                self._warn_backend_failure("triton_proj", err)
                x = attn_output.reshape(B_, N, C)
                x = self.proj(x)
        else:
            if used_backend is not None and used_backend.startswith("triton") and used_out_layout == "NH":
                x = attn_output.reshape(B_, N, C)
            else:
                x = attn_output.transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x)
        x = self.proj_drop(x)
        return x


_PATCHED = False


def patch_elsa_window_attention() -> None:
    global _PATCHED
    if _PATCHED:
        return
    base.WindowAttention = WindowAttentionFused
    _PATCHED = True
