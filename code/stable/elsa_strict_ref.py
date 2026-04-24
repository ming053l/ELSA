from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch


def default_strict_ref_block_n(seq_len: int, *, training: bool) -> int:
    if training:
        if seq_len <= 1024:
            return 256
        return 512
    if seq_len <= 1024:
        return 512
    if seq_len <= 4096:
        return int(seq_len)
    return 4096


@dataclass(frozen=True)
class ElsaStrictState:
    m: torch.Tensor
    s: torch.Tensor
    w: torch.Tensor


def merge_states(lhs: ElsaStrictState, rhs: ElsaStrictState) -> ElsaStrictState:
    """Associative merge for the ELSA online-softmax state triple.

    Each state represents an unnormalized summary over a disjoint key range:
      u = (m, S, W)
    where:
      - m: running max over logits
      - S: sum(exp(logits - m))
      - W: sum(exp(logits - m) * V)

    The merge is exact and associative:
      u_ab = u_a ⊕ u_b
    """
    m = torch.maximum(lhs.m, rhs.m)
    exp_l = torch.exp(lhs.m - m)
    exp_r = torch.exp(rhs.m - m)
    s = lhs.s * exp_l + rhs.s * exp_r
    w = lhs.w * exp_l.unsqueeze(-1) + rhs.w * exp_r.unsqueeze(-1)
    return ElsaStrictState(m=m, s=s, w=w)


def _masked_scores_block(
    q: torch.Tensor,
    k_block: torch.Tensor,
    *,
    scale: float,
    q_offsets: torch.Tensor,
    k_offsets: torch.Tensor,
    is_causal: bool,
    attn_bias_block: torch.Tensor | None = None,
    attn_bias_extra_block: torch.Tensor | None = None,
) -> torch.Tensor:
    scores = torch.matmul(q, k_block.transpose(-1, -2)) * scale
    if attn_bias_block is not None:
        scores = scores + attn_bias_block
    if attn_bias_extra_block is not None:
        scores = scores + attn_bias_extra_block
    if is_causal:
        causal = q_offsets[..., :, None] >= k_offsets[..., None, :]
        neg_inf = torch.full_like(scores, float("-inf"))
        scores = torch.where(causal, scores, neg_inf)
    return scores


def _slice_attn_bias(
    attn_bias: torch.Tensor | None,
    *,
    q_start: int | None = None,
    q_end: int | None = None,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if attn_bias is None:
        return None
    if attn_bias.ndim == 2:
        if q_start is None:
            raise ValueError("Compact label attention bias requires q_start/q_end.")
        labels_q = attn_bias[:, q_start:q_end]
        labels_k = attn_bias[:, start:end]
        return torch.where(
            labels_q[:, None, :, None] == labels_k[:, None, None, :],
            torch.zeros((), device=attn_bias.device, dtype=torch.float32),
            torch.full((), -100.0, device=attn_bias.device, dtype=torch.float32),
        )
    if attn_bias.ndim not in (3, 4):
        raise ValueError(
            f"Expected additive attention bias with shape [H,N,N] or [B,H,N,N], got {tuple(attn_bias.shape)}."
        )
    if attn_bias.ndim == 3:
        if (
            q_start is not None
            and attn_bias.shape[1] == attn_bias.shape[2]
            and attn_bias.shape[1] > 1
            and attn_bias.shape[1] % 2 == 1
        ):
            win_w = (int(attn_bias.shape[2]) + 1) // 2
            q_offsets = torch.arange(q_start, q_end, device=attn_bias.device)
            k_offsets = torch.arange(start, end, device=attn_bias.device)
            max_offset = int(torch.maximum(q_offsets[-1], k_offsets[-1]).item()) if q_offsets.numel() and k_offsets.numel() else -1
            if win_w * win_w >= max_offset + 1:
                q_h = q_offsets // win_w
                q_w = q_offsets - q_h * win_w
                k_h = k_offsets // win_w
                k_w = k_offsets - k_h * win_w
                rel_h = q_h[:, None] - k_h[None, :] + (win_w - 1)
                rel_w = q_w[:, None] - k_w[None, :] + (win_w - 1)
                return attn_bias[:, rel_h, rel_w]
        if q_start is None:
            return attn_bias[:, :, start:end]
        return attn_bias[:, q_start:q_end, start:end]
    if q_start is None:
        return attn_bias[:, :, :, start:end]
    return attn_bias[:, :, q_start:q_end, start:end]


def build_block_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_n: int = 128,
    group_blocks: int = 1,
    scale: float | None = None,
    is_causal: bool = False,
    attn_bias: torch.Tensor | None = None,
) -> List[ElsaStrictState]:
    """Build monoid summaries over disjoint K/V blocks.

    Shapes:
      q, k, v: [B, H, N, D]
    Returns:
      list of block-local states, each with shapes:
      - m: [B, H, N]
      - s: [B, H, N]
      - w: [B, H, N, Dv]
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("Expected q, k, v with shape [B, H, N, D].")
    if q.shape[:3] != k.shape[:3] or k.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on [B, H, N].")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    bsz, heads, seq_len, _ = q.shape
    dv = v.shape[-1]
    device = q.device

    q_offsets = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len)
    group_blocks = max(1, int(group_blocks))
    summaries: List[ElsaStrictState] = []
    group_span = block_n * group_blocks
    for group_start in range(0, seq_len, group_span):
        group_end = min(group_start + group_span, seq_len)
        k_group = k[:, :, group_start:group_end, :]
        v_group = v[:, :, group_start:group_end, :]
        k_offsets_group = torch.arange(group_start, group_end, device=device, dtype=torch.long).view(1, 1, group_end - group_start)
        scores_group = _masked_scores_block(
            q,
            k_group,
            scale=scale,
            q_offsets=q_offsets,
            k_offsets=k_offsets_group,
            is_causal=is_causal,
            attn_bias_block=_slice_attn_bias(attn_bias, start=group_start, end=group_end),
        )

        local_width = group_end - group_start
        for local_start in range(0, local_width, block_n):
            local_end = min(local_start + block_n, local_width)
            scores = scores_group[..., local_start:local_end]
            v_block = v_group[:, :, local_start:local_end, :]
            m = scores.max(dim=-1).values
            probs = torch.exp(scores - m.unsqueeze(-1))
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            s = probs.sum(dim=-1)
            w = torch.matmul(probs, v_block)
            summaries.append(ElsaStrictState(m=m, s=s, w=w))
    return summaries


def build_block_summaries_stacked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_n: int = 128,
    group_blocks: int = 1,
    scale: float | None = None,
    is_causal: bool = False,
    attn_bias: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build stacked monoid summaries directly, avoiding Python list+stack overhead."""
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("Expected q, k, v with shape [B, H, N, D].")
    if q.shape[:3] != k.shape[:3] or k.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on [B, H, N].")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    bsz, heads, seq_len, _ = q.shape
    dv = v.shape[-1]
    device = q.device
    q_offsets = torch.arange(seq_len, device=device, dtype=torch.long).view(1, 1, seq_len)
    group_blocks = max(1, int(group_blocks))
    num_blocks = (seq_len + block_n - 1) // block_n
    stack_m = torch.empty((num_blocks, bsz, heads, seq_len), device=device, dtype=torch.float32)
    stack_s = torch.empty((num_blocks, bsz, heads, seq_len), device=device, dtype=torch.float32)
    stack_w = torch.empty((num_blocks, bsz, heads, seq_len, dv), device=device, dtype=torch.float32)

    group_span = block_n * group_blocks
    block_idx = 0
    for group_start in range(0, seq_len, group_span):
        group_end = min(group_start + group_span, seq_len)
        k_group = k[:, :, group_start:group_end, :]
        v_group = v[:, :, group_start:group_end, :]
        k_offsets_group = torch.arange(group_start, group_end, device=device, dtype=torch.long).view(1, 1, group_end - group_start)
        scores_group = _masked_scores_block(
            q,
            k_group,
            scale=scale,
            q_offsets=q_offsets,
            k_offsets=k_offsets_group,
            is_causal=is_causal,
            attn_bias_block=_slice_attn_bias(attn_bias, start=group_start, end=group_end),
        )

        local_width = group_end - group_start
        for local_start in range(0, local_width, block_n):
            local_end = min(local_start + block_n, local_width)
            scores = scores_group[..., local_start:local_end]
            v_block = v_group[:, :, local_start:local_end, :]
            m = scores.max(dim=-1).values
            probs = torch.exp(scores - m.unsqueeze(-1))
            probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            stack_m[block_idx] = m
            stack_s[block_idx] = probs.sum(dim=-1)
            stack_w[block_idx] = torch.matmul(probs, v_block)
            block_idx += 1
    return stack_m, stack_s, stack_w


def hillis_steele_scan(states: List[ElsaStrictState]) -> List[ElsaStrictState]:
    """Inclusive Hillis-Steele scan over block summaries."""
    if not states:
        return []
    out = list(states)
    step = 1
    while step < len(out):
        prev = out
        nxt = list(prev)
        for idx in range(step, len(prev)):
            nxt[idx] = merge_states(prev[idx - step], prev[idx])
        out = nxt
        step <<= 1
    return out


def hillis_steele_scan_stacked(
    m: torch.Tensor,
    s: torch.Tensor,
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inclusive Hillis-Steele scan on stacked block summaries.

    Shapes:
      m: [T, B, H, N]
      s: [T, B, H, N]
      w: [T, B, H, N, D]
    """
    if m.ndim != 4 or s.ndim != 4 or w.ndim != 5:
        raise ValueError("Expected stacked state tensors with shapes [T,B,H,N] / [T,B,H,N,D].")
    out_m = m.clone()
    out_s = s.clone()
    out_w = w.clone()
    step = 1
    total = out_m.shape[0]
    while step < total:
        lhs_m = out_m[:-step]
        rhs_m = out_m[step:]
        lhs_s = out_s[:-step]
        rhs_s = out_s[step:]
        lhs_w = out_w[:-step]
        rhs_w = out_w[step:]

        merged_m = torch.maximum(lhs_m, rhs_m)
        exp_l = torch.exp(lhs_m - merged_m)
        exp_r = torch.exp(rhs_m - merged_m)
        merged_s = lhs_s * exp_l + rhs_s * exp_r
        merged_w = lhs_w * exp_l.unsqueeze(-1) + rhs_w * exp_r.unsqueeze(-1)

        out_m[step:] = merged_m
        out_s[step:] = merged_s
        out_w[step:] = merged_w
        step <<= 1
    return out_m, out_s, out_w


def tree_reduce_stacked(
    m: torch.Tensor,
    s: torch.Tensor,
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced associative reduction on stacked block summaries.

    For non-causal full attention we only need the total reduction over the
    K/V block summaries, not the full prefix array. This keeps the strict ELSA
    monoid semantics while avoiding the extra work and memory traffic of a
    full inclusive scan.
    """
    if m.ndim != 4 or s.ndim != 4 or w.ndim != 5:
        raise ValueError("Expected stacked state tensors with shapes [T,B,H,N] / [T,B,H,N,D].")
    cur_m = m
    cur_s = s
    cur_w = w
    while cur_m.shape[0] > 1:
        pair_count = cur_m.shape[0] // 2
        if pair_count:
            lhs_m = cur_m[0 : 2 * pair_count : 2]
            rhs_m = cur_m[1 : 2 * pair_count : 2]
            lhs_s = cur_s[0 : 2 * pair_count : 2]
            rhs_s = cur_s[1 : 2 * pair_count : 2]
            lhs_w = cur_w[0 : 2 * pair_count : 2]
            rhs_w = cur_w[1 : 2 * pair_count : 2]

            merged_m = torch.maximum(lhs_m, rhs_m)
            exp_l = torch.exp(lhs_m - merged_m)
            exp_r = torch.exp(rhs_m - merged_m)
            merged_s = lhs_s * exp_l + rhs_s * exp_r
            merged_w = lhs_w * exp_l.unsqueeze(-1) + rhs_w * exp_r.unsqueeze(-1)

            if cur_m.shape[0] & 1:
                cur_m = torch.cat([merged_m, cur_m[-1:]], dim=0)
                cur_s = torch.cat([merged_s, cur_s[-1:]], dim=0)
                cur_w = torch.cat([merged_w, cur_w[-1:]], dim=0)
            else:
                cur_m = merged_m
                cur_s = merged_s
                cur_w = merged_w
        else:
            break
    return cur_m[0], cur_s[0], cur_w[0]


def reduce_block_summaries_streaming(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_n: int = 128,
    group_blocks: int = 1,
    scale: float | None = None,
    is_causal: bool = False,
    attn_bias: torch.Tensor | None = None,
    attn_bias_extra: torch.Tensor | None = None,
    q_chunk_size: int = 256,
) -> torch.Tensor:
    """Reference reduction that streams over query chunks.

    This keeps the strict ELSA monoid semantics while avoiding the large
    stacked-state tensors used by tree_reduce_stacked on long masked windows.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError("Expected q, k, v with shape [B, H, N, D].")
    if q.shape[:3] != k.shape[:3] or k.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on [B, H, N].")
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])

    _, _, seq_len, _ = q.shape
    device = q.device
    out = torch.empty_like(v)
    q_chunk_size = max(1, int(q_chunk_size))
    group_blocks = max(1, int(group_blocks))
    if attn_bias is not None and seq_len >= 8192:
        # Huge masked windows are dominated by temporary score tiles; keep the
        # K/V group narrow when the exact reference is only used as a safety
        # fallback.
        group_blocks = 1
    group_span = block_n * group_blocks

    for q_start in range(0, seq_len, q_chunk_size):
        q_end = min(q_start + q_chunk_size, seq_len)
        q_chunk = q[:, :, q_start:q_end, :]
        q_offsets = torch.arange(q_start, q_end, device=device, dtype=torch.long).view(1, 1, q_end - q_start)
        total_state: ElsaStrictState | None = None

        for group_start in range(0, seq_len, group_span):
            group_end = min(group_start + group_span, seq_len)
            k_group = k[:, :, group_start:group_end, :]
            v_group = v[:, :, group_start:group_end, :]
            k_offsets_group = torch.arange(
                group_start,
                group_end,
                device=device,
                dtype=torch.long,
            ).view(1, 1, group_end - group_start)
            scores_group = _masked_scores_block(
                q_chunk,
                k_group,
                scale=scale,
                q_offsets=q_offsets,
                k_offsets=k_offsets_group,
                is_causal=is_causal,
                attn_bias_block=_slice_attn_bias(
                    attn_bias,
                    q_start=q_start,
                    q_end=q_end,
                    start=group_start,
                    end=group_end,
                ),
                attn_bias_extra_block=_slice_attn_bias(
                    attn_bias_extra,
                    q_start=q_start,
                    q_end=q_end,
                    start=group_start,
                    end=group_end,
                ),
            )

            local_width = group_end - group_start
            for local_start in range(0, local_width, block_n):
                local_end = min(local_start + block_n, local_width)
                scores = scores_group[..., local_start:local_end]
                v_block = v_group[:, :, local_start:local_end, :]
                m = scores.max(dim=-1).values
                probs = torch.exp(scores - m.unsqueeze(-1))
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                state = ElsaStrictState(
                    m=m,
                    s=probs.sum(dim=-1),
                    w=torch.matmul(probs, v_block),
                )
                total_state = state if total_state is None else merge_states(total_state, state)

            del scores_group

        if total_state is None:
            out[:, :, q_start:q_end, :] = 0
        else:
            out[:, :, q_start:q_end, :] = total_state.w / total_state.s.clamp_min(1e-32).unsqueeze(-1)

    return out


def elsa_strict_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_n: int = 128,
    group_blocks: int | None = None,
    scale: float | None = None,
    is_causal: bool = False,
    attn_bias: torch.Tensor | None = None,
    attn_bias_extra: torch.Tensor | None = None,
    return_prefix_states: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, List[ElsaStrictState]]:
    """Strict ELSA reference via block summaries + associative scan.

    This is a correctness/reference implementation for the strong ELSA claim:
    it explicitly constructs monoid summaries over disjoint blocks and resolves
    them through an associative prefix scan.
    """
    if q.shape[-2] == 0:
        out = torch.empty_like(v)
        return (out, []) if return_prefix_states else out
    if group_blocks is None:
        override = os.environ.get("ELSA_STRICT_REF_GROUP_BLOCKS")
        if override is not None:
            try:
                group_blocks = int(override)
            except ValueError:
                group_blocks = 4
        else:
            seq_len = q.shape[-2]
            bh = q.shape[0] * q.shape[1]
            if block_n >= seq_len:
                group_blocks = 1
            elif seq_len <= 1024:
                group_blocks = 1 if bh <= 8 else 2
            elif seq_len <= 4096:
                group_blocks = 8 if bh <= 8 else 4
            else:
                group_blocks = 4
        group_blocks = min(max(int(group_blocks), 1), 8)
    stream_min_n = int(os.environ.get("ELSA_STRICT_REF_STREAM_MIN_N", "8192"))
    stream_q_chunk = int(os.environ.get("ELSA_STRICT_REF_STREAM_Q_CHUNK", "128"))
    stream_block_n_cap = int(os.environ.get("ELSA_STRICT_REF_STREAM_BLOCK_N", "512"))
    seq_len = q.shape[-2]
    compact_attn_bias = (
        (attn_bias is not None and attn_bias.ndim == 2)
        or (attn_bias_extra is not None and attn_bias_extra.ndim == 2)
        or (
            attn_bias is not None
            and attn_bias.ndim == 3
            and attn_bias.shape[1] == attn_bias.shape[2]
            and attn_bias.shape[1] != seq_len
        )
        or (
            attn_bias_extra is not None
            and attn_bias_extra.ndim == 3
            and attn_bias_extra.shape[1] == attn_bias_extra.shape[2]
            and attn_bias_extra.shape[1] != seq_len
        )
    )
    if attn_bias is not None and seq_len >= 16384:
        stream_q_chunk = min(stream_q_chunk, 32)
        stream_block_n_cap = min(max(stream_block_n_cap, 1024), block_n)
    elif attn_bias is not None and seq_len >= 8192:
        stream_q_chunk = min(stream_q_chunk, 128)
        stream_block_n_cap = min(stream_block_n_cap, 512)
    if (not is_causal) and (not return_prefix_states) and (seq_len >= stream_min_n or compact_attn_bias):
        return reduce_block_summaries_streaming(
            q,
            k,
            v,
            block_n=min(block_n, stream_block_n_cap),
            group_blocks=group_blocks,
            scale=scale,
            is_causal=is_causal,
            attn_bias=attn_bias,
            attn_bias_extra=attn_bias_extra,
            q_chunk_size=stream_q_chunk,
        )
    if attn_bias_extra is not None:
        if attn_bias is None:
            attn_bias = attn_bias_extra
        else:
            attn_bias = attn_bias + attn_bias_extra
    stack_m, stack_s, stack_w = build_block_summaries_stacked(
        q,
        k,
        v,
        block_n=block_n,
        group_blocks=group_blocks,
        scale=scale,
        is_causal=is_causal,
        attn_bias=attn_bias,
    )
    if not is_causal and not return_prefix_states:
        total_m, total_s, total_w = tree_reduce_stacked(stack_m, stack_s, stack_w)
        out = total_w / total_s.clamp_min(1e-32).unsqueeze(-1)
        return out
    pref_m, pref_s, pref_w = hillis_steele_scan_stacked(stack_m, stack_s, stack_w)
    total = ElsaStrictState(m=pref_m[-1], s=pref_s[-1], w=pref_w[-1])
    out = total.w / total.s.clamp_min(1e-32).unsqueeze(-1)
    if not return_prefix_states:
        return out
    prefixes = [ElsaStrictState(m=pref_m[i], s=pref_s[i], w=pref_w[i]) for i in range(pref_m.shape[0])]
    return out, prefixes


__all__ = [
    "ElsaStrictState",
    "default_strict_ref_block_n",
    "merge_states",
    "build_block_summaries",
    "build_block_summaries_stacked",
    "reduce_block_summaries_streaming",
    "hillis_steele_scan",
    "hillis_steele_scan_stacked",
    "tree_reduce_stacked",
    "elsa_strict_reference",
]
