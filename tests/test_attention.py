from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from elsa_twopass_clean import twopass_attention
from elsa_twopass_clean.full_model import make_model_pair


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sdpa_ref(q, k, v, *, bias=None, is_causal=False):
    return F.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=None if bias is None else bias.float(),
        is_causal=is_causal,
        dropout_p=0.0,
        scale=1.0 / math.sqrt(q.shape[-1]),
    ).to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1, 2, 64, 32), (1, 3, 196, 64), (2, 4, 257, 32)])
def test_twopass_matches_sdpa_no_bias(dtype, shape):
    torch.manual_seed(0)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    out = twopass_attention(q, k, v)
    ref = _sdpa_ref(q, k, v)

    atol = 2e-4 if dtype is torch.float32 else 3e-2
    rtol = 2e-4 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_twopass_matches_sdpa_bias(dtype):
    torch.manual_seed(1)
    shape = (2, 3, 128, 32)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)
    bias = torch.randn((1, 3, 128, 128), device="cuda", dtype=torch.float32) * 0.05

    out = twopass_attention(q, k, v, bias=bias)
    ref = _sdpa_ref(q, k, v, bias=bias)

    atol = 2e-4 if dtype is torch.float32 else 3e-2
    rtol = 2e-4 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_twopass_matches_sdpa_causal(dtype):
    torch.manual_seed(2)
    shape = (1, 2, 129, 32)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    out = twopass_attention(q, k, v, is_causal=True)
    ref = _sdpa_ref(q, k, v, is_causal=True)

    atol = 2e-4 if dtype is torch.float32 else 3e-2
    rtol = 2e-4 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_twopass_fp16_summary_variant_matches_sdpa():
    torch.manual_seed(3)
    shape = (1, 3, 196, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    out = twopass_attention(q, k, v, summary_dtype=torch.float16)
    ref = _sdpa_ref(q, k, v)

    torch.testing.assert_close(out, ref, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_twopass_matches_sdpa_multiple_q_chunks(dtype):
    torch.manual_seed(4)
    shape = (1, 2, 513, 64)
    q = torch.randn(shape, device="cuda", dtype=dtype)
    k = torch.randn(shape, device="cuda", dtype=dtype)
    v = torch.randn(shape, device="cuda", dtype=dtype)

    out = twopass_attention(q, k, v, q_chunk_size=128)
    ref = _sdpa_ref(q, k, v)

    atol = 2e-4 if dtype is torch.float32 else 3e-2
    rtol = 2e-4 if dtype is torch.float32 else 3e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def test_non_scan_algorithm_is_rejected():
    torch.manual_seed(5)
    shape = (1, 2, 128, 32)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="paper_scan"):
        twopass_attention(q, k, v, algorithm="not_available")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_full_model_pair_matches(dtype):
    torch.manual_seed(6)
    baseline, elsa = make_model_pair(dim=96, depth=2, heads=3)
    baseline = baseline.eval().to(device="cuda", dtype=dtype)
    elsa = elsa.eval().to(device="cuda", dtype=dtype)
    x = torch.randn((1, 128, 96), device="cuda", dtype=dtype)

    with torch.no_grad():
        ref = baseline(x)
        out = elsa(x)

    atol = 5e-3 if dtype is torch.float32 else 8e-2
    rtol = 5e-3 if dtype is torch.float32 else 8e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
