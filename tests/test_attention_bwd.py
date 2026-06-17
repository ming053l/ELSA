from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from elsa_twopass_clean.attention_bwd import twopass_attention_train


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("is_causal", [False, True])
def test_twopass_attention_train_backward_matches_sdpa(dtype, is_causal):
    torch.manual_seed(11)
    shape = (1, 2, 64, 32)
    q = (torch.randn(shape, device="cuda", dtype=dtype) * 0.5).requires_grad_(True)
    k = (torch.randn(shape, device="cuda", dtype=dtype) * 0.5).requires_grad_(True)
    v = torch.randn(shape, device="cuda", dtype=dtype).requires_grad_(True)
    dout = torch.randn(shape, device="cuda", dtype=dtype) * 0.25

    q_ref = q.float().detach().requires_grad_(True)
    k_ref = k.float().detach().requires_grad_(True)
    v_ref = v.float().detach().requires_grad_(True)
    ref = F.scaled_dot_product_attention(
        q_ref,
        k_ref,
        v_ref,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=1.0 / math.sqrt(shape[-1]),
    )
    ref.backward(dout.float())

    out = twopass_attention_train(q, k, v, is_causal=is_causal)
    out.backward(dout)

    atol = 2.5e-3 if dtype is torch.float32 else 8e-2
    rtol = 2.5e-3 if dtype is torch.float32 else 8e-2
    torch.testing.assert_close(out.float(), ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad.float(), q_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad.float(), k_ref.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad.float(), v_ref.grad, atol=atol, rtol=rtol)

