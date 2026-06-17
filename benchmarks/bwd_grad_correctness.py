"""Verify ELSA backward gradients (dq,dk,dv) on ViT ODD seqs vs PyTorch autograd reference.
Critical: the backward kernels process partial blocks too — silent wrong gradients are worse
than a crash. If max grad error is tiny, backward is correct for non-divisible seq."""
import sys; sys.path.insert(0, "src")
import torch
import torch.nn.functional as Fnn
from elsa_twopass_clean.attention_bwd import twopass_attention_train

dev = "cuda"
print(f"{'B,H,N,D':16} {'dt':5} {'dq_err':>10} {'dk_err':>10} {'dv_err':>10} {'verdict':8}")
for (B, H, N, D, dt) in [(8, 3, 1025, 64, torch.float32),
                          (8, 3, 2305, 64, torch.float32),
                          (8, 3, 4097, 64, torch.float32),
                          (8, 3, 1025, 64, torch.float16),
                          (8, 3, 2305, 64, torch.float16),
                          (1, 8, 4096, 64, torch.float32)]:  # divisible control
    torch.manual_seed(0)
    q = torch.randn(B, H, N, D, device=dev, dtype=dt, requires_grad=True)
    k = torch.randn(B, H, N, D, device=dev, dtype=dt, requires_grad=True)
    v = torch.randn(B, H, N, D, device=dev, dtype=dt, requires_grad=True)
    dout = torch.randn(B, H, N, D, device=dev, dtype=dt) * 0.1
    # reference (math, exact)
    qr, kr, vr = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    ref = Fnn.scaled_dot_product_attention(qr, kr, vr, scale=1.0 / (D ** 0.5))
    ref.backward(dout)
    # elsa
    out = twopass_attention_train(q, k, v)
    out.backward(dout)
    tol = 5e-2 if dt == torch.float16 else 5e-4

    def rel(a, b):
        return (a - b).abs().max().item()
    dqe, dke, dve = rel(q.grad, qr.grad), rel(k.grad, kr.grad), rel(v.grad, vr.grad)
    ok = "OK" if max(dqe, dke, dve) < tol else "WRONG"
    print(f"{str((B,H,N,D)):16} {str(dt)[6:]:5} {dqe:10.2e} {dke:10.2e} {dve:10.2e} {ok:8}", flush=True)
    del q, k, v, qr, kr, vr; torch.cuda.empty_cache()
print("GRADCHECK_DONE")
