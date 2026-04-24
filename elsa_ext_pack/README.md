ELSA Ultra (A100, CUDA 12.6)
============================
This is a drop-in PyTorch CUDA extension exposing a single function:
    elsa_ext_ultra.forward(q, k, v, causal=False, d_scale=0.0) -> out

Shapes: [B, H, N, D], contiguous. Dtypes: float32, float16.
Default temperature scale = 1/sqrt(D) if d_scale == 0.

Build (inside this folder):
    export TORCH_CUDA_ARCH_LIST="8.0"
    python setup.py bdist_wheel
    python -m pip install -v --no-build-isolation dist/*.whl

Example:
    import torch, elsa_ext_ultra
    B,H,N,D = 1, 12, 4096, 64
    q = torch.randn(B,H,N,D, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    out = elsa_ext_ultra.forward(q,k,v,False,0.0)