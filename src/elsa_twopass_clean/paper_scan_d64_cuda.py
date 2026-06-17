from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _paper_scan_d64_extension():
    root = Path(__file__).resolve().parent
    cuda_home = Path("/usr/local/cuda-12.8")
    if cuda_home.exists() and "CUDA_HOME" not in os.environ:
        os.environ["CUDA_HOME"] = str(cuda_home)
    venv_bin = Path(torch.__file__).resolve().parents[4] / "bin"
    if (venv_bin / "ninja").exists():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
    build_dir = root.parent.parent / "build" / "paper_scan_d64_ext"
    build_dir.mkdir(parents=True, exist_ok=True)
    return load(
        name="elsa_paper_scan_d64_ext",
        sources=[
            str(root / "cuda" / "paper_scan_d64.cpp"),
            str(root / "cuda" / "paper_scan_d64_kernel.cu"),
        ],
        build_directory=str(build_dir),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        with_cuda=True,
        verbose=False,
    )


def paper_scan_d64_final_reduce(
    m: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
    out: torch.Tensor,
    row_m: torch.Tensor,
    row_z: torch.Tensor,
    *,
    q_start: int,
    seq_len: int,
    k_blocks: int,
    q_blocks: int,
    bh: int,
    block_m: int,
    store_state: bool,
) -> None:
    _paper_scan_d64_extension().final_reduce(
        m,
        z,
        s,
        out,
        row_m,
        row_z,
        int(q_start),
        int(seq_len),
        int(k_blocks),
        int(q_blocks),
        int(bh),
        int(block_m),
        bool(store_state),
    )
