"""
ELSA — Exact Linear-Scan Attention
===================================

Public API
----------
Kernel functions (no timm required):
    ELSA_triton          — Triton kernel, FP16/BF16, forward only
    ELSA_triton_fp32     — Triton kernel, FP32 inference
    ELSA_triton_fp32_train — Triton kernel, FP32 training (with backward)
    ELSA_pytorch         — Pure-PyTorch reference (supports autograd)

Model classes (requires timm):
    ElsaAttention        — Drop-in attention module (B, N, C) → (B, N, C)
    ElsaViT              — Full ViT model with ELSA attention

Backend helpers:
    set_default_elsa_backend(backend)
    get_default_elsa_backend() -> str
"""

# ---------------------------------------------------------------------------
# Kernel-level functions — always available (only torch + triton required)
# ---------------------------------------------------------------------------
from .elsa_triton import (
    ELSA_pytorch,
    ELSA_triton,
    ELSA_triton_fp32,
    ELSA_triton_fp32_train,
)

# ---------------------------------------------------------------------------
# Model-level classes — require timm to be installed
# ---------------------------------------------------------------------------
try:
    from .elsa import (
        ElsaAttention,
        ElsaViT,
        ElsaDistilled,
        set_default_elsa_backend,
        get_default_elsa_backend,
    )
    _MODEL_AVAILABLE = True
except Exception as _e:  # pragma: no cover
    _MODEL_AVAILABLE = False
    _MODEL_IMPORT_ERROR = _e

    class _Unavailable:  # type: ignore[no-redef]
        def __init__(self, *a, **kw):
            raise ImportError(
                "ElsaAttention / ElsaViT require timm. "
                "Install it with:  pip install timm>=0.9\n"
                f"Original error: {_e}"
            )

    ElsaAttention = _Unavailable  # type: ignore[assignment,misc]
    ElsaViT = _Unavailable  # type: ignore[assignment,misc]
    ElsaDistilled = _Unavailable  # type: ignore[assignment,misc]

    def set_default_elsa_backend(backend: str) -> None:  # type: ignore[misc]
        raise ImportError("timm not available") from _e

    def get_default_elsa_backend() -> str:  # type: ignore[misc]
        raise ImportError("timm not available") from _e

# ---------------------------------------------------------------------------
# Swin integration — optional
# ---------------------------------------------------------------------------
try:
    from .elsa_swin import ElsaSwinTransformerV2
    _SWIN_AVAILABLE = True
except Exception:  # pragma: no cover
    _SWIN_AVAILABLE = False

__version__ = "1.0.0"

__all__ = [
    # kernels
    "ELSA_triton",
    "ELSA_triton_fp32",
    "ELSA_triton_fp32_train",
    "ELSA_pytorch",
    # model classes
    "ElsaAttention",
    "ElsaViT",
    "ElsaDistilled",
    # helpers
    "set_default_elsa_backend",
    "get_default_elsa_backend",
]
