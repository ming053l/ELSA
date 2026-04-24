"""Entry module exposing ELSA kernels without requiring full timm import."""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

try:
    _module = importlib.import_module("timm.models.elsa_triton")
except Exception:
    # Fall back to loading the module directly to avoid heavy torchvision deps during benchmarking.
    _path = Path(__file__).resolve().parent / "timm" / "models" / "elsa_triton.py"
    spec = importlib.util.spec_from_file_location("timm.models.elsa_triton", _path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to locate elsa_triton at {_path}")
    _module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(_module)

CAN_triton = _module.ELSA_triton
CAN_triton_fp32 = _module.ELSA_triton_fp32
CAN_triton_mem = _module.ELSA_triton_mem
CAN_pytorch = _module.ELSA_pytorch
CAN_triton_new = getattr(_module, "elsa_triton_new", None)
CAN_triton_new_fp32_legacy = getattr(_module, "elsa_triton_new_fp32_legacy", None)
CAN_triton_new_fp32_fast = getattr(_module, "elsa_triton_new_fp32_fast", None)
CAN_triton_new_fp16 = getattr(_module, "elsa_triton_new_fp16", None)
CAN_triton_new_bf16 = getattr(_module, "elsa_triton_new_bf16", None)
CAN_triton_new_tf32 = getattr(_module, "elsa_triton_new_tf32", None)
