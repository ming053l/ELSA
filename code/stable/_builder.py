"""
Minimal stub for timm's internal _builder module.

Attempts to import the real implementation from an installed timm first.
Falls back to a stub that raises a clear error when
timm.create_model / build_model_with_cfg is called.

ElsaAttention can be used standalone without this function; it is only
needed when constructing ElsaViT via timm's model factory.
"""
try:
    from timm.models._builder import build_model_with_cfg
except ImportError:
    def build_model_with_cfg(*args, **kwargs):
        raise RuntimeError(
            "build_model_with_cfg requires timm to be installed.\n"
            "Install with:  pip install timm>=0.9\n\n"
            "Alternatively, use ElsaAttention directly without the timm "
            "model factory:\n\n"
            "    from elsa import ElsaAttention\n"
            "    attn = ElsaAttention(dim=768, num_heads=12)"
        )

__all__ = ["build_model_with_cfg"]
