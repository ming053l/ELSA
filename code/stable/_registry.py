"""
Minimal stub for timm's internal _registry module.

Attempts to import the real implementations from an installed timm first.
Falls back to no-op stubs so that ElsaAttention can be imported standalone
even when the files are not placed inside timm's model directory.

Note: model *registration* (timm.create_model support) is only available
when the real timm registry is present.
"""
try:
    # If this package is installed as part of timm (i.e. files are inside
    # timm/models/), the real implementations are available via the parent
    # package.
    from timm.models._registry import (
        generate_default_cfgs,
        register_model,
        register_model_deprecations,
    )
except ImportError:
    # Standalone stub — model registration is silently skipped.
    def register_model(fn=None, **kwargs):
        """No-op decorator: model will not be registered with timm.create_model."""
        if fn is None:
            return lambda f: f
        return fn

    def generate_default_cfgs(configs):
        """No-op: returns empty dict when timm registry is unavailable."""
        return {}

    def register_model_deprecations(model_name, deprecation_map):
        """No-op stub."""
        pass

__all__ = ["generate_default_cfgs", "register_model", "register_model_deprecations"]
