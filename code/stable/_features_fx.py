"""Minimal stub for timm's internal _features_fx module."""
try:
    from timm.models._features_fx import register_notrace_function
except ImportError:
    def register_notrace_function(fn):
        """No-op decorator stub when timm is not available."""
        return fn

__all__ = ["register_notrace_function"]
