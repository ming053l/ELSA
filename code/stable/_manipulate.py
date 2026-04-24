"""Minimal stub for timm's internal _manipulate module."""
try:
    from timm.models._manipulate import checkpoint
except ImportError:
    from torch.utils.checkpoint import checkpoint  # standard PyTorch equivalent

__all__ = ["checkpoint"]
