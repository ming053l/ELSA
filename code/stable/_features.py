"""Minimal stub for timm's internal _features module."""
try:
    from timm.models._features import feature_take_indices
except ImportError:
    def feature_take_indices(num_features, indices):
        """Passthrough stub when timm is not available."""
        if indices is None:
            return list(range(num_features)), num_features
        indices = [i % num_features for i in indices]
        return indices, max(indices) + 1 if indices else 0

__all__ = ["feature_take_indices"]
