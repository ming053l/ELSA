"""Legacy shim that re-exports the new ELSA Triton attention kernels.

The fully featured implementation lives in `timm.models.elsa_triton`.  This file is
kept to avoid breaking long-running scripts that still import `sic_triton`.
"""

from warnings import warn

warn(
    "timm.models.sic_triton is deprecated. Please import timm.models.elsa_triton "
    "and update your code to the ELSA naming.",
    DeprecationWarning,
    stacklevel=2,
)

from .elsa_triton import *  # noqa: F401,F403
