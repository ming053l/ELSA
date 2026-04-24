from __future__ import annotations

# Re-export the original Triton ELSA (formerly CAN) baseline so experiments can
# compare against the upstream snapshot without depending on internal paths.
from elsa_cuda.versions.original_20251021_195305.sic_triton import *  # noqa: F401,F403
