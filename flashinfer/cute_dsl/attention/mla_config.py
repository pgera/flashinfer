# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAConfig — configuration for Multi-Latent Attention decode kernels.

Sibling of AttentionConfig (not extending it), following the C++ CUTLASS pattern
of separate concrete types per kernel variant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Type

import cutlass


@dataclass(frozen=True)
class MLAConfig:
    """Core configuration for MLA decode kernels.

    Groups constructor parameters that define the problem shape, data types,
    tile shapes, and execution mode. Derived iteration counts are computed
    as properties.
    """

    latent_dim: int
    rope_dim: int
    num_heads: int

    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32
    lse_dtype: Type[cutlass.Numeric] = cutlass.Float32

    mma_qk_tiler_mn: Tuple[int, int] = (128, 128)
    mma_pv_tiler_mn: Tuple[int, int] = (128, 256)

    max_active_clusters: int = 1
    is_persistent: bool = True
    is_cpasync: bool = False
    use_page_table: bool = True
    is_var_seq: bool = False
    is_var_split_kv: bool = False
    use_2cta_instrs: bool = True
    cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1)

    warps_in_n: int = 2

    @property
    def mma_qk_tiler(self) -> Tuple[int, int, int]:
        return (self.mma_qk_tiler_mn[0], self.mma_qk_tiler_mn[1], self.rope_dim)

    @property
    def mma_pv_tiler(self) -> Tuple[int, int, int]:
        return (self.mma_pv_tiler_mn[0], self.mma_pv_tiler_mn[1], 32)

    @property
    def iterations_qk_latent(self) -> int:
        return self.latent_dim // self.mma_qk_tiler[2]

    @property
    def iterations_qk_rope(self) -> int:
        return self.rope_dim // self.mma_qk_tiler[2]

    @property
    def iterations_qk(self) -> int:
        return self.iterations_qk_latent + self.iterations_qk_rope

    @property
    def iterations_pv_k(self) -> int:
        return self.mma_qk_tiler[1] // self.mma_pv_tiler[2]

    @property
    def iterations_pv_n(self) -> int:
        return self.latent_dim // self.mma_pv_tiler[1]
