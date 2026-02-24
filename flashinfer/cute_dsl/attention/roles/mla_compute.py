# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""MLAComputeRole — orchestrator for MLA decode compute warps.

Thin dispatcher that delegates to MLASoftmaxRole, MLARescaleRole, and
MLAEpilogueRole, matching the C++ CUTLASS collectives pattern of
separate concrete role types.
"""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline

from .mla_softmax import MLASoftmaxRole
from .mla_rescale import MLARescaleRole
from .mla_epilogue import MLAEpilogueRole


class MLAComputeRole:
    def __init__(self, config, mainloop, schedule, exchange_sync_bar):
        self.mma_qk_tiler = config.mma_qk_tiler
        self.acc_dtype = config.acc_dtype

        self.softmax_role = MLASoftmaxRole(
            config, mainloop, schedule, exchange_sync_bar
        )
        self.rescale_role = MLARescaleRole(config, mainloop, schedule)
        self.epilogue_role = MLAEpilogueRole(
            config, mainloop, schedule, exchange_sync_bar
        )

        self._q_dtype = None
        self._o_dtype = None

    @property
    def q_dtype(self):
        return self._q_dtype

    @q_dtype.setter
    def q_dtype(self, value):
        self._q_dtype = value
        self.softmax_role.q_dtype = value

    @property
    def o_dtype(self):
        return self._o_dtype

    @o_dtype.setter
    def o_dtype(self, value):
        self._o_dtype = value
        self.epilogue_role.o_dtype = value

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        rescale_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        mma_o_consumer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])

        row_max = -self.acc_dtype.inf
        row_sum = self.acc_dtype(0)
        correction_factor = self.acc_dtype(1)
        k_index_init = k_index
        while k_tile_count > 0:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax_role.dispatch_apply_mask(
                common_params,
                softmax_params,
                k_index,
                k_tile_total,
                mma_s_consumer_state,
                p_mma_producer_state,
                row_max,
                row_sum,
                correction_factor,
            )
            if k_index > k_index_init:
                mma_o_consumer_state = self.rescale_role.run(
                    common_params,
                    rescale_params,
                    mma_o_consumer_state,
                    correction_factor,
                )
            k_index = k_index + 1
            k_tile_count = k_tile_count - 1

        mma_o_consumer_state = self.epilogue_role.run(
            common_params, epilogue_params, mma_o_consumer_state, row_max, row_sum
        )
        return mma_s_consumer_state, p_mma_producer_state, mma_o_consumer_state
