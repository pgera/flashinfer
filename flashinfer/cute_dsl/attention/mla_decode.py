# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import math
from typing import Type, Tuple, Optional, Union, overload, Literal
from types import SimpleNamespace

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch

from cutlass.cute.runtime import from_dlpack

from .mla_config import MLAConfig
from .mla_warp_schedule import MLAWarpSchedule, MLA_DECODE_SCHEDULE
from .pipeline_topology import PipelineTopology, make_mla_topology
from .mainloop_spec import MLAMainloopSpec, make_mla_mainloop_spec
from .collective_builder import build_mla_launch_params
from .roles.mla_loader import MLALoaderRole
from .roles.mla_mma import MLAMmaRole
from .roles.mla_compute import MLAComputeRole
from .scheduler.mla_persistent import (
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler,
    create_mla_static_tile_scheduler_params,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


LOG2_E = 1.4426950408889634074


class BlackwellMultiLatentAttentionForward:
    def __init__(
        self,
        latent_dim: int,
        rope_dim: int,
        num_heads: int,
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        is_persistent: bool,
        is_cpasync: bool,
        use_page_table: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        use_2cta_instrs: bool,
        cluster_shape_mnk: Tuple[int, int, int],
        *,
        config: Optional[MLAConfig] = None,
        warp_schedule: Optional[MLAWarpSchedule] = None,
    ):
        if config is None:
            config = MLAConfig(
                latent_dim=latent_dim,
                rope_dim=rope_dim,
                num_heads=num_heads,
                acc_dtype=acc_dtype,
                lse_dtype=lse_dtype,
                mma_qk_tiler_mn=mma_qk_tiler_mn,
                mma_pv_tiler_mn=mma_pv_tiler_mn,
                max_active_clusters=max_active_clusters,
                is_persistent=is_persistent,
                is_cpasync=is_cpasync,
                use_page_table=use_page_table,
                is_var_seq=is_var_seq,
                is_var_split_kv=is_var_split_kv,
                use_2cta_instrs=use_2cta_instrs,
                cluster_shape_mnk=cluster_shape_mnk,
            )

        self.config = config
        self.schedule = warp_schedule if warp_schedule is not None else MLA_DECODE_SCHEDULE
        self.mainloop = make_mla_mainloop_spec(config, self.schedule)
        self.loader_role = MLALoaderRole(config)
        self.mma_role = None  # initialized after mainloop.resolve() sets stage counts
        self.compute_role = None  # initialized after mainloop.resolve() and dtypes known

        self.tmem_ptr_sync_bar = pipeline.NamedBarrier(
            barrier_id=self.schedule.tmem_ptr_sync_bar_id,
            num_threads=self.schedule.tmem_ptr_sync_num_threads,
        )
        self.exchange_sync_bar = pipeline.NamedBarrier(
            barrier_id=self.schedule.exchange_sync_bar_id,
            num_threads=self.schedule.exchange_sync_num_threads,
        )

    def _setup_attributes(self):
        """Set up configurations and parameters for the MLA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the multi-head latent attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        self.mainloop.resolve(self.k_dtype.width)
        self.mma_role = MLAMmaRole(self.config, self.mainloop)
        self.compute_role = MLAComputeRole(
            self.config, self.mainloop, self.schedule, self.exchange_sync_bar
        )
        self.compute_role.q_dtype = self.q_dtype
        self.compute_role.o_dtype = self.o_dtype

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Execute the Multi-Head Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for temporary split KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [num_head, latent_dim, batch_size]
        :type q_latent: cute.Tensor
        :param q_rope: The query RoPE tensor with shape [num_head, rope_dim, batch_size]
        :type q_rope: cute.Tensor
        :param c_latent: The key tensor with shape [seq_len, latent_dim, batch_size]
        :type c_latent: cute.Tensor
        :param c_rope: The key RoPE tensor with shape [seq_len, rope_dim, batch_size]
        :type c_rope: cute.Tensor
        :param page_table: The page table tensor with shape [page_count, batch_size]
        :type page_table: cute.Tensor
        :param o: The output tensor with shape [num_head, latent_dim, batch_size]
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [num_head, batch_size]
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The block split KV tensor with shape [batch_size]
        :type block_split_kvs: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent.element_type
        self.v_dtype = c_latent.element_type
        self.o_dtype = o.element_type

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype
        ):
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} or {self.q_dtype} != {self.v_dtype}"
            )
        # check leading dimensions of input/output
        if cutlass.const_expr(q_latent.stride[1] != 1 or q_rope.stride[1] != 1):
            raise ValueError("q_latent and q_rope must have leading dimension 1")
        if cutlass.const_expr(c_latent.stride[1] != 1 or c_rope.stride[1] != 1):
            raise ValueError("c_latent and c_rope must have leading dimension 1")
        if cutlass.const_expr(o.stride[1] != 1):
            raise ValueError("o must have leading dimension 1")
        if cutlass.const_expr(lse.stride[0] != 1):
            raise ValueError("lse must have leading dimension 0")

        acc_o, acc_lse = self.initialize_workspace(
            q_latent.shape[0],
            q_latent.shape[1],
            q_latent.shape[2],
            split_kv,
            self.config.acc_dtype,
            workspace,
        )

        c_latent_tranpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_tranpose_layout
        )

        self._setup_attributes()

        lp = build_mla_launch_params(
            self.mainloop, self.schedule,
            q_latent, q_rope, c_latent, c_rope, c_latent_transpose,
            self.q_dtype, self.k_dtype, self.v_dtype,
        )
        self.tma_copy_q_bytes = lp.tma_copy_q_bytes
        self.tma_copy_kc_bytes = lp.tma_copy_kc_bytes

        tile_sched_params, grid = self._compute_grid(
            o, split_kv,
            self.config.cluster_shape_mnk,
            self.config.max_active_clusters,
            self.config.is_persistent,
        )

        softmax_scale_log2 = softmax_scale * LOG2_E
        self.split_kv_kernel(
            lp.qk_tiled_mma,
            lp.pv_tiled_mma,
            lp.tma_atom_q_latent,
            lp.tma_tensor_q_latent,
            lp.tma_atom_q_rope,
            lp.tma_tensor_q_rope,
            lp.tma_atom_c_latent,
            lp.tma_tensor_c_latent,
            lp.tma_atom_c_rope,
            lp.tma_tensor_c_rope,
            lp.tma_atom_c_latent_transpose,
            lp.tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            lp.q_smem_layout_staged,
            lp.kc_smem_layout_staged,
            lp.p_smem_layout_staged,
            lp.vc_smem_layout_staged,
            lp.cta_layout_vmnk,
            tile_sched_params,
            lp.SharedStorage,
        ).launch(
            grid=grid,
            block=[self.schedule.threads_per_cta(self.config.is_cpasync), 1, 1],
            cluster=self.config.cluster_shape_mnk,
            smem=lp.SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        if cutlass.const_expr(acc_o is not None):
            self.reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
            ).launch(
                grid=(q_latent.shape[0], 1, q_latent.shape[2]),
                block=[self.schedule.threads_per_warp * self.schedule.num_compute_warps, 1, 1],
                smem=split_kv * self.config.acc_dtype.width // 8,
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.jit
    def _create_pipelines(self, storage, cta_layout_vmnk):
        """Create all inter-warp pipelines from the topology and storage barriers."""
        barrier_ptrs = {
            "load_q": storage.load_q_mbar_ptr.data_ptr(),
            "load_kv": storage.load_kv_mbar_ptr.data_ptr(),
            "mma_s": storage.mma_s_mbar_ptr.data_ptr(),
            "p_mma": storage.p_mma_mbar_ptr.data_ptr(),
            "mma_o": storage.mma_o_mbar_ptr.data_ptr(),
        }
        tx_counts = {"q": self.tma_copy_q_bytes, "kv": self.tma_copy_kc_bytes}
        return self.mainloop.pipeline_topology.create_pipelines_native(
            barrier_ptrs, tx_counts, self.schedule.threads_per_warp, cta_layout_vmnk,
        )

    @cute.kernel
    def split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: cute.CopyAtom,
        mQL: cute.Tensor,
        tma_atom_q_rope: cute.CopyAtom,
        mQR: cute.Tensor,
        tma_atom_c_latent: cute.CopyAtom,
        mCL: cute.Tensor,
        tma_atom_c_rope: cute.CopyAtom,
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: cute.CopyAtom,
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        kc_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: MLAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
    ):
        """MLA split-KV device kernel: warp-specialized decode with pipelined TMA loads."""

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Coords inside cluster
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Tensor memory dealloc barrier init
        if warp_idx == self.schedule.load_tma_warp_id:
            num_tmem_dealloc_threads = self.schedule.threads_per_warp * self.schedule.num_compute_warps
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        pipelines = self._create_pipelines(storage, cta_layout_vmnk)
        load_q_pipeline = pipelines["load_q"]
        load_kv_pipeline = pipelines["load_kv"]
        mma_s_pipeline = pipelines["mma_s"]
        p_mma_pipeline = pipelines["p_mma"]
        mma_o_pipeline = pipelines["mma_o"]

        # Cluster arrive after barrier init
        if cutlass.const_expr(cute.size(self.config.cluster_shape_mnk) > 1):
            cute.arch.cluster_arrive_relaxed()

        sQ = storage.smem_q.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sKC = storage.smem_kc.get_tensor(kc_smem_layout_staged.outer, swizzle=kc_smem_layout_staged.inner)
        sVC = cute.make_tensor(cute.recast_ptr(sKC.iterator, vc_smem_layout_staged.inner), vc_smem_layout_staged.outer)
        sP = storage.smem_p.get_tensor(p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner)
        smem_exchange = storage.smem_exchange.get_tensor(
            cute.make_layout(self.schedule.num_compute_warps * self.schedule.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        if cutlass.const_expr(cute.size(self.config.cluster_shape_mnk) > 1):
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including page table and data tensors
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.config.is_cpasync):
            # TODO: add cp async load variant.
            #  Load page table when isasync is true
            # if warp_idx == self.schedule.load_pt_warp_id:
            #     self.load_page_table()
            # if (
            #     warp_idx == self.load_cpasync_warp_id[0]
            #     and warp_idx == self.load_cpasync_warp_id[1]
            # ):
            #     load_cpasync()
            pass
        else:
            if warp_idx == self.schedule.load_tma_warp_id:
                load_q_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.mainloop.load_q_stages
                )
                load_kv_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.mainloop.load_kv_stages
                )
                tile_sched = create_mla_static_tile_scheduler(
                    tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
                )
                work_tile = tile_sched.initial_work_tile_info()
                while work_tile.is_valid_tile:
                    blk_coord = work_tile.tile_idx
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv,
                        cache_seqs,
                        block_split_kvs,
                        blk_coord,
                    )
                    if k_tile_count > 0:
                        # Construct fixed common/tma_qk/tma_pv params for load_tma
                        tma_common_params = SimpleNamespace(
                            blk_coord=blk_coord,
                            local_split_kv=local_split_kv,
                            load_q_pipeline=load_q_pipeline,
                            load_kv_pipeline=load_kv_pipeline,
                            mPT=mPT,
                        )
                        tma_qk_params = SimpleNamespace(
                            tiled_mma_qk=tiled_mma_qk,
                            tma_atom_q_latent=tma_atom_q_latent,
                            tma_atom_q_rope=tma_atom_q_rope,
                            tma_atom_c_latent=tma_atom_c_latent,
                            tma_atom_c_rope=tma_atom_c_rope,
                            mQL=mQL,
                            mQR=mQR,
                            mCL=mCL,
                            mKR=mKR,
                            sQ=sQ,
                            sKC=sKC,
                        )
                        tma_pv_params = SimpleNamespace(
                            tiled_mma_pv=tiled_mma_pv,
                            tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                            mCL=mCL,
                            mKR=mKR,
                            mCLT=mCLT,
                            sVC=sVC,
                        )
                        # Load tma
                        load_q_producer_state, load_kv_producer_state = self.loader_role.load_tma(
                            tma_common_params,
                            tma_qk_params,
                            tma_pv_params,
                            k_index,
                            k_tile_count,
                            load_q_producer_state,
                            load_kv_producer_state,
                        )
                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                load_q_pipeline.producer_tail(load_q_producer_state)
                load_kv_pipeline.producer_tail(load_kv_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.mma_warp_id:
            # Alloc tensor memory buffer
            cute.arch.alloc_tmem(
                cute.arch.SM100_TMEM_CAPACITY_COLUMNS,
                tmem_holding_buf,
                is_two_cta=self.config.use_2cta_instrs,
            )

            # sync with compute warp before tmem ptr is retrieved
            self.tmem_ptr_sync_bar.arrive()

            # Retrieving tensor memory ptr and make accumulator tensor
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.config.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )

            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mainloop.load_q_stages
            )
            load_kv_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mainloop.load_kv_stages
            )
            mma_s_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mainloop.mma_s_stages
            )
            p_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mainloop.p_mma_stages
            )
            mma_o_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mainloop.mma_o_stages
            )
            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    mma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_kv_pipeline=load_kv_pipeline,
                        tmem_ptr=tmem_ptr,
                        is_leader_cta=is_leader_cta,
                        L=mCL.shape[1],
                    )
                    mma_qk_params = SimpleNamespace(
                        mma_s_pipeline=mma_s_pipeline,
                        sQ=sQ,
                        sKC=sKC,
                    )
                    mma_pv_params = SimpleNamespace(
                        p_mma_pipeline=p_mma_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                        sP=sP,
                        sVC=sVC,
                    )
                    (
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_kv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    ) = self.mma_role.mma(
                        mma_common_params,
                        mma_qk_params,
                        mma_pv_params,
                        k_tile_count,
                        tiled_mma_qk,
                        tiled_mma_pv,
                        load_q_consumer_state,
                        load_kv_consumer_state,
                        mma_s_producer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mma_s_pipeline.producer_tail(mma_s_producer_state)
            mma_o_pipeline.producer_tail(mma_o_producer_state)

            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.config.use_2cta_instrs)
            # Dealloc the tensor memory buffer
            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)

            cute.arch.dealloc_tmem(
                tmem_ptr,
                cute.arch.SM100_TMEM_CAPACITY_COLUMNS,
                is_two_cta=self.config.use_2cta_instrs,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.compute_warp_ids[0]
            and warp_idx <= self.schedule.compute_warp_ids[-1]
        ):
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mainloop.mma_s_stages
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mainloop.p_mma_stages
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mainloop.mma_o_stages
            )
            # sync with mma warp before retrieving tmem ptr
            self.tmem_ptr_sync_bar.wait()

            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.config.acc_dtype,
                alignment=16,
                ptr_to_buffer_holding_addr=tmem_holding_buf,
            )

            tile_sched = create_mla_static_tile_scheduler(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            while work_tile.is_valid_tile:
                blk_coord = work_tile.tile_idx
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv, cache_seqs, block_split_kvs, blk_coord
                )
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        L=mCL.shape[1],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    compute_rescale_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        mma_o_pipeline=mma_o_pipeline,
                    )
                    compute_epilogue_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        mma_o_pipeline=mma_o_pipeline,
                        output_scale=output_scale,
                        softmax_scale_log2=softmax_scale_log2,
                        mAccLSE=mAccLSE,
                        mLSE=mLSE,
                    )
                    mma_s_consumer_state, p_mma_producer_state, mma_o_consumer_state = (
                        self.compute_role.compute(
                            compute_common_params,
                            compute_softmax_params,
                            compute_rescale_params,
                            compute_epilogue_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            mma_o_consumer_state=mma_o_consumer_state,
                        )
                    )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Arrive for the tensor memory deallocation barrier
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr, cta_rank_in_cluster ^ 1)

        return

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
    ):
        """The reduction kernel for Multi-Head Latent Attention (MLA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor (for variable split_kv)
        :type block_split_kvs: cute.Tensor
        """
        # avoid register indexing on array.
        MAX_SPLITS = 256
        bidx, _, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord = (bidx, 0, bidz)
        local_split_kv = (
            block_split_kvs[blk_coord[2]] if self.config.is_var_split_kv else split_kv
        )
        k_tile_total = cute.ceil_div(cache_seqs[blk_coord[2]], self.config.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, local_split_kv)
        local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

        # Alloc shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(MAX_SPLITS * self.config.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.config.acc_dtype)
        smem_lse_scale = cute.make_tensor(lse_scale_ptr, cute.make_layout(MAX_SPLITS))

        gLSE = mAccLSE[blk_coord[0], None, blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            # calculate the global lse and exp ^ (local_lse - global_lse)
            lse_per_thread = cute.ceil_div(MAX_SPLITS, self.schedule.threads_per_warp)

            local_lse = cute.make_fragment(
                cute.make_layout(lse_per_thread), self.config.lse_dtype
            )
            lse_max = -self.config.lse_dtype.inf
            # find the max lse
            for i in range(lse_per_thread):
                split_kv_idx = tidx + i * self.schedule.threads_per_warp
                local_lse[i] = (
                    gLSE[split_kv_idx]
                    if cute.elem_less(split_kv_idx, local_split_kv)
                    else -self.config.lse_dtype.inf
                )
                # reduce the local lse
                lse_max = cute.arch.fmax(lse_max, local_lse[i])
            lse_max = cute.arch.warp_reduction_max(lse_max)
            lse_max = lse_max if lse_max != -self.config.lse_dtype.inf else 0.0
            # calculate sum_lse
            sum_lse = 0.0
            for i in range(lse_per_thread):
                sum_lse += cute.arch.exp2(local_lse[i] - lse_max)
            sum_lse = cute.arch.warp_reduction_sum(sum_lse)
            # calculate the global_lse
            global_lse = (
                lse_max + cute.math.log2(sum_lse)
                if sum_lse != self.config.lse_dtype(0.0) or sum_lse != sum_lse
                else self.config.lse_dtype.inf
            )
            if tidx == 0:
                mLSE[blk_coord[0], blk_coord[2]] = global_lse
            # store the scale to shared memory
            for i in range(lse_per_thread):
                split_kv_idx = tidx + i * self.schedule.threads_per_warp
                if cute.elem_less(split_kv_idx, local_split_kv):
                    smem_lse_scale[split_kv_idx] = cute.arch.exp2(
                        local_lse[i] - global_lse
                    )

        cute.arch.barrier()

        elements_per_thread = cute.ceil_div(
            self.config.latent_dim, self.schedule.threads_per_warp * self.schedule.num_compute_warps
        )
        gAccO = mAccO[blk_coord[0], None, None, blk_coord[2]]
        rAccO = cute.make_fragment(
            cute.make_layout(elements_per_thread), self.config.acc_dtype
        )
        rAccO.fill(0.0)
        for i in range(local_split_kv):
            for j in range(elements_per_thread):
                element_idx = tidx + j * self.schedule.threads_per_warp * self.schedule.num_compute_warps
                rAccO[j] += gAccO[i, element_idx] * smem_lse_scale[i]
        for j in range(elements_per_thread):
            element_idx = tidx + j * self.schedule.threads_per_warp * self.schedule.num_compute_warps
            mO[blk_coord[0], element_idx, blk_coord[2]] = rAccO[j].to(self.o_dtype)
        return

    @staticmethod
    def get_split_kv(
        B: int, K: int, mma_qk_tiler_mn: tuple, max_active_blocks: int
    ) -> int:
        """Get the proper split_kv value for the MLA kernel based on parameters.

        :param B: Batch size
        :type B: int
        :param K: Sequence length
        :type K: int
        :param mma_qk_tiler_mn: MLA tiling parameters
        :type mma_qk_tiler_mn: tuple
        :param max_active_blocks: Maximum number of active blocks
        :type max_active_blocks: int
        :return: Split_kv value
        :rtype: int
        """
        max_splits = ceil_div(K, mma_qk_tiler_mn[1])
        blocks_per_batch = max(1, max_active_blocks // B)
        split_heur = min(max_splits, blocks_per_batch)
        # {$nv-internal-release begin}
        # TODO: figure out the error of make_tile with dynamic int_tuple
        # {$nv-internal-release end}
        k_waves = ceil_div(max_splits, split_heur)
        split_wave_aware = ceil_div(max_splits, k_waves)
        return split_wave_aware

    @cute.jit
    def get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get the current k_index, k_tile_count, and local split_kv value for the MLA kernel.

        :param split_kv: Split_kv value
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param blk_coord: Block coordinate
        :type blk_coord: cute.Coord
        :return: k_index, k_tile_count, split_kv
        :rtype: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        """
        K = cache_seqs[blk_coord[2]]
        if cutlass.const_expr(self.config.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[2]]

        k_tile_total = cute.ceil_div(K, self.config.mma_qk_tiler[1])
        # {$nv-internal-release begin}
        # TODO: figure out the error of make_tile with dynamic int_tuple
        # {$nv-internal-release end}
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[3] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)
        return k_index, k_tile_count, split_kv

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
    ) -> Tuple[MLAStaticTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Tile scheduler parameters and grid shape.
        :rtype: tuple[MLAStaticTileSchedulerParams, tuple[int, int, int]]
        """
        o_shape = o.shape
        tile_sched_params = create_mla_static_tile_scheduler_params(
            is_persistent,
            cute.size(o_shape[2]),
            cluster_shape_mnk,
            split_kv,
        )
        grid = MLAStaticTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

    @staticmethod
    def get_workspace_size(
        H: int,
        D: int,
        B: int,
        split_kv: int,
        acc_dtype: Type[cutlass.Numeric],
    ) -> int:
        """Get the extra workspace(device memory) size for the MLA kernel when split_kv is not 1.

        :param H: The height of the output tensor C
        :type H: int
        :param D: The depth of the output tensor C
        :type D: int
        :param B: The batch size of the output tensor C
        :type B: int
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]

        :return: The workspace size for the MLA kernel
        :rtype: int
        """
        if split_kv == 1:
            return 0
        return B * H * split_kv * (D + 1) * acc_dtype.width // 8

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize the workspace for the MLA kernel. Construct the intermediate tensors
        acc_o and acc_lse.

        :param H: The height of the output tensor C
        :type H: cutlass.Int32
        :param D: The depth of the output tensor C
        :type D: cutlass.Int32
        :param B: The batch size of the output tensor C
        :type B: cutlass.Int32
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: cutlass.Int32
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]
        :param workspace: The workspace tensor
        :type workspace: cute.Tensor

        :return: The output tensor C and the workspace tensor
        :rtype: tuple[cute.Tensor, cute.Tensor]
        """
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            align = 128 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(H * split_kv * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, B), stride=(split_kv, 1, H * split_kv)
            )
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator + cute.cosize(acc_o_layout) * acc_dtype.width // 8,
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @staticmethod
    def can_implement(
        B: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_cpasync: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        use_page_table: bool,
        page_size: int,
    ) -> bool:
        """Check if the MLA kernel can be implemented.

        :param H: The height of the output tensor C
        :type H: int
        :param K: The width of the output tensor C
        :type K: int
        :param L: The length of the output tensor C
        :type L: int
        :param R: The row of the output tensor C
        :type R: int
        :param B: The batch size of the output tensor C
        :type B: int
        :param in_dtype: The data type of the input tensor
        :type in_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: The data type of the log-sum-exp
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_qk_tiler_mn: The tile shape of the query-key matrix multiplication
        :type mma_qk_tiler_mn: Tuple[int, int]
        :param mma_pv_tiler_mn: The tile shape of the probability-value matrix multiplication
        :type mma_pv_tiler_mn: Tuple[int, int]
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param is_persistent: Whether to use persistent kernel optimization
        :type is_persistent: bool
        :param is_cpasync: Whether to use cpasync
        :type is_cpasync: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split_kv
        :type is_var_split_kv: bool
        :param use_page_table: Whether to use page table
        :type use_page_table: bool
        :param page_size: The page size of the page table
        :type page_size: int

        :return: Whether the MLA kernel can be implemented
        :rtype: bool
        """
        if L != 512 or R != 64:
            return False
        if in_dtype not in [cutlass.Float8E4M3FN, cutlass.Float16, cutlass.BFloat16]:
            return False
        if out_dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
            return False
        if is_cpasync:
            if not use_page_table:
                return False
            if page_size & (page_size - 1) != 0:
                return False
            if page_size > mma_qk_tiler_mn[1]:
                return False
        else:
            if use_page_table and page_size != mma_qk_tiler_mn[1]:
                return False
        if mma_qk_tiler_mn[0] != 128 or mma_pv_tiler_mn[0] != 128:
            return False
        if mma_pv_tiler_mn[1] * 32 != mma_qk_tiler_mn[1] * R:
            return False
        if is_var_split_kv and (not use_page_table or not is_var_seq):
            return False
        if is_var_seq and not use_page_table:
            return False
        # if H != 128:
        #     return False
        if K <= 0:
            return False
        return True


def create_page_table(
    batch_size,
    seq_len,
    is_var_seq,
    use_page_table,
    page_size,
    cache_seqs_torch,
    kv_indptr=None,
    kv_indices=None,
):
    page_table_ref, page_table, page_table_gpu = None, None, None
    if use_page_table:
        max_seq_len = seq_len if not is_var_seq else torch.max(cache_seqs_torch)
        page_count = ceil_div(max_seq_len, page_size)
        page_table_ref = torch.empty([batch_size, page_count], dtype=torch.int32)
        if kv_indptr is not None and kv_indices is not None:
            kv_indptr_cpu = kv_indptr.cpu()
            kv_indices_cpu = kv_indices.cpu()
            for b in range(batch_size):
                start = kv_indptr_cpu[b].item()
                end = kv_indptr_cpu[b + 1].item()
                for j in range(page_count):
                    if start + j < end:
                        page_table_ref[b, j] = kv_indices_cpu[start + j].item()
                    else:
                        page_table_ref[b, j] = 0
        else:
            for b in range(batch_size):
                for j in range(page_count):
                    page_table_ref[b, j] = b + j * batch_size
        page_table_gpu = page_table_ref.permute(1, 0).cuda()
        page_table = from_dlpack(page_table_gpu, assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        )
    return page_table_ref, page_table, page_table_gpu


def create_block_split_kvs(
    batch_size,
    split_kv,
    cache_seqs_ref,
    is_var_split_kv,
    mma_qk_tiler_mn,
    cluster_shape_mnk,
    max_active_clusters,
):
    block_split_kvs_ref, block_split_kvs, block_split_kvs_gpu = None, None, None
    # check if split_kv is valid otherwise do auto setting of split_kv
    if is_var_split_kv:
        block_split_kvs_ref = torch.zeros([batch_size], dtype=torch.int32)
        for b in range(batch_size):
            block_split_kvs_ref[b] = BlackwellMultiLatentAttentionForward.get_split_kv(
                batch_size,
                cache_seqs_ref[b].item(),
                mma_qk_tiler_mn,
                max_active_clusters * cluster_shape_mnk[0],
            )
        split_kv = torch.max(block_split_kvs_ref).item()
        block_split_kvs_gpu = block_split_kvs_ref.cuda()
        block_split_kvs = from_dlpack(
            block_split_kvs_gpu, assumed_align=16
        ).mark_layout_dynamic()
    elif split_kv <= 0:
        split_kv = BlackwellMultiLatentAttentionForward.get_split_kv(
            batch_size,
            cache_seqs_ref[0].item(),
            mma_qk_tiler_mn,
            max_active_clusters * cluster_shape_mnk[0],
        )
    return split_kv, block_split_kvs_ref, block_split_kvs, block_split_kvs_gpu


def create_workspace(num_heads, latent_dim, batch_size, split_kv, acc_dtype):
    workspace_size = BlackwellMultiLatentAttentionForward.get_workspace_size(
        num_heads,
        latent_dim,
        batch_size,
        split_kv,
        acc_dtype,
    )

    workspace, workspace_torch = None, None
    if workspace_size > 0:
        workspace_torch = torch.empty([workspace_size], dtype=torch.int8).cuda()
        workspace = from_dlpack(workspace_torch, assumed_align=16)
    return workspace, workspace_torch


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def torch_to_cute(
    torch_tensor_gpu,
    dtype,
    is_dynamic_layout=True,
    page_table=None,
    page_size=None,
    cache_seqs=None,
    is_lse=False,
):
    if is_lse:
        shape = torch_tensor_gpu.shape
        B, HK = shape
        permute_order = (1, 0)
        stride_order = (1, 0)
        leading_dim = 0
    else:
        shape = torch_tensor_gpu.shape
        B, HK, D = shape
        permute_order = (1, 2, 0)
        stride_order = (2, 0, 1)
        leading_dim = 1
    if page_table is not None:
        if cache_seqs is not None:
            max_seq_len = torch.max(cache_seqs)
            shape = (B * ceil_div(max_seq_len, page_size), page_size, D)
        else:
            shape = (B * ceil_div(HK, page_size), page_size, D)

    torch_tensor_gpu = torch_tensor_gpu.permute(permute_order)

    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype
    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=leading_dim
        ).mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=stride_order,
            divisibility=(128 // dtype.width),
        )

    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )

    return cute_tensor, torch_tensor_gpu


def create_tensor(
    B,
    HK,
    D,
    dtype,
    is_dynamic_layout=True,
    page_table=None,
    cache_seqs=None,
    is_lse=False,
    page_size=None,
):
    shape = (B, HK, D)
    if page_table is not None:
        if cache_seqs is not None:
            max_seq_len = torch.max(cache_seqs)
            shape = (B * ceil_div(max_seq_len, page_size), page_size, D)
        else:
            shape = (B * ceil_div(HK, page_size), page_size, D)
    permute_order = (1, 2, 0)
    stride_order = (2, 0, 1)
    leading_dim = 1
    if is_lse:
        shape = (B, HK)
        permute_order = (1, 0)
        stride_order = (1, 0)
        leading_dim = 0
    init_config = cutlass.torch.RandomInitConfig(min_val=-2, max_val=2)
    torch_dtype = cutlass_torch.dtype(dtype)
    torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        shape,
        torch_dtype,
        permute_order=permute_order,
        init_type=cutlass.torch.TensorInitType.RANDOM,
        init_config=init_config,
    )
    torch_tensor_gpu = torch_tensor_cpu.cuda()

    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype

    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=leading_dim
        ).mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=stride_order,
            divisibility=(128 // dtype.width),
        )
    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )
    return cute_tensor, torch_tensor_gpu
