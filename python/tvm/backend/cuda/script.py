# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""CUDA TVMScript namespaces."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tvm.backend.cuda import op as _cuda_op
from tvm.tirx import Buffer
from tvm.tirx import op as _tir_op
from tvm.tirx.script.builder.ir import _dtype_forward, _op_wrapper

# pylint: disable=protected-access


def _ptx_ldg32(reg, guard, addr, local_addr):
    if isinstance(addr, Buffer):
        addr = addr[0]
    return _tir_op.call_intrin(reg.ty, "tirx.ptx.ldg32", reg, guard, addr, local_addr)


_ptx_ldg32.__tir_op_name__ = "ptx.ldg32"


class PTXNamespace:
    """The PTX instruction submodule."""

    def __init__(self):
        self.ldg32 = _ptx_ldg32
        self.ldmatrix = _dtype_forward(_cuda_op.ptx_ldmatrix)
        # Apache-compatible variant. Same lowered intrinsic as
        # ``ldmatrix`` but accepts the historical ``(trans, num, dtype,
        # local_ptr, local_offset, smem_ptr, smem_offset)`` form. Coexists
        # with the fork-native version so upstream-derived tests keep
        # working without rewriting their tirx code.
        self.ldmatrix_legacy = _dtype_forward(_cuda_op.ptx_ldmatrix_legacy)
        self.stmatrix = _op_wrapper(_cuda_op.ptx_stmatrix)
        self.setmaxnreg: Callable[..., Any] = _op_wrapper(_cuda_op.ptx_setmaxnreg)
        self.elect_sync: Callable[..., Any] = _op_wrapper(_cuda_op.ptx_elect_sync)
        self.clc_try_cancel = _op_wrapper(_cuda_op.ptx_clc_try_cancel)
        self.clc_query_cancel = _op_wrapper(_cuda_op.ptx_clc_query_cancel)
        self.fetch_register: Callable[..., Any] = _op_wrapper(_cuda_op.ptx_fetch_register)
        self.ld = _op_wrapper(_cuda_op.ptx_ld)
        self.ld_acquire = _op_wrapper(_cuda_op.ptx_ld_acquire)
        self.ld_relaxed = _op_wrapper(_cuda_op.ptx_ld_relaxed)
        self.ld_volatile = _op_wrapper(_cuda_op.ptx_ld_volatile)
        self.ld_mmio = _op_wrapper(_cuda_op.ptx_ld_mmio)
        self.ld_global_acquire = _op_wrapper(_cuda_op.ptx_ld_global_acquire)
        self.red_scalar = _op_wrapper(_cuda_op.ptx_red_scalar)
        self.atom_scalar = _op_wrapper(_cuda_op.ptx_atom_scalar)
        self.prefetch_tensormap = _op_wrapper(_cuda_op.ptx_prefetch_tensormap)
        self.mbarrier_test_wait_parity = _op_wrapper(_cuda_op.ptx_mbarrier_test_wait_parity)
        self.cp_async_bulk_g2s_cta = _op_wrapper(_cuda_op.ptx_cp_async_bulk_g2s_cta)
        self.cp_async_bulk_g2s_cluster = _op_wrapper(_cuda_op.ptx_cp_async_bulk_g2s_cluster)
        self.cp_async_bulk_s2s_cluster = _op_wrapper(_cuda_op.ptx_cp_async_bulk_s2s_cluster)
        self.cp_async_bulk_s2g = _op_wrapper(_cuda_op.ptx_cp_async_bulk_s2g)
        self.st = _op_wrapper(_cuda_op.ptx_st)
        self.st_relaxed = _op_wrapper(_cuda_op.ptx_st_relaxed)
        self.st_release = _op_wrapper(_cuda_op.ptx_st_release)
        self.st_volatile = _op_wrapper(_cuda_op.ptx_st_volatile)
        self.st_mmio = _op_wrapper(_cuda_op.ptx_st_mmio)
        self.st_bulk = _op_wrapper(_cuda_op.ptx_st_bulk)
        self.fns_b32 = _op_wrapper(_cuda_op.ptx_fns_b32)
        self.add_rn_f32_bf16 = _op_wrapper(_cuda_op.ptx_add_rn_f32_bf16)
        self.mapa = _op_wrapper(_cuda_op.ptx_mapa)
        self.map_shared_rank = _op_wrapper(_cuda_op.ptx_map_shared_rank)
        self.any_sync = _op_wrapper(_cuda_op.ptx_any_sync)
        # Math operations
        self.exp2 = _op_wrapper(_cuda_op.ptx_exp2)
        self.rcp = _op_wrapper(_cuda_op.ptx_rcp)
        self.reduce3_min_f32 = _op_wrapper(_cuda_op.ptx_reduce3_min_f32)
        self.reduce3_max_f32 = _op_wrapper(_cuda_op.ptx_reduce3_max_f32)
        # add/sub/mul/fma DPS form: (d_addr, a, b[, c], *, rounding, ftz[, sat])
        self.add_f32 = _op_wrapper(_cuda_op.ptx_add_f32)
        self.add_f32x2 = _op_wrapper(_cuda_op.ptx_add_f32x2)
        self.add_f64 = _op_wrapper(_cuda_op.ptx_add_f64)
        self.sub_f32 = _op_wrapper(_cuda_op.ptx_sub_f32)
        self.sub_f32x2 = _op_wrapper(_cuda_op.ptx_sub_f32x2)
        self.sub_f64 = _op_wrapper(_cuda_op.ptx_sub_f64)
        self.mul_f32 = _op_wrapper(_cuda_op.ptx_mul_f32)
        self.mul_f32x2 = _op_wrapper(_cuda_op.ptx_mul_f32x2)
        self.mul_f64 = _op_wrapper(_cuda_op.ptx_mul_f64)
        self.fma_f32 = _op_wrapper(_cuda_op.ptx_fma_f32)
        self.fma_f32x2 = _op_wrapper(_cuda_op.ptx_fma_f32x2)
        self.fma_f64 = _op_wrapper(_cuda_op.ptx_fma_f64)
        self.max_f32 = _op_wrapper(_cuda_op.ptx_max_f32)
        self.mma = MmaNamespace()
        self.cp_async = CpAsyncNamespace()
        self.wgmma = WgmmaNamespace()
        self.mbarrier = MbarrierNamespace()
        self.tcgen05 = Tcgen05Namespace()
        self.bar = BarNamespace()
        self.barrier = BarrierNamespace()
        self.fence = FenceNamespace()
        self.griddepcontrol = GriddepcontrolNamespace()


class MmaNamespace:
    """The MMA instruction submodule."""

    def __init__(self):
        self.sp = _dtype_forward(_cuda_op.ptx_mma_sp)
        # Apache-compatible variant of ptx_mma. Coexists with the
        # fork-native ``__call__`` form (``T.ptx.mma(...)``).
        self.legacy = _dtype_forward(_cuda_op.ptx_mma_legacy)
        # __call__ corresponds to ptx_mma
        self.__tir_call_op_name__ = "ptx_mma"

    def __call__(self, *args, **kwds):
        return _dtype_forward(_cuda_op.ptx_mma)(*args, **kwds)


class CpAsyncNamespace:
    """The CpAsync instruction submodule."""

    def __init__(self):
        self.commit_group = _op_wrapper(_cuda_op.ptx_cp_async_commit_group)
        self.wait_group = _op_wrapper(_cuda_op.ptx_cp_async_wait_group)
        # Legacy variant: takes (dst_ptr, dst_offset, src_ptr, src_offset,
        # cp_size). Offsets are folded into the pointers; coexists with
        # the fork-native ``__call__`` form.
        self.legacy = _dtype_forward(_cuda_op.ptx_cp_async_legacy)
        self.bulk = CpAsyncBulkNamespace()
        self.mbarrier = CpAsyncMbarrierNamespace()

    def __call__(self, *args, **kwds):
        # Accept the legacy 6-arg form ``(elem_dtype, dst, dst_off, src,
        # src_off, cp_size)`` that the printer round-trips for the raw
        # ``tirx.ptx.cp_async`` Call emitted by
        # ``tvm.backend.cuda.transform.InjectPTXAsyncCopy``. The pass-emitted
        # Call has 5 args (no ``tvm_access_ptr`` fold) and a
        # per-element-dtype Call.dtype, so build it directly.
        if len(args) == 6 and isinstance(args[0], str) and "dtype" not in kwds:
            import tvm

            elem_dtype, dst, dst_off, src, src_off, cp_size = args
            return tvm.tirx.Call(
                tvm.DataType(elem_dtype),
                tvm.ir.Op.get("tirx.ptx.cp_async_raw"),
                [dst, dst_off, src, src_off, cp_size],
            )
        return _dtype_forward(_cuda_op.ptx_cp_async)(*args, **kwds)

    # __call__ corresponds to ptx_cp_async
    __tir_call_op_name__ = "ptx_cp_async"


class CpAsyncBulkNamespace:
    """The CpAsyncBulk instruction submodule."""

    def __init__(self):
        self.commit_group = _op_wrapper(_cuda_op.ptx_cp_async_bulk_commit_group)
        self.wait_group = _op_wrapper(_cuda_op.ptx_cp_async_bulk_wait_group)
        self.tensor = CpAsyncBulkTensorNamespace()
        self.s2c = _op_wrapper(_cuda_op.ptx_cp_async_bulk_shared_to_cluster)

    def __call__(self, *args, **kwds):
        return _dtype_forward(_cuda_op.ptx_cp_async_bulk)(*args, **kwds)

    # __call__ corresponds to ptx_cp_async_bulk
    __tir_call_op_name__ = "ptx_cp_async_bulk"


class CpAsyncBulkTensorNamespace:
    """The CpAsyncBulkTensor instruction submodule."""

    def __init__(self):
        self.g2c = _op_wrapper(_cuda_op.ptx_cp_async_bulk_tensor_global_to_cluster)
        self.g2c_tile_gather4 = _op_wrapper(
            _cuda_op.ptx_cp_async_bulk_tensor_tile_gather4_global_to_cluster
        )
        self.s2g = _op_wrapper(_cuda_op.ptx_cp_async_bulk_tensor_shared_to_global)
        self.s2g_reduce = _op_wrapper(_cuda_op.ptx_cp_async_bulk_tensor_shared_to_global_reduce)
        self.g2c_prefetch = _op_wrapper(
            _cuda_op.ptx_cp_async_bulk_tensor_global_to_cluster_prefetch
        )

    @staticmethod
    def g2c_bar_addr(
        dim,
        dst_ptr,
        bar_addr,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_hint,
        *coords,
        cache_policy=None,
    ):
        _cuda_op._choice("cta_group", cta_group, _cuda_op._TCGEN05_CTA_GROUP)
        cache_policy, has_cache_policy = _cuda_op._resolve_cache_policy(cache_hint, cache_policy)
        return _tir_op.call_intrin(
            "",
            "tirx.ptx.cp_async_bulk_tensor_global_to_cluster",
            dim,
            dst_ptr,
            bar_addr,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_policy,
            int(has_cache_policy),
            1,
            *coords,
        )

    @staticmethod
    def g2c_tile_gather4_bar_addr(
        dim,
        dst_ptr,
        bar_addr,
        tensormap_addr,
        cta_mask,
        cta_group,
        cache_hint,
        *coords,
        cache_policy=None,
    ):
        _cuda_op._choice("cta_group", cta_group, _cuda_op._TCGEN05_CTA_GROUP)
        cache_policy, has_cache_policy = _cuda_op._resolve_cache_policy(cache_hint, cache_policy)
        return _tir_op.call_intrin(
            "",
            "tirx.ptx.cp_async_bulk_tensor_tile_gather4_global_to_cluster",
            dim,
            dst_ptr,
            bar_addr,
            tensormap_addr,
            cta_mask,
            cta_group,
            cache_policy,
            int(has_cache_policy),
            1,
            *coords,
        )


class CpAsyncMbarrierNamespace:
    """The CpAsyncMbarrier instruction submodule."""

    def __init__(self):
        self.arrive = _op_wrapper(_cuda_op.ptx_cp_async_mbarrier_arrive)


class WgmmaNamespace:
    """The WGMMA instruction submodule."""

    def __init__(self):
        self.fence: Callable[..., Any] = _op_wrapper(_cuda_op.ptx_wgmma_fence)
        self.commit_group = _op_wrapper(_cuda_op.ptx_wgmma_commit_group)
        self.wait_group = _op_wrapper(_cuda_op.ptx_wgmma_wait_group)
        self.noop_barrier = _op_wrapper(_cuda_op.ptx_wgmma_noop_barrier)
        self.mma_async = WgmmaMmaAsyncNamespace()
        self.encode_matrix_descriptor = _op_wrapper(_cuda_op.ptx_wgmma_encode_matrix_descriptor)


class WgmmaMmaAsyncNamespace:
    """The WGMMA MMAAsync instruction submodule."""

    def __init__(self):
        self.ss = _op_wrapper(_cuda_op.ptx_wgmma_mma_async_ss)
        self.rs = _op_wrapper(_cuda_op.ptx_wgmma_mma_async_rs)


class MbarrierNamespace:
    """The Mbarrier instruction submodule."""

    def __init__(self):
        self.init = _op_wrapper(_cuda_op.ptx_mbarrier_init)
        self.try_wait = _op_wrapper(_cuda_op.ptx_mbarrier_try_wait)
        self.try_wait_once = _op_wrapper(_cuda_op.ptx_mbarrier_try_wait_once)
        self.try_wait_acquire_cluster = _op_wrapper(_cuda_op.ptx_mbarrier_try_wait_acquire_cluster)
        self.arrive = MbarrierArriveNamespace()


class MbarrierArriveNamespace:
    """The Mbarrier Arrive instruction submodule."""

    def __init__(self):
        self.expect_tx = _op_wrapper(_cuda_op.ptx_mbarrier_arrive_expect_tx)
        self.cluster_count = _op_wrapper(_cuda_op.ptx_mbarrier_arrive_cluster_count)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_cuda_op.ptx_mbarrier_arrive)(*args, **kwds)

    # __call__ corresponds to ptx_mbarrier_arrive
    __tir_call_op_name__ = "ptx_mbarrier_arrive"


class Tcgen05Namespace:
    """The Tcgen05 instruction submodule."""

    def __init__(self):
        self.alloc = _op_wrapper(_cuda_op.ptx_tcgen05_alloc)
        self.dealloc = _op_wrapper(_cuda_op.ptx_tcgen05_dealloc)
        self.relinquish_alloc_permit = _op_wrapper(_cuda_op.ptx_tcgen05_relinquish_alloc_permit)
        self.encode_matrix_descriptor = _op_wrapper(_cuda_op.ptx_tcgen05_encode_matrix_descriptor)
        self.encode_instr_descriptor = _op_wrapper(_cuda_op.ptx_tcgen05_encode_instr_descriptor)
        self.encode_instr_descriptor_block_scaled = _op_wrapper(
            _cuda_op.ptx_tcgen05_encode_instr_descriptor_block_scaled
        )
        self.ld = _op_wrapper(_cuda_op.ptx_tcgen05_ld)
        self.st = _op_wrapper(_cuda_op.ptx_tcgen05_st)
        self.cp = _op_wrapper(_cuda_op.ptx_tcgen05_cp)
        self.shift = _op_wrapper(_cuda_op.ptx_tcgen05_shift)
        self.commit = _op_wrapper(_cuda_op.ptx_tcgen05_commit)
        self.wait = Tcgen05WaitNamespace()
        self.mma = Tcgen05MmaNamespace()
        self.fence = Tcgen05FenceNamespace()


class Tcgen05FenceNamespace:
    """The Tcgen05 Fence instruction submodule."""

    def __init__(self):
        self.before_thread_sync = _op_wrapper(_cuda_op.ptx_tcgen05_fence_before_thread_sync)
        self.after_thread_sync = _op_wrapper(_cuda_op.ptx_tcgen05_fence_after_thread_sync)


class Tcgen05MmaNamespace:
    """The Tcgen05 MMA instruction submodule."""

    def __init__(self):
        self.block_scale = _op_wrapper(_cuda_op.ptx_tcgen05_mma_block_scale)
        self.sp = Tcgen05MmaSpNamespace()

    def __call__(self, *args, **kwds):
        return _op_wrapper(_cuda_op.ptx_tcgen05_mma)(*args, **kwds)

    # __call__ corresponds to ptx_tcgen05_mma
    __tir_call_op_name__ = "ptx_tcgen05_mma"


class Tcgen05MmaSpNamespace:
    """Tcgen05 Sparse MMA instruction submodule."""

    def __init__(self):
        self.block_scale = _op_wrapper(_cuda_op.ptx_tcgen05_mma_sp_block_scale)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_cuda_op.ptx_tcgen05_mma_sp)(*args, **kwds)

    # __call__ corresponds to ptx_tcgen05_mma_sp
    __tir_call_op_name__ = "ptx_tcgen05_mma_sp"


class Tcgen05WaitNamespace:
    """The Tcgen05 Wait instruction submodule."""

    def __init__(self):
        self.ld = _op_wrapper(_cuda_op.ptx_tcgen05_wait_ld)
        self.st = _op_wrapper(_cuda_op.ptx_tcgen05_wait_st)


class BarNamespace:
    """The Bar instruction submodule."""

    def __init__(self):
        self.arrive = _op_wrapper(_cuda_op.ptx_bar_arrive)
        self.sync = _op_wrapper(_cuda_op.ptx_bar_sync)


class BarrierNamespace:
    """The Barrier instruction submodule."""

    def __init__(self):
        self.cluster = BarrierClusterNamespace()


class BarrierClusterNamespace:
    """The BarrierCluster instruction submodule."""

    def __init__(self):
        self.arrive = _op_wrapper(_cuda_op.ptx_barrier_cluster_arrive)
        self.wait = _op_wrapper(_cuda_op.ptx_barrier_cluster_wait)


class FenceNamespace:
    """PTX fence instruction submodule."""

    def __init__(self):
        self.proxy_async = _op_wrapper(_cuda_op.ptx_fence_proxy_async)
        self.mbarrier_init = _op_wrapper(_cuda_op.ptx_fence_mbarrier_init)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_cuda_op.ptx_fence)(*args, **kwds)

    __tir_call_op_name__ = "ptx_fence"


class GriddepcontrolNamespace:
    """PTX griddepcontrol instruction submodule (sm_90+)."""

    def __init__(self):
        self.wait = _op_wrapper(_cuda_op.ptx_griddepcontrol_wait)
        self.launch_dependents = _op_wrapper(_cuda_op.ptx_griddepcontrol_launch_dependents)


class CUDANamespace:
    """The CUDA intrinsics submodule."""

    def __init__(self):
        self.atomic_add = _op_wrapper(_cuda_op.cuda_atomic_add)
        self.thread_fence = _op_wrapper(_cuda_op.cuda_thread_fence)
        self.warpgroup_sync = _op_wrapper(_cuda_op.cuda_warpgroup_sync)
        self.warp_sync = _op_wrapper(_cuda_op.cuda_warp_sync)
        self.warp_reduce = _op_wrapper(_cuda_op.cuda_warp_reduce)
        self.warp_sum = _op_wrapper(_cuda_op.cuda_warp_sum)
        self.warp_max = _op_wrapper(_cuda_op.cuda_warp_max)
        self.warp_min = _op_wrapper(_cuda_op.cuda_warp_min)
        self.cta_reduce = _op_wrapper(_cuda_op.cuda_cta_reduce)
        self.cta_sum = _op_wrapper(_cuda_op.cuda_cta_sum)
        self.cta_max = _op_wrapper(_cuda_op.cuda_cta_max)
        self.cta_min = _op_wrapper(_cuda_op.cuda_cta_min)
        self.cta_sync = _op_wrapper(_cuda_op.cuda_cta_sync)
        self.grid_sync = _op_wrapper(_cuda_op.cuda_grid_sync)
        self.cluster_sync = _op_wrapper(_cuda_op.cuda_cluster_sync)
        self.thread_rank = _op_wrapper(_cuda_op.cuda_thread_rank)
        self.trap_when_assert_failed = _op_wrapper(_cuda_op.cuda_trap_when_assert_failed)
        self.runtime_instr_desc = _op_wrapper(_cuda_op.cuda_runtime_instr_desc)
        self.half2float = _op_wrapper(_cuda_op.cuda_half2float)
        self.bfloat162float = _op_wrapper(_cuda_op.cuda_bfloat162float)
        self.float22half2 = _op_wrapper(_cuda_op.cuda_float22half2)
        self.half8tofloat8 = _op_wrapper(_cuda_op.cuda_half8tofloat8)
        self.float8tohalf8 = _op_wrapper(_cuda_op.cuda_float8tohalf8)
        self.syncthreads_and = _op_wrapper(_cuda_op.cuda_syncthreads_and)
        self.syncthreads_or = _op_wrapper(_cuda_op.cuda_syncthreads_or)
        self.nano_sleep = _op_wrapper(_cuda_op.cuda_nano_sleep)
        self.atomic_cas = _op_wrapper(_cuda_op.cuda_atomic_cas)
        self.func_call = _op_wrapper(_cuda_op.cuda_func_call)
        self.printf = _op_wrapper(_cuda_op.cuda_printf)
        self.ldg = _op_wrapper(_cuda_op.cuda_ldg)
        self.get_tmem_addr = _op_wrapper(_cuda_op.cuda_get_tmem_addr)
        self.cvta_generic_to_shared = _op_wrapper(_cuda_op.cuda_cvta_generic_to_shared)
        self.smem_addr_from_uint64 = _op_wrapper(_cuda_op.cuda_smem_addr_from_uint64)
        self.sm100_tma_2sm_mbarrier_addr = _op_wrapper(_cuda_op.cuda_sm100_tma_2sm_mbarrier_addr)
        self.uint_as_float = _op_wrapper(_cuda_op.cuda_uint_as_float)
        self.float_as_uint = _op_wrapper(_cuda_op.cuda_float_as_uint)
        self.ballot_sync = _op_wrapper(_cuda_op.cuda_ballot_sync)
        self.ffs_u32 = _op_wrapper(_cuda_op.cuda_ffs_u32)
        self.reduce_add_sync_u32 = _op_wrapper(_cuda_op.cuda_reduce_add_sync_u32)
        self.reduce_min_sync_u32 = _op_wrapper(_cuda_op.cuda_reduce_min_sync_u32)
        self.clock64 = _op_wrapper(_cuda_op.cuda_clock64)
        self.make_float2 = _op_wrapper(_cuda_op.cuda_make_float2)
        self.float2_x = _op_wrapper(_cuda_op.cuda_float2_x)
        self.float2_y = _op_wrapper(_cuda_op.cuda_float2_y)
        self.fmul2_rn = _op_wrapper(_cuda_op.cuda_fmul2_rn)
        self.fadd2_rn = _op_wrapper(_cuda_op.cuda_fadd2_rn)
        self.float22bfloat162_rn = _op_wrapper(_cuda_op.cuda_float22bfloat162_rn)
        self.float22bfloat162_rn_from_float2 = _op_wrapper(
            _cuda_op.cuda_float22bfloat162_rn_from_float2
        )
        self.bfloat1622float2 = _op_wrapper(_cuda_op.cuda_bfloat1622float2)
        self.hmin2 = _op_wrapper(_cuda_op.cuda_hmin2)
        self.hmax2 = _op_wrapper(_cuda_op.cuda_hmax2)
        self.fp8x4_e4m3_from_float4 = _op_wrapper(_cuda_op.cuda_fp8x4_e4m3_from_float4)
        self.timer_init = _op_wrapper(_cuda_op.timer_init_cuda)
        self.timer_start = _op_wrapper(_cuda_op.timer_start_cuda)
        self.timer_end = _op_wrapper(_cuda_op.timer_end_cuda)
        self.timer_finalize = _op_wrapper(_cuda_op.timer_finalize_cuda)
        self.mma_store = _dtype_forward(_cuda_op.mma_store)
        self.mma_fill = _dtype_forward(_cuda_op.mma_fill)
        self.mma_store_legacy = _dtype_forward(_cuda_op.mma_store_legacy)
        self.mma_fill_legacy = _dtype_forward(_cuda_op.mma_fill_legacy)
        setattr(self, "__shfl_sync", self._shfl_sync)
        setattr(self, "__shfl_up_sync", self._shfl_up_sync)
        setattr(self, "__shfl_down_sync", self._shfl_down_sync)
        setattr(self, "__shfl_xor_sync", self._shfl_xor_sync)
        setattr(self, "__activemask", self._activemask)

    @staticmethod
    def _shfl_sync(mask, var, lane, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_sync", mask, var, lane, width)

    @staticmethod
    def _shfl_up_sync(mask, var, delta, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_up_sync", mask, var, delta, width)

    @staticmethod
    def _shfl_down_sync(mask, var, delta, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_down_sync", mask, var, delta, width)

    @staticmethod
    def _shfl_xor_sync(mask, var, lane_mask, width):
        if isinstance(var, Buffer):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_xor_sync", mask, var, lane_mask, width)

    @staticmethod
    def _activemask():
        return _tir_op.call_intrin("uint32", "tirx.cuda.__activemask")


class NVSHMEMNamespace:
    """The NVSHMEM intrinsics submodule."""

    def __init__(self):
        self.my_pe = _op_wrapper(_cuda_op.nvshmem_my_pe)
        self.n_pes = _op_wrapper(_cuda_op.nvshmem_n_pes)
        self.signal_op = _op_wrapper(_cuda_op.nvshmem_signal_op)
        self.wait_until = _op_wrapper(_cuda_op.nvshmem_wait_until)
        self.quiet = _op_wrapper(_cuda_op.nvshmem_quiet)
        self.fence = _op_wrapper(_cuda_op.nvshmem_fence)
        self.barrier_all = _op_wrapper(_cuda_op.nvshmem_barrier_all)
        self.getmem_nbi = NVSHMEMGetMemNBINamespace()
        self.putmem_nbi = NVSHMEMPutMemNBINamespace()
        self.putmem_signal_nbi = NVSHMEMPutMemSignalNBINamespace()


class NVSHMEMGetMemNBINamespace:
    """The NVSHMEM GetMemNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _op_wrapper(_cuda_op.nvshmem_getmem_nbi_warp)
        self.block = _op_wrapper(_cuda_op.nvshmem_getmem_nbi_block)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_cuda_op.nvshmem_getmem_nbi)(*args, **kwds)

    # __call__ corresponds to nvshmem_getmem_nbi
    __tir_call_op_name__ = "nvshmem_getmem_nbi"


class NVSHMEMPutMemNBINamespace:
    """The NVSHMEM PutMemNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _op_wrapper(_cuda_op.nvshmem_putmem_nbi_warp)
        self.block = _op_wrapper(_cuda_op.nvshmem_putmem_nbi_block)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_cuda_op.nvshmem_putmem_nbi)(*args, **kwds)

    # __call__ corresponds to nvshmem_putmem_nbi
    __tir_call_op_name__ = "nvshmem_putmem_nbi"


class NVSHMEMPutMemSignalNBINamespace:
    """The NVSHMEM PutMemSignalNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _op_wrapper(_cuda_op.nvshmem_putmem_signal_nbi_warp)
        self.block = _op_wrapper(_cuda_op.nvshmem_putmem_signal_nbi_block)

    def __call__(self, *args, **kwds):
        return _op_wrapper(_cuda_op.nvshmem_putmem_signal_nbi)(*args, **kwds)

    # __call__ corresponds to nvshmem_putmem_signal_nbi
    __tir_call_op_name__ = "nvshmem_putmem_signal_nbi"


__all__ = ["CUDANamespace", "NVSHMEMNamespace", "PTXNamespace"]
