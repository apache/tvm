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
from tvm.tirx import is_buffer_var
from tvm.tirx import op as _tir_op
from tvm.tirx.script.builder.ir import _dtype_forward, _op_wrapper

# pylint: disable=protected-access


def _s_tir_ldg32(reg, guard, addr, local_addr):
    if is_buffer_var(addr):
        addr = addr[0]
    return _tir_op.call_intrin(reg.ty, "tirx.s_tir.ldg32", reg, guard, addr, local_addr)


_s_tir_ldg32.__tir_op_name__ = "s_tir.ldg32"


class _CpAsyncRaw:
    """The raw cp.async node's printer surface."""

    def __init__(self):
        # Legacy variant: takes (dst_ptr, dst_offset, src_ptr, src_offset,
        # cp_size). Offsets are folded into the pointers; lowers through the
        # raw op below.
        self.legacy = _dtype_forward(_cuda_op.ptx_cp_async_legacy)

    def __call__(self, *args, **kwds):
        # The 6-arg form ``(elem_dtype, dst, dst_off, src, src_off, cp_size)``
        # the printer round-trips for the raw ``tirx.s_tir.cp_async_raw`` Call
        # emitted by ``tvm.backend.cuda.transform.InjectPTXAsyncCopy``. The
        # pass-emitted Call has 5 args (no ``tvm_access_ptr`` fold) and a
        # per-element-dtype Call.dtype, so build it directly.
        if len(args) == 6 and isinstance(args[0], str) and "dtype" not in kwds:
            import tvm

            elem_dtype, dst, dst_off, src, src_off, cp_size = args
            return tvm.ir.Call(
                tvm.ir.Op.get("tirx.s_tir.cp_async_raw"),
                [dst, dst_off, src, src_off, cp_size],
                ret_ty=tvm.ir.PrimType(elem_dtype),
            )
        raise TypeError(
            "T.s_tir.cp_async_raw only accepts the printed 6-arg raw form; "
            'issue new copies through T.ptxd["cp.async..."]'
        )


class STIRNamespace:
    """Nodes the s_tir pipeline's own passes build.

    Nothing here is meant to be written by hand: ``InjectPTXLDG32`` and
    ``InjectPTXAsyncCopy`` construct these, later passes match on them, and
    codegen turns them into asm. They have a script spelling only so printed
    IR round-trips.
    """

    def __init__(self):
        self.ldg32 = _s_tir_ldg32
        self.cp_async_raw = _CpAsyncRaw()


class PTXLegacyNamespace:
    """Apache-compatible spellings of instructions the dialect already covers.

    They take the historical argument order and are pattern-matched by the
    passes that lower them, which is why they are not simply deleted: tests
    inherited from upstream still write them.
    """

    def __init__(self):
        # Same lowered asm as T.ptxd.mma, but the accumulator doubles as the
        # destination and the offsets are explicit.
        self.mma = _dtype_forward(_cuda_op.ptx_legacy_mma)
        # (trans, num, dtype, local_ptr, local_offset, smem_ptr, smem_offset)
        self.ldmatrix = _dtype_forward(_cuda_op.ptx_legacy_ldmatrix)


class CudaWgmmaNamespace:
    """WGMMA companions that are not PTX instructions (pure-C / empty-asm helpers)."""

    def __init__(self):
        self.noop_barrier = _op_wrapper(_cuda_op.cuda_wgmma_noop_barrier)
        self.encode_matrix_descriptor = _op_wrapper(_cuda_op.cuda_wgmma_encode_matrix_descriptor)


class CudaTcgen05Namespace:
    """tcgen05 companions that are not PTX instructions (pure-C descriptor packers)."""

    def __init__(self):
        self.encode_matrix_descriptor = _op_wrapper(_cuda_op.cuda_tcgen05_encode_matrix_descriptor)
        self.encode_instr_descriptor = _op_wrapper(_cuda_op.cuda_tcgen05_encode_instr_descriptor)
        self.encode_instr_descriptor_block_scaled = _op_wrapper(
            _cuda_op.cuda_tcgen05_encode_instr_descriptor_block_scaled
        )


class IketNamespace:
    """Frontend-only NVIDIA IKET annotations."""

    def __init__(self):
        self.mark = _op_wrapper(_cuda_op.cuda_iket_mark)
        self.range_start = _op_wrapper(_cuda_op.cuda_iket_range_start)
        self.range_end = _op_wrapper(_cuda_op.cuda_iket_range_end)
        self.range_push = _op_wrapper(_cuda_op.cuda_iket_range_push)
        self.range_pop = _op_wrapper(_cuda_op.cuda_iket_range_pop)
        self.sentinel_token = _op_wrapper(_cuda_op.cuda_iket_sentinel_token)
        self.official_event = _op_wrapper(_cuda_op.cuda_iket_official_event)


class CUDANamespace:
    """The CUDA intrinsics submodule."""

    def __init__(self):
        self.iket = IketNamespace()
        self.wgmma = CudaWgmmaNamespace()
        self.tcgen05 = CudaTcgen05Namespace()
        self.any_sync = _op_wrapper(_cuda_op.cuda_any_sync)
        # elect.sync plus the two movs that materialize its d|p pair: a
        # multi-statement asm block, so it belongs here rather than T.ptxd.
        # The warp-specialization passes match this op to build predicates.
        self.elect_sync: Callable[..., Any] = _op_wrapper(_cuda_op.cuda_elect_sync)
        # `mov.u32 d, %sreg` -- one PTX instruction, but the special-register
        # name is baked into the asm text, so it is a helper per register
        # rather than a ptxd entry with a register operand.
        self.mov_sreg: Callable[..., Any] = _op_wrapper(_cuda_op.cuda_mov_sreg)
        # Spin-until-ready mbarrier waits: label-loop asm blocks, not single
        # PTX instructions -- which is why they live here and not in T.ptxd.
        self.mbarrier_wait = _op_wrapper(_cuda_op.cuda_mbarrier_wait)
        self.mbarrier_wait_acquire_cluster = _op_wrapper(
            _cuda_op.cuda_mbarrier_wait_acquire_cluster
        )
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
        self.fdividef = _op_wrapper(_cuda_op.cuda_fdividef)
        self.get_tmem_addr = _op_wrapper(_cuda_op.cuda_get_tmem_addr)
        self.cvta_generic_to_shared = _op_wrapper(_cuda_op.cuda_cvta_generic_to_shared)
        self.smem_addr_from_uint64 = _op_wrapper(_cuda_op.cuda_smem_addr_from_uint64)
        self.sm100_2sm_leader_smem_addr = _op_wrapper(_cuda_op.cuda_sm100_2sm_leader_smem_addr)
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
        if is_buffer_var(var):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_sync", mask, var, lane, width)

    @staticmethod
    def _shfl_up_sync(mask, var, delta, width):
        if is_buffer_var(var):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_up_sync", mask, var, delta, width)

    @staticmethod
    def _shfl_down_sync(mask, var, delta, width):
        if is_buffer_var(var):
            var = var[0]
        return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_down_sync", mask, var, delta, width)

    @staticmethod
    def _shfl_xor_sync(mask, var, lane_mask, width):
        if is_buffer_var(var):
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


__all__ = ["CUDANamespace", "NVSHMEMNamespace", "PTXLegacyNamespace", "STIRNamespace"]
