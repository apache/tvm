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

from tvm import ir as _ir
from tvm.backend.cuda import op as _cuda_op
from tvm.tirx import is_buffer_var
from tvm.tirx import op as _tir_op
from tvm.tirx.script.ir_builder.op import _dtype_forward, _op_wrapper

# pylint: disable=protected-access


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
                ty=tvm.ir.PrimType(elem_dtype),
            )
        raise TypeError(
            "T.s_tir.cp_async_raw only accepts the printed 6-arg raw form; "
            'issue new copies through T.ptx["cp.async..."]'
        )


class STIRNamespace:
    """Nodes the s_tir pipeline's own passes build.

    Nothing here is meant to be written by hand: ``InjectPTXAsyncCopy``
    constructs these, later passes match on them, and
    codegen turns them into asm. They have a script spelling only so printed
    IR round-trips.
    """

    def __init__(self):
        self.cp_async_raw = _CpAsyncRaw()


class PTXLegacyNamespace:
    """Apache-compatible spellings of instructions the dialect already covers.

    They take the historical argument order and are pattern-matched by the
    passes that lower them, which is why they are not simply deleted: tests
    inherited from upstream still write them.
    """

    def __init__(self):
        # Same lowered asm as T.ptx.mma, but the accumulator doubles as the
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


def _shfl_sync(mask, var, lane, width):
    if is_buffer_var(var):
        var = var[0]
    return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_sync", mask, var, lane, width)


def _shfl_up_sync(mask, var, delta, width):
    if is_buffer_var(var):
        var = var[0]
    return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_up_sync", mask, var, delta, width)


def _shfl_down_sync(mask, var, delta, width):
    if is_buffer_var(var):
        var = var[0]
    return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_down_sync", mask, var, delta, width)


def _shfl_xor_sync(mask, var, lane_mask, width):
    if is_buffer_var(var):
        var = var[0]
    return _tir_op.call_intrin(var.ty, "tirx.cuda.__shfl_xor_sync", mask, var, lane_mask, width)


def _activemask():
    return _tir_op.call_intrin("uint32", "tirx.cuda.__activemask")


iket = IketNamespace()
wgmma = CudaWgmmaNamespace()
tcgen05 = CudaTcgen05Namespace()
mov_sreg: Callable[..., Any] = _cuda_op.cuda_mov_sreg
wait_until = _cuda_op.cuda_wait_until
atomic_add = _cuda_op.cuda_atomic_add
warp_reduce = _cuda_op.cuda_warp_reduce
warp_sum = _cuda_op.cuda_warp_sum
warp_max = _cuda_op.cuda_warp_max
warp_min = _cuda_op.cuda_warp_min
cta_reduce = _cuda_op.cuda_cta_reduce
cta_sum = _cuda_op.cuda_cta_sum
cta_max = _cuda_op.cuda_cta_max
cta_min = _cuda_op.cuda_cta_min
atomic_cas = _cuda_op.cuda_atomic_cas
func_call = _cuda_op.cuda_func_call
ldg = _cuda_op.cuda_ldg
sm100_2sm_leader_smem_addr_composed = _cuda_op.cuda_sm100_2sm_leader_smem_addr
timer_init = _cuda_op.timer_init_cuda
timer_start = _cuda_op.timer_start_cuda
timer_end = _cuda_op.timer_end_cuda
timer_finalize = _cuda_op.timer_finalize_cuda
mma_store = _dtype_forward(_cuda_op.mma_store)
mma_fill = _dtype_forward(_cuda_op.mma_fill)
mma_store_legacy = _dtype_forward(_cuda_op.mma_store_legacy)
mma_fill_legacy = _dtype_forward(_cuda_op.mma_fill_legacy)
mov_sreg.__tvm_op__ = _ir.Op.get("tirx.cuda.mov_sreg")
wait_until.__tvm_op__ = _ir.Op.get("tirx.cuda.wait_until")
atomic_add.__tvm_op__ = _ir.Op.get("tirx.cuda.atomic_add")
warp_reduce.__tvm_op__ = _ir.Op.get("tirx.cuda.warp_reduce")
cta_reduce.__tvm_op__ = _ir.Op.get("tirx.cuda.cta_reduce")
atomic_cas.__tvm_op__ = _ir.Op.get("tirx.cuda.atomic_cas")
func_call.__tvm_op__ = _ir.Op.get("tirx.cuda.func_call")
ldg.__tvm_op__ = _ir.Op.get("tirx.cuda.ldg")
__shfl_sync = _shfl_sync
__shfl_sync.__tvm_op__ = _ir.Op.get("tirx.cuda.__shfl_sync")
__shfl_up_sync = _shfl_up_sync
__shfl_up_sync.__tvm_op__ = _ir.Op.get("tirx.cuda.__shfl_up_sync")
__shfl_down_sync = _shfl_down_sync
__shfl_down_sync.__tvm_op__ = _ir.Op.get("tirx.cuda.__shfl_down_sync")
__shfl_xor_sync = _shfl_xor_sync
__shfl_xor_sync.__tvm_op__ = _ir.Op.get("tirx.cuda.__shfl_xor_sync")
__activemask = _activemask
__activemask.__tvm_op__ = _ir.Op.get("tirx.cuda.__activemask")

_ir.op._init_op_api("tirx.cuda", __name__)


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
