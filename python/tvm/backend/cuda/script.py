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

from tvm import ir as _ir
from tvm.backend.cuda import op as _cuda_op

# pylint: disable=protected-access


class _CpAsyncRaw:
    """The raw cp.async node's printer surface."""

    def __init__(self):
        # Legacy variant: takes (dst_ptr, dst_offset, src_ptr, src_offset,
        # cp_size). Offsets are folded into the pointers; lowers through the
        # raw op below.
        self.legacy = _cuda_op.ptx_cp_async_legacy
        op = _ir.Op.get("tirx.s_tir.cp_async_raw")
        self._construct = _ir.op._make_op_api(op, __name__)

    def __call__(self, *args, **kwds):
        return self._construct(*args, **kwds)


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
        self.mma = _cuda_op.ptx_legacy_mma
        # (trans, num, dtype, local_ptr, local_offset, smem_ptr, smem_offset)
        self.ldmatrix = _cuda_op.ptx_legacy_ldmatrix


class CudaWgmmaNamespace:
    """WGMMA companions that are not PTX instructions (pure-C / empty-asm helpers)."""

    def __init__(self):
        self.noop_barrier = _cuda_op.cuda_wgmma_noop_barrier
        self.encode_matrix_descriptor = _cuda_op.cuda_wgmma_encode_matrix_descriptor


class CudaTcgen05Namespace:
    """tcgen05 companions that are not PTX instructions (pure-C descriptor packers)."""

    def __init__(self):
        self.encode_matrix_descriptor = _cuda_op.cuda_tcgen05_encode_matrix_descriptor
        self.encode_instr_descriptor = _ir.op._make_op_api(
            _ir.Op.get("tirx.cuda.tcgen05_encode_instr_descriptor"), __name__
        )
        self.encode_instr_descriptor_block_scaled = _ir.op._make_op_api(
            _ir.Op.get("tirx.cuda.tcgen05_encode_instr_descriptor_block_scaled"), __name__
        )


class IketNamespace:
    """Frontend-only NVIDIA IKET annotations."""

    def __init__(self):
        self.mark = _cuda_op.cuda_iket_mark
        self.range_start = _cuda_op.cuda_iket_range_start
        self.range_end = _cuda_op.cuda_iket_range_end
        self.range_push = _cuda_op.cuda_iket_range_push
        self.range_pop = _cuda_op.cuda_iket_range_pop
        self.sentinel_token = _cuda_op.cuda_iket_sentinel_token
        self.official_event = _cuda_op.cuda_iket_official_event


TensorMapEncodeTiledAttr = _cuda_op.TensorMapEncodeTiledAttr
TCGen05InstrDescriptorAttrs = _cuda_op.TCGen05InstrDescriptorAttrs
TCGen05InstrDescriptorBlockScaledAttrs = _cuda_op.TCGen05InstrDescriptorBlockScaledAttrs

iket = IketNamespace()
wgmma = CudaWgmmaNamespace()
tcgen05 = CudaTcgen05Namespace()
warp_sum = _cuda_op.cuda_warp_sum
warp_max = _cuda_op.cuda_warp_max
warp_min = _cuda_op.cuda_warp_min
cta_sum = _cuda_op.cuda_cta_sum
cta_max = _cuda_op.cuda_cta_max
cta_min = _cuda_op.cuda_cta_min
func_call = _cuda_op.cuda_func_call
sm100_2sm_leader_smem_addr = _cuda_op.cuda_sm100_2sm_leader_smem_addr
timer_init = _cuda_op.timer_init_cuda
timer_start = _cuda_op.timer_start_cuda
timer_end = _cuda_op.timer_end_cuda
timer_finalize = _cuda_op.timer_finalize_cuda
mma_store = _ir.op._make_op_api(_ir.Op.get("tirx.mma_store"), __name__)
mma_fill = _ir.op._make_op_api(_ir.Op.get("tirx.mma_fill"), __name__)
bmma_sync = _cuda_op.bmma_sync
mma_store_legacy = _cuda_op.mma_store_legacy
mma_fill_legacy = _cuda_op.mma_fill_legacy
atomic_add = _cuda_op.cuda_atomic_add
atomic_cas = _cuda_op.cuda_atomic_cas
cta_reduce = _cuda_op.cuda_cta_reduce
ldg = _cuda_op.cuda_ldg
mov_sreg = _cuda_op.cuda_mov_sreg
wait_until = _cuda_op.cuda_wait_until
warp_reduce = _cuda_op.cuda_warp_reduce
__shfl_sync = _cuda_op.__shfl_sync
__shfl_up_sync = _cuda_op.__shfl_up_sync
__shfl_down_sync = _cuda_op.__shfl_down_sync
__shfl_xor_sync = _cuda_op.__shfl_xor_sync
__activemask = _cuda_op.__activemask
_ir.op._init_op_api("tirx.cuda", __name__)


class NVSHMEMNamespace:
    """The NVSHMEM intrinsics submodule."""

    def __init__(self):
        self.my_pe = _cuda_op.nvshmem_my_pe
        self.n_pes = _cuda_op.nvshmem_n_pes
        self.signal_op = _cuda_op.nvshmem_signal_op
        self.wait_until = _cuda_op.nvshmem_wait_until
        self.quiet = _cuda_op.nvshmem_quiet
        self.fence = _cuda_op.nvshmem_fence
        self.barrier_all = _cuda_op.nvshmem_barrier_all
        self.getmem_nbi = NVSHMEMGetMemNBINamespace()
        self.putmem_nbi = NVSHMEMPutMemNBINamespace()
        self.putmem_signal_nbi = NVSHMEMPutMemSignalNBINamespace()


class NVSHMEMGetMemNBINamespace:
    """The NVSHMEM GetMemNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _cuda_op.nvshmem_getmem_nbi_warp
        self.block = _cuda_op.nvshmem_getmem_nbi_block

    def __call__(self, *args, **kwds):
        return _cuda_op.nvshmem_getmem_nbi(*args, **kwds)

    # __call__ corresponds to nvshmem_getmem_nbi


class NVSHMEMPutMemNBINamespace:
    """The NVSHMEM PutMemNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _cuda_op.nvshmem_putmem_nbi_warp
        self.block = _cuda_op.nvshmem_putmem_nbi_block

    def __call__(self, *args, **kwds):
        return _cuda_op.nvshmem_putmem_nbi(*args, **kwds)

    # __call__ corresponds to nvshmem_putmem_nbi


class NVSHMEMPutMemSignalNBINamespace:
    """The NVSHMEM PutMemSignalNBI intrinsics submodule."""

    def __init__(self):
        self.warp = _cuda_op.nvshmem_putmem_signal_nbi_warp
        self.block = _cuda_op.nvshmem_putmem_signal_nbi_block

    def __call__(self, *args, **kwds):
        return _cuda_op.nvshmem_putmem_signal_nbi(*args, **kwds)

    # __call__ corresponds to nvshmem_putmem_signal_nbi
