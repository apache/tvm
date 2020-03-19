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
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Compute and Schedule definition for dense tensorcore with cuda backend"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
import tvm.autotvm as autotvm
from .. import tag
from ..util import traverse_inline, get_const_tuple
from .tensor_intrin import intrin_wmma_load_matrix_A, \
        intrin_wmma_load_matrix_W, intrin_wmma_store_matrix


@autotvm.register_topi_compute("dense_tensorcore.cuda")
def dense_tensorcore(cfg, data, weight, bias=None, out_dtype=None):
    """Dense tensorcore operator on CUDA"""
    matmul = dense_tensorcore_cuda(data, weight, bias, out_dtype)
    return matmul


@autotvm.register_topi_schedule("dense_tensorcore.cuda")
def schedule_dense_tensorcore(cfg, outs):
    """Schedule dense operator using Tensorcore"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == 'dense_tensorcore':
            _schedule_dense_tensorcore(cfg, s, op.output(0))
    traverse_inline(s, outs[0].op, _callback)
    return s


def dense_tensorcore_cuda(data, weight, bias=None, out_dtype=None):
    """Dense tensorcore operator on CUDA"""
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    k = te.reduce_axis((0, in_dim), name='k')
    data_16 = te.compute((batch, in_dim), lambda b, i: data[b, i].astype('float16'))
    weight_16 = te.compute((out_dim, in_dim), lambda o, i: weight[o, i].astype('float16'))
    matmul = te.compute((batch, out_dim), \
                         lambda i, j: te.sum(data_16[i, k].astype(out_dtype) * \
                                              weight_16[j, k].astype(out_dtype), axis=k), \
                         name='T_dense', tag='dense_tensorcore')
    if bias is not None:
        matmul = te.compute((batch, out_dim), \
                             lambda i, j: matmul[i, j] + bias[j].astype(out_dtype), \
                             tag=tag.BROADCAST)
    return matmul


def _schedule_dense_tensorcore(cfg, s, C):
    """Schedule dense operator using Tensorcore"""
    A, B = s[C].op.input_tensors
    batch, out_dim = get_const_tuple(C.shape)
    out_dtype = C.dtype
    s[A].compute_inline()
    s[B].compute_inline()

    # Explicit memory access
    AS = s.cache_read(A, 'shared', [C])
    BS = s.cache_read(B, 'shared', [C])
    AF = s.cache_read(AS, 'wmma.matrix_a', [C])
    BF = s.cache_read(BS, 'wmma.matrix_b', [C])
    CF = s.cache_write(C, 'wmma.accumulator')
    CS = s.cache_read(CF, 'shared', [C])

    # fallback support
    target = tvm.target.Target.current()
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.target_name, target.model, 'dense_tensorcore.cuda')
        cfg.fallback_with_reference_log(ref_log)

    # Deal with op fusion, such as bias and relu
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # create tuning space
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("offset", [0, 8])
    cfg.define_knob("offsetCS", [0, 8])
    cfg.define_knob("vec", [1, 2, 4, 8])

    #Make it available by default
    if (batch % 32 == 0 and out_dim % 8 == 0):
        cfg.define_knob("wmma_m", [32, 16, 8])
    elif (batch%16 == 0 and out_dim % 16 == 0):
        cfg.define_knob("wmma_m", [16, 8, 32])
    elif (batch % 8 == 0 and out_dim % 32 == 0):
        cfg.define_knob("wmma_m", [8, 16, 32])

    warp_size = 32
    wmma_k = 16
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offset = cfg["offset"].val
    offsetCS = cfg["offsetCS"].val
    wmma_m = cfg["wmma_m"].val
    vec = cfg["vec"].val

    if wmma_m == 16:
        wmma_n = 16
    elif wmma_m == 8:
        wmma_n = 32
    elif wmma_m == 32:
        wmma_n = 8

    #Define the stride of intrin functions
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    thread_z = te.thread_axis('threadIdx.z')

    #Schedule for dense computation
    block_factor_b = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
    b, o = C.op.axis
    block_i, bc = s[C].split(b, factor=block_factor_b)
    block_j, oc = s[C].split(o, factor=block_factor_o)
    s[C].reorder(block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    #Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)

    #Schedule for wmma computation
    s[CF].compute_at(s[CS], oo)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    k, = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    #Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    #Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(o, i, o_ii, i_ii)

    #Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    shape = (wmma_m, wmma_n, wmma_k)

    #lower the computation loops down to TensorCore hardware intrinsics
    #by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(b_ii, intrin_wmma_load_matrix_A( \
            AF_stride, AS_stride, shape, "row_major", (wmma_m, wmma_k), (wmma_m, wmma_k)))
    s[BF].tensorize(o_ii, intrin_wmma_load_matrix_W( \
            BF_stride, BS_stride, shape, "col_major", (wmma_n, wmma_k), (wmma_n, wmma_k)))
    s[CF].tensorize(_ii, intrin_wmma_gemm( \
            AF_stride, BF_stride, CF_stride, shape, out_dtype))
    s[CS].tensorize(bbi, intrin_wmma_store_matrix( \
            CS_stride, CF_stride, shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)))


def intrin_wmma_gemm(strides_A, strides_B, strides_C, shape, out_dtype):
    """Intrin function of wmma::mma.sync"""
    wmma_m, wmma_n, wmma_k = shape
    n = 16
    A = te.placeholder((wmma_m, wmma_k), name='A', dtype='float16')
    B = te.placeholder((wmma_n, wmma_k), name='B', dtype='float16')
    k = te.reduce_axis((0, wmma_k), name="k")
    C = te.compute((wmma_m, wmma_n),
                   lambda ii, jj:
                   te.sum(A[ii, k].astype(out_dtype) * B[jj, k].astype(out_dtype), axis=k),
                   name='C')
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, name='BA',
                             scope='wmma.matrix_a', data_alignment=32,
                             offset_factor=n, strides=strides_A)
    BB = tvm.tir.decl_buffer(B.shape, B.dtype, name='BB',
                             scope='wmma.matrix_b', data_alignment=32,
                             offset_factor=n, strides=strides_B)
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, name='BC',
                             scope='wmma.accumulator', data_alignment=32,
                             offset_factor=n, strides=strides_C)

    def intrin_func(ins, outs):
        BA, BB = ins
        BC, = outs

        def warp_index(offset, rows, cols):
            row = rows * cols
            return offset // row + offset % row // cols

        warp_index_A = warp_index(BA.elem_offset, wmma_m, wmma_k)
        warp_index_B = warp_index(BB.elem_offset, wmma_k, wmma_n)
        warp_index_C = warp_index(BC.elem_offset, wmma_m, wmma_n)

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_intrin('handle', 'tvm_fill_fragment', \
                    BC.data, wmma_m, wmma_n, wmma_k, warp_index_C, 0.0))
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_intrin('handle', 'tvm_mma_sync',
                                        BC.data, warp_index_C,
                                        BA.data, warp_index_A,
                                        BB.data, warp_index_B,
                                        BC.data, warp_index_C))
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})
