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
# pylint: disable=invalid-name, too-many-locals, too-many-function-args
# pylint: disable=too-many-statements, unused-argument, too-many-arguments
"""Tensorcore template for cuda backend"""
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from ..util import get_const_tuple, traverse_inline, simplify
from ..nn.pad import pad
from ..nn.util import get_pad_tuple3d
from .tensor_intrin import intrin_wmma_load_matrix_A
from .tensor_intrin import intrin_wmma_load_matrix_W
from .tensor_intrin import intrin_wmma_store_matrix
from .tensor_intrin import intrin_wmma_gemm


def ndhwc_tensorcore_cuda(cfg, Input, Filter, stride, padding, dilation, out_dtype):
    """Compute declaration for conv3d tensorcore function"""
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3

    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_depth, in_height, in_width, in_channel = get_const_tuple(Input.shape)
    kernel_d, kernel_h, kernel_w, _, num_filter = get_const_tuple(Filter.shape)
    assert (batch % 16 == 0 and in_channel % 16 == 0 and num_filter % 16 == 0) or \
               (batch % 8 == 0 and in_channel % 16 == 0 and num_filter % 32 == 0) or \
               (batch % 32 == 0 and in_channel % 16 == 0 and num_filter % 8 == 0), \
               "The shape of (batch, in_channel, num_filter) "\
               "must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32) for now"

    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w))
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_front, pad_top, pad_left, 0]
    pad_after = [0, pad_back, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = te.reduce_axis((0, in_channel), name='rc')
    rz = te.reduce_axis((0, kernel_d), name='rz')
    ry = te.reduce_axis((0, kernel_h), name='ry')
    rx = te.reduce_axis((0, kernel_w), name='rx')
    # convert data type of input feature maps and weights
    TransPaddedInput = te.compute(
        PaddedInput.shape,
        lambda n, d, h, w, c: PaddedInput[n, d, h, w, c].astype('float16'))
    TransFilter = te.compute(
        Filter.shape, lambda d, h, w, i, o: Filter[d, h, w, i, o].astype('float16'))
    Output = te.compute(
        (batch, out_depth, out_height, out_width, out_channel),
        lambda nn, zz, yy, xx, ff: te.sum(
            TransPaddedInput[nn,
                             zz * stride_d + rz * dilation_d,
                             yy * stride_h + ry * dilation_h,
                             xx * stride_w + rx * dilation_w,
                             rc].astype(out_dtype) *
            TransFilter[rz, ry, rx, rc, ff].astype(out_dtype),
            axis=[rz, ry, rx, rc]),
        name="Conv3dOutput", tag="conv3d_ndhwc_tensorcore")
    return Output


def schedule_ndhwc_tensorcore_cuda(cfg, s, Conv):
    """Schedule tensorcore template"""
    kd, kh, kw, ic = s[Conv].op.reduce_axis
    out_dtype = Conv.dtype
    trans_paddata, kernel = s[Conv].op.input_tensors
    in_dtype = trans_paddata.dtype
    batch, _, _, _, _ = get_const_tuple(Conv.shape)
    _, _, _, _, out_channels = get_const_tuple(kernel.shape)
    paddata = s[trans_paddata].op.input_tensors

    # inline the pad and dtype transform
    s[trans_paddata].compute_inline()
    s[kernel].compute_inline()
    s[paddata[0]].compute_inline()

    # Designate the memory hierarchy
    AS = s.cache_read(trans_paddata, 'shared', [Conv])
    WS = s.cache_read(kernel, 'shared', [Conv])
    AF = s.cache_read(AS, 'wmma.matrix_a', [Conv])
    WF = s.cache_read(WS, 'wmma.matrix_b', [Conv])
    ConvF = s.cache_write(Conv, 'wmma.accumulator')

    if Conv.op in s.outputs:
        output = Conv
        ConvS = s.cache_read(ConvF, 'shared', [Conv])
        OL = ConvS
    else:
        output = s.outputs[0].output(0)
        s[Conv].set_scope('shared')
        OL = Conv

    # Schedule for autotvm
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("offset", [0, 8])
    cfg.define_knob("vector_width", [1, 2, 4, 8])

    if (batch % 16 == 0 and out_channels % 16 == 0):
        cfg.define_knob("wmma_m", [16, 8, 32])
    elif (batch % 8 == 0 and out_channels % 32 == 0):
        cfg.define_knob("wmma_m", [8, 16, 32])
    elif (batch % 32 == 0 and out_channels % 8 == 0):
        cfg.define_knob("wmma_m", [32, 16, 8])

    # fallback support
    target = tvm.target.Target.current()
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.target_name, target.model, 'conv3d_ndhwc_tensorcore.cuda')
        cfg.fallback_with_reference_log(ref_log)

    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offset = cfg["offset"].val
    wmma_m = cfg["wmma_m"].val
    vector_width = cfg["vector_width"].val

    wmma_k = 16
    if wmma_m == 16:
        wmma_n = 16
    elif wmma_m == 8:
        wmma_n = 32
    elif wmma_m == 32:
        wmma_n = 8

    warp_size = 32

    block_x = te.thread_axis('blockIdx.x')
    block_y = te.thread_axis('blockIdx.y')
    block_z = te.thread_axis('blockIdx.z')
    thread_x = te.thread_axis('threadIdx.x')
    thread_y = te.thread_axis('threadIdx.y')
    thread_z = te.thread_axis('threadIdx.z')

    # Define the intrin strides
    def get_strides(extents):
        return [np.prod(extents[i:]).tolist() for i in range(len(extents))]

    AS_align = chunk * wmma_k + offset
    WS_align = warp_col_tiles * block_col_warps * wmma_n + offset
    block_factor_n = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
    CS_align = block_factor_o + offset
    AS_strides = get_strides([1, 1, 1, AS_align, 1])
    AL_strides = get_strides([1, 1, 1, wmma_k, 1])
    WS_strides = get_strides([WS_align, 1])
    WL_strides = get_strides([wmma_n * warp_col_tiles, 1])
    CL_strides = get_strides([1, 1, 1, wmma_n * warp_col_tiles, 1])
    CS_strides = get_strides([1, 1, 1, CS_align, 1])

    # Schedule for output
    nc, dc, hc, wc, oc = output.op.axis
    block_k = s[output].fuse(dc, hc, wc)
    s[output].bind(block_k, block_z)
    block_i, nc = s[output].split(nc, factor=block_factor_n)
    block_j, oc = s[output].split(oc, factor=block_factor_o)
    s[output].reorder(block_k, block_i, block_j, nc, oc)
    t = s[output].fuse(nc, oc)
    t, ti = s[output].split(t, factor=vector_width)
    t, tx = s[output].split(t, factor=warp_size)
    t, ty = s[output].split(t, factor=block_row_warps)
    t, tz = s[output].split(t, factor=block_col_warps)
    s[output].bind(block_i, block_x)
    s[output].bind(block_j, block_y)
    s[output].bind(tz, thread_z)
    s[output].bind(ty, thread_y)
    s[output].bind(tx, thread_x)
    s[output].vectorize(ti)

    # Schedule wmma store
    s[OL].compute_at(s[output], block_j)
    nc, dc, hc, wc, oc = OL.op.axis
    s[OL].reorder(dc, hc, wc, nc, oc)
    s[OL].storage_align(wc, CS_align - 1, CS_align)
    oc, ooc = s[OL].split(oc, factor=wmma_n)
    oc, oci = s[OL].split(oc, factor=warp_col_tiles)
    _, oc = s[OL].split(oc, factor=block_col_warps)
    nc, nnc = s[OL].split(nc, factor=wmma_m)
    nc, nci = s[OL].split(nc, factor=warp_row_tiles)
    _, nc = s[OL].split(nc, factor=block_row_warps)
    s[OL].reorder(nc, oc, nci, oci, nnc, ooc)
    s[OL].bind(nc, thread_y)
    s[OL].bind(oc, thread_z)

    # Schedule wmma computation
    s[ConvF].compute_at(s[OL], oc)
    n, d, h, w, o = ConvF.op.axis
    n, nnf = s[ConvF].split(n, factor=wmma_m)
    o, oof = s[ConvF].split(o, factor=wmma_n)
    ic, ii = s[ConvF].split(ic, factor=wmma_k)
    ko, ki = s[ConvF].split(ic, factor=chunk)
    s[ConvF].reorder(kd, kh, kw, ko, ki, n, o, nnf, oof, ii)

    s[AF].compute_at(s[ConvF], ki)
    s[WF].compute_at(s[ConvF], ki)

    # Schedule wmma load
    n, d, h, w, i = AF.op.axis
    n, nn = s[AF].split(n, factor=wmma_m)
    i, ii = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(n, i, nn, ii)

    kd, kh, kw, i, o = WF.op.axis
    i, ii = s[WF].split(i, factor=wmma_k)
    o, oo = s[WF].split(o, factor=wmma_n)
    s[WF].reorder(o, i, oo)
    s[WF].reorder(i, o, ii, oo)

    s[WS].compute_at(s[ConvF], ko)
    s[AS].compute_at(s[ConvF], ko)

    # Schedule for data's share memory
    n, d, h, w, i = AS.op.axis
    s[AS].reorder(d, h, w, n, i)
    s[AS].storage_align(w, AS_align - 1, AS_align)
    t = s[AS].fuse(n, i)
    t, ti = s[AS].split(t, factor=vector_width)
    t, tx = s[AS].split(t, factor=warp_size)
    t, ty = s[AS].split(t, factor=block_row_warps)
    _, tz = s[AS].split(t, factor=block_col_warps)
    s[AS].bind(ty, thread_y)
    s[AS].bind(tz, thread_z)
    s[AS].bind(tx, thread_x)
    s[AS].vectorize(ti)

    # Schedule for kernel's share memory
    kd, kh, kw, ic, o = WS.op.axis
    t = s[WS].fuse(ic, o)
    s[WS].storage_align(ic, WS_align - 1, WS_align)
    t, ti = s[WS].split(t, factor=vector_width)
    t, tx = s[WS].split(t, factor=warp_size)
    t, ty = s[WS].split(t, factor=block_row_warps)
    _, tz = s[WS].split(t, factor=block_col_warps)
    s[WS].bind(ty, thread_y)
    s[WS].bind(tz, thread_z)
    s[WS].bind(tx, thread_x)
    s[WS].vectorize(ti)

    shape = (wmma_m, wmma_n, wmma_k)

    # tensorize the wmma process
    AS_shape = (wmma_m, 1, 1, 1, wmma_k)
    AL_shape = (wmma_m, 1, 1, 1, wmma_k)
    WS_shape = (wmma_k, wmma_n)
    WL_shape = (wmma_k, wmma_n)
    CL_shape = (wmma_m, 1, 1, 1, wmma_n)
    CS_shape = (wmma_m, 1, 1, 1, wmma_n)

    AL_gemm = te.placeholder(AL_shape, name='A', dtype=in_dtype)
    WL_gemm = te.placeholder(WL_shape, name='B', dtype=in_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k")
    CL_compute = te.compute(CL_shape, lambda ii, t0, t1, t2, jj:
                            te.sum(AL_gemm[ii, t0, t1, t2, k_gemm].astype(out_dtype) * \
                                   WL_gemm[k_gemm, jj].astype(out_dtype), axis=k_gemm),
                            name='C')

    s[AF].tensorize(nn, intrin_wmma_load_matrix_A(AL_strides, AS_strides, shape,
                                                  "row_major", AS_shape, AL_shape, in_dtype))
    s[WF].tensorize(ii, intrin_wmma_load_matrix_W(WL_strides, WS_strides, shape,
                                                  "row_major", WS_shape, WL_shape, in_dtype))
    s[OL].tensorize(nnc, intrin_wmma_store_matrix(CS_strides, CL_strides,
                                                  shape, out_dtype, CL_shape, CS_shape))
    s[ConvF].tensorize(nnf, intrin_wmma_gemm(AL_gemm, WL_gemm, CL_compute, AL_strides,
                                             WL_strides, CL_strides, shape))

    N, OD, OH, OW, CO = get_const_tuple(output.shape)
    KD, KH, KW, CI, _ = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OD * OH * OW * CO * CI * KD * KH * KW)


@autotvm.register_topi_compute("conv3d_ndhwc_tensorcore.cuda")
def conv3d_ndhwc_tensorcore(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv3d with tensorcore for NDHWC layout"""
    return ndhwc_tensorcore_cuda(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv3d_ndhwc_tensorcore.cuda")
def schedule_conv3d_ndhwc_tensorcore(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv3d_ndhwc_tensorcore' in op.tag:
            schedule_ndhwc_tensorcore_cuda(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s
