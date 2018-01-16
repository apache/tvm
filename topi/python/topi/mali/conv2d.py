"""conv2d schedule on ARM Mali GPU"""

from __future__ import absolute_import as _abs
import tvm

from .. import generic
from .. import util
from .. import tag
from ..nn import pad
from ..nn.conv2d import conv2d
from ..nn.util import get_pad_tuple

##### SCHEDULE UTILITIES #####
def fuse_and_bind(s, tensor, axis=None, num_thread=None):
    """ fuse all the axis and bind to GPU threads """
    axis = axis or s[tensor].op.axis
    fused = s[tensor].fuse(*axis)
    max_threads = tvm.target.current_target(allow_none=False).max_num_threads
    bx, tx = s[tensor].split(fused, num_thread or max_threads)
    s[tensor].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(tx, tvm.thread_axis("threadIdx.x"))
    return bx, tx

def tile_and_bind(s, tensor, y, x, y_factor, x_factor=None):
    """ tile and bind to GPU threads """
    x_factor= x_factor or y_factor
    yo, xo, yi, xi = s[tensor].tile(y, x, y_factor, x_factor)
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    return yo, xo, yi, xi

def tile_and_bind3d(s, tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
    """ tile and bind 3d """
    y_factor = y_factor or z_factor
    x_factor = x_factor or y_factor
    zo, zi = s[tensor].split(z, z_factor)
    yo, yi = s[tensor].split(y, y_factor)
    xo, xi = s[tensor].split(x, x_factor)
    s[tensor].bind(zo, tvm.thread_axis("blockIdx.z"))
    s[tensor].bind(zi, tvm.thread_axis("threadIdx.z"))
    s[tensor].bind(yo, tvm.thread_axis("blockIdx.y"))
    s[tensor].bind(yi, tvm.thread_axis("threadIdx.y"))
    s[tensor].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[tensor].bind(xi, tvm.thread_axis("threadIdx.x"))

def pack_tensor(s, tensor, factor, readers):
    """ do transform X[n, m] -> X[n / factor, m, factor] """
    tmp = s.cache_read(tensor, 'global', readers)
    y, x = s[tmp].op.axis
    yo, yi = s[tmp].split(y, factor)
    s[tmp].reorder(yo, x, yi)
    s[tmp].compute_inline()
    return s.cache_write(tmp, 'global')

def transpose(s, tensor, readers):
    """ do transform X[n, m] -> X[m, n] """
    tmp = s.cache_read(tensor, 'global', readers)
    y, x = s[tmp].op.axis
    s[tmp].reorder(x, y)
    s[tmp].compute_inline()
    return s.cache_write(tmp, "global"), tmp

@conv2d.register("mali")
def decl_conv2d(input, filter, stride, padding, layout='NCHW', out_dtype='float32'):
    """Conv2D operator for ARM Mali GPU backend.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    filter : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    assert layout == 'NCHW', "only support NCHW convolution on mali"
    assert input.shape[0].value == 1, "only support batch size=1 convolution on mali"
    assert input.dtype == filter.dtype, "Do not support inputs with different data types now."

    out_dtype = input.dtype
    if util.get_const_int(filter.shape[2]) == 1:
        return _decl_im2col(input, filter, stride, padding, layout, out_dtype)
    else:
        return _decl_direct(input, filter, stride, padding, layout, out_dtype)

@generic.schedule_conv2d_nchw.register(["mali"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw for ARM Mali GPU

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(op):
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'im2col_conv_output' in op.tag:
            _schedule_im2col_conv2d(s, op)

        if 'direct_conv_output' in op.tag:
            _schedule_direct_conv2d(s, op)

    traverse(outs[0].op)
    return s

def _decl_direct(data, kernel, stride, padding, layout, out_dtype):
    """declare the direct method (spatial packing) for conv2d"""
    _, CI, IH, IW = [util.get_const_int(x) for x in data.shape]
    CO, _, KH, KW = [util.get_const_int(x) for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)
    HCAT, WCAT = KH - 1, KW - 1

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    N = 1
    TH = IH + 2*HPAD
    TW = IW + 2*WPAD
    OH = (IH + 2*HPAD - KH) // HSTR + 1
    OW = (IW + 2*WPAD - KW) // WSTR + 1

    DO_PAD = (HPAD != 0 and WPAD != 0)
    if DO_PAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    # set tunable parameters (tile factor, ...)
    tune_config = getattr(tvm.target.current_target(), "tune_config", None)
    if tune_config is None:
        VH = 1
        VW, VC = 4, 4
        # correct tile factor
        if OW % VW != 0:
            if OW == 14:
                VW = 2
                VC = 8
            elif OW == 7:
                VW = 7
    else:
        VH = tune_config['VH']
        VW = tune_config['VW']
        VC = tune_config['VC']

    if data.dtype == 'float16':
        VC *= 2

    assert CO % VC == 0
    assert OH % VH == 0, "OH: %d  VH : %d" % (OH, VH)
    assert OW % VW == 0, "OW: %d  VW : %d" % (OW, VW)

    dvshape = (N, TH//(VH*HSTR), TW//(VW*WSTR), CI, VH*HSTR+HCAT, VW*WSTR+WCAT)
    kvshape = (CO // VC, CI, KH, KW, VC)
    ovshape = (N, CO // VC, OH // VH, OW // VW, VH, VW, VC)
    oshape = (N, CO, OH, OW)

    data_vec = tvm.compute(dvshape, lambda n, h, w, ci, vh, vw:
        data_pad[n][ci][h*VH*HSTR+vh][w*VW*WSTR+vw], name='data_vec')

    kernel_vec = tvm.compute(kvshape, lambda co, ci, kh, kw, vc:
        kernel[co*VC+vc][ci][kh][kw], name='kernel_vec')

    ci = tvm.reduce_axis((0, CI), name='ci')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    conv = tvm.compute(ovshape, lambda n, co, h, w, vh, vw, vc:
        tvm.sum(data_vec[n, h, w, ci, vh*HSTR+kh, vw*WSTR+kw].astype(out_dtype) *
                kernel_vec[co, ci, kh, kw, vc].astype(out_dtype),
                axis=[ci, kh, kw]), name='conv')

    output = tvm.compute(oshape, lambda n, co, h, w:
                         conv[n][co//VC][h/VH][w//VW][h%VH][w%VW][co%VC],
                         name='output_unpack', tag='direct_conv_output')

    return output

def _schedule_direct_conv2d(s, op):
    """schedule the direct method (spatial packing) for conv2d"""
    # get ops and tensors
    output = op.output(0)
    output_height = util.get_const_int(output.shape[2])

    conv = op.input_tensors[0]
    data_vec = s[conv].op.input_tensors[0]
    kernel_vec = s[conv].op.input_tensors[1]
    data = s[data_vec].op.input_tensors[0]
    kernel = s[kernel_vec].op.input_tensors[0]

    # set tunable parameters (tile factor, ...)
    tune_config = getattr(tvm.target.current_target(), "tune_config", None)
    if tune_config is None:
        num_thread = 8

        out_channel = util.get_const_int(kernel.shape[0])
        in_channel  = util.get_const_int(kernel.shape[1])
        in_width    = util.get_const_int(data.shape[2])

        if in_width >= 224:
            pass
        elif in_width >= 112:
            pass
        elif in_width >= 56:
            if out_channel != in_channel:
                num_thread = 16
        elif in_width >= 28:
            if out_channel >= 256:
                num_thread = 16
        elif in_width >= 14:
            if in_channel == out_channel:
                num_thread = 8
            else:
                num_thread = 4
    else:
        num_thread = tune_config["num_thread"]

    last = 1
    if output_height == 28:
        last = 7
        num_thread = 32

    if data.dtype == 'float16' and (util.get_const_int(conv.shape[1]) == 4 or output_height == 28):
        num_thread /= 2

    # schedule padding
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
        data = data_pad.op.input_tensors[0]
        s[data_pad].compute_inline()

    # schedule data packing
    _, h, w, ci, vh, vw = s[data_vec].op.axis
    tile_and_bind3d(s, data_vec, h, w, ci, 1)
    s[data_vec].unroll(vw)

    # schedule kernel packing
    co, ci, kh, kw, vc = s[kernel_vec].op.axis
    tile_and_bind(s, kernel_vec, co, ci, 1)
    s[kernel_vec].unroll(kh)
    s[kernel_vec].unroll(kw)
    s[kernel_vec].vectorize(vc)

    # schedule convolution
    _, c, h, w, vh, vw, vc = s[conv].op.axis
    kc, kh, kw = s[conv].op.reduce_axis
    s[conv].reorder(_, c, h, w, vh, kc, kh, kw, vw, vc)
    tile_and_bind3d(s, conv, c, h, w, num_thread, 1, last)
    s[conv].unroll(kh)
    s[conv].unroll(kw)
    s[conv].unroll(vw)
    s[conv].vectorize(vc)

    # schedule output
    if output.op not in s.outputs:  # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    _, co, oh, ow = s[output].op.axis
    tile_and_bind3d(s, output, co, oh, ow, num_thread, 1, last)

    #print(tvm.lower(s, [data, kernel, output], simple_mode=True))

def _decl_im2col(data, kernel, stride, padding, layout='NCHW', out_dtype='float32'):
    """declare the Im2Col method for conv2d"""
    _, CI, IH, IW = [x.value for x in data.shape]
    CO, _, KH, KW = [x.value for x in kernel.shape]
    HPAD, WPAD, _, _ = get_pad_tuple(padding, kernel)

    if isinstance(stride, (tuple, list)):
        HSTR, WSTR = stride
    else:
        HSTR, WSTR = stride, stride

    N = 1
    OH = (IH + 2*HPAD - KH) // HSTR + 1
    OW = (IW + 2*WPAD - KW) // WSTR + 1

    DO_PAD = (HPAD != 0 and WPAD != 0)
    if DO_PAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    ALIGN = 16
    def upround(x, align):
        return (x + align - 1) / align * align

    # A [CO, CI * KH * KW]
    reduce_len = upround(CI * KH * KW, ALIGN)
    A = tvm.compute((upround(CO, ALIGN), reduce_len), lambda i, j : 
            #tvm.select(tvm.all(i < CO, j < CI * KH * KW),
                       kernel[i][j / KW / KH][j / KW % KH][j % KW], name='A')
                       #tvm.const(0, kernel.dtype)), name='A')

    # B [CI * KH * KW, N * OH * OW]
    B = tvm.compute((reduce_len, upround(N * OH * OW, ALIGN)), lambda i, j:
            tvm.select(tvm.all(i < CI * KH * KW, j < N * OH * OW),
                               data_pad[j / (OH*OW)][i / (KH*KW)][j / OW % OH*HSTR + i / KW % KH]
                                                                 [j % OW*WSTR + i % KW],
                               tvm.const(0, data_pad.dtype)), name='B')

    gemm_n, gemm_l, gemm_m = A.shape[0], reduce_len, B.shape[1]

    # C [CO, N * OH * OW]
    k = tvm.reduce_axis((0, gemm_l), name='k')
    C = tvm.compute((gemm_n, gemm_m), lambda i, j: tvm.sum(A[i,k] * B[k,j], axis=k), name='C')

    # output
    # the last term C[gemm_n-1, gemm_m-1] is for enabling the alignment, otherwise the alignemt above
    # will be eliminated by bound inference
    output = tvm.compute((N, CO, OH, OW), lambda n, co, h, w:
                 C[co][n * OW * OW + h * OW + w] + tvm.const(0, C.dtype) * C[gemm_n-1,gemm_m-1],
                 name='output', tag='im2col_conv_output')

    return output

def _schedule_im2col_conv2d(s, op):
    """schedule the Im2Col method for conv2d"""

    # get ops and tensors
    output = op.output(0)
    C = op.input_tensors[0]
    A, B = C.op.input_tensors
    kernel = A.op.input_tensors[0]
    data = B.op.input_tensors[0]

    # tuning parameter config
    tune_config = getattr(tvm.target.current_target(), "tune_config", None)
    if tune_config is None: # use rule
        bn = 4
        unroll_step = 16

        total_work = util.get_const_int(C.shape[0] * C.shape[1])
        reduce_work = util.get_const_int(A.shape[1])
        if total_work > 200000:
            last_work = util.get_const_int(C.shape[1])
            if last_work > 10000:
                num_thread = 16
            elif last_work > 3000:
                num_thread = 8
            elif reduce_work > 100:
                num_thread = 4
            else:
                num_thread = 2

            if reduce_work < 50 and last_work < 30000:
                num_thread = 4
        elif total_work > 150000:
            num_thread = 8
        elif total_work > 50000:
            num_thread = 4
        else:
            num_thread = 2

        if num_thread == 4:
            unroll_step = 2
    else:
        bn = tune_config["bn"]
        num_thread = tune_config["num_thread"]
        unroll_step = tune_config["unroll_step"]

    bna = bnb = bn
    num_thread1 = num_thread2 = num_thread
    if data.dtype == 'float16':
        bnb *= 2
        last_work = util.get_const_int(C.shape[1])
        if last_work % (bnb * num_thread2) != 0:
            num_thread1 = num_thread * 2
            num_thread2 = num_thread / 2

    # schedule padding
    if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
        data_pad = data
        s[data_pad].compute_inline()

    ##### SCHEDULE A #####
    if util.get_const_int(kernel.shape[2]) == 1 and util.get_const_int(kernel.shape[3]) == 1:
        s[A].compute_inline()
    else:
        y, x = s[A].op.axis
        yo, xo, yi, xi = s[A].tile(y, x, bna, util.get_const_int(kernel.shape[3]))
        s[A].vectorize(xi)
        fuse_and_bind(s, A, [yo, xo])

    # pack to vector form
    packedA = pack_tensor(s, A, bna, [C])

    # vectorize load
    y, x = s[packedA].op.axis[:2]
    tmp = s.cache_write(packedA, "local")
    x, xt = s[packedA].split(x, bna)
    _, _, _, xi = tile_and_bind(s, packedA, y, x, num_thread)
    s[tmp].compute_at(s[packedA], xi)
    s[tmp].vectorize(s[tmp].op.axis[1])
    s[tmp].unroll(s[tmp].op.axis[2])
    s[packedA].vectorize(s[packedA].op.axis[2])
    s[packedA].unroll(xt)

    ##### SCHEDULE B #####
    y, x = s[B].op.axis
    yo, xo, yi, xi = s[B].tile(y, x, 1, 1 * bnb)
    fuse_and_bind(s, B, [yo, xo])

    # transpose and pack to vector form
    B_transpose, B_tmp = transpose(s, B, [C])
    s[B_transpose].compute_inline()
    packedB = pack_tensor(s, B_transpose, bnb, [B_tmp])

    # vectorize load
    s[packedB].vectorize(s[packedB].op.axis[2])
    y, x = s[packedB].op.axis[:2]
    tile_and_bind(s, packedB, y, x, num_thread)

    ##### SCHEDULE C #####
    # vectorize and unroll dot
    y, x = s[C].op.axis
    y, x, yt, xt = s[C].tile(y, x, bna, bnb)

    k = s[C].op.reduce_axis[0]
    s[C].reorder(k, yt, xt)
    if unroll_step != 1:
        k, k_unroll = s[C].split(k, unroll_step)
        s[C].unroll(k_unroll)
    s[C].unroll(yt)
    s[C].vectorize(xt)

    tile_and_bind(s, C, y, x, num_thread1, num_thread2)

    ##### COPY TO OUTPUT #####
    if output.op in s.outputs:  # no bias
        output = output
    else:                       # has bias
        s[output].compute_inline()
        output = s.outputs[0]

    n, co, h, w = s[output].op.axis
    h, w, vh, vw = s[output].tile(h, w, 1, bnb)
    s[output].unroll(vh)
    if util.get_const_int(s[output].op.output(0).shape[3]) % bnb != 0:
        pass
    else:
        s[output].vectorize(vw)
    fuse_and_bind(s, output, [n, co, h, w])

    #print(tvm.lower(s, [data, kernel], simple_mode=True))
