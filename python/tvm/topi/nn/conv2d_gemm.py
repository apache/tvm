import tvm
from tvm import te
from tvm.topi import nn
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import get_const_tuple
from tvm import autotvm

def conv2d_gemm_nchw(data, weights, strides, padding, dilation,
                     layout):
    """Compute conv2d by transforming the input,
    executing GEMM and not transforming the output back yet"""
    batches, IC, IH, IW = get_const_tuple(data.shape)
    OC, IC, KH, KW = get_const_tuple(weights.shape)

    K = KH * KW

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = \
        get_pad_tuple(padding, (dilated_kernel_h, dilated_kernel_w))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1

    N = OC
    K = KH * KW * IC
    M = OH * OW


    # --- Weight reshape
    A = tvm.topi.nn.flatten(weights)

    if pad_top or pad_left:
        data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_down, pad_right],
                          name="data_pad")
    else:
        data_pad = data

    # --- Im2col

    B_shape = (batches, K, M)
    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv

    B = te.compute(B_shape, lambda n, k, m:
                   data_pad[n, (k // (KH*KW)) % IC,
                            (k // KH) % KW + ((m // OW) * HSTR),
                            (k % KW) + ((m % OW) * WSTR)],
                       name='data_im2col')


    # --- GEMM: A*B'
    k = te.reduce_axis((0, K), 'k')
    # C = te.compute(
    #            (batches, N, M),
    #            lambda b, n, m: te.sum(A[n, k] * B[b, k, m], axis=k),
    #            name='C')

    oshape = (batches, OC, OH, OW)
    # C = te.compute(
    #     oshape,
    #     lambda b, c, h, w: C[b, c, h*OW + w]
    # )

    C = te.compute(
        oshape,
        lambda b, c, h, w: te.sum(A[c, k] * B[b, k, h*OW + w], axis=k),
        name='C')
    return C

# @autotvm.register_topi_schedule("conv2d_NCHW.x86")
def schedule_gemm_conv2d_nchw(outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    return s
