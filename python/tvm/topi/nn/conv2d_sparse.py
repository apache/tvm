import tvm
from tvm import te
from tvm.topi import nn
from tvm.topi.nn.util import get_pad_tuple
from tvm.topi.util import get_const_tuple
from tvm import autotvm
from ..nn.conv2d import conv2d_infer_layout, _get_workload as _get_conv2d_workload
from ..util import get_const_tuple, traverse_inline
from tvm.topi.sparse import batch_csrmm, csrmm_default

def _fallback_schedule(cfg, wkl):
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False,
                        layout='NCHW'):
    """
    Get default schedule config for the workload
    """
    static_data_shape = []
    for dim in get_const_tuple(data.shape):
        if isinstance(dim, tvm.tir.Var):
            static_data_shape.append(1)
        else:
            static_data_shape.append(dim)
    data = te.placeholder(static_data_shape, dtype=data.dtype)
    wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype, layout)
    is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
    _fallback_schedule(cfg, wkl)

def conv2d_sparse_gemm_nchw(data, w_data, w_indices, w_indptr,
                            OC, KH, KW,
                            strides, padding, dilation,
                            out_dtype='float32'):
    """Compute conv2d by transforming the input,
    executing GEMM and not transforming the output back yet"""
    batches, IC, IH, IW = get_const_tuple(data.shape)

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

    if pad_top or pad_left:
        data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_down, pad_right],
                          name="data_pad")
    else:
        data_pad = data

    # --- Im2col

    B_shape = (batches, K, M)
    idxmod = tvm.tir.indexmod
    idxdiv = tvm.tir.indexdiv
    # print(KH, KW, IC, OW, HSTR)

    B = te.compute(B_shape, lambda n, k, m:
                   data_pad[n, (k // (KH*KW)) % IC,
                            (k // KH) % KW + ((m // OW) * HSTR),
                            (k % KW) + ((m % OW) * WSTR)],
                       name='data_im2col')


    # --- GEMM: A*B'
    # oshape = (batches, N, M)
    oshape = (batches, OC, OH, OW)
    # B = te.compute((N,M), lambda n, m:
    #                B[0, n, m],
    #                name='data_flatten')
    C = batch_csrmm(w_data, w_indices, w_indptr, B, oshape)
    # C = csrmm_default(w_data, w_indices, w_indptr, B)


    # placeholder reshape
    # k = te.reduce_axis((0, K), 'k')
    # C = te.compute(
    #     oshape,
    #     lambda b, c, h, w: te.sum(C[b, c, w] * C[b, c, w], axis=k),
    #     name='C')

    return C

def csrdc(data, indices, indptr, inputs, oshape, kdim, strides, padding):
    # pylint: disable=invalid-name
    """The default implementation of csrdc in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        1-D with shape [nonzeros]

    indices : tvm.te.Tensor
        1-D with shape [nonzeros]

    indptr : tvm.te.Tensor
        1-D with shape [m+1]

    inputs : tvm.te.Tensor
        4-D with shape [N, IC, IH, IW]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [N, OC, OH, OW]
    """

    KH, KW = kdim
    HSTR, WSTR = strides
    pad_top, pad_left, pad_down, pad_right = padding
    def csrdc_ir(weight_data, indices, indptr, inputs, out):
        """define ir for csrdc"""
        irb = tvm.tir.ir_builder.create()
        weight_data_ptr = irb.buffer_ptr(weight_data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        inputs_ptr = irb.buffer_ptr(inputs)
        out_ptr = irb.buffer_ptr(out)

        batches, OC, OH, OW = out.shape
        _, IC, IH, IW = inputs.shape

        def offset_fn(c,h,w,H,W):
            return (c*H + h)*W + w

        with irb.for_range(0, batches, name='n') as n:
            with irb.for_range(0, OC, name='oc', for_type="parallel") as oc:
                with irb.for_range(0, OH, name='oh') as oh:
                    with irb.for_range(0, OW, name='ow') as ow:
                        out_idx = n*OC*OW*OH + oc*OW*OH + oh*OW + ow
                        out_ptr[out_idx] = 0.

                tmp = irb.allocate('int32', (1,), name='tmp', scope='local')
                tmp[0] = indptr_ptr[oc+1] - indptr_ptr[oc]

                with irb.for_range(0, tmp[0], name='j') as j:
                    true_j = irb.allocate('int32', (1,), name='true_j', scope='local')
                    true_j[0] = j + indptr_ptr[oc]
                    off = indices_ptr[true_j[0]]
                    coeff = weight_data_ptr[true_j[0]]
                    kh = (off % (KH * KW)) // KH
                    kw = (off % (KH * KW)) % KW
                    ic = off // (KH * KW)

                    with irb.for_range(0, OH, name='oh') as oh:
                        with irb.for_range(0, OW, name='ow') as ow:
                            ih = (oh * HSTR + kh) - pad_top
                            iw = (ow * WSTR + kw) - pad_left
                            with irb.if_scope(tvm.tir.all(IH > ih, ih >= 0)):
                                with irb.if_scope(tvm.tir.all(IW > iw, iw >= 0)):
                                    data_idx = offset_fn(ic,ih*HSTR,iw*WSTR,IH,IW) + n*IC*IW*IH
                                    out_idx = offset_fn(oc,oh,ow,OH,OW)
                                    out_ptr[out_idx] +=  coeff * inputs_ptr[data_idx]

        return irb.get()

    out = te.extern(oshape, [data, indices, indptr, inputs],
                    lambda ins, outs: csrdc_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                    tag="conv2d_sparse_nchw_direct", dtype='float32',
                    name='conv2d_sparse_nchw_direct')
    return out


def conv2d_sparse_direct_nchw(data, w_data, w_indices, w_indptr,
                              OC, KH, KW, strides, padding, dilation,
                              out_dtype='float32'):
    return conv2d_sparse_direct_NCHW(data, w_data, w_indices, w_indptr,
                                     OC, KH, KW, strides, padding, dilation,
                                     out_dtype)


@autotvm.register_topi_compute("conv2d.nn")
def conv2d_sparse_direct_NCHW(cfg, data, w_data, w_indices, w_indptr,
                              OC, KH, KW,
                              strides, padding, dilation,
                              out_dtype='float32'):
    """Compute conv2d by transforming the input,
    executing GEMM and not transforming the output back yet"""
    batches, IC, IH, IW = get_const_tuple(data.shape)

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
    padding = pad_top, pad_left, pad_down, pad_right

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1

    # If no config was set, we can fallback to default config.
    if cfg.is_fallback:
        _get_default_config(cfg, te.placeholder((batches, IC, IH, IW), dtype=data.dtype),
                            te.placeholder((OC, IC, KH, KW),
                                           dtype=w_data.dtype),
                            (HSTR, WSTR), padding, out_dtype)

    # if pad_top or pad_left:
    #     data_pad = nn.pad(data, [0, 0, pad_top, pad_left], [0, 0, pad_down, pad_right],
    #                       name="data_pad")
    # else:
    #     data_pad = data

    oshape = (batches, OC, OH, OW)
    out = csrdc(w_data, w_indices, w_indptr, data, oshape, (KH,KW), (HSTR,WSTR), padding)

    return out


@autotvm.register_topi_schedule("conv2d_sparse_nchw_direct.nn")
def schedule_conv2d_direct_sparse_nchw(cfg, outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    print('We are in the sparse schedule!\n\n\n\n')
    def _callback(op):
        print('hello sir:', op.tag)
        if 'broadcast' in op.tag:
            conv_out = op.output(0)

            data = conv_out.op.input_tensors[0]
            kernels = conv_out.op.input_tensors[1]

            args = [s, cfg, data, kernels, conv_out, outs[0]]
            # ic, oc, kh, kw = get_const_tuple(kernel.shape)
            schedule_conv2d_sparse_nchw(*args)

    traverse_inline(s, outs[0].op, _callback)
    return s

def schedule_conv2d_sparse_nchw(s, cfg, data, kernels, conv_out, last):
    """Create schedule for tensors"""
    print('hello there brother')
    print(last.shape)
    _, OC, _, _ = s[last].op.axis
    s[last].parallel(OC)
    return s
