# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D schedule on x86"""
import tvm
from tvm import autotvm
from tvm.autotvm.task.nnvm_integration import deserialize_args
from tvm.autotvm.task import get_config
from .. import generic, tag
from .. import nn
from ..util import get_const_tuple, get_const_int, traverse_inline
from ..nn.conv2d import (
    conv2d,
    conv2d_NCHWc,
    conv2d_NCHWc_winograd_weight_transform,
    conv2d_NCHWc_winograd_without_weight_transform,
    conv2d_alter_layout,
    _get_workload as _get_conv2d_workload)
from ..nn.depthwise_conv2d import _get_workload as _get_depthwise_conv2d_workload
from ..nn.depthwise_conv2d import depthwise_conv2d_NCHWc, depthwise_conv2d_nchw
from ..nn.pad import pad
from ..nn.util import get_pad_tuple
from ..nn.winograd_util import winograd_transform_matrices

from . import conv2d_avx_1x1, conv2d_avx_common



def _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False):
    """
    Get default schedule config for the workload
    """
    if is_depthwise:
        wkl = _get_depthwise_conv2d_workload(data, kernel, strides, padding, out_dtype)
        from depthwise_conv2d import _fallback_schedule
        _fallback_schedule(cfg, wkl)
    else:
        wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype)
        is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
        if is_kernel_1x1:
            conv2d_avx_1x1._fallback_schedule(cfg, wkl)
        else:
            conv2d_avx_common._fallback_schedule(cfg, wkl)


def _create_tuning_space(cfg, data, kernel, strides, padding, dilation, layout):
    """Create schedule configuration from input arguments"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    if layout == 'NCHW':
        n, ic, h, w = dshape
        oc, _, kh, kw = kshape
    else:
        raise ValueError("Not support this layout {} with "
                         "schedule template.".format(layout))
    is_kernel_1x1 = kh == 1 and kw == 1
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (h - kh + 2 * ph) // sh + 1
    ow = (w - kw + 2 * pw) // sw + 1

    # Create schedule config
    cfg.define_split("tile_ic", ic, num_outputs=2)
    cfg.define_split("tile_oc", oc, num_outputs=2)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])


@autotvm.register_topi_compute(conv2d, 'cpu', 'direct')
def _declaration_conv(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    padding = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dilation = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    if layout == 'NCHW':
        _create_tuning_space(cfg, data, kernel, strides, padding, dilation, layout)
        if cfg.is_fallback:
            _get_default_config(cfg, data, kernel, strides, padding, out_dtype)
        return _declaration_conv_impl(cfg, data, kernel, strides,
                                      padding, dilation, layout, out_dtype)
    elif layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, dilation, out_dtype)
    elif layout == 'NHWC':
        return nn.conv2d_nhwc(data, kernel, strides, padding, dilation, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


def _declaration_conv_impl(cfg, data, kernel, strides, padding, dilation, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    assert layout == 'NCHW', "only support NCHW convolution for AVX"

    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(dilation, int):
        dilation_h, dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    HPAD, WPAD = padding
    HSTR, WSTR = strides

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    dilated_kernel_h = (kernel_height - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_width - 1) * dilation_w + 1
    out_height = (in_height + 2 * HPAD - dilated_kernel_h) // HSTR + 1
    out_width = (in_width + 2 * WPAD - dilated_kernel_w) // WSTR + 1

    # pack data
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD), name="data_pad")
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    shape = (batch_size, in_channel // ic_bn, pad_height, ic_bn, pad_width)
    data_vec = tvm.compute(shape,
                           lambda n, C, h, c, w: data_pad[n, C * ic_bn + c, h, w],
                           name='data_vec')

    # pack kernel
    shape = (num_filter//oc_bn, in_channel//ic_bn,
             kernel_height, kernel_width, ic_bn, oc_bn)
    kernel_vec = tvm.compute(shape,
                             lambda CO, CI, h, w, ci, co:
                             kernel[CO * oc_bn + co, CI * ic_bn + ci, h, w],
                             name='kernel_vec')

    # convolution
    oshape = (batch_size, num_filter//oc_bn, out_height, out_width, oc_bn)
    unpack_shape = (batch_size, num_filter, out_height, out_width)

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_vec[n, ic//ic_bn, oh*HSTR+kh*dilation_h, ic%ic_bn,
                                        ow*WSTR+kw*dilation_w].astype(out_dtype) *
                               kernel_vec[oc_chunk, ic//ic_bn, kh, kw, ic%ic_bn,
                                          oc_block].astype(out_dtype),
                               axis=[ic, kh, kw]), name='conv')

    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, c // oc_bn, h, w, c % oc_bn]
                         .astype(out_dtype),
                         name='output_unpack',
                         tag='conv2d_nchw')
    return unpack


@autotvm.register_topi_schedule(generic.schedule_conv2d_nchw, 'cpu', ['direct'])
def schedule_conv2d(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_nchw' in op.tag:
            output = op.output(0)
            conv_out = op.input_tensors[0]
            kernel_vec = conv_out.op.input_tensors[1]
            kernel = kernel_vec.op.input_tensors[0]
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            _, _, kh, kw = get_const_tuple(kernel.shape)
            is_kernel_1x1 = kh == 1 and kw == 1
            args = [s, cfg, data, data_pad, data_vec, kernel_vec, conv_out, output, outs[0]]
            if is_kernel_1x1:
                conv2d_avx_1x1._schedule_conv(*args)
            else:
                conv2d_avx_common._schedule_conv(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


@generic.schedule_conv2d_nhwc.register("cpu")
def schedule_conv2d_nhwc(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else: # inject custom schedule
                if len(op.axis) == 4: # schedule bias + bn + relu
                    n, h, w, c = op.axis
                    fused = s[op].fuse(n, h, w)
                    s[op].parallel(fused)
                    s[op].vectorize(c)
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_nhwc' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()

            data = op.input_tensors[0]
            data_pad = None
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            n_pad, h_pad, w_pad, c_pad = data_pad.op.axis
            pad_fused = s[data_pad].fuse(n_pad, h_pad)
            s[data_pad].parallel(pad_fused)
            C = conv
            n, h, w, c = C.op.axis
            ry, rx, rc = C.op.reduce_axis
            n_out, h_out, w_out, c_out = output_op.axis
            s[C].vectorize(c)
            if op != output_op: # fuse bias + bn + relu into conv
                s[C].compute_at(s[output_op], c_out)
            else:
                fused = s[C].fuse(n, h, w)
                s[C].parallel(fused)

        scheduled_ops.append(op)

    traverse(output_op)
    return s


# Define template function for autotvm task
# We define schedule template in this function instead of
# declaration function since actual input arguments need
# to be altered by the schedule selected.
@autotvm.task.register("topi_x86_conv2d_NCHWc")
def _topi_nn_conv2d_NCHWc(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    data, kernel, strides, padding, dilation, origin_layout, dtype = deserialize_args(args)
    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_tuning_space(cfg, data, kernel, strides, padding, dilation, origin_layout)

    # change shape with the value in config
    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])
    new_data_shape = (raw_data_shape[0], raw_data_shape[1] // ic_bn,
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    data_layout = "NCHW%dc" % ic_bn
    out_layout = "NCHW%dc" % oc_bn
    new_kernel_shape = (raw_kernel_shape[0] // oc_bn, raw_kernel_shape[1] // ic_bn,
                        raw_kernel_shape[2], raw_kernel_shape[3], ic_bn, oc_bn)
    new_data = tvm.placeholder(new_data_shape, data.dtype)
    new_kernel = tvm.placeholder(new_kernel_shape, kernel.dtype)

    C = _declaration_conv_NCHWc(cfg, new_data, new_kernel, strides, padding, dilation,
                                data_layout, out_layout, dtype)
    s = _schedule_conv2d_NCHWc(cfg, [C])
    return s, [new_data, new_kernel, C]


@autotvm.task.register("topi_x86_conv2d_NCHWc_winograd")
def _topi_nn_conv2d_NCHWc_winograd(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    data, kernel, strides, padding, dilation, origin_layout, dtype = deserialize_args(args)
    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_tuning_space(cfg, data, kernel, strides, padding, dilation, origin_layout)

    # change shape with the value in config
    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])
    new_data_shape = (raw_data_shape[0], raw_data_shape[1] // ic_bn,
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    data_layout = "NCHW%dc" % ic_bn
    out_layout = "NCHW%dc" % oc_bn
    new_kernel_shape = (raw_kernel_shape[0] // oc_bn, raw_kernel_shape[1] // ic_bn,
                        raw_kernel_shape[2], raw_kernel_shape[3], ic_bn, oc_bn)
    new_data = tvm.placeholder(new_data_shape, data.dtype)
    new_kernel = tvm.placeholder(new_kernel_shape, kernel.dtype)

    C = _declaration_conv_NCHWc_winograd_impl(
        cfg, new_data, new_kernel, strides, padding, dilation,
        data_layout, out_layout, dtype,
        transform_kernel=True, tile_size=None)
    s = tvm.create_schedule([C.op])
    s = _schedule_conv2d_NCHWc_winograd(cfg, s, C, C)
    return s, [new_data, new_kernel, C]


@conv2d_alter_layout.register("cpu")
def _alter_conv2d_layout(attrs, inputs, tinfo):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    data, kernel = tinfo[0], tinfo[1]
    batch_size, in_channel, height, width = get_const_tuple(data.shape)

    groups = attrs.get_int("groups")
    out_channel = attrs.get_int("channels")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    layout = attrs['layout']
    kh, kw = attrs.get_int_tuple("kernel_size")

    dtype = data.dtype
    out_dtype = dtype if attrs["out_dtype"] == "same" else attrs["out_dtype"]
    is_depthwise = groups == in_channel and groups == out_channel

    # only optimize for NCHW
    if layout != 'NCHW':
        return None
    if groups != 1 and not is_depthwise:
        return None

    dispatch_ctx = autotvm.task.DispatchContext.current
    target = tvm.target.current_target()
    # query schedule and fallback if necessary
    workload = autotvm.task.args_to_workload(
        [data, kernel, strides, padding, dilation, out_dtype], depthwise_conv2d_nchw) \
        if is_depthwise else \
        autotvm.task.args_to_workload(
            [data, kernel, strides, padding, dilation, layout, out_dtype], conv2d)
    cfg = dispatch_ctx.query(target, workload)
    if cfg.is_fallback:
        _get_default_config(cfg, data, kernel, strides, padding, out_dtype, is_depthwise)

    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    new_attrs['layout'] = 'NCHW%dc' % ic_bn
    new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

    new_data = tvm.placeholder((batch_size, in_channel//ic_bn, height, width, ic_bn),
                               dtype=data.dtype)
    if is_depthwise:
        # channel, channel_multiplier, kh, kw -> out_channel_chunk, kh, kw, out_channel_block
        # in which out_channel = merge(channel, channel_multiplier)
        kernel_sym = copy_inputs[1]
        kernel_sym = sym.reshape(kernel_sym, shape=(out_channel//oc_bn, oc_bn, kh, kw))
        kernel_sym = sym.transpose(kernel_sym, axes=(0, 2, 3, 1))
        copy_inputs[1] = kernel_sym

        # Store altered operator's config
        new_kernel = tvm.placeholder((out_channel//oc_bn, kh, kw, oc_bn), dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_attrs['layout'],
             new_attrs['out_layout'], out_dtype], depthwise_conv2d_NCHWc)
        dispatch_ctx.update(target, new_workload, cfg)
        return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)

    elif cfg.is_fallback or cfg.template_key == "direct":
        out_channel, _, kh, kw = get_const_tuple(kernel.shape)
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)

        # Store altered operator's config
        new_kernel = tvm.placeholder((out_channel//oc_bn, in_channel//ic_bn, kh, kw, ic_bn, oc_bn),
                                     dtype=kernel.dtype)
        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_attrs['layout'],
             new_attrs['out_layout'], out_dtype], conv2d_NCHWc)
        dispatch_ctx.update(target, new_workload, cfg)
        return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)
    elif cfg.template_key == "winograd":
        tile_size = cfg["tile_size"].val
        out_channel, _, kh, kw = get_const_tuple(kernel.shape)
        assert (kh, kw) == (3, 3)
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)
        new_attrs['tile_size'] = tile_size
        # Store altered operator's config
        new_kernel = tvm.placeholder(
            (out_channel//oc_bn, in_channel//ic_bn, ic_bn,
             tile_size + 3 - 1, tile_size + 3 - 1, oc_bn),
            dtype=kernel.dtype)

        new_kernel_workload = autotvm.task.args_to_workload(
            [kernel, new_attrs['kernel_layout'], out_dtype, tile_size],
            conv2d_NCHWc_winograd_weight_transform)

        new_kernel_sym = sym.contrib.conv2d_NCHWc_winograd_weight_transform(
            copy_inputs[1],
            kernel_layout=new_attrs['kernel_layout'],
            tile_size=tile_size)
        dispatch_ctx.update(target, new_kernel_workload, cfg)
        copy_inputs[1] = new_kernel_sym

        new_workload = autotvm.task.args_to_workload(
            [new_data, new_kernel, strides, padding, dilation, new_attrs['layout'],
             new_attrs['out_layout'], out_dtype],
            conv2d_NCHWc_winograd_without_weight_transform)
        dispatch_ctx.update(target, new_workload, cfg)
        return sym.contrib.conv2d_NCHWc_winograd_without_weight_transform(
            *copy_inputs,
            **new_attrs)
    else:
        raise RuntimeError("Unknown template: {}".format(cfg.template_key))


@autotvm.register_topi_compute(conv2d_NCHWc, 'cpu', 'direct')
def _declaration_conv_NCHWc(cfg, data, kernel, strides,
                            padding, dilation, layout, out_layout, out_dtype):
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload
    HPAD, WPAD = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    assert (dh, dw) == (1, 1), "Does not support dilation"

    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    if data.dtype == 'uint8':
        oc_chunk, _, kernel_height, kernel_width, _, oc_bn, _ = get_const_tuple(kernel.shape)
    else:
        oc_chunk, _, kernel_height, kernel_width, _, oc_bn = get_const_tuple(kernel.shape)
    num_filter = oc_chunk * oc_bn

    if cfg.is_fallback:
        _get_default_config(cfg, tvm.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
                            tvm.placeholder((num_filter, in_channel, kernel_height, kernel_width),
                                            dtype=kernel.dtype),
                            strides, padding, out_dtype)

    # output shape
    out_height = (ih + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (iw + 2 * WPAD - kernel_width) // WSTR + 1
    oshape = (n, oc_chunk, out_height, out_width, oc_bn)

    # DOPAD
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    ic = tvm.reduce_axis((0, in_channel), name='ic')
    kh = tvm.reduce_axis((0, kernel_height), name='kh')
    kw = tvm.reduce_axis((0, kernel_width), name='kw')

    if data.dtype == 'uint8':
        assert out_dtype == "int32", \
            "INT8 convolution requires input dtype = uint8 and output dtype=int32"
        # Intel performs dot product of 2 "4" Int8 values
        # Current implementation requires ic_bn to be a multiple of 4
        n_elems = 4
        assert ic_bn % n_elems == 0

        ic_outer = tvm.reduce_axis((0, in_channel//ic_bn), name='ic_outer')
        ic_f_inner = tvm.reduce_axis((0, ic_bn//n_elems), name='ic_f_inner')
        ic_s_inner = tvm.reduce_axis((0, n_elems), name='ic_s_inner')
        return tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                           tvm.sum(data_pad[n, ic_outer, oh*HSTR+kh, ow*WSTR+kw,
                                            ic_f_inner * n_elems +  ic_s_inner]
                                   .astype(out_dtype) *
                                   kernel[oc_chunk, ic_outer, kh, kw, ic_f_inner,
                                          oc_block, ic_s_inner].astype(out_dtype),
                                   axis=[kh, kw, ic_outer, ic_f_inner, ic_s_inner]),
                           name='conv2d_NCHWc_int8', tag="conv2d_NCHWc_int8")
    # else: fp implementation
    return tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic//ic_bn, oh*HSTR+kh, ow*WSTR+kw,
                                        ic%ic_bn].astype(out_dtype) *
                               kernel[oc_chunk, ic//ic_bn, kh, kw, ic%ic_bn, oc_block],
                               axis=[ic, kh, kw]),
                       name='conv2d_NCHWc', tag="conv2d_NCHWc")


@autotvm.register_topi_compute(conv2d_NCHWc, 'cpu', 'winograd')
def _declaration_conv_NCHWc_winograd(cfg, data, kernel, strides,
                                     padding, dilation, layout, out_layout, out_dtype):
    return _declaration_conv_NCHWc_winograd_impl(
        cfg, data, kernel, strides, padding, dilation,
        layout, out_layout, out_dtype,
        transform_kernel=True, tile_size=None)


def _declaration_conv_NCHWc_winograd_impl(
        cfg, data, kernel, strides,
        padding, dilation, layout, out_layout, out_dtype,
        transform_kernel, tile_size):
    out_dtype = out_dtype or data.dtype
    N, CII, IH, IW, CIII = get_const_tuple(data.shape)

    if transform_kernel:
        COO, CII, KH, KW, CIII_, VC = get_const_tuple(kernel.shape)
    else:
        COO, CII, CIII_, _, _, VC = get_const_tuple(kernel.shape)
        KH = 3
        KW = 3

    cfg.define_knob("tile_size", [2, 4, 6])
    m = tile_size if tile_size else cfg["tile_size"].val
    r = 3
    alpha = m + r - 1

    pad_top, pad_left, pad_bottom, pad_right = get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_bottom - KH) // HSTR + 1
    OW = (IW + pad_left + pad_right - KW) // WSTR + 1
    data_pad = pad(
        data,
        [0, 0, pad_top, pad_left, 0],
        [0, 0, pad_bottom, pad_right, 0],
        name="data_pad"
    )

    A, B, G = winograd_transform_matrices(m, out_dtype)

    def div_round_up(a, b):
        return (a + b - 1) // b

    # assert all(k == 3 for k in (KH, KW))
    assert all(p == 1 for p in (pad_top, pad_left, pad_bottom, pad_right))
    assert all(s == 1 for s in (HSTR, WSTR))
    assert OH == IH
    assert OW == IW

    OH_M = div_round_up(OH, m)
    OW_M = div_round_up(OW, m)
    # Layouts:

    # input            = (N, CII, IH, IW, CIII)
    # -> transpose
    ############################################################
    # input_tile_shape = (N, CII, OH // m, OH // m, alpha, alpha, CIII)
    # U_shape          = (COO, CII, CIII, alpha, alpha, COOO)
    # V_shape          = (N, CII, OH // m, OW // m, alpha, alpha, CIII)
    # M_shape          = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    # Y_shape          = (N, COO, OH // m, OW // m, m, m, COOO)
    ############################################################
    # -> transpose
    # O_shape          = (N, COO, OH, OW, COOO)

    n, coo, oh, ow, oh_m, ow_m, vc = \
        cfg.axis(N), cfg.axis(COO), cfg.axis(OH), cfg.axis(OW), \
        cfg.axis(OH_M), cfg.axis(OW_M), cfg.axis(VC)
    cii, ciii, kh, kw = cfg.reduce_axis(CII), cfg.reduce_axis(CIII), \
                        cfg.reduce_axis(KH), cfg.reduce_axis(KW)

    eps, nu = cfg.axis(alpha), cfg.axis(alpha)
    vh, vw = cfg.axis(m), cfg.axis(m)
    r_eps, r_nu = cfg.axis(alpha), cfg.axis(alpha)
    cfg.define_reorder("reorder_M",
                       [n, coo, oh_m, ow_m, eps, nu, vc, cii, ciii],
                       policy='candidate', candidate=[
                           [n, coo, cii, oh_m, ow_m, eps, ciii, nu, vc],
                           [n, coo, oh_m, ow_m, eps, cii, ciii, nu, vc],
                           # [n, coo, cii, oh_m, ow_m, ciii, nu, eps, vc],
                           # [n, coo, cii, oh_m, ow_m, nu, eps, ciii, vc],
                           # [n, coo, oh_m, ow_m, nu, eps, cii, ciii, vc],
                       ])

    cfg.define_reorder("reorder_V",
                       [n, cii, oh_m, ow_m, eps, nu, ciii, r_eps, r_nu],
                       policy='candidate', candidate=[
                           [n, cii, oh_m, ow_m, eps, r_eps, r_nu, nu, ciii],
                           # [n, cii, oh_m, ow_m, eps, nu, r_eps, r_nu, ciii],
                           # [n, cii, oh_m, ow_m, r_eps, r_nu, eps, nu, ciii],
                           # [n, cii, oh_m, ow_m, r_eps, r_nu, eps, nu, ciii],
                       ])

    cfg.define_reorder("reorder_Y",
                       [n, coo, oh_m, ow_m, vh, vw, vc, r_eps, r_nu],
                       policy='candidate', candidate=[
                           [n, coo, oh_m, ow_m, vh, r_eps, r_nu, vw, vc],
                           # [n, coo, oh_m, ow_m, vh, vw, r_eps, r_nu, vc],
                           # [n, coo, oh_m, ow_m, r_eps, r_nu, vh, vw, vc],
                           # [n, coo, oh_m, ow_m, r_eps, r_nu, vh, vw, vc],
                       ])


    input_tile = tvm.compute((N, CII, OH_M, OW_M, alpha, alpha, CIII),
                             lambda n, cii, oh_m, ow_m, eps, nu, ciii:
                             data_pad[n][cii][oh_m * m + eps][ow_m * m + nu][ciii],
                             name='input_tile')

    # transform kernel
    if transform_kernel:
        r_kh = tvm.reduce_axis((0, KH), 'r_kh')
        r_kw = tvm.reduce_axis((0, KW), 'r_kw')
        U = tvm.compute((COO, CII, CIII, alpha, alpha, VC),
                        lambda coo, cii, ciii, eps, nu, vc:
                        tvm.sum(kernel[coo][cii][r_kh][r_kw][ciii][vc].astype(out_dtype) *
                                G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]),
                        name='U')
    else:
        U = kernel

    # transform image
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    V = tvm.compute((N, CII, OH_M, OW_M, alpha, alpha, CIII),
                    lambda n, cii, oh_m, ow_m, eps, nu, ciii:
                    tvm.sum(input_tile[n][cii][oh_m][ow_m][r_eps][r_nu][ciii].astype(out_dtype) *
                            B[r_eps][eps] * B[r_nu][nu], axis=[r_eps, r_nu]), name='V')
    cii = tvm.reduce_axis((0, CII), name='cii')
    ciii = tvm.reduce_axis((0, CIII), name='ciii')

    # M_shape = (N, COO, OH // m, OW // m, alpha, alpha, COOO)
    M = tvm.compute((N, COO, OH_M, OW_M, alpha, alpha, VC),
                    lambda n, coo, oh_m, ow_m, eps, nu, vc:
                    tvm.sum(U[coo][cii][ciii][eps][nu][vc] * V[n][cii][oh_m][ow_m][eps][nu][ciii],
                            axis=[cii, ciii]),
                    name='M')

    # inverse transform
    r_eps = tvm.reduce_axis((0, alpha), 'r_eps')
    r_nu = tvm.reduce_axis((0, alpha), 'r_nu')
    # Y_shape = (N, COO, OH // m, OW // m, m, m, COOO)
    Y = tvm.compute((N, COO, OH_M, OW_M, m, m, VC),
                    lambda n, coo, oh_m, ow_m, vh, vw, vc:
                    tvm.sum(M[n][coo][oh_m][ow_m][r_eps][r_nu][vc] * A[r_eps][vh] * A[r_nu][vw],
                            axis=[r_eps, r_nu]),
                    name='Y')

    output = tvm.compute((N, COO, OH, OW, VC),
                         lambda n, coo, oh, ow, vc:
                         Y[n][coo][oh // m][ow // m][oh % m][ow % m][vc],
                         name='output', tag='conv2d_NCHWc_winograd')
    cfg.add_flop(2 * N * COO * VC * OH * OW * KH * KW * CII * CIII)
    return output

@autotvm.register_topi_compute(
    conv2d_NCHWc_winograd_without_weight_transform, 'cpu', 'winograd')
def _declaration_conv_NCHWc_winograd_without_weight_transform(
        cfg, data, transformed_kernel, strides,
        padding, dilation, layout, out_layout, out_dtype, tile_size):
    return _declaration_conv_NCHWc_winograd_impl(
        cfg, data, transformed_kernel, strides, padding, dilation,
        layout, out_layout, out_dtype, transform_kernel=False, tile_size=tile_size)


@autotvm.register_topi_schedule(
    generic.schedule_conv2d_NCHWc, 'cpu', ['direct', 'winograd'])
def _schedule_conv2d_NCHWc(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _callback(op):
        if 'conv2d_NCHWc_winograd' in op.tag:
            _schedule_conv2d_NCHWc_winograd(cfg, s, op.output(0), outs[0])
        elif 'conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, cfg, data_vec, conv_out, outs[0]]
            if data.dtype == 'uint8':
                # int8 conv kernel is 7-dim
                _, _, kh, kw, _, _, _ = get_const_tuple(kernel.shape)
                if kh == 1 and kw == 1:
                    conv2d_avx_1x1._schedule_conv_NCHWc_int8(*args)
                else:
                    conv2d_avx_common._schedule_conv_NCHWc_int8(*args)
            else:
                _, _, kh, kw, _, _, = get_const_tuple(kernel.shape)
                if kh == 1 and kw == 1:
                    conv2d_avx_1x1._schedule_conv_NCHWc(*args)
                else:
                    conv2d_avx_common._schedule_conv_NCHWc(*args)
        scheduled_ops.append(op)

    traverse_inline(s, outs[0].op, _callback)
    return s

@autotvm.register_topi_schedule(
    generic.schedule_conv2d_NCHWc_winograd_without_weight_transform,
    'cpu', ['winograd'])
def schedule_conv2d_winograd_without_weight_transform_(cfg, outs):
    """TOPI schedule callback"""
    s = tvm.create_schedule([x.op for x in outs])
    def _callback(op):
        if 'conv2d_NCHWc_winograd' in op.tag:

            output = op.output(0)
            _schedule_conv2d_NCHWc_winograd(cfg, s, output, outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s

def _schedule_conv2d_NCHWc_winograd(cfg, s, output, last):
    Y = output.op.input_tensors[0]
    M, A = Y.op.input_tensors
    U, V = M.op.input_tensors
    input_tile, B = V.op.input_tensors
    data_pad = input_tile.op.input_tensors[0]

    # Inline the constants.
    s[A].compute_inline()
    s[B].compute_inline()

    # transform kernel
    if isinstance(U.op, tvm.tensor.ComputeOp):
        kernel, G = U.op.input_tensors
        s[G].compute_inline()
        coo, cii, eps, nu, ciii, vc = s[U].op.axis
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # kernel transformation will be pre-computed during compilation, so we skip
            # this part to make tuning records correct
            s[U].pragma(eps, 'debug_skip_region')
        else:
            pass
            # r_kh, r_kw = s[U].op.reduce_axis
            # s[U].reorder(k, c, eps, nu, r_kh, r_kw, kk)
            # for axis in [eps, nu, r_kh, r_kw]:
            #     s[U].unroll(axis)
            # s[U].vectorize(kk)
            # s[U].parallel(k)

        if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
            s[kernel].compute_inline()

    ############################################################
    # input tile
    n, cii, oh_m, ow_m, eps, nu, ciii = s[input_tile].op.axis
    # Vectorize the input tile
    s[input_tile].vectorize(ciii)

    cfg.define_knob('data_pad_compute_location', [0, 1, 2, 3])
    if cfg['data_pad_compute_location'].val == 0:
        parallel_axis = s[input_tile].fuse(n)
        s[data_pad].compute_inline()
    if cfg['data_pad_compute_location'].val == 1:
        parallel_axis = s[input_tile].fuse(n)
        s[data_pad].compute_at(s[input_tile], cii)
        (_, _, _, _, dpcii) = s[data_pad].op.axis
        s[data_pad].vectorize(dpcii)
    if cfg['data_pad_compute_location'].val == 2:
        parallel_axis = s[input_tile].fuse(n, cii)
        s[data_pad].compute_at(s[input_tile], oh_m)
        (_, _, _, _, dpcii) = s[data_pad].op.axis
        s[data_pad].vectorize(dpcii)
    if cfg['data_pad_compute_location'].val == 3:
        parallel_axis = s[input_tile].fuse(n, cii, oh_m)
        s[data_pad].compute_at(s[input_tile], ow_m)
        (_, _, _, _, dpcii) = s[data_pad].op.axis
        s[data_pad].vectorize(dpcii)

    # s[input_tile].parallel(parallel_axis)
    ############################################################

    ############################################################
    # data_pad
    # s[data_pad].compute_inline()
    ############################################################

    ############################################################
    # transform image
    n, cii, oh_m, ow_m, eps, nu, ciii = s[V].op.axis
    r_eps, r_nu = s[V].op.reduce_axis

    s[V].vectorize(ciii)
    cfg["reorder_V"].apply(s, V, [n, cii, oh_m, ow_m, eps, nu, ciii, r_eps, r_nu])

    cfg.define_annotate("reduce_V", [r_eps, r_nu, eps, nu],
                        policy='unroll')
    cfg['reduce_V'].apply(s, V, [r_eps, r_nu, eps, nu], cfg=cfg)


    cfg.define_knob('input_tile_compute_location', [0, 1, 2, 3])
    if cfg['input_tile_compute_location'].val == 0:
        parallel_axis = s[V].fuse(n)
    if cfg['input_tile_compute_location'].val == 1:
        parallel_axis = s[V].fuse(n)
        s[input_tile].compute_at(s[V], cii)
    if cfg['input_tile_compute_location'].val == 2:
        parallel_axis = s[V].fuse(n, cii)
        s[input_tile].compute_at(s[V], oh_m)
    if cfg['input_tile_compute_location'].val == 3:
        parallel_axis = s[V].fuse(n, cii, oh_m)
        s[input_tile].compute_at(s[V], ow_m)

    # s[V].parallel(parallel_axis)
    ############################################################

    ############################################################
    # batch gemm
    n, coo, oh_m, ow_m, eps, nu, vc = s[M].op.axis
    cii, ciii = s[M].op.reduce_axis
    s[M].vectorize(vc)

    cfg["reorder_M"].apply(s, M, [n, coo, oh_m, ow_m, eps, nu, vc, cii, ciii])

    cfg.define_annotate("reduce_M", [eps, nu],
                        policy='unroll')
    cfg['reduce_M'].apply(s, M, [eps, nu], cfg=cfg)

    cfg.define_knob('V_compute_location', [0, 1, 2, 3])
    if cfg['V_compute_location'].val == 0:
        parallel_axis = s[M].fuse(n)
    if cfg['V_compute_location'].val == 1:
        parallel_axis = s[M].fuse(n)
        s[V].compute_at(s[M], coo)
    if cfg['V_compute_location'].val == 2:
        parallel_axis = s[M].fuse(n, coo)
        s[V].compute_at(s[M], oh_m)
    if cfg['V_compute_location'].val == 3:
        parallel_axis = s[M].fuse(n, coo)
        s[V].compute_at(s[M], ow_m)

    # s[M].parallel(parallel_axis)
    ############################################################

    ############################################################
    # inverse transform
    s[A].compute_inline()
    n, coo, oh_m, ow_m, vh, vw, vc = s[Y].op.axis
    r_eps, r_nu = s[Y].op.reduce_axis
    s[Y].vectorize(vc)

    cfg['reorder_Y'].apply(s, Y, [n, coo, oh_m, ow_m, vh, vw, vc, r_eps, r_nu])

    cfg.define_annotate("reduce_Y", [r_eps, r_nu, vh, vw],
                        policy='unroll')
    cfg['reduce_Y'].apply(s, Y, [r_eps, r_nu, vh, vw], cfg=cfg)

    cfg.define_knob('M_compute_location', [0, 1, 2, 3])
    if cfg['M_compute_location'].val == 0:
        parallel_axis = s[Y].fuse(n, coo, oh_m)
    if cfg['M_compute_location'].val == 1:
        s[M].compute_at(s[Y], coo)
        parallel_axis = s[Y].fuse(n)
    if cfg['M_compute_location'].val == 2:
        s[M].compute_at(s[Y], oh_m)
        parallel_axis = s[Y].fuse(n, coo)
    if cfg['M_compute_location'].val == 3:
        parallel_axis = s[Y].fuse(n, coo, oh_m)
        s[M].compute_at(s[Y], ow_m)

    # s[Y].parallel(parallel_axis)
    ############################################################

    ############################################################
    # output

    if output != last:
        s[output].compute_inline()

    n, coo, oh, ow, vc = s[last].op.axis
    s[last].vectorize(vc)

    OH = get_const_int(oh.dom.extent)
    OW = get_const_int(ow.dom.extent)
    mh = get_const_int(vh.dom.extent)
    mw = get_const_int(vw.dom.extent)
    cfg.define_knob('output_tile', [1])
    cfg.define_annotate('reduce_output', [cfg.axis(mh), cfg.axis(mw)], policy="unroll")
    if OH % mh == 0 and OW % mw == 0 and cfg['output_tile'].val == 1:
        # We can tile in OH
        oh, ow, ohi, owi = s[last].tile(oh, ow, mh, mw)
        cfg["reduce_output"].apply(s, last, [ohi, owi], cfg=cfg)

    cfg.define_knob('Y_compute_location', [0, 1, 2, 3])
    if cfg['Y_compute_location'].val == 0:
        parallel_axis = s[last].fuse(n)
    if cfg['Y_compute_location'].val == 1:
        parallel_axis = s[last].fuse(n)
        s[Y].compute_at(s[last], coo)
    if cfg['Y_compute_location'].val == 2:
        parallel_axis = s[last].fuse(n, coo)
        s[Y].compute_at(s[last], oh)
    if cfg['Y_compute_location'].val == 3:
        parallel_axis = s[last].fuse(n, coo, oh)
        s[Y].compute_at(s[last], ow)
    s[last].parallel(parallel_axis)

    ############################################################

    return s
