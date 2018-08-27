# pylint: disable=invalid-name,unused-variable,invalid-name,unused-argument
"""Conv2D schedule on x86"""
import tvm
from tvm import autotvm
from .. import generic, tag
from .. import nn
from ..util import get_const_tuple
from ..nn.conv2d import conv2d, conv2d_NCHWc, conv2d_alter_layout

from . import conv2d_avx_1x1, conv2d_avx_common
from tvm.autotvm.task import register, get_config, ConfigEntity
from tvm.autotvm.task.nnvm_integration import deserialize_args
from tvm.autotvm.task.dispatcher import ApplyGraphBest


def _create_schedule_template(cfg, data, kernel, strides, padding, layout):
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


def conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype):
    """convert argument to workload"""
    if len(kernel.shape) == 4:
        raw_kernel = kernel
    else:  # the input kernel is transformed by alter_op_layout
        shape = get_const_tuple(kernel.shape)
        raw_kernel = tvm.placeholder((shape[0] * shape[4], shape[1], shape[2], shape[3]),
                                     dtype=kernel.dtype)
    return ('conv2d', ) + autotvm.task.args_to_workload(
        [data, raw_kernel, strides, padding, layout, out_dtype])


@conv2d.register("cpu")
@autotvm.task.dispatcher
def conv2d_x86(data, kernel, strides, padding, layout, out_dtype):
    return conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype)

@conv2d_x86.register(["direct"])
def _declaration_conv(cfg, data, kernel, strides, padding, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    if layout == 'NCHW':
        _create_schedule_template(cfg, data, kernel, strides, padding, layout)
        args = [cfg, data, kernel, strides, padding, layout, out_dtype]
        _, _, kh, kw = get_const_tuple(kernel.shape)
        is_kernel_1x1 = kh == 1 and kw == 1
        if is_kernel_1x1:
            return conv2d_avx_1x1._declaration_conv(*args)
        else:
            return conv2d_avx_common._declaration_conv(*args)
    elif layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, out_dtype)
    elif layout == 'NHWC':
        return nn.conv2d_nhwc(data, kernel, strides, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


@autotvm.task.register_topi_schedule(generic.schedule_conv2d_nchw, 'cpu', ['direct'])
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
            args = [s, cfg, data, data_pad, data_vec, kernel_vec, conv_out,
                    output, outs[0]]
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
@register("topi_x86_conv2d_NCHWc")
def topi_nn_conv2d_NCHWc(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    kernel_size = args[3]
    strides = args[4]
    padding = args[5]
    layout = args[6]
    kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)
                                       ) else (kernel_size, kernel_size)
    is_kernel_1x1 = kh == 1 and kw == 1
    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_schedule_template(cfg, data, kernel, strides, padding, layout)

    # change shape with the value in config
    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])
    new_data_shape = (raw_data_shape[0], raw_data_shape[1] // ic_bn,
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    data_layout = "NCHW%dc" % ic_bn
    out_layout = "NCHW%dc" % oc_bn
    if is_kernel_1x1:
        new_kernel_shape = (raw_kernel_shape[0] // oc_bn, raw_kernel_shape[1] // ic_bn,
                            ic_bn, oc_bn, raw_kernel_shape[2], raw_kernel_shape[3])
    else:
        new_kernel_shape = (raw_kernel_shape[0] // oc_bn, raw_kernel_shape[1] // ic_bn,
                            raw_kernel_shape[2], raw_kernel_shape[3], ic_bn, oc_bn)
    args[0] = tvm.placeholder(new_data_shape, data.dtype)
    args[1] = tvm.placeholder(new_kernel_shape, kernel.dtype)
    args[6] = data_layout
    args[7] = out_layout

    C = _declaration_conv_NCHWc(cfg, *args, **kwargs)
    s = _schedule_conv2d_NCHWc(cfg, args[2], args[3], args[4], args[5],
                              args[6], args[7], [C])
    return s, [args[0], args[1], C]


def conv_NCHWc_arg_to_workload(data, kernel, kernel_size, strides,
                               padding, layout, out_dtype):
    """convert argument to workload"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)
                                       ) else (kernel_size, kernel_size)
    is_kernel_1x1 = kh == 1 and kw == 1
    if len(dshape) > 4:
        raw_data = tvm.placeholder((dshape[0], dshape[1] * dshape[4], dshape[2],
                                    dshape[3]), dtype=kernel.dtype)
    else:
        raw_data = data
    if len(kshape) > 4:
        if is_kernel_1x1:
            raw_kernel = tvm.placeholder((kshape[0] * kshape[3], kshape[1] * kshape[2],
                                          kshape[4], kshape[5]), dtype=kernel.dtype)
        else:
            raw_kernel = tvm.placeholder((kshape[0] * kshape[5], kshape[1] * kshape[4],
                                          kshape[2], kshape[3]), dtype=kernel.dtype)
    else:
        raw_kernel = kernel
    return ('conv2d_NCHWc', ) + autotvm.task.args_to_workload(
        [raw_data, raw_kernel, strides, padding, layout, out_dtype])


@conv2d_alter_layout.register("cpu")
def _alter_conv2d_layout(attrs, inputs, _):
    dispatch_ctx = autotvm.task.DispatchContext.current
    if not isinstance(dispatch_ctx, ApplyGraphBest):
        raise RuntimeError("Intel AVX conv2d requires ApplyGraphBest to be dispatch context."
                           "Add 'with ApplyGraphBest(records):' before building function.")

    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    # only optimize for NCHW, groups=1 conv
    if attrs['layout'] != 'NCHW' or attrs.get_int("groups") != 1:
        return None

    import ast
    kernel_size = ast.literal_eval(attrs["kernel_size"])
    kh, kw = kernel_size
    is_kernel_1x1 = kh == 1 and kw == 1

    dispatch_ctx = autotvm.task.DispatchContext.current
    cfg = dispatch_ctx.query(None, None)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    new_attrs['layout'] = 'NCHW%dc' % ic_bn
    new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

    if is_kernel_1x1:
        # (oc, ic, h, w) -> (OC, IC, ic, oc, h, w)
        new_attrs['kernel_layout'] = 'OI%di%doHW' % (ic_bn, oc_bn)
    else:
        # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
        new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)

    return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)

@conv2d_NCHWc.register("cpu")
@autotvm.task.dispatcher
def conv2d_NCHWc_cpu(data, kernel, num_filter, kernel_size, strides,
                     padding, layout, out_layout, out_dtype):
    """TOPI compute callback. Mark this function as a dispatcher, so
    this template can assign config according to workload"""
    return conv_NCHWc_arg_to_workload(data, kernel, kernel_size, strides,
                                      padding, layout, out_dtype)

@conv2d_NCHWc_cpu.register(['direct'])
def _declaration_conv_NCHWc(cfg, data, kernel, num_filter, kernel_size, strides,
                            padding, layout, out_layout, out_dtype):
    kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)
                                       ) else (kernel_size, kernel_size)
    is_kernel_1x1 = kh == 1 and kw == 1
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    args = [cfg, data, kernel, (kh, kw), (sh, sw), (ph, pw), layout, out_dtype]
    if is_kernel_1x1:
        return conv2d_avx_1x1._declaration_conv_NCHWc(*args)
    else:
        return conv2d_avx_common._declaration_conv_NCHWc(*args)

@generic.schedule_conv2d_NCHWc.register("cpu")
def schedule_conv2d_NCHWc(num_filter, kernel_size, strides, padding,
                          layout, out_layout, outs):
    return _schedule_conv2d_NCHWc(None, num_filter, kernel_size, strides, padding,
                                  layout, out_layout, outs)

def _schedule_conv2d_NCHWc(cfg, num_filter, kernel_size, strides, padding,
                          layout, out_layout, outs):
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

        if 'conv2d_NCHWc' in op.tag:
            conv_out = op.output(0)
            if cfg is None:
                if "cfg" not in op.attrs:
                    raise RuntimeError("cfg not found for conv2d_NCHWc schedule.")
                serialized_cfg = []
                for item in op.attrs['cfg']['e']:
                    val = item[2]
                    if isinstance(val, tvm.container.Array):
                        serialized_cfg.append(val[1].value)
                    else:
                        serialized_cfg.append(val.value > 0)
            data_vec = conv_out.op.input_tensors[0]
            kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)
                                               ) else (kernel_size, kernel_size)
            is_kernel_1x1 = kh == 1 and kw == 1
            args = [s, cfg if cfg else tuple(serialized_cfg),
                    data_vec, conv_out, outs[0]]
            if is_kernel_1x1:
                conv2d_avx_1x1._schedule_conv_NCHWc(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
