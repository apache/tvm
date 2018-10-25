# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D schedule on x86"""
import tvm
from tvm import autotvm
from tvm.autotvm.task.dispatcher import ApplyGraphBest
from tvm.autotvm.task.nnvm_integration import deserialize_args
from tvm.autotvm.task import register, get_config
from .. import generic, tag
from .. import nn
from ..util import get_const_tuple
from ..nn.conv2d import conv2d, conv2d_NCHWc, conv2d_alter_layout, \
    _get_workload_int8, _get_schedule, _get_schedule_NCHWc, \
    _get_schedule_NCHWc_int8, _get_alter_layout_schedule, Workload
from ..nn.pad import pad

from . import conv2d_avx_1x1, conv2d_avx_common
from .conv2d_avx_common import AVXConvCommonFwd
from .conv2d_avx_1x1 import AVXConv1x1Fwd
from .check_targets import check_skylake

@_get_schedule.register("cpu")
def _get_schedule_conv(wkl):
    _WORKLOADS_AVX = [
        # workloads of resnet18_v1 on imagenet
        Workload('float32', 'float32', 224, 224, 3, 64, 7, 7, 3, 3, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('float32', 'float32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        # workloads of resnet34_v1 on imagenet, no extra workload required
        # workloads of resnet50_v1 on imagenet
        Workload('float32', 'float32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
        Workload('float32', 'float32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
        Workload('float32', 'float32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),
        # workloads of resnet101_v1 on imagenet, no extra workload required
        # workloads of resnet152_v1 on imagenet, no extra workload required
        # workloads of resnet18_v2 on imagenet, no extra workload required
        # workloads of resnet34_v2 on imagenet, no extra workload required
    ]

    fp32_vec_len = 8
    target = tvm.target.current_target(allow_none=False)
    for opt in target.options:
        if opt == '-mcpu=skylake-avx512':
            fp32_vec_len = 16

    _SCHEDULES_AVX = [
        # workloads of resnet18_v1 on imagenet
        AVXConvCommonFwd(3, fp32_vec_len, 28, False),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 28),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 28),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 14, False),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 14, True),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 7, True),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 7),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 7, True),
        # workloads of resnet34_v1 on imagenet, no extra workload required
        # workloads of resnet50_v1 on imagenet
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        # workloads of resnet101_v1 on imagenet, no extra workload required
        # workloads of resnet152_v1 on imagenet, no extra workload required
        # workloads of resnet18_v2 on imagenet, no extra workload required
        # workloads of resnet34_v2 on imagenet, no extra workload required
    ]

    if wkl not in _WORKLOADS_AVX:
        if wkl.hkernel == 1 and wkl.wkernel == 1:
            return conv2d_avx_1x1._get_default_schedule(wkl, fp32_vec_len)
        return conv2d_avx_common._get_default_schedule(wkl, fp32_vec_len)
    idx = _WORKLOADS_AVX.index(wkl)
    sch = _SCHEDULES_AVX[idx]
    return sch

def _get_schedule_conv_int8(wkl):
    _WORKLOADS_AVX = [
        ## Following are for INT8 kernels
        Workload('uint8', 'int32', 56, 56, 64, 64, 3, 3, 1, 1, 1, 1),
        Workload('uint8', 'int32', 56, 56, 64, 64, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 56, 56, 64, 128, 3, 3, 1, 1, 2, 2),
        Workload('uint8', 'int32', 56, 56, 64, 128, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 28, 28, 128, 128, 3, 3, 1, 1, 1, 1),
        Workload('uint8', 'int32', 28, 28, 128, 256, 3, 3, 1, 1, 2, 2),
        Workload('uint8', 'int32', 28, 28, 128, 256, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 14, 14, 256, 256, 3, 3, 1, 1, 1, 1),
        Workload('uint8', 'int32', 14, 14, 256, 512, 3, 3, 1, 1, 2, 2),
        Workload('uint8', 'int32', 14, 14, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 7, 7, 512, 512, 3, 3, 1, 1, 1, 1),
        # workloads of resnet34_v1 on imagenet, no extra workload required
        # workloads of resnet50_v1 on imagenet
        Workload('uint8', 'int32', 56, 56, 64, 256, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 56, 56, 256, 64, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 56, 56, 256, 128, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 28, 28, 128, 512, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 56, 56, 256, 512, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 28, 28, 512, 128, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 28, 28, 512, 256, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 14, 14, 256, 1024, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 28, 28, 512, 1024, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 14, 14, 1024, 256, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 14, 14, 1024, 512, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 7, 7, 512, 2048, 1, 1, 0, 0, 1, 1),
        Workload('uint8', 'int32', 14, 14, 1024, 2048, 1, 1, 0, 0, 2, 2),
        Workload('uint8', 'int32', 7, 7, 2048, 512, 1, 1, 0, 0, 1, 1),
    ]

    fp32_vec_len = 8
    target = tvm.target.current_target(allow_none=False)
    if check_skylake(target):
        fp32_vec_len = 16

    _SCHEDULES_AVX = [
        # Following are for INT8 operations
        # workloads of resnet18_v1 on imagenet
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 28),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 28),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 28, False),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 14, False),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 14, True),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 7, True),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 1, 7),
        AVXConvCommonFwd(fp32_vec_len, fp32_vec_len, 7, True),
        # workloads of resnet34_v1 on imagenet, no extra workload required
        # workloads of resnet50_v1 on imagenet
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 28),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        AVXConv1x1Fwd(fp32_vec_len, fp32_vec_len, 2, 7),
        # workloads of resnet101_v1 on imagenet, no extra workload required
        # workloads of resnet152_v1 on imagenet, no extra workload required
        # workloads of resnet18_v2 on imagenet, no extra workload required
        # workloads of resnet34_v2 on imagenet, no extra workload required
    ]

    if wkl not in _WORKLOADS_AVX:
        if wkl.hkernel == 1 and wkl.wkernel == 1:
            return conv2d_avx_1x1._get_default_schedule(wkl, fp32_vec_len)
        return conv2d_avx_common._get_default_schedule(wkl, fp32_vec_len)
    idx = _WORKLOADS_AVX.index(wkl)
    sch = _SCHEDULES_AVX[idx]
    return sch

@_get_schedule_NCHWc.register("cpu")
def _get_schedule_NCHWc_x86(wkl, layout, out_layout):
    return _get_schedule_conv(wkl)

@_get_schedule_NCHWc_int8.register("cpu")
def _get_schedule_NCHWc_x86_int8(wkl, layout, out_layout):
    return _get_schedule_conv_int8(wkl)

@_get_alter_layout_schedule.register("cpu")
def _get_alter_layout_schedule_x86(wkl):
    return _get_schedule_conv(wkl)


def _get_fp32_len():
    fp32_vec_len = 8
    target = tvm.target.current_target()
    if target is not None:
        for opt in target.options:
            if opt == '-mcpu=skylake-avx512':
                fp32_vec_len = 16
    return fp32_vec_len


def _get_default_sch(workload):
    fp32_vec_len = _get_fp32_len()
    _, _, kh, kw, _ = workload[2]
    is_kernel_1x1 = kh == 1 and kw == 1
    if is_kernel_1x1:
        cfg = conv2d_avx_1x1._fallback_schedule(workload, fp32_vec_len)
    else:
        cfg = conv2d_avx_common._fallback_schedule(workload, fp32_vec_len)
    return cfg


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
    """x86 conv2d declaration."""
    return conv_arg_to_workload(data, kernel, strides, padding, layout, out_dtype)


@conv2d_x86.register(["direct"])
def _declaration_conv(cfg, data, kernel, strides, padding, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    padding = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    strides = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    if layout == 'NCHW':
        _create_schedule_template(cfg, data, kernel, strides, padding, layout)
        if cfg.is_fallback:
            workload = conv_arg_to_workload(data, kernel, strides, padding,
                                            layout, out_dtype)
            cfg = _get_default_sch(workload)
        args = [cfg, data, kernel, strides, padding, layout, out_dtype]
        return _declaration_conv_impl(*args)
    elif layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, strides, padding, out_dtype)
    elif layout == 'NHWC':
        return nn.conv2d_nhwc(data, kernel, strides, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


def _declaration_conv_impl(cfg, data, kernel, strides, padding, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    assert layout == 'NCHW', "only support NCHW convolution for AVX"

    HPAD, WPAD = padding
    HSTR, WSTR = strides

    batch_size, in_channel, in_height, in_width = get_const_tuple(data.shape)
    num_filter, _, kernel_height, kernel_width = get_const_tuple(kernel.shape)

    pad_height = in_height + 2 * HPAD
    pad_width = in_width + 2 * WPAD

    out_height = (in_height + 2 * HPAD - kernel_height) // HSTR + 1
    out_width = (in_width + 2 * WPAD - kernel_width) // WSTR + 1

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
                       tvm.sum(data_vec[n, ic//ic_bn, oh*HSTR+kh, ic%ic_bn,
                                        ow*WSTR+kw].astype(out_dtype) *
                               kernel_vec[oc_chunk, ic//ic_bn, kh, kw, ic%ic_bn,
                                          oc_block].astype(out_dtype),
                               axis=[ic, kh, kw]), name='conv')

    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, c // oc_bn, h, w, c % oc_bn]
                         .astype(out_dtype),
                         name='output_unpack',
                         tag='conv2d_nchw',
                         attrs={'workload':
                                    conv_arg_to_workload(data, kernel, strides,
                                                         padding, layout,
                                                         out_dtype)})
    return unpack


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
            current_cfg = cfg
            if cfg.is_fallback:
                workload_attr = op.attrs["workload"]
                strides = (int(workload_attr[3][0].value), int(workload_attr[3][1].value))
                padding = (int(workload_attr[4][0].value), int(workload_attr[4][1].value))
                layout = workload_attr[5].value
                out_dtype = workload_attr[6].value
                workload = conv_arg_to_workload(data, kernel, strides, padding,
                                                layout, out_dtype)
                current_cfg = _get_default_sch(workload)
            args = [s, current_cfg, data, data_pad, data_vec, kernel_vec, conv_out,
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
def _topi_nn_conv2d_NCHWc(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)
    data, kernel = args[:2]
    strides = args[4]
    padding = args[5]
    layout = args[6]
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
                               padding, layout, out_layout, out_dtype):
    """convert argument to workload"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    if len(dshape) > 4:
        raw_data = tvm.placeholder((dshape[0], dshape[1] * dshape[4], dshape[2],
                                    dshape[3]), dtype=kernel.dtype)
    else:
        raw_data = data
    if len(kshape) > 4:
        raw_kernel = tvm.placeholder((kshape[0] * kshape[5], kshape[1] * kshape[4],
                                      kshape[2], kshape[3]), dtype=kernel.dtype)
    else:
        raw_kernel = kernel
    return ('conv2d_NCHWc', ) + autotvm.task.args_to_workload(
        [raw_data, raw_kernel, strides, padding, layout, out_layout,
         out_dtype])


def _query_dispatcher(workload, in_alter_op=False):
    dispatch_ctx = autotvm.task.DispatchContext.current
    if isinstance(dispatch_ctx, ApplyGraphBest):
        if in_alter_op:
            cfg = dispatch_ctx.query(None, None)
        else:
            cfg = dispatch_ctx.query_global_dict(workload)
    else:
        target = tvm.target.current_target()
        cfg = dispatch_ctx.query(target, workload)
        if cfg.is_fallback:
            cfg = _get_default_sch(workload)
    return cfg


@conv2d_alter_layout.register("cpu")
def _alter_conv2d_layout(attrs, inputs, tinfo):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    data, kernel = tinfo[0], tinfo[1]
    # only optimize for NCHW, groups=1 conv
    if attrs['layout'] != 'NCHW' or attrs.get_int("groups") != 1:
        return None

    kernel_size = attrs.get_int_tuple("kernel_size")
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    layout = attrs['layout']
    out_layout = layout if attrs["out_layout"] == "__undef__" else attrs["out_layout"]

    dtype = data.dtype
    out_dtype = dtype if attrs["out_dtype"] == "same" else attrs["out_dtype"]
    workload = conv_NCHWc_arg_to_workload(data, kernel, kernel_size, strides,
                                          padding, layout, out_layout, out_dtype)
    cfg = _query_dispatcher(workload, True)
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    new_attrs['layout'] = 'NCHW%dc' % ic_bn
    new_attrs['out_layout'] = 'NCHW%dc' % oc_bn

    # Store global schedule dictionary for ApplyGraphBest dispatcher
    dispatch_ctx = autotvm.task.DispatchContext.current
    if isinstance(dispatch_ctx, ApplyGraphBest):
        workload = conv_NCHWc_arg_to_workload(data, kernel, kernel_size, strides,
                                              padding, new_attrs['layout'],
                                              new_attrs['out_layout'], out_dtype)
        global_dict_key = workload
        dispatch_ctx.update_global_dict(global_dict_key, cfg)

    # (oc, ic, h, w) -> (OC, IC, h, w, ic, oc)
    new_attrs['kernel_layout'] = 'OIHW%di%do' % (ic_bn, oc_bn)

    return sym.contrib.conv2d_NCHWc(*copy_inputs, **new_attrs)


@conv2d_NCHWc.register("cpu")
def conv2d_NCHWc_cpu(data, kernel, num_filter, kernel_size, strides,
                     padding, layout, out_layout, out_dtype):
    """x86 conv2d_NCHWc declaration."""
    dispatch_ctx = autotvm.task.DispatchContext.current
    if not isinstance(dispatch_ctx, ApplyGraphBest):
        layout = out_layout = "NCHW"
    workload = conv_NCHWc_arg_to_workload(data, kernel, kernel_size, strides,
                                          padding, layout, out_layout, out_dtype)
    cfg = _query_dispatcher(workload)
    return _declaration_conv_NCHWc(cfg, data, kernel, num_filter, kernel_size, strides,
                                   padding, layout, out_layout, out_dtype)


def _declaration_conv_NCHWc(cfg, data, kernel, num_filter, kernel_size, strides,
                            padding, layout, out_layout, out_dtype):
    n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
    ic = ic_chunk * ic_block
    kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)) else \
        (kernel_size, kernel_size)
    is_kernel_1x1 = kh == 1 and kw == 1
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    if data.dtype == 'uint8':
        wkl = _get_workload_int8(tvm.placeholder((n, ic, h, w), dtype=data.dtype),
                                 tvm.placeholder((num_filter, ic, kh, kw),
                                                 dtype=kernel.dtype),
                                 strides, padding, out_dtype)
        sch = _get_schedule_NCHWc_int8(wkl, layout, out_layout)
        return conv2d_avx_1x1._declaration_conv_NCHWc_int8(wkl, sch, data, kernel) \
            if is_kernel_1x1 \
            else conv2d_avx_common._declaration_conv_NCHWc_int8(wkl, sch, data, kernel)

    args = [cfg, data, kernel, (kh, kw), (sh, sw), (ph, pw), layout, out_layout, out_dtype]
    return _declaration_conv_NCHWc_impl(*args)


def _declaration_conv_NCHWc_impl(cfg, data, kernel, kernel_size, strides, padding, layout,
                                 out_layout, out_dtype):
    HPAD, WPAD = padding
    HSTR, WSTR = strides

    n, ic_chunk, ih, iw, ic_block = get_const_tuple(data.shape)
    ic = ic_chunk * ic_block
    kh, kw = kernel_size
    oc_chunk, _, _, _, _, oc_block = get_const_tuple(kernel.shape)
    oc = oc_chunk * oc_block
    oh = (ih + 2 * HPAD - kh) // HSTR + 1
    ow = (iw + 2 * WPAD - kw) // WSTR + 1

    # DOPAD
    DOPAD = (HPAD != 0 or WPAD != 0)
    if DOPAD:
        data_pad = pad(data, (0, 0, HPAD, WPAD, 0), name="data_pad")
    else:
        data_pad = data

    # fetch schedule
    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]
    if ic_bn != ic_block:
        raise RuntimeError("ic_bn in config is not equal to actual data ic_block: %d vs %d."
                           % (ic_bn, ic_block))
    if oc_bn != oc_block:
        raise RuntimeError("oc_bn in config is not equal to actual kernel oc_block: %d vs %d."
                           % (oc_bn, oc_block))

    # convolution
    oshape = (n, oc//oc_bn, oh, ow, oc_bn)

    ic = tvm.reduce_axis((0, ic), name='ic')
    kh = tvm.reduce_axis((0, kernel_size[0]), name='kh')
    kw = tvm.reduce_axis((0, kernel_size[1]), name='kw')

    workload = conv_NCHWc_arg_to_workload(data, kernel, kernel_size,
                                          strides, padding, layout,
                                          out_layout, out_dtype),
    attrs = {'workload': workload}
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(data_pad[n, ic//ic_bn, oh*HSTR+kh, ow*WSTR+kw,
                                        ic%ic_bn].astype(out_dtype) *
                               kernel[oc_chunk, ic//ic_bn, kh, kw, ic%ic_bn, oc_block],
                               axis=[ic, kh, kw]),
                       name='conv2d_NCHWc', tag="conv2d_NCHWc", attrs=attrs)
    return conv


@generic.schedule_conv2d_NCHWc.register("cpu")
def schedule_conv2d_NCHWc(num_filter, kernel_size, strides, padding,
                          layout, out_layout, outs):
    """x86 conv2d_NCHWc schedule"""
    return _schedule_conv2d_NCHWc(None, num_filter, kernel_size, strides, padding,
                                  layout, out_layout, outs)


def _schedule_conv2d_NCHWc(cfg, num_filter, kernel_size, strides, padding,
                           layout, out_layout, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    dispatch_ctx = autotvm.task.DispatchContext.current
    if not isinstance(dispatch_ctx, ApplyGraphBest):
        layout = out_layout = "NCHW"

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
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            kh, kw = kernel_size if isinstance(kernel_size, (tuple, list)) else \
                (kernel_size, kernel_size)
            is_kernel_1x1 = kh == 1 and kw == 1
            n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
            ic = ic_chunk * ic_block
            original_data = tvm.placeholder((n, ic, h, w), dtype=data.dtype)

            kh, kw = kernel_size
            original_kernel = tvm.placeholder((num_filter, ic, kh, kw),
                                              dtype=kernel.dtype)
            if data.dtype == 'uint8':
                wkl = _get_workload_int8(original_data, original_kernel,
                                         strides, padding, conv_out.dtype)
                sch = _get_schedule_NCHWc_int8(wkl, layout, out_layout)
                args = [s, wkl, sch, data_vec, kernel, conv_out, outs[0]]
                if is_kernel_1x1:
                    conv2d_avx_1x1._schedule_conv_NCHWc_int8(*args)
                else:
                    conv2d_avx_common._schedule_conv_NCHWc_int8(*args)
            else:
                current_cfg = cfg
                if current_cfg is None:
                    workload = conv_NCHWc_arg_to_workload(data, kernel, kernel_size, strides,
                                                          padding, layout, out_layout,
                                                          conv_out.dtype)
                    current_cfg = _query_dispatcher(workload)
                args = [s, current_cfg, data_vec, conv_out, outs[0]]
                if is_kernel_1x1:
                    conv2d_avx_1x1._schedule_conv_NCHWc(*args)
                else:
                    conv2d_avx_common._schedule_conv_NCHWc(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
