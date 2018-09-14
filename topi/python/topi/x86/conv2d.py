# pylint: disable=invalid-name,unused-variable,invalid-name,unused-argument
"""Conv2D schedule on x86"""
import tvm
from .. import generic, tag
from .. import nn
from ..nn.util import infer_pad, infer_stride
from ..nn.conv2d import conv2d, conv2d_NCHWc, conv2d_alter_layout, \
    _get_workload, _get_workload_int8, _get_schedule, _get_schedule_NCHWc, \
    _get_schedule_NCHWc_int8, _get_alter_layout_schedule, Workload

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

@conv2d.register("cpu")
def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
    _AVX_SCH_TO_DECL_FUNC = {
        AVXConvCommonFwd: conv2d_avx_common._declaration_conv,
        AVXConv1x1Fwd: conv2d_avx_1x1._declaration_conv
    }
    out_dtype = data.dtype if out_dtype is None else out_dtype
    target = tvm.target.current_target(allow_none=False)
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    if layout == 'NCHW':
        sch = _get_schedule(wkl)
        return _AVX_SCH_TO_DECL_FUNC[type(sch)](data, kernel, stride, padding, layout, out_dtype)
    elif layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, stride, padding, out_dtype)
    elif layout == 'NHWC':
        return nn.conv2d_nhwc(data, kernel, stride, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


@conv2d_alter_layout.register("cpu")
def _alter_conv2d_layout(attrs, inputs, tinfos):
    import nnvm.symbol as sym
    copy_inputs = [s for s in inputs]
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    # only optimize for NCHW, groups=1 conv
    if attrs['layout'] != 'NCHW' or attrs.get_int("groups") != 1:
        return None

    data = tinfos[0]
    kernel = tinfos[1]

    import ast
    padding = ast.literal_eval(attrs['padding'])
    stride = ast.literal_eval(attrs['strides'])

    wkl = _get_workload(data, kernel, stride, padding, data.dtype)
    sch = _get_alter_layout_schedule(wkl)
    is_kernel_1x1 = isinstance(sch, AVXConv1x1Fwd)
    ic_bn, oc_bn = sch.ic_bn, sch.oc_bn

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
def _declaration_conv_NCHWc(data, kernel, num_filter, kernel_size, stride,
                            padding, layout, out_layout, out_dtype):
    _AVX_SCH_TO_DECL_FUNC = {
        AVXConvCommonFwd: conv2d_avx_common._declaration_conv_NCHWc,
        AVXConv1x1Fwd: conv2d_avx_1x1._declaration_conv_NCHWc
    }

    # Use int8 schedules if the input data is of int8 dtype
    if data.dtype == 'uint8':
        _AVX_SCH_TO_DECL_FUNC = {
            AVXConvCommonFwd: conv2d_avx_common._declaration_conv_NCHWc_int8,
            AVXConv1x1Fwd: conv2d_avx_1x1._declaration_conv_NCHWc_int8
        }

    n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
    ic = ic_chunk * ic_block
    kh, kw = kernel_size
    if data.dtype == 'uint8':
        wkl = _get_workload_int8(tvm.placeholder((n, ic, h, w), dtype=data.dtype),
                                 tvm.placeholder((num_filter, ic, kh, kw),
                                                 dtype=kernel.dtype),
                                 stride, padding, out_dtype)
        sch = _get_schedule_NCHWc_int8(wkl, layout, out_layout)
    else:
        wkl = _get_workload(tvm.placeholder((n, ic, h, w), dtype=data.dtype),
                            tvm.placeholder((num_filter, ic, kh, kw),
                                            dtype=kernel.dtype),
                            stride, padding, out_dtype)
        sch = _get_schedule_NCHWc(wkl, layout, out_layout)
    return _AVX_SCH_TO_DECL_FUNC[type(sch)](wkl, sch, data, kernel)


@generic.schedule_conv2d_nchw.register(["cpu"])
def schedule_conv2d(outs):
    """Create schedule for tensors"""
    _AVX_SCH_TO_SCH_FUNC = {
        AVXConvCommonFwd: conv2d_avx_common._schedule_conv,
        AVXConv1x1Fwd: conv2d_avx_1x1._schedule_conv
    }
    s = tvm.create_schedule([x.op for x in outs])
    target = tvm.target.current_target(allow_none=False)
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
            padding = infer_pad(data, data_pad)
            if data_pad is None:
                stride = infer_stride(data, kernel, output)
            else:
                stride = infer_stride(data_pad, kernel, output)

            wkl = _get_workload(data, kernel, stride, padding, output.dtype)
            sch = _get_schedule(wkl)
            _AVX_SCH_TO_SCH_FUNC[type(sch)](s, data, data_pad, data_vec,
                                            kernel, kernel_vec, conv_out, output, outs[0])

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s


@generic.schedule_conv2d_nhwc.register(["cpu"])
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


@generic.schedule_conv2d_NCHWc.register(["cpu"])
def schedule_conv2d_NCHWc(num_filter, kernel_size, stride, padding,
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
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            _AVX_SCH_TO_SCH_FUNC = {
                AVXConvCommonFwd: conv2d_avx_common._schedule_conv_NCHWc,
                AVXConv1x1Fwd: conv2d_avx_1x1._schedule_conv_NCHWc
            }

            # Use int8 schedules if the input data is of int8 dtype
            if data.dtype == 'uint8':
                _AVX_SCH_TO_SCH_FUNC = {
                    AVXConvCommonFwd: conv2d_avx_common._schedule_conv_NCHWc_int8,
                    AVXConv1x1Fwd: conv2d_avx_1x1._schedule_conv_NCHWc_int8
                }

            n, ic_chunk, h, w, ic_block = [x.value for x in data.shape]
            ic = ic_chunk * ic_block
            original_data = tvm.placeholder((n, ic, h, w), dtype=data.dtype)

            kh, kw = kernel_size
            original_kernel = tvm.placeholder((num_filter, ic, kh, kw),
                                              dtype=kernel.dtype)

            if data.dtype == 'uint8':
                wkl = _get_workload_int8(original_data, original_kernel,
                                         stride, padding, conv_out.dtype)
                sch = _get_schedule_NCHWc_int8(wkl, layout, out_layout)
            else:
                wkl = _get_workload(original_data, original_kernel, stride, padding, conv_out.dtype)
                sch = _get_schedule_NCHWc(wkl, layout, out_layout)
            _AVX_SCH_TO_SCH_FUNC[type(sch)](s, wkl, sch, data_vec,
                                            kernel, conv_out, outs[0])

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
