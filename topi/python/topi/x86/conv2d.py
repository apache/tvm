# pylint: disable=invalid-name,unused-variable,invalid-name
"""Conv2D schedule on x86"""
import tvm
from .. import generic, tag
from .. import nn
from ..nn.util import infer_pad, infer_stride
from ..nn.conv2d import conv2d, _get_workload, _get_schedule, _WORKLOADS

from . import conv2d_avx_1x1, conv2d_avx_common
from .conv2d_avx_common import AVXConvCommonFwd
from .conv2d_avx_1x1 import AVXConv1x1Fwd

_AVX_SCH_TO_DECL_FUNC = {
    AVXConvCommonFwd: conv2d_avx_common._declaration_conv,
    AVXConv1x1Fwd: conv2d_avx_1x1._declaration_conv
}

_AVX_SCH_TO_SCH_FUNC = {
    AVXConvCommonFwd: conv2d_avx_common._schedule_conv,
    AVXConv1x1Fwd: conv2d_avx_1x1._schedule_conv
}

@_get_schedule.register("cpu")
def _get_schedule_conv(wkl):
    if wkl not in _WORKLOADS:
        raise ValueError("no schedule for such workload: {}".format(wkl))
    idx = _WORKLOADS.index(wkl)

    fp32_vec_len = 8
    target = tvm.target.current_target(allow_none=False)
    for opt in target.options:
        if opt == '-mcpu=skylake-avx512':
            fp32_vec_len = 16

    _SCHEDULES_AVX_NCHW = [
        # float32 resnet-18
        AVXConvCommonFwd(3, fp32_vec_len, 28, False),
        AVXConvCommonFwd(16, fp32_vec_len, 28, False),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 28),
        AVXConvCommonFwd(16, fp32_vec_len, 28, False),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 28),
        AVXConvCommonFwd(16, fp32_vec_len, 28, False),
        AVXConvCommonFwd(16, fp32_vec_len, 14, False),
        AVXConv1x1Fwd(16, fp32_vec_len, 2, 14),
        AVXConvCommonFwd(16, fp32_vec_len, 14, True),
        AVXConvCommonFwd(16, 32, 7, True),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 7),
        AVXConvCommonFwd(16, fp32_vec_len, 7, True),
        # float32 mobilenet
        AVXConvCommonFwd(3, fp32_vec_len, 28, False),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 28),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 28),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 28),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 28),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 28),
        AVXConv1x1Fwd(16, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(16, fp32_vec_len, 2, 14),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 7),
        AVXConv1x1Fwd(16, fp32_vec_len, 1, 7),
    ]

    sch = _SCHEDULES_AVX_NCHW[idx]
    return sch


@conv2d.register("cpu")
def _declaration_conv(data, kernel, stride, padding, layout, out_dtype):
    out_dtype = data.dtype if out_dtype is None else out_dtype
    target = tvm.target.current_target(allow_none=False)
    wkl = _get_workload(data, kernel, stride, padding, out_dtype)
    if wkl in _WORKLOADS and 'avx' in str(target) and layout == 'NCHW':
        sch = _get_schedule(wkl)
        return _AVX_SCH_TO_DECL_FUNC[type(sch)](data, kernel, stride, padding, layout, out_dtype)
    elif layout == 'NCHW':
        return nn.conv2d_nchw(data, kernel, stride, padding, out_dtype)
    elif layout == 'HWCN':
        return nn.conv2d_hwcn(data, kernel, stride, padding, out_dtype)
    elif layout == 'NHWC':
        return nn.conv2d_nhwc(data, kernel, stride, padding, out_dtype)
    else:
        raise ValueError("not support this layout {} yet".format(layout))


@generic.schedule_conv2d_nchw.register(["cpu"])
def schedule_conv2d(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    target = tvm.target.current_target(allow_none=False)

    def default_schedule(op):
        """NCHW conv2d schedule for non imagenet workloads"""
        conv = op.output(0)
        kernel = op.input_tensors[1]
        data = op.input_tensors[0]
        data_pad = None
        if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
            data_pad = data
            data = data_pad.op.input_tensors[0]

        n_pad, c_pad, h_pad, w_pad = data_pad.op.axis
        pad_fused = s[data_pad].fuse(n_pad, c_pad)
        s[data_pad].parallel(pad_fused)
        C = conv
        n, c, h, w = C.op.axis
        rc, ry, rx = C.op.reduce_axis
        fused = s[C].fuse(n, c)
        s[C].parallel(fused)
        wo, wi = s[C].split(w, factor=16)
        s[C].reorder(fused, rc, h, wo, ry, rx, wi)  # move rc to outer loop
        s[C].unroll(rx)
        s[C].unroll(ry)
        s[C].vectorize(wi)

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else: # inject custom schedule
                if len(op.axis) == 4 and 'avx' not in str(target): # schedule bias + bn + relu
                    n, c, h, w = op.axis
                    fused = s[op].fuse(n, c)
                    s[op].parallel(fused)
                    s[op].vectorize(w)
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'conv2d_nchw' in op.tag:
            if 'avx' in str(target):
                try:
                    output = op.output(0)
                    conv_out = op.input_tensors[0]
                    kernel_vec = conv_out.op.input_tensors[1]
                    kernel = kernel_vec.op.input_tensors[0]
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
                except IndexError:
                    default_schedule(op)
            else:
                default_schedule(op)

    traverse(outs[0].op)
    return s


@generic.schedule_conv2d_nhwc.register(["cpu"])
def schedule_conv2d_nhwc(outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op

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
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        if 'conv2d_nhwc' in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
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

    traverse(output_op)
    return s
