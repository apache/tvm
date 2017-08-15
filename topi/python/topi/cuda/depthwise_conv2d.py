# pylint: disable=invalid-name
"""Schedule for depthwise_conv2d with auto fusion"""
import tvm
from ..util import get_const_tuple

def schedule_depthwise_conv2d_nchw(outs):
    """Schedule for depthwise_conv2d nchw forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(PaddedInput, Filter, DepthwiseConv2d):
        out_shape = get_const_tuple(DepthwiseConv2d.shape)
        out_height = out_shape[2]
        out_width = out_shape[3]
        channel_multiplier = get_const_tuple(Filter.shape)[1]
        s[PaddedInput].compute_inline()
        IS = s.cache_read(PaddedInput, "shared", [DepthwiseConv2d])
        FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])
        IL = s.cache_read(IS, "local", [DepthwiseConv2d])
        FL = s.cache_read(FS, "local", [DepthwiseConv2d])
        if DepthwiseConv2d.op in s.outputs:
            Output = DepthwiseConv2d
            CL = s.cache_write(DepthwiseConv2d, "local")
        else:
            Output = outs[0].op.output(0)
            s[DepthwiseConv2d].set_scope("local")
        # schedule parameters
        num_thread_x = 8
        num_thread_y = 8
        num_vthread_x = 1
        num_vthread_y = 1
        blocking_h = out_height
        blocking_w = out_width
        if out_height % 32 == 0:
            blocking_h = 32
            num_thread_x = 2
            num_vthread_x = 2
        if out_width % 32 == 0:
            blocking_w = 32
            num_thread_y = 16
            num_vthread_y = 2
        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
        thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
        thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")
        # split and bind
        bx, bxi = s[Output].split(Output.op.axis[1], factor=channel_multiplier)
        s[Output].reorder(Output.op.axis[2], Output.op.axis[3], bxi)
        bx = s[Output].fuse(Output.op.axis[0], bx)
        s[Output].bind(bx, block_x)
        by1, y1i = s[Output].split(Output.op.axis[2], factor=blocking_h)
        tvx, vxi = s[Output].split(y1i, nparts=num_vthread_x)
        tx, xi = s[Output].split(vxi, nparts=num_thread_x)
        by2, y2i = s[Output].split(Output.op.axis[3], factor=blocking_w)
        tvy, vyi = s[Output].split(y2i, nparts=num_vthread_y)
        ty, yi = s[Output].split(vyi, nparts=num_thread_y)
        s[Output].reorder(by1, by2, tvx, tvy, tx, ty, xi, yi)
        by = s[Output].fuse(by1, by2)
        s[Output].bind(tvx, thread_vx)
        s[Output].bind(tvy, thread_vy)
        s[Output].bind(tx, thread_x)
        s[Output].bind(ty, thread_y)
        s[Output].bind(by, block_y)
        # local memory load
        s[IL].compute_at(s[Output], ty)
        s[FL].compute_at(s[Output], ty)
        if DepthwiseConv2d.op in s.outputs:
            s[CL].compute_at(s[Output], ty)
        else:
            s[DepthwiseConv2d].compute_at(s[Output], ty)
        # input's shared memory load
        s[IS].compute_at(s[Output], by)
        tx, xi = s[IS].split(IS.op.axis[2], nparts=num_thread_x)
        ty, yi = s[IS].split(IS.op.axis[3], nparts=num_thread_y)
        s[IS].bind(tx, thread_x)
        s[IS].bind(ty, thread_y)
        # filter's shared memory load
        s[FS].compute_at(s[Output], by)
        s[FS].reorder(FS.op.axis[2], FS.op.axis[3], FS.op.axis[1])
        tx, xi = s[FS].split(FS.op.axis[2], nparts=num_thread_x)
        ty, yi = s[FS].split(FS.op.axis[3], nparts=num_thread_y)
        s[FS].bind(tx, thread_x)
        s[FS].bind(ty, thread_y)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if 'ewise' in OP.tag or 'bcast' in OP.tag:
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        if OP.tag == 'depthwise_conv2d_nchw':
            PaddedInput = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            DepthwiseConv2d = OP.output(0)
            _schedule(PaddedInput, Filter, DepthwiseConv2d)

    traverse(outs[0].op)
    return s

def schedule_depthwise_conv2d_nhwc(outs):
    """Schedule for depthwise_conv2d nhwc forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(temp, Filter, DepthwiseConv2d):

        s[temp].compute_inline()
        FS = s.cache_read(Filter, "shared", [DepthwiseConv2d])
        if DepthwiseConv2d.op in s.outputs:
            Output = DepthwiseConv2d
            CL = s.cache_write(DepthwiseConv2d, "local")
        else:
            Output = outs[0].op.output(0)
            s[DepthwiseConv2d].set_scope("local")

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")

        b, h, w, c = s[Output].op.axis

        h_val = tvm.ir_pass.Simplify(Output.shape[1]).value
        b_val = tvm.ir_pass.Simplify(Output.shape[0]).value
        ic_val = tvm.ir_pass.Simplify(temp.shape[3]).value
        xoc, xic = s[Output].split(c, factor=ic_val)
        s[Output].reorder(xoc, b, h, w, xic)
        xo, yo, xi, yi = s[Output].tile(h, w, x_factor=2, y_factor=2)
        fused = s[Output].fuse(yo, xo)
        fused = s[Output].fuse(fused, b)
        fused = s[Output].fuse(fused, xoc)

        s[Output].bind(fused, block_x)
        s[Output].bind(xic, thread_x)

        if DepthwiseConv2d.op in s.outputs:
            s[CL].compute_at(s[Output], xic)
        else:
            s[DepthwiseConv2d].compute_at(s[Output], xic)

        yi, xi, ci, fi = s[FS].op.axis
        s[FS].compute_at(s[Output], fused)
        fused = s[FS].fuse(fi, ci)
        s[FS].bind(fused, thread_x)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if 'ewise' in OP.tag or 'bcast' in OP.tag:
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule depthwise_conv2d
        if OP.tag == 'depthwise_conv2d_nhwc':
            PaddedInput = OP.input_tensors[0]
            Filter = OP.input_tensors[1]
            DepthwiseConv2d = OP.output(0)
            _schedule(PaddedInput, Filter, DepthwiseConv2d)

    traverse(outs[0].op)
    return s
