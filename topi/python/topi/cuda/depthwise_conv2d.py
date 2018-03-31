# pylint: disable=invalid-name
"""Schedule for depthwise_conv2d with auto fusion"""
import tvm
from ..util import get_const_tuple
from .. import tag
from .. import generic

@generic.schedule_depthwise_conv2d_nchw.register(["cuda", "gpu"])
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
        in_shape = get_const_tuple(PaddedInput.shape)
        out_shape = get_const_tuple(DepthwiseConv2d.shape)
        in_height = in_shape[2]
        in_width = in_shape[3]
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
        num_thread_y = 8
        num_thread_x = 8
        num_vthread_y = 1
        num_vthread_x = 1
        blocking_h = out_height
        blocking_w = out_width
        if out_height % 32 == 0 or in_height >= 108:
            blocking_h = 32
        if out_width % 32 == 0:
            blocking_w = 32
            num_thread_x = 16
            num_vthread_x = 2
        elif in_width >= 108:
            blocking_w = 32
        block_y = tvm.thread_axis("blockIdx.y")
        block_x = tvm.thread_axis("blockIdx.x")
        thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
        thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
        thread_vy = tvm.thread_axis((0, num_vthread_y), "vthread", name="vy")
        thread_vx = tvm.thread_axis((0, num_vthread_x), "vthread", name="vx")
        # split and bind
        by, byi = s[Output].split(Output.op.axis[1], factor=channel_multiplier)
        s[Output].reorder(Output.op.axis[2], Output.op.axis[3], byi)
        by = s[Output].fuse(Output.op.axis[0], by)
        s[Output].bind(by, block_y)
        bx1, x1i = s[Output].split(Output.op.axis[2], factor=blocking_h)
        tvy, vyi = s[Output].split(x1i, nparts=num_vthread_y)
        ty, yi = s[Output].split(vyi, nparts=num_thread_y)
        bx2, x2i = s[Output].split(Output.op.axis[3], factor=blocking_w)
        tvx, vxi = s[Output].split(x2i, nparts=num_vthread_x)
        tx, xi = s[Output].split(vxi, nparts=num_thread_x)
        s[Output].reorder(bx1, bx2, tvy, tvx, ty, tx, yi, xi)
        bx = s[Output].fuse(bx1, bx2)
        s[Output].bind(bx, block_x)
        s[Output].bind(tvy, thread_vy)
        s[Output].bind(tvx, thread_vx)
        s[Output].bind(ty, thread_y)
        s[Output].bind(tx, thread_x)
        # local memory load
        s[IL].compute_at(s[Output], tx)
        s[FL].compute_at(s[Output], tx)
        if DepthwiseConv2d.op in s.outputs:
            s[CL].compute_at(s[Output], tx)
        else:
            s[DepthwiseConv2d].compute_at(s[Output], tx)
        # input's shared memory load
        s[IS].compute_at(s[Output], bx)
        ty, yi = s[IS].split(IS.op.axis[2], nparts=num_thread_y)
        tx, xi = s[IS].split(IS.op.axis[3], nparts=num_thread_x)
        s[IS].bind(ty, thread_y)
        s[IS].bind(tx, thread_x)
        # filter's shared memory load
        s[FS].compute_at(s[Output], bx)
        s[FS].reorder(FS.op.axis[2], FS.op.axis[3], FS.op.axis[1])
        ty, yi = s[FS].split(FS.op.axis[2], nparts=num_thread_y)
        tx, xi = s[FS].split(FS.op.axis[3], nparts=num_thread_x)
        s[FS].bind(ty, thread_y)
        s[FS].bind(tx, thread_x)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
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

@generic.schedule_depthwise_conv2d_nhwc.register(["cuda", "gpu"])
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

        # num_thread here could be 728, it is larger than cuda.max_num_threads
        num_thread = tvm.ir_pass.Simplify(temp.shape[3]).value
        target = tvm.target.current_target()
        if target and target.target_name != "cuda":
            num_thread = target.max_num_threads
        xoc, xic = s[Output].split(c, factor=num_thread)
        s[Output].reorder(xoc, b, h, w, xic)
        xo, yo, _, _ = s[Output].tile(h, w, x_factor=2, y_factor=2)
        fused = s[Output].fuse(yo, xo)
        fused = s[Output].fuse(fused, b)
        fused = s[Output].fuse(fused, xoc)

        s[Output].bind(fused, block_x)
        s[Output].bind(xic, thread_x)

        if DepthwiseConv2d.op in s.outputs:
            s[CL].compute_at(s[Output], xic)
        else:
            s[DepthwiseConv2d].compute_at(s[Output], xic)

        _, _, ci, fi = s[FS].op.axis
        s[FS].compute_at(s[Output], fused)
        fused = s[FS].fuse(fi, ci)
        s[FS].bind(fused, thread_x)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
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


def schedule_depthwise_conv2d_backward_input_nhwc(outs):
    """Schedule for depthwise_conv2d nhwc backward wrt input.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        backward wrt input in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d backward
        wrt input with layout nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Padded_out_grad, In_grad):
        s[Padded_out_grad].compute_inline()

        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis("threadIdx.x")
        _, h, w, c = In_grad.op.axis

        fused_hwc = s[In_grad].fuse(h, w, c)
        xoc, xic = s[In_grad].split(fused_hwc, factor=128)

        s[In_grad].bind(xoc, block_x)
        s[In_grad].bind(xic, thread_x)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if OP.tag == 'depthwise_conv2d_backward_input_nhwc':
            Padded_out_grad = OP.input_tensors[0]
            Dilated_out_grad = Padded_out_grad.op.input_tensors[0]
            s[Dilated_out_grad].compute_inline()
            In_grad = OP.output(0)
            _schedule(Padded_out_grad, In_grad)
        else:
            raise ValueError("Depthwise conv backward wrt input for non-NHWC is not supported.")

    traverse(outs[0].op)
    return s

def schedule_depthwise_conv2d_backward_weight_nhwc(outs):
    """Schedule for depthwise_conv2d nhwc backward wrt weight.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        backward wrt weight in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d backward
        wrt weight with layout nhwc.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Weight_grad):
        block_x = tvm.thread_axis("blockIdx.x")
        thread_y = tvm.thread_axis("threadIdx.y")
        thread_x = tvm.thread_axis("threadIdx.x")

        db, dh, dw = Weight_grad.op.reduce_axis

        fused_dbdhdw = s[Weight_grad].fuse(db, dh, dw)
        _, ki = s[Weight_grad].split(fused_dbdhdw, factor=8)
        BF = s.rfactor(Weight_grad, ki)

        fused_fwcm = s[Weight_grad].fuse(*s[Weight_grad].op.axis)

        xo, xi = s[Weight_grad].split(fused_fwcm, factor=32)

        s[Weight_grad].bind(xi, thread_x)
        s[Weight_grad].bind(xo, block_x)

        s[Weight_grad].bind(s[Weight_grad].op.reduce_axis[0], thread_y)
        s[BF].compute_at(s[Weight_grad], s[Weight_grad].op.reduce_axis[0])

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if OP.tag == 'depthwise_conv2d_backward_weight_nhwc':
            Padded_in = OP.input_tensors[1]
            s[Padded_in].compute_inline()
            Weight_grad = OP.output(0)
            _schedule(Weight_grad)
        else:
            raise ValueError("Depthwise conv backward wrt weight for non-NHWC is not supported.")

    traverse(outs[0].op)
    return s
