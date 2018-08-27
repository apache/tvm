# pylint: disable=invalid-name,unused-variable,unused-argument
"""HLS nn operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic


def _schedule_conv2d(outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule conv2d
        elif OP.tag.find("conv2d") >= 0:
            Conv2d = OP.output(0)
            if not Conv2d.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Conv2d].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, tvm.thread_axis("pipeline"))
    return s


@generic.schedule_conv2d_nchw.register(["hls"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


@generic.schedule_conv2d_nhwc.register(["hls"])
def schedule_conv2d_nhwc(outs):
    """Schedule for conv2d_nhwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


@generic.schedule_conv2d_NCHWc.register(["hls"])
def schedule_conv2d_NCHWc(num_filter, kernel_size, strides,
                          padding, layout, out_layout, outs):
    """Schedule for conv2d_NCHW[x]c

    Parameters
    ----------
    num_filter : int
        The number of filter, i.e., the output channel.

    kernel_size : tuple of int
        (kernel_height, kernel_width)

    strides : tuple of int
        (stride_of_height, stride_of_width)

    padding : tuple of int
        (pad_of_height, pad_of_width)

    layout : str
        Input data layout

    out_layout : str
        Output data layout

    outs : Array of Tensor
        The computation graph description of conv2d_NCHWc
        in the format of an array of tensors.

    Returns
    -------
    sch : Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


@generic.schedule_conv2d_transpose_nchw.register(["hls"])
def schedule_conv2d_transpose_nchw(outs):
    """Schedule for conv2d_transpose_nchw

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_transpose_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


@generic.schedule_depthwise_conv2d_nchw.register(["hls"])
def schedule_depthwise_conv2d_nchw(outs):
    """Schedule for depthwise_conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of depthwise_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


@generic.schedule_depthwise_conv2d_nhwc.register(["hls"])
def schedule_depthwise_conv2d_nhwc(outs):
    """Schedule for depthwise_conv2d_nhwc
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of depthwise_conv2d_nhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)

@generic.schedule_bitserial_conv2d_nchw.register(["hls"])
def schedule_bitserial_conv2d_nchw(outs):
    """Schedule for bitserial_conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of bitserial_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


@generic.schedule_bitserial_conv2d_nhwc.register(["hls"])
def schedule_bitserial_conv2d_nhwc(outs):
    """Schedule for bitserial_conv2d_nhwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of bitserial_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _schedule_conv2d(outs)


@generic.schedule_reduce.register(["hls"])
def schedule_reduce(outs):
    """Schedule for reduction

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif OP.tag in ["comm_reduce", "comm_reduce_idx"]:
            if OP.tag == "comm_reduce":
                Reduce = OP.output(0)
            else:
                Reduce = OP.input_tensors[0]
            if not Reduce.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Reduce].compute_at(s[Out], s[Out].op.axis[0])
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)

    fused = s[outs[0]].fuse()
    px, x = s[outs[0]].split(fused, nparts=1)
    s[outs[0]].bind(px, tvm.thread_axis("pipeline"))
    return s


@generic.schedule_softmax.register(["hls"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)

    softmax = outs[0]
    max_elem = softmax.op.input_tensors[1]
    expsum = softmax.op.input_tensors[2]

    s[expsum].compute_at(s[softmax], s[softmax].op.axis[1])
    s[max_elem].compute_at(s[softmax], s[softmax].op.axis[1])

    px, x = s[softmax].split(softmax.op.axis[0], nparts=1)
    s[softmax].bind(px, tvm.thread_axis("pipeline"))
    return s


@generic.schedule_dense.register(["hls"])
def schedule_dense(outs):
    """Schedule for dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            Dense = OP.output(0)
            if not Dense.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Dense].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, tvm.thread_axis("pipeline"))
    return s


@generic.schedule_pool.register(["hls"])
def schedule_pool(outs, layout):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith('pool'):
            Pool = OP.output(0)
            if not Pool.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Pool].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, tvm.thread_axis("pipeline"))
    return s


@generic.schedule_global_pool.register(["hls"])
def schedule_global_pool(outs):
    """Schedule for global pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of global pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule global_pool
        elif OP.tag.startswith('global_pool'):
            Pool = OP.output(0)
            if not Pool.op in s.outputs:
                Out = outs[0].op.output(0)
                s[Pool].compute_at(s[Out], s[Out].op.axis[1])
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)

    px, x = s[outs[0]].split(outs[0].op.axis[0], nparts=1)
    s[outs[0]].bind(px, tvm.thread_axis("pipeline"))
    return s
