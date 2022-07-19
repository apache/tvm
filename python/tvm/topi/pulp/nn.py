from __future__ import absolute_import as _abs
import logging
from os import wait
from tvm import te, tir, autotvm
from tvm.target import Target
import re
from ..nn.pad import pad
from ..utils import simplify, get_const_tuple, traverse_inline
from ..nn.utils import get_pad_tuple1d, get_pad_tuple
from .. import tag

logger = logging.getLogger(__name__)

def sdotp(data_dtype, kernel_dtype, out_dtype, vec_length, data_is_last_axis=True, kernel_is_last_axis=True, dilation=1):

    flip = False
    if data_dtype.startswith("u"):
        if kernel_dtype.startswith("u"):
            intrinsic_name = "llvm.riscv.pulp.sdotup"
        else:
            intrinsic_name = "llvm.riscv.pulp.sdotusp"
    else:
        if kernel_dtype.startswith("u"):
            intrinsic_name = "llvm.riscv.pulp.sdotusp"
            flip = True
        else:
            intrinsic_name = "llvm.riscv.pulp.sdotsp"

    intrinsic_name += str(vec_length)

    a_size = (vec_length - 1) * dilation + 1
    a_shape = (a_size,) if data_is_last_axis else (a_size, 1)
    a = te.placeholder(a_shape, data_dtype, "a")

    b_shape = (vec_length,) if kernel_is_last_axis else (vec_length, 1)
    b = te.placeholder(b_shape, kernel_dtype, "b")

    k = te.reduce_axis((0, vec_length), "k")

    ak = a[k * dilation] if data_is_last_axis else a[k * dilation, 0]
    ak = ak.astype(out_dtype)

    bk = b[k] if kernel_is_last_axis else b[k, 0]
    bk = bk.astype(out_dtype)

    c = te.compute((1,), lambda _: te.sum(ak * bk, [k]), "c")

    def intrin_func(ins, outs):
        aa, bb = ins
        cc,    = outs

        #no easier way to build a vector?
        if data_is_last_axis and dilation == 1:
            aaval = aa.vload([0], data_dtype + "x" + str(vec_length))
        else:
            aaval = tir.Load(data_dtype + "x" + str(vec_length), aa.data,
                             tir.Ramp(aa.elem_offset, dilation * aa.strides[0], vec_length))

        if kernel_is_last_axis:
            bbval = bb.vload([0], kernel_dtype + "x" + str(vec_length))
        else:
            bbval = tir.Load(kernel_dtype + "x" + str(vec_length), bb.data,
                             tir.Ramp(bb.elem_offset, bb.strides[0], vec_length))

        if flip:
            aaval, bbval = bbval, aaval

        ccval = cc.vload([0]).astype("int32")

        call = tir.call_llvm_pure_intrin(
            "int32",
            intrinsic_name,
            3,
            aaval, bbval, ccval
        ).astype(out_dtype)

        body = cc.vstore([0], call)

        reduce_init = cc.vstore([0], tir.IntImm(out_dtype, 0))
        return None, reduce_init, body

    Ab = tir.decl_buffer(a.shape, a.dtype, name="A",
                         offset_factor=1, strides=[1] if data_is_last_axis else [te.var("batch"), 1], data_alignment=1)
    Bb = tir.decl_buffer(b.shape, b.dtype, name="B",
                         offset_factor=1, strides=[1] if kernel_is_last_axis else [te.var("out_channels"), 1], data_alignment=1)
    Cb = tir.decl_buffer(c.shape, c.dtype, name="C",
                         offset_factor=1, strides=[1], data_alignment=1)

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def get_vec_length(out, data, weight):
    tgt = Target.current()
    if tgt.kind.name == "llvm" and "+xpulpv" in tgt.mattr and re.match("u?int(8|16|32)", out):
        if re.match("u?int8", data) and re.match("u?int8", weight):
            return 4
        elif re.match("u?int16", data) and re.match("u?int16", weight):
            return 2

    return 1


@autotvm.register_topi_schedule("conv1d_ncw.pulp")
def schedule_conv1d_ncw(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_ncw":
            data, weight = op.input_tensors
            out = op.output(0)
            n, c, w = op.axis
            rc, rw = op.reduce_axis

            workload = op.attrs["workload"]
            dilation = get_const_tuple(workload[5])

            s[data.op.input_tensors[0]].compute_inline()
            s[data].compute_at(s[out], rw)
            s[weight].compute_at(s[out], rw)

            vec_length = get_vec_length(out.dtype, data.dtype, weight.dtype)

            if vec_length != 1:
                rwo, rwi = s[out].split(rw, vec_length)
                t = sdotp(data.dtype, weight.dtype, out.dtype, vec_length, dilation=dilation[0])
                s[out].tensorize(rwi, t)

                cfg.define_split("tile_w", w, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                wo, wi = cfg["tile_w"].apply(s, out, w)
                cfg.define_split("tile_rwo", rwo, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                rwoo, rwoi = cfg["tile_rwo"].apply(s, out, rwo)

                s[out].reorder(n, wo, rwoo, c, rc, wi, rwoi, rwi)

                s[data].compute_at(s[out], rwoi)
                s[weight].compute_at(s[out], rwoi)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_compute("conv1d_ncw.pulp")
def conv1d_ncw(cfg, data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution forward operator for NCW layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_channel, in_width]

    kernel : tvm.te.Tensor
        3-D with shape [num_filter, in_channel, filter_size]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    batch, in_channels, data_width = data.shape
    out_channels, _, kernel_size = kernel.shape


    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_channels = simplify(out_channels)
    out_width = simplify((data_width - dilated_kernel_size + pad_left + pad_right) // strides + 1)

    # Compute flop
    cfg.add_flop(2 * batch * out_channels * out_width * kernel_size * in_channels)

    # Apply padding
    pad_before = [0, 0, pad_left]
    pad_after = [0, 0, pad_right]
    temp = pad(data, pad_before, pad_after, name="pad_temp")

    # Apply padding for tensorization
    pad_tensorize = -kernel_size % get_vec_length(out_dtype, data.dtype, kernel.dtype)
    temp = pad(temp, (0, 0, 0), (0, 0, pad_tensorize * dilation))
    kernel = pad(kernel, (0, 0, 0), (0, 0, pad_tensorize))

    # Compute graph
    rc = te.reduce_axis((0, in_channels), name="rc")
    rw = te.reduce_axis((0, kernel_size + pad_tensorize), name="rw")

    return te.compute(
        (batch, out_channels, out_width),
        lambda b, c, w: te.sum(
            temp[b, rc, w * strides + rw * dilation].astype(out_dtype)
            * kernel[c, rc, rw].astype(out_dtype),
            axis=[rc, rw],
        ),
        tag="conv1d_ncw",
    )


@autotvm.register_topi_schedule("conv1d_nwc.pulp")
def schedule_conv1d_nwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_nwc":
            data, weight = op.input_tensors
            out = op.output(0)
            n, w, c = op.axis
            rw, rc = op.reduce_axis

            s[data.op.input_tensors[0]].compute_inline()
            s[data].compute_at(s[out], rw)
            s[weight].compute_at(s[out], rw)

            vec_length = get_vec_length(out.dtype, data.dtype, weight.dtype)

            if vec_length != 1:
                t = sdotp(data.dtype, weight.dtype, out.dtype, vec_length, kernel_is_last_axis=False)
                rco, rci = s[out].split(rc, vec_length)
                s[out].tensorize(rci, t)

                cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                co, ci = cfg["tile_c"].apply(s, out, c)
                cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

                s[out].reorder(n, co, rcoo, w, rw, ci, rcoi)

                s[data].compute_at(s[out], rcoi)
                s[weight].compute_at(s[out], rcoi)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_compute("conv1d_nwc.pulp")
def conv1d_nwc(cfg, data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution forward operator for NWC layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_width, in_channel]

    kernel : tvm.te.Tensor
        3-D with shape [filter_size, in_channel, num_filter]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    batch, data_width, in_channels = data.shape
    kernel_size, _, out_channels = kernel.shape

    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_channels = simplify(out_channels)
    out_width = simplify((data_width - dilated_kernel_size + pad_left + pad_right) // strides + 1)

    # Compute flop
    cfg.add_flop(2 * batch * out_channels * out_width * kernel_size * in_channels)

    # Apply padding
    pad_before = [0, pad_left, 0]
    pad_after = [0, pad_right, 0]
    temp = pad(data, pad_before, pad_after, name="pad_temp")

    # Apply padding for tensorization
    pad_tensorize = -in_channels % get_vec_length(out_dtype, data.dtype, kernel.dtype)
    temp = pad(temp, (0, 0, 0), (0, 0, pad_tensorize))
    kernel = pad(kernel, (0, 0, 0), (0, pad_tensorize, 0))

    # Compute graph
    rc = te.reduce_axis((0, in_channels + pad_tensorize), name="rc")
    rw = te.reduce_axis((0, kernel_size), name="rw")

    return te.compute(
        (batch, out_width, out_channels),
        lambda b, w, c: te.sum(
            temp[b, w * strides + rw * dilation, rc].astype(out_dtype)
            * kernel[rw, rc, c].astype(out_dtype),
            axis=[rw, rc],
        ),
        tag="conv1d_nwc",
    )


@autotvm.register_topi_schedule("conv1d_nwc_owi.pulp")
def schedule_conv1d_nwc_owi(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_nwc_owi":
            data, weight = op.input_tensors
            out = op.output(0)
            n, w, c = op.axis
            rw, rc = op.reduce_axis

            s[data.op.input_tensors[0]].compute_inline()
            s[data].compute_at(s[out], rc)
            s[weight].compute_at(s[out], rc)

            vec_length = get_vec_length(out.dtype, data.dtype, weight.dtype)

            if vec_length != 1:
                t = sdotp(data.dtype, weight.dtype, out.dtype, vec_length)
                rco, rci = s[out].split(rc, vec_length)
                s[out].tensorize(rci, t)

                cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                co, ci = cfg["tile_c"].apply(s, out, c)
                cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

                s[out].reorder(n, rcoo, w, rw, co, ci, rcoi, rci)

                s[data].compute_at(s[out], rcoi)
                s[weight].compute_at(s[out], rcoi)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_compute("conv1d_nwc_owi.pulp")
def conv1d_nwc_owi(cfg, data, kernel, strides=1, padding="VALID", dilation=1, out_dtype=None):
    """1D convolution forward operator for NWC layout.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D with shape [batch, in_width, in_channel]

    kernel : tvm.te.Tensor
        3-D with shape [num_filter, filter_size, in_channel]

    strides : int or tuple
        The spatial stride along width

    padding : int, tuple, or str
        Padding size can be an integer for equal padding,
        a tuple of (left, right) or a string in ['VALID', 'SAME'].

    dilation : int or tuple
        Dilation rate if convolution should be dilated.

    out_dtype : str
        The output data type. If None then output is same type as input.
    """
    if out_dtype is None:
        out_dtype = data.dtype
    if isinstance(strides, (tuple, list)):
        strides = strides[0]
    if isinstance(dilation, (tuple, list)):
        dilation = dilation[0]

    batch, data_width, in_channels = data.shape
    out_channels, kernel_size, _ = kernel.shape

    # Compute the output shape
    dilated_kernel_size = (kernel_size - 1) * dilation + 1
    pad_left, pad_right = get_pad_tuple1d(padding, (dilated_kernel_size,))
    out_channels = simplify(out_channels)
    out_width = simplify((data_width - dilated_kernel_size + pad_left + pad_right) // strides + 1)

    # Compute flop
    cfg.add_flop(2 * batch * out_channels * out_width * kernel_size * in_channels)

    # Apply padding
    pad_before = [0, pad_left, 0]
    pad_after = [0, pad_right, 0]
    temp = pad(data, pad_before, pad_after, name="pad_temp")

    # Apply padding for tensorization
    pad_tensorize = -in_channels % get_vec_length(out_dtype, data.dtype, kernel.dtype)
    temp = pad(temp, (0, 0, 0), (0, 0, pad_tensorize))
    kernel = pad(kernel, (0, 0, 0), (0, 0, pad_tensorize))

    # Compute graph
    rc = te.reduce_axis((0, in_channels + pad_tensorize), name="rc")
    rw = te.reduce_axis((0, kernel_size), name="rw")

    return te.compute(
        (batch, out_width, out_channels),
        lambda b, w, c: te.sum(
            temp[b, w * strides + rw * dilation, rc].astype(out_dtype)
            * kernel[c, rw, rc].astype(out_dtype),
            axis=[rw, rc],
        ),
        tag="conv1d_nwc_owi",
    )


@autotvm.register_topi_schedule("conv2d_nchw.pulp")
def schedule_conv2d_nchw(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_nchw":
            data, weight = op.input_tensors
            out = op.output(0)
            n, c, h, w = op.axis
            rc, ry, rx = op.reduce_axis

            workload = op.attrs["workload"]
            dilation = get_const_tuple(workload[5])

            s[data.op.input_tensors[0]].compute_inline()
            s[data].compute_at(s[out], rx)
            s[weight].compute_at(s[out], rx)

            vec_length = get_vec_length(out.dtype, data.dtype, weight.dtype)

            if vec_length != 1:
                t = sdotp(data.dtype, weight.dtype, out.dtype, vec_length, dilation=dilation[1])
                rxo, rxi = s[out].split(rx, vec_length)
                s[out].tensorize(rxi, t)

                cfg.define_split("tile_w", w, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                wo, wi = cfg["tile_w"].apply(s, out, w)
                cfg.define_split("tile_rxo", rxo, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                rxoo, rxoi = cfg["tile_rxo"].apply(s, out, rxo)

                s[out].reorder(n, wo, rxoo, h, c, rc, ry, wi, rxoi)

                s[data].compute_at(s[out], rxoi)
                s[weight].compute_at(s[out], rxoi)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_compute("conv2d_nchw.pulp")
def conv2d_nchw(cfg, Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHW layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    Returns
    -------
    Output : tvm.te.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel, in_height, in_width = Input.shape
    num_filter, channel, kernel_h, kernel_w = Filter.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # Compute flop
    cfg.add_flop(2 * batch * out_channel * out_height * out_width * kernel_h * kernel_w * in_channel)

    # Apply padding
    pad_before = [0, 0, pad_top, pad_left]
    pad_after = [0, 0, pad_down, pad_right]
    temp = pad(Input, pad_before, pad_after, name="pad_temp")

    # Apply padding for tensorization
    pad_tensorize = -kernel_w % get_vec_length(out_dtype, Input.dtype, Filter.dtype)
    temp = pad(temp, (0, 0, 0, 0), (0, 0, 0, pad_tensorize * dilation_w))
    Filter = pad(Filter, (0, 0, 0, 0), (0, 0, 0, pad_tensorize))

    # compute graph
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w + pad_tensorize), name="rx")
    return te.compute(
        (batch, out_channel, out_height, out_width),
        lambda nn, ff, yy, xx: te.sum(
            temp[nn, rc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w].astype(
                out_dtype
            )
            * Filter[ff, rc, ry, rx].astype(out_dtype),
            axis=[rc, ry, rx],
        ),
        tag="conv2d_nchw",
    )


@autotvm.register_topi_schedule("conv2d_nhwc.pulp")
def schedule_conv2d_nhwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv1d_nhwc":
            data, weight = op.input_tensors
            out = op.output(0)
            n, h, w, c = op.axis
            ry, rx, rc = op.reduce_axis

            s[data.op.input_tensors[0]].compute_inline()
            s[data].compute_at(s[out], rc)
            s[weight].compute_at(s[out], rc)

            vec_length = get_vec_length(out.dtype, data.dtype, weight.dtype)

            if vec_length != 1:
                t = sdotp(data.dtype, weight.dtype, out.dtype, vec_length, kernel_is_last_axis=False)
                rco, rci = s[out].split(rc, vec_length)
                s[out].tensorize(rci, t)

                cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                co, ci = cfg["tile_c"].apply(s, out, c)
                cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

                s[out].reorder(n, co, rcoo, h, w, ry, rx, ci, rcoi)

                s[data].compute_at(s[out], rcoi)
                s[weight].compute_at(s[out], rcoi)

    traverse_inline(s, outs[0].op, _callback)

    return s


@autotvm.register_topi_compute("conv2d_nhwc.pulp")
def conv2d_nhwc(
    cfg,
    Input,
    Filter,
    stride,
    padding,
    dilation,
    out_dtype="float32",
):
    """Convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        4-D with shape [filter_height, filter_width, in_channel, num_filter]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str = "float32",
        The type of output tensor

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    kernel_h, kernel_w, channel, num_filter = Filter.shape
    batch, in_height, in_width, in_channel = Input.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # Compute flop
    cfg.add_flop(2 * batch * out_channel * out_height * out_width * kernel_h * kernel_w * in_channel)

    # Apply padding
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")

    # Apply padding for tensorization
    pad_tensorize = -in_channel % get_vec_length(out_dtype, Input.dtype, Filter.dtype)
    PaddedInput = pad(PaddedInput, (0, 0, 0, 0), (0, 0, 0, pad_tensorize))
    Filter = pad(Filter, (0, 0, 0, 0), (0, 0, pad_tensorize, 0))

    rc = te.reduce_axis((0, in_channel + pad_tensorize), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            PaddedInput[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * Filter[ry, rx, rc, ff].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="Conv2dOutput",
        tag="conv2d_nhwc",
    )

    return Output


@autotvm.register_topi_schedule("conv2d_nhwc_ohwi.pulp")
def schedule_conv2d_nhwc_ohwi(cfg : autotvm.ConfigSpace, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv2d_nhwc_ohwi":
            data, weight = op.input_tensors
            out = op.output(0)
            n, h, w, c = op.axis
            ry, rx, rc = op.reduce_axis


    
            #s[data.op.input_tensors[0]].compute_inline()
            #s[data].compute_at(s[out], rc)
            #s[weight].compute_at(s[out], rc)

            vec_length = get_vec_length(out.dtype, data.dtype, weight.dtype)
            from tvm.target import codegen
            llvm_id = codegen.llvm_lookup_intrinsic_id(llvm.riscv.pulp.sdotsp4)
            if llvm_id == 0:
                logger.critical("llvm version does not support llvm intrinsics")
            if vec_length != 1 and llvm_id != 0:
                t = sdotp(data.dtype, weight.dtype, out.dtype, vec_length)
                rco, rci = s[out].split(rc, vec_length)
                
                
                s[out].tensorize(rci, t)


                cfg.define_split("tile_c", c, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                co, ci = cfg["tile_c"].apply(s, out, c)
                cfg.define_split("tile_rco", rco, num_outputs=2, policy="candidate", candidate=[[-1, 1],[-1, 2],[-1, 4],[-1, 8]])
                rcoo, rcoi = cfg["tile_rco"].apply(s, out, rco)

                s[out].reorder(n, co, rcoo, h, w, ry, rx, ci, rcoi)
                cfg.define_reorder("axis_order", [co, rcoo, h, w, ry, rx], policy="all")
                

                #s[data].compute_at(s[out], ci)
                #s[weight].compute_at(s[out], co)

    traverse_inline(s, outs[0].op, _callback)

    return s

@autotvm.register_topi_compute("conv2d_nhwc_ohwi.pulp")
def conv2d_nhwc_ohwi(
    cfg,
    Input,
    Filter,
    stride,
    padding,
    dilation,
    out_dtype="float32",
):
    """Convolution operator in NHWC layout.

    Parameters
    ----------
    Input : tvm.te.Tensor
        4-D with shape [batch, in_height, in_width, in_channel]

    Filter : tvm.te.Tensor
        4-D with shape [num_filter, filter_height, filter_width, in_channel]

    stride : int or a list/tuple of two ints
        Stride size, or [stride_height, stride_width]

    padding : int or a list/tuple of 2 or 4 ints
        padding size, or
        [pad_height, pad_width] for 2 ints, or
        [pad_top, pad_left, pad_bottom, pad_right] for 4 ints

    dilation: int or a list/tuple of two ints
        dilation size, or [dilation_height, dilation_width]

    out_dtype: str = "float32",
        The type of output tensor

    Returns
    -------
    output : tvm.te.Tensor
        4-D with shape [batch, out_height, out_width, out_channel]
    """
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    num_filter, kernel_h, kernel_w, channel = Filter.shape
    batch, in_height, in_width, in_channel = Input.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)

    # Compute flop
    cfg.add_flop(2 * batch * out_channel * out_height * out_width * kernel_h * kernel_w * in_channel)

    # Apply padding
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")

    # Apply padding for tensorization
    pad_tensorize = -in_channel % get_vec_length(out_dtype, Input.dtype, Filter.dtype)
    if pad_tensorize != 0:
        PaddedInput = pad(PaddedInput, (0, 0, 0, 0), (0, 0, 0, pad_tensorize))
        Filter = pad(Filter, (0, 0, 0, 0), (0, 0, 0, pad_tensorize))

    rc = te.reduce_axis((0, in_channel + pad_tensorize), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            PaddedInput[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * Filter[ff, ry, rx, rc].astype(out_dtype),
            axis=[ry, rx, rc],
        ),
        name="Conv2dOutput",
        tag="conv2d_nhwc_ohwi",
    )

    return Output


def schedule_pool(outs, layout):
    """Schedule for pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of pool
        in the format of an array of tensors.

    layout: str
        Data layout.

    Returns
    -------
    s: Schedule
        The computation schedule for pool.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, te.ComputeOp):
            s[PaddedInput].compute_inline()

    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith("pool"):
            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
