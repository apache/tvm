# pylint: disable=invalid-name
"""Int8 conv2d in NCHWc layout"""
import tvm
from tvm import autotvm

from .injective import _schedule_injective
from ..generic import schedule_conv2d_NCHWc_int8_prepacked
from .tensor_intrin import dp4a
from ..nn.conv2d import conv2d_NCHWc_int8_prepacked
from ..nn.pad import pad
from ..nn.util import get_pad_tuple
from ..util import get_const_tuple, get_const_int, traverse_inline


def _conv2d_NCHWc_int8_arg_to_workload(data, kernel, stride, padding, out_dtype):
    """convert argument to workload"""
    shape = get_const_tuple(data.shape)
    if len(shape) == 5:
        N, ic_chunk, H, W, ic_block = shape
        raw_data = tvm.placeholder(
            (N, ic_chunk*ic_block, H, W), dtype=data.dtype)
    else:
        raw_data = data

    shape = get_const_tuple(kernel.shape)
    if len(shape) == 6:
        oc_chunk, ic_chunk, KH, KW, oc_block, ic_block = shape
        raw_kernel = tvm.placeholder(
            (oc_chunk*oc_block, ic_chunk*ic_block, KH, KW), dtype=kernel.dtype)
    else:
        raw_kernel = kernel

    return ('conv2d', ) + autotvm.task.task.args_to_workload(
        [raw_data, raw_kernel, stride, padding, "NCHW", out_dtype])


def conv2d_NCHWc_int8(cfg, data, kernel, stride, padding, layout, out_dtype, pre_computed):
    """Convolution operator in NCHW[x]c layout for int8.

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    data : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width] or
        5-D with shape [batch, in_channel_chunk, in_height, in_width, in_channel_block]

    kernel : tvm.Tensor
        4-D with shape [num_filter, in_channel, filter_height, filter_width] or
        6-D with shape [num_filter_chunk, in_channel_chunk, filter_height,
        filter_width, num_filter_block, in_channel_block]

    stride : int or a list/tuple of two ints
        stride size, or [stride_height, stride_width]

    padding: int or a list/tuple of two ints
        padding size, or [pad_height, pad_width]

    layout : str
        layout of data

    out_dtype : str
        The output type. This is used for mixed precision.

    pre_computed : str
        Whether packed data and kernel are pre-computed

    Returns
    -------
    output : tvm.Tensor
        5-D with shape [batch, out_channel_chunk, out_height, out_width, out_channel_block]
    """
    assert layout in ["NCHW", "NCHW4c"]

    ic_block_factor = 4
    oc_block_factor = 4

    if not pre_computed:
        batch, channels, height, width = get_const_tuple(data.shape)
        assert channels % ic_block_factor == 0, \
            "Number of input channels should be multiple of {}".format(
                ic_block_factor)
        packed_data = tvm.compute((batch, channels // ic_block_factor, height, width,
                                   ic_block_factor),
                                  lambda n, c, h, w, vc: data[n, c*ic_block_factor + vc, h, w],
                                  name="packed_data")

        out_channels, in_channels, kernel_h, kernel_w = get_const_tuple(
            kernel.shape)
        assert out_channels % 4 == 0, \
            "Number of output channels should be multiple of {}".format(
                oc_block_factor)
        packed_kernel = tvm.compute(
            (out_channels // oc_block_factor, in_channels // ic_block_factor, kernel_h, kernel_w,
             oc_block_factor, ic_block_factor),
            lambda oc_chunk, ic_chunk, kh, kw, oc_block, ic_block:
            kernel[oc_chunk * oc_block_factor + oc_block,
                   ic_chunk * ic_block_factor + ic_block, kh, kw],
            name="packed_kernel")

    else:
        packed_data = data
        packed_kernel = kernel

    batch, ic_chunk, in_height, in_width, ic_block = get_const_tuple(
        packed_data.shape)
    oc_chunk, ic_chunk, kernel_h, kernel_w, oc_block, ic_block = get_const_tuple(
        packed_kernel.shape)

    if isinstance(stride, int):
        stride_h, stride_w = stride
    else:
        stride_h, stride_w = stride

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (kernel_h, kernel_w))
    # compute graph
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    pad_data = pad(packed_data, pad_before, pad_after, name="pad_data")

    # compute the output shape
    out_height = (in_height - kernel_h + pad_top + pad_down) // stride_h + 1
    out_width = (in_width - kernel_w + pad_left + pad_right) // stride_w + 1

    oshape = (batch, oc_chunk, out_height, out_width, oc_block)

    icc = tvm.reduce_axis((0, ic_chunk), name='ic_chunk')
    icb = tvm.reduce_axis((0, ic_block), name='ic_block')
    kh = tvm.reduce_axis((0, kernel_h), name='kh')
    kw = tvm.reduce_axis((0, kernel_w), name='kw')

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                       tvm.sum(pad_data[n, icc, oh*stride_h+kh, ow*stride_w+kw, icb]
                               .astype('int32') *
                               packed_kernel[oc_chunk, icc,
                                             kh, kw, oc_block, icb]
                               .astype('int32'),
                               axis=[icc, kh, kw, icb]))

    output = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
                         conv[n, oc_chunk, oh, ow, oc_block].astype(out_dtype),
                         tag="conv2d_NCHWc_int8",
                         attrs={"workload": _conv2d_NCHWc_int8_arg_to_workload(
                             data, kernel, stride, padding, out_dtype)})

    # num flop
    num_flop = batch * oc_chunk * oc_block * out_height * out_width * \
        ic_chunk * ic_block * kernel_h * kernel_w * 2
    cfg.add_flop(num_flop)

    return output


_dp4a = dp4a('shared', 'shared', 'local')


def schedule_conv2d_NCHWc_int8(cfg, s, output, pre_computed):
    """Schedule conv2d int8 NCHWc template"""
    workload = output.op.attrs["workload"]

    stride = workload[3]

    conv = output.op.input_tensors[0]
    packed_data, packed_kernel = conv.op.input_tensors

    if isinstance(packed_data.op, tvm.tensor.ComputeOp) and "pad" in packed_data.op.tag:
        pad_data = packed_data
        packed_data = pad_data.op.input_tensors[0]
    else:
        pad_data = packed_data

    if not pre_computed:
        kernel, = packed_kernel.op.input_tensors
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # skip this part during tuning to make recrods accurate
            # this part will be pre-computed during NNVM's pre-compute optimization pass
            s[packed_data].pragma(s[packed_data].op.axis[0], "debug_skip_region")
            s[packed_kernel].pragma(
                s[packed_kernel].op.axis[0], "debug_skip_region")
        else:
            _schedule_injective(packed_data.op, s)
            _schedule_injective(packed_kernel.op, s)
    else:
        kernel = packed_data

    if isinstance(kernel.op, tvm.tensor.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    if pad_data != packed_data:
        s[pad_data].compute_inline()

    batch = get_const_int(packed_data.shape[0])
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [conv])
    WW = s.cache_read(packed_kernel, 'shared', [conv])

    s[conv].set_scope('local')

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    # tile and bind spatial axes
    n, f, y, x, c = s[output].op.axis
    cfg.define_split("tile_f", cfg.axis(f), num_outputs=4)
    cfg.define_split("tile_y", cfg.axis(y), num_outputs=4)
    cfg.define_split("tile_x", cfg.axis(x), num_outputs=4)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, output, f)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    max_block_z = 128
    if batch > max_block_z:
        _, n = s[output].split(n, factor=max_block_z)
    s[output].reorder(n, bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi)
    fused_byx = s[output].fuse(by, bx)
    s[output].bind(n, tvm.thread_axis("blockIdx.z"))
    s[output].bind(bf, tvm.thread_axis("blockIdx.y"))
    s[output].bind(fused_byx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(vf, tvm.thread_axis("vthread"))
    s[output].bind(vy, tvm.thread_axis("vthread"))
    s[output].bind(vx, tvm.thread_axis("vthread"))
    s[output].bind(tf, tvm.thread_axis("threadIdx.z"))
    s[output].bind(ty, tvm.thread_axis("threadIdx.y"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))

    s[conv].compute_at(s[output], tx)

    # tile and bind reduction axes
    n, f, y, x, c = s[conv].op.axis

    rc, ry, rx, rc_block = s[conv].op.reduce_axis
    cfg.define_split("tile_rc", cfg.axis(rc), num_outputs=2)
    cfg.define_split("tile_ry", cfg.axis(ry), num_outputs=2)
    cfg.define_split("tile_rx", cfg.axis(rx), num_outputs=2)
    rco, rci = cfg['tile_rc'].apply(s, conv, rc)
    ryo, ryi = cfg['tile_ry'].apply(s, conv, ry)
    rxo, rxi = cfg['tile_rx'].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x, c, rc_block)

    _, rc_block = s[conv].split(rc_block, factor=4)
    s[conv].tensorize(rc_block, _dp4a)

    s[AA].compute_at(s[conv], rxo)
    s[WW].compute_at(s[conv], rxo)

    # cooperative fetching
    for load in [AA, WW]:
        if load == AA:
            n, f, y, x, c = s[load].op.axis
            if pad_data == packed_data and stride_h == 1 and stride_w == 1:
                s[load].vectorize(c)
                fused = s[load].fuse(n, f, y, x)
            else:
                c, _ = s[load].split(c, factor=4)
                fused = s[load].fuse(n, f, y, x, c)
        else:
            n, f, y, x, oc_chunk, c = s[load].op.axis
            fused = s[load].fuse(n, f, y, x, oc_chunk)
            s[load].vectorize(c)

        fused, tx = s[load].split(fused, factor=cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, factor=cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, factor=cfg["tile_f"].size[2])
        s[load].bind(tz, tvm.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.thread_axis("threadIdx.x"))

    # double buffer
    cfg.define_knob('AA_double_buffer', [0, 1])
    cfg.define_knob('WW_double_buffer', [0, 1])
    if cfg['AA_double_buffer'].val:
        s[AA].double_buffer()
    if cfg['WW_double_buffer'].val:
        s[WW].double_buffer()

    # unroll
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    s[output].pragma(kernel_scope, 'auto_unroll_max_step',
                     cfg['auto_unroll_max_step'].val)
    s[output].pragma(kernel_scope, 'unroll_explicit', False)

    return s


@conv2d_NCHWc_int8_prepacked.register(["cuda"])
@autotvm.task.dispatcher
def conv2d_NCHWc_int8_prepacked_dispatcher(data, kernel, stride, padding, layout, out_dtype):
    assert layout == 'NCHW4c'
    return _conv2d_NCHWc_int8_arg_to_workload(data, kernel, stride, padding, out_dtype)


@conv2d_NCHWc_int8_prepacked_dispatcher.register("int8")
def _decl_conv2d_NCHWc_int8_prepacked(cfg, data, kernel, stride, padding, layout, out_dtype):
    return conv2d_NCHWc_int8(cfg, data, kernel, stride, padding, layout, out_dtype,
                             pre_computed=True)

@autotvm.register_topi_schedule(schedule_conv2d_NCHWc_int8_prepacked, ["cuda"], ["int8"])
def schedule_conv2d_NCHWc_int8_prepacked_cuda(cfg, outs):
    """TOPI schedule callback of conv2d for cuda

    Parameters
    ----------
    cfg: ConfigEntity
        The config for this template

    outs: Array of Tensor
        The computation graph description of conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d.
    """
    s = tvm.create_schedule([x.op for x in outs])

    def _callback(op):
        if 'conv2d_NCHWc_int8' in op.tag:
            schedule_conv2d_NCHWc_int8(cfg, s, op.output(0), pre_computed=True)

    traverse_inline(s, outs[0].op, _callback)
    return s
