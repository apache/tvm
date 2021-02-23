# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""conv2d schedule on Qualcomm Adreno GPU"""
import tvm
from tvm import te
from tvm import autotvm

from tvm.topi import nn
from tvm.topi.utils import simplify
from ..utils import get_const_tuple, traverse_inline


@autotvm.register_topi_compute("conv2d_nchwc.image2d")
def conv2d_nchwc(cfg, data, kernel, strides, padding, dilation, out_dtype="float16"):
    """Compute conv2d with NCHWc layout"""
    args={"memory" : "texture", "shared" : False, "accumulator" : "float16"}
    return compute_conv2d_NCHWc_KCRSk(data, kernel, strides, padding, dilation, out_dtype, args=args)

@autotvm.register_topi_compute("conv2d_nchwc_acc32.image2d")
def conv2d_nchwc_acc32(cfg, data, kernel, strides, padding, dilation, out_dtype="float16"):
    """Compute conv2d with NCHWc layout"""
    args={"memory" : "texture", "shared" : False, "accumulator" : "float32"}
    return compute_conv2d_NCHWc_KCRSk(data, kernel, strides, padding, dilation, out_dtype, args=args)

@autotvm.register_topi_schedule("conv2d_nchwc.image2d")
def schedule_conv2d_nchwc(cfg, outs):
    return schedule_conv2d_nchwc_impl(cfg, outs, tag="cast_from_acc16")

@autotvm.register_topi_schedule("conv2d_nchwc_acc32.image2d")
def schedule_conv2d_nchwc_acc32(cfg, outs):
    return schedule_conv2d_nchwc_impl(cfg, outs, tag="cast_from_acc32")

@autotvm.register_topi_compute("depthwise_conv2d_nchwc.image2d")
def depthwise_conv2d_nchwc(cfg, data, kernel, strides, padding, dilation, out_dtype="float16"):
    """Compute depthwise_conv2d with NCHWc layout"""
    args={"memory" : "texture", "shared" : False, "accumulator" : "float16"}
    return compute_depthwise_conv2d_NCHWc_KCRSk(data, kernel, strides, padding, dilation, out_dtype, args=args)

@autotvm.register_topi_compute("depthwise_conv2d_nchwc_acc32.image2d")
def depthwise_conv2d_nchwc_acc32(cfg, data, kernel, strides, padding, dilation, out_dtype="float16"):
    """Compute depthwise_conv2d with NCHWc layout"""
    args={"memory" : "texture", "shared" : False, "accumulator" : "float32"}
    return compute_depthwise_conv2d_NCHWc_KCRSk(data, kernel, strides, padding, dilation, out_dtype, args=args)

@autotvm.register_topi_schedule("depthwise_conv2d_nchwc.image2d")
def schedule_depthwise_conv2d_nchwc(cfg, outs):
    return schedule_depthwise_conv2d_nchwc_impl(cfg, outs, tag="cast_from_acc16")

@autotvm.register_topi_schedule("depthwise_conv2d_nchwc_acc32.image2d")
def schedule_depthwise_conv2d_nchwc_acc32(cfg, outs):
    return schedule_depthwise_conv2d_nchwc_impl(cfg, outs, tag="cast_from_acc32")

def schedule_conv2d_nchwc_impl(cfg, outs, tag):
    """Create the schedule for conv2d_nchw"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    def _callback(op):
        if op.tag == tag:
            args={"memory" : "texture", "shared" : False}
            schedule_conv2d_NCHWc_KCRSk(cfg, s, op.output(0), args)

    traverse_inline(s, outs[0].op, _callback)
    return s

def compute_conv2d_NCHWc_KCRSk(Input, Filter, stride, padding, dilation, out_dtype=None, args={}):
    """Convolution operator in NCHWc layout. """

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

    batch, in_channel_chunk, in_height, in_width, in_channel_block = Input.shape
    num_filter_chunk, channel, kernel_h, kernel_w, num_filter_block = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    temp = nn.pad(Input, pad_before, pad_after, name="pad_temp")

    rcc = te.reduce_axis((0, in_channel_chunk), name="rc")
    rcb = te.reduce_axis((0, in_channel_block), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    if args["memory"] != None:
        # NCHWc x KCRSk
        # texture: NCH|W|c
        # texture: K|CRS|k
        Filter_tx = te.compute(
            (num_filter_chunk, channel * kernel_h * kernel_w, num_filter_block),
            lambda ffc, crs, ffb: Filter[ffc, crs // (kernel_h * kernel_w), (crs // kernel_w) % kernel_h, crs % kernel_w, ffb],
            name = "packed_filter"
        )
        conv = te.compute(
            (batch, num_filter_chunk, out_height, out_width, num_filter_block),
            lambda nn, ffc, yy, xx, ffb: te.sum(
                (temp[nn, rcc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rcb]
                * Filter_tx[ffc, ((rcc * in_channel_block + rcb)*kernel_h + ry)*kernel_w + rx, ffb]).astype(args["accumulator"]),
                axis=[rcc, rcb, ry, rx],
            ),
            tag="conv2d_nchwc",
        )
    else:
        conv = te.compute(
            (batch, num_filter_chunk, out_height, out_width, num_filter_block),
            lambda nn, ffc, yy, xx, ffb: te.sum(
                (temp[nn, rcc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rcb]
                * Filter[ffc, rcc * in_channel_block + rcb, ry, rx, ffb]).astype(args["accumulator"]),
                axis=[rcc, rcb, ry, rx],
            ),
            tag="conv2d_nchwc",
        )
    return te.compute(conv.shape, lambda n,fc,y,x,fb: conv[n,fc,y,x,fb].astype("float16"), tag="cast_from_acc" + args["accumulator"][-2:])

def schedule_conv2d_NCHWc_KCRSk(cfg, s, output, args={}):
    """schedule optimized for batch size = 1"""
    conv = output.op.input_tensors[0]

    ##### space definition begin #####
    n, fc, y, x, fb = s[conv].op.axis
    rcc, rcb, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_fc", fc, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rcc", rcc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    if args["memory"] != None:
        pad_data, flattened_kernel = s[conv].op.input_tensors
        kernel = s[flattened_kernel].op.input_tensors[0]
        s[flattened_kernel].compute_inline()
    else:
        pad_data, kernel = s[conv].op.input_tensors
        flattened_kernel = kernel

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    kernel = flattened_kernel

    # conv only
    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    # conv -> output (e.g. when casting conv output)
    elif output.op in s.outputs:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv
    # conv -> injective -> ... -> injective -> output
    else:
        # Explicitly mark the output cast to be computed inline
        # the other injective ops are inlined via traverse_inline.
        s[output].compute_inline()
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    if args["memory"] != None:
        AT = s.cache_read(pad_data, args["memory"], [OL])
        WT = s.cache_read(kernel, args["memory"], [OL])
        def copy_to_texture(stage):
            axes = s[stage].op.axis
            fused = s[stage].fuse(*axes[:-1])
            block, thread = s[stage].split(fused, factor=32)
            s[stage].vectorize(axes[-1])
            s[stage].bind(block, te.thread_axis("blockIdx.x"))
            s[stage].bind(thread, te.thread_axis("threadIdx.x"))
        copy_to_texture(AT)
        copy_to_texture(WT)

        if args["shared"]:
            AA = s.cache_read(AT, "shared", [OL])
            WW = s.cache_read(WT, "shared", [OL])
    else:
        AA = s.cache_read(pad_data, "shared", [OL])
        WW = s.cache_read(kernel, "shared", [OL])

    # tile and bind spatial axes
    n, fc, y, x, fb = s[output].op.axis

    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_fc"].apply(s, output, fc)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi, fb)
    s[output].vectorize(fb)

    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, fc, y, x, fb = s[OL].op.axis

    rcc, rcb, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rcc"].apply(s, OL, rcc)
    ryo, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, OL, rx)

    # TODO(csullivan): check position of rcb
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, rcb, n, fc, y, x, fb)
    s[OL].vectorize(fb)
    s[OL].unroll(rcb)

    if args["memory"] == None or args["shared"]:
        s[AA].compute_at(s[OL], rxo)
        s[WW].compute_at(s[OL], rxo)
        # cooperative fetching
        for load in [AA, WW]:
            if args["memory"] != None and load == WW:
                n, fyx, v = s[load].op.axis
                fused = s[load].fuse(n, fyx)
            else:
                n, f, y, x, v = s[load].op.axis
                fused = s[load].fuse(n, f, y, x)
            tz, fused = s[load].split(fused, nparts=cfg["tile_fc"].size[2])
            ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
            tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
            s[load].bind(tz, te.thread_axis("threadIdx.z"))
            s[load].bind(ty, te.thread_axis("threadIdx.y"))
            s[load].bind(tx, te.thread_axis("threadIdx.x"))
            s[load].vectorize(v)

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    N, OCC, OH, OW, OCB = get_const_tuple(output.shape)
    if args["memory"] != None:
        _, ICKHKW, _ = get_const_tuple(kernel.shape)
    else:
        _, IC, KH, KW, _ = get_const_tuple(kernel.shape)
        ICKHKW = IC*KH*KW


    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * OCB * ICKHKW)


def schedule_depthwise_conv2d_nchwc_impl(cfg, outs, tag):
    """Create the schedule for depthwise conv2d_nchw4c_ohwi4o"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    def _callback(op):
        if op.tag == tag:
            args={"memory" : "texture", "shared" : False}
            schedule_depthwise_conv2d_NCHWc_KCRSk(cfg, s, op.output(0), args)

    traverse_inline(s, outs[0].op, _callback)
    return s

def compute_depthwise_conv2d_NCHWc_KCRSk(Input, Filter, stride, padding, dilation, out_dtype=None, args={}):
    """Depthwise convolution operator in NCHWc layout. """
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

    batch, channel_chunk, in_height, in_width, channel_block = Input.shape
    _, channel_multiplier, kernel_h, kernel_w, _ = Filter.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel_chunk = simplify(channel_chunk * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    temp = nn.pad(Input, pad_before, pad_after, name="pad_temp")

    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")


    if args["memory"] != None:
        # NCHWc x CMRSc = [N,(C//4)M,OH,OW, 4c]
        # NCHWc x CMRS
        # texture: NCH|W|c
        # texture: C|MRS|c
        Filter_tx = te.compute(
            (channel_chunk, channel_multiplier * kernel_h * kernel_w, channel_block),
            lambda ffc, mrs, ffb: Filter[ffc, mrs // (kernel_h * kernel_w), (mrs // kernel_w) % kernel_h, mrs % kernel_w, ffb],
            name = "packed_filter"
        )

        conv = te.compute(
            (batch, out_channel_chunk, out_height, out_width, channel_block),
            lambda nn, ffc, yy, xx, ffb: te.sum(
                (temp[nn, ffc//channel_multiplier, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, ffb]
                 * Filter_tx[ffc//channel_multiplier, ((ffc % channel_multiplier) * kernel_h + ry) * kernel_w + rx, ffb]).astype(args["accumulator"]),
                axis=[ry, rx],
            ),
            tag="depthwise_conv2d_nchwc_kcrsk_texture",
        )
    else:
        conv = te.compute(
            (batch, out_channel_chunk, out_height, out_width, channel_block),
            lambda nn, ffc, yy, xx, ffb: te.sum(
                (temp[nn, ffc//channel_multiplier, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, ffb]
                * Filter[ffc//channel_multiplier, ffc % channel_multiplier, ry, rx, ffb]).astype(args["accumulator"]),
                axis=[ry, rx],
            ),
            tag="depthwise_conv2d_nchwc_kcrsk",
        )
    return te.compute(conv.shape, lambda n,ffc,y,x,ffb: conv[n,ffc,y,x,ffb].astype("float16"), tag="cast_from_acc" + args["accumulator"][-2:])

def schedule_depthwise_conv2d_NCHWc_KCRSk(cfg, s, output, args={}):
    """schedule optimized for batch size = 1"""
    conv = output.op.input_tensors[0]

    ##### space definition begin #####
    n, fc, y, x, fb = s[conv].op.axis
    ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_fc", fc, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    target = tvm.target.Target.current()
    if target.kind.name in ["nvptx", "rocm"]:
        cfg.define_knob("unroll_explicit", [1])
    else:
        cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    if args["memory"] != None:
        pad_data, flattened_kernel = s[conv].op.input_tensors
        kernel = s[flattened_kernel].op.input_tensors[0]
        s[flattened_kernel].compute_inline()
    else:
        pad_data, kernel = s[conv].op.input_tensors
        flattened_kernel = kernel

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    kernel = flattened_kernel

    # conv only
    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    # conv -> output (e.g. when casting conv output)
    elif output.op in s.outputs:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv
    # conv -> injective -> ... -> injective -> output
    else:
        # Explicitly mark the output cast to be computed inline
        # the other injective ops are inlined via traverse_inline.
        s[output].compute_inline()
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    if args["memory"] != None:
        AT = s.cache_read(pad_data, args["memory"], [OL])
        WT = s.cache_read(kernel, args["memory"], [OL])
        def copy_to_texture(stage):
            axes = s[stage].op.axis
            fused = s[stage].fuse(*axes[:-1])
            block, thread = s[stage].split(fused, factor=32)
            s[stage].vectorize(axes[-1])
            s[stage].bind(block, te.thread_axis("blockIdx.x"))
            s[stage].bind(thread, te.thread_axis("threadIdx.x"))
        copy_to_texture(AT)
        copy_to_texture(WT)

        if args["shared"]:
            AA = s.cache_read(AT, "shared", [OL])
            WW = s.cache_read(WT, "shared", [OL])
    else:
        AA = s.cache_read(pad_data, "shared", [OL])
        WW = s.cache_read(kernel, "shared", [OL])

    # tile and bind spatial axes
    n, fc, y, x, fb = s[output].op.axis

    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_fc"].apply(s, output, fc)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi, fb)
    s[output].vectorize(fb)

    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, fc, y, x, fb = s[OL].op.axis

    ry, rx = s[OL].op.reduce_axis
    ryo, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, OL, rx)

    s[OL].reorder(ryo, rxo, ryi, rxi, n, fc, y, x, fb)
    s[OL].vectorize(fb)
    #s[OL].unroll()

    if args["memory"] == None or args["shared"]:
        s[AA].compute_at(s[OL], rxo)
        s[WW].compute_at(s[OL], rxo)
        # cooperative fetching
        for load in [AA, WW]:
            if args["memory"] != None and load == WW:
                n, fyx, v = s[load].op.axis
                fused = s[load].fuse(n, fyx)
            else:
                n, f, y, x, v = s[load].op.axis
                fused = s[load].fuse(n, f, y, x)
            tz, fused = s[load].split(fused, nparts=cfg["tile_fc"].size[2])
            ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
            tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
            s[load].bind(tz, te.thread_axis("threadIdx.z"))
            s[load].bind(ty, te.thread_axis("threadIdx.y"))
            s[load].bind(tx, te.thread_axis("threadIdx.x"))
            s[load].vectorize(v)

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    N, OCC, OH, OW, OCB = get_const_tuple(output.shape)
    # OC = OCC * OCB = IC * M
    # M = OC // IC == (OCC * OCB) // ICC * ICB
    if args["memory"] != None:
        ICC, MKHKW, ICB = get_const_tuple(kernel.shape)
        M = (OCC * OCB) // (ICC * ICB)
        KHKW = MKHKW // M
    else:
        ICC, M, KH, KW, ICB = get_const_tuple(kernel.shape)
        KHKW = KH*KW

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * OCB * KHKW)
