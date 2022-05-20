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
"""depthwise_conv2d_nchw(c) schedule on Qualcomm Adreno GPU"""
import tvm
from tvm import te
from tvm import autotvm

from ..utils import get_const_tuple, traverse_inline
from .utils import (
    split_to_chunks,
    pack_input,
    pack_filter,
    expand_spatial_dimensions,
    add_pad,
    bind_data_copy,
)


@autotvm.register_topi_compute("depthwise_conv2d_nchwc.image2d")
def depthwise_conv2d_nchwc(cfg, data, kernel, strides, padding, dilation, out_dtype="float16"):
    """Compute depthwise_conv2d with NCHWc layout"""
    args = {"shared": False, "accumulator": "float16"}
    return compute_depthwise_conv2d_NCHWc_KCRSk(
        data, kernel, strides, padding, dilation, out_dtype, args=args
    )


@autotvm.register_topi_compute("depthwise_conv2d_nchwc_acc32.image2d")
def depthwise_conv2d_nchwc_acc32(
    cfg, data, kernel, strides, padding, dilation, out_dtype="float16"
):
    """Compute depthwise_conv2d with NCHWc layout"""
    args = {"shared": False, "accumulator": "float32"}
    return compute_depthwise_conv2d_NCHWc_KCRSk(
        data, kernel, strides, padding, dilation, out_dtype, args=args
    )


@autotvm.register_topi_schedule("depthwise_conv2d_nchwc.image2d")
def schedule_depthwise_conv2d_nchwc(cfg, outs):
    return schedule_depthwise_conv2d_nchwc_impl(cfg, outs, tag="cast_from_acc16")


@autotvm.register_topi_schedule("depthwise_conv2d_nchwc_acc32.image2d")
def schedule_depthwise_conv2d_nchwc_acc32(cfg, outs):
    return schedule_depthwise_conv2d_nchwc_impl(cfg, outs, tag="cast_from_acc32")


def schedule_depthwise_conv2d_nchwc_impl(cfg, outs, tag):
    """Create the schedule for depthwise conv2d_nchw4c_ohwi4o"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == tag:
            schedule_depthwise_conv2d_NCHWc_KCRSk(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def compute_depthwise_conv2d_NCHWc_KCRSk(Input, Filter, stride, padding, dilation, out_dtype, args):
    """
    Depthwise convolution operator in NCHWc layout.
    Algo:
      1. Convert into blocked format if we have 4d original tensor.
         In case of AutoTVM we override the convert by just tensors since such conversion
         will be absent for real blocked convolution, no sense to include into tuning
      2. Expand spatial dimensions to have width and height be dividable by factor 4
         This leads to slightly bigger amount of compute but allow utilize GPU much better
      3. Add paddings. This happens even if we do not need pad originaly. This is useful
         due to work arounding of the gaps of texture annotation between Primary Functions
         and limited support of textures in schedules. Later on this pad will be executed
         separately and will produce texture
      4. 5d Convolution compute with accumulating into out_dtype
      5. Cast to the origin output data type
      6. For case of 4d convolution: convert of output from 5d to 4d
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

    convert_from4d = False
    if len(Input.shape) == 4:
        batch, in_channels, in_height, in_width = Input.shape
        out_channles, in_filter_channels, kernel_h, kernel_w = Filter.shape

        in_channel_chunks, in_channel_block, in_channel_tail = split_to_chunks(in_channels, 4)
        out_channel_chunks, out_channel_block, out_channel_tail = split_to_chunks(out_channles, 4)

        if autotvm.GLOBAL_SCOPE.in_tuning:
            dshape = (batch, in_channel_chunks, in_height, in_width, in_channel_block)
            Input = tvm.te.placeholder(dshape, Input.dtype, name="data_placeholder")
            kshape = (out_channel_chunks, in_filter_channels, kernel_h, kernel_w, out_channel_block)
            Filter = tvm.te.placeholder(kshape, Filter.dtype, name="kernel_placeholder")
        else:
            convert_from4d = True
            Input = pack_input(
                Input,
                "NCHW",
                batch,
                in_channel_chunks,
                in_channel_block,
                in_channel_tail,
                in_height,
                in_width,
            )
            Filter = pack_filter(
                Filter,
                "OIHW",
                out_channel_chunks,
                out_channel_block,
                out_channel_tail,
                in_filter_channels,
                in_channel_chunks,
                in_channel_block,
                in_channel_tail,
                kernel_h,
                kernel_w,
            )

    else:
        batch, in_channel_chunks, in_height, in_width, in_channel_block = Input.shape
        out_channel_chunks, in_filter_channels, kernel_h, kernel_w, out_channel_block = Filter.shape

    out_height_orig, out_height, out_width_orig, out_width = expand_spatial_dimensions(
        in_height, in_width, kernel_h, kernel_w, dilation_h, dilation_w, padding, stride_h, stride_w
    )

    temp = add_pad(
        Input,
        "NCHW",
        out_height_orig,
        out_width_orig,
        kernel_h,
        kernel_w,
        dilation_h,
        dilation_w,
        padding,
        stride_h,
        stride_w,
    )

    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    conv = te.compute(
        (batch, out_channel_chunks, out_height, out_width, out_channel_block),
        lambda nn, ffc, yy, xx, ffb: te.sum(
            (
                temp[
                    nn,
                    ffc // in_filter_channels,
                    yy * stride_h + ry * dilation_h,
                    xx * stride_w + rx * dilation_w,
                    ffb,
                ]
                * Filter[ffc // in_filter_channels, ffc % in_filter_channels, ry, rx, ffb]
            ).astype(args["accumulator"]),
            axis=[ry, rx],
        ),
        tag="depthwise_conv2d_nchwc_kcrsk",
    )

    if convert_from4d and not autotvm.GLOBAL_SCOPE.in_tuning:
        dummy_cast = te.compute(
            (batch, out_channel_chunks, out_height_orig, out_width_orig, out_channel_block),
            lambda n, fc, y, x, fb: conv[n, fc, y, x, fb].astype(out_dtype),
            tag="dummy_cast",
        )
        return te.compute(
            (batch, out_channles, out_height_orig, out_width_orig),
            lambda n, c, y, x: dummy_cast[n, c // out_channel_block, y, x, c % out_channel_block],
            tag="cast_from_acc" + args["accumulator"][-2:],
        )
    else:
        return te.compute(
            (batch, out_channel_chunks, out_height_orig, out_width_orig, out_channel_block),
            lambda n, ffc, y, x, ffb: conv[n, ffc, y, x, ffb].astype(out_dtype),
            tag="cast_from_acc" + args["accumulator"][-2:],
        )


def schedule_depthwise_conv2d_NCHWc_KCRSk(cfg, s, output):
    """
    schedule optimized for batch size = 1

    Algo:
    1. Split output axis to three parts: global work size, vthread, local worksize.
       The limitations for tuning includes heuristics from some tuned networks to limit
       search space and not pay much time for useles configurations.
    2. For depthwise convolution it's better to inline pad into the conv2d compute, the
       divergence in opencl kernel will not so significant as for regular conv2d.
    3. For 5d convolution we schedule the latest op with binding 5d axis and vectorize
       for textures
       For 4d tensor we are doing the same for the latest blocked stage, i.e. conversion
       of data type
    4. In case of 4d conv we need to schedule postops as well
    """
    latest = s.outputs[0].output(0)
    if len(latest.op.axis) == 4:
        latest_blocked = dummy = output.op.input_tensors[0]
        conv = dummy.op.input_tensors[0]
    else:
        conv = output.op.input_tensors[0]
        latest_blocked = latest

    ##### space definition begin #####
    n, fc, y, x, fb = s[conv].op.axis
    ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_fc", fc, num_outputs=3)
    cfg.define_split("tile_y", y, num_outputs=3)
    cfg.define_split("tile_x", x, num_outputs=3)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])
    cfg.define_knob("unroll_explicit", [0, 1])
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors
    if (
        isinstance(kernel.op, tvm.te.ComputeOp) and "filter_pack" in kernel.op.tag
    ):  # len(latest.op.axis) == 4:
        # manage scheduling of datacopy
        pad_data, kernel = s[conv].op.input_tensors
        pack_data = pad_data.op.input_tensors[0]
        bind_data_copy(s[pack_data])
        bind_data_copy(s[kernel])

    pad_data, kernel = s[conv].op.input_tensors

    s[pad_data].compute_inline()

    s[conv].set_scope("local")
    if latest_blocked == latest and output != latest:
        s[output].compute_inline()

    # create cache stage
    AT = s.cache_read(pad_data, "global.texture", [conv])
    WT = s.cache_read(kernel, "global.texture-weight", [conv])
    bind_data_copy(s[AT])
    bind_data_copy(s[WT])

    # tile and bind spatial axes
    n, fc, y, x, fb = s[latest_blocked].op.axis
    kernel_scope, n = s[latest_blocked].split(n, nparts=1)

    bf, vf, tf = cfg["tile_fc"].apply(s, latest_blocked, fc)
    by, vy, ty = cfg["tile_y"].apply(s, latest_blocked, y)
    bx, vx, tx = cfg["tile_x"].apply(s, latest_blocked, x)

    bf = s[latest_blocked].fuse(n, bf)
    s[latest_blocked].bind(bf, te.thread_axis("blockIdx.z"))
    s[latest_blocked].bind(by, te.thread_axis("blockIdx.y"))
    s[latest_blocked].bind(bx, te.thread_axis("blockIdx.x"))
    s[latest_blocked].bind(vf, te.thread_axis("vthread"))
    s[latest_blocked].bind(vy, te.thread_axis("vthread"))
    s[latest_blocked].bind(vx, te.thread_axis("vthread"))
    s[latest_blocked].bind(tf, te.thread_axis("threadIdx.z"))
    s[latest_blocked].bind(ty, te.thread_axis("threadIdx.y"))
    s[latest_blocked].bind(tx, te.thread_axis("threadIdx.x"))
    s[latest_blocked].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fb)
    s[latest_blocked].vectorize(fb)

    s[conv].compute_at(s[latest_blocked], tx)

    # tile reduction axes
    n, fc, y, x, fb = s[conv].op.axis

    ry, rx = s[conv].op.reduce_axis
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(ryo, rxo, ryi, rxi, n, fc, y, x, fb)
    s[conv].vectorize(fb)

    # unroll
    s[latest_blocked].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[latest_blocked].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)
    if latest_blocked != latest:
        s[latest].compute_root()
        bind_data_copy(s[latest], 1)
        if latest != output:
            s[output].compute_inline()

    N, OCC, OH, OW, OCB = get_const_tuple(latest_blocked.shape)
    _, _, KH, KW, ICB = get_const_tuple(kernel.shape)
    KHKW = KH * KW

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * OCB * KHKW * ICB)
