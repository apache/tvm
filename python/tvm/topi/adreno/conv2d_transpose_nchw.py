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
"""conv2d_transpose nchw schedule on Qualcomm Adreno GPU"""
import tvm
from tvm import te
from tvm import autotvm
from .. import nn


from ..utils import get_const_tuple, traverse_inline
from .utils import (
    split_to_chunks,
    pack_input,
    pack_filter,
    bind_data_copy,
    get_default_conv2d_config,
    get_texture_storage,
)


@autotvm.register_topi_compute("conv2d_transpose_nchwc.image2d")
def conv2d_transpose_nchwc(
    cfg, Input, Filter, stride, padding, out_dtype, output_padding, groups=1
):
    """
    Transposed Convolution operator in NCHWc layout.
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
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    outpad_height, outpad_width = output_padding
    assert outpad_height < stride_h and outpad_width < stride_w

    convert_from4d = False
    if len(Input.shape) == 4:
        batch, in_channels, in_height, in_width = Input.shape
        in_channel_chunks, in_channel_block, in_channel_tail = split_to_chunks(in_channels, 4)

        if autotvm.GLOBAL_SCOPE.in_tuning:
            dshape = (batch, in_channel_chunks, in_height, in_width, in_channel_block)
            Input = tvm.te.placeholder(dshape, Input.dtype, name="data_placeholder")
        else:
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
    else:
        batch, in_channel_chunks, in_height, in_width, in_channel_block = Input.shape

    if len(Filter.shape) == 4:
        in_filter_channels, out_channels, kernel_h, kernel_w = Filter.shape
        out_channel_chunks, out_channel_block, out_channel_tail = split_to_chunks(out_channels, 4)

        if autotvm.GLOBAL_SCOPE.in_tuning:
            kshape = (in_filter_channels, out_channel_chunks, kernel_h, kernel_w, out_channel_block)
            Filter = tvm.te.placeholder(kshape, Filter.dtype, name="kernel_placeholder")
        else:
            convert_from4d = True
            Filter = pack_filter(
                Filter,
                "IOHW",
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
        in_filter_channels, out_channel_chunks, kernel_h, kernel_w, out_channel_block = Filter.shape

    cfg.stride = stride

    pad_top, pad_left, pad_bottom, pad_right = nn.get_pad_tuple(padding, (kernel_h, kernel_w))

    out_width_orig = out_width = (
        (in_width - 1) * stride_w + kernel_w - pad_left - pad_right + outpad_width
    )
    pad_left = kernel_w - 1 - pad_left
    pad_right = kernel_w - 1 - pad_right + outpad_width
    dilated_width = stride_w * (in_width - 1) + 1

    out_height_orig = out_height = (
        (in_height - 1) * stride_h + kernel_h - pad_top - pad_bottom + outpad_height
    )
    pad_top = kernel_h - 1 - pad_top
    pad_bottom = kernel_h - 1 - pad_bottom + outpad_height
    dilated_height = stride_h * (in_height - 1) + 1

    if out_height % 2 != 0:
        out_height += 1
    if out_width % 2 != 0:
        out_width += 1

    if out_height % 4 != 0:
        out_height += 2
    if out_width % 4 != 0:
        out_width += 2

    # compute pad
    temp = te.compute(
        (
            batch,
            in_channel_chunks,
            pad_top + dilated_height + pad_bottom,
            pad_left + dilated_width + pad_right,
            in_channel_block,
        ),
        lambda n, c, y, x, cb: tvm.tir.if_then_else(
            tvm.tir.all(
                x >= pad_left,
                x < pad_left + dilated_width,
                tvm.tir.indexmod(x - pad_left, stride_w).equal(0),
                y >= pad_top,
                y < pad_top + dilated_height,
                tvm.tir.indexmod(y - pad_top, stride_h).equal(0),
            ),
            Input[
                n,
                c,
                tvm.tir.indexdiv(y - pad_top, stride_h),
                tvm.tir.indexdiv(x - pad_left, stride_w),
                cb,
            ],
            tvm.tir.const(0.0, Input.dtype),
        ),
        name="pad_temp",
    )

    # compute transposed conv
    dcc = te.reduce_axis((0, in_channel_chunks), name="dcc")
    dcb = te.reduce_axis((0, in_channel_block), name="dcb")
    dh = te.reduce_axis((0, kernel_h), name="dh")
    dw = te.reduce_axis((0, kernel_w), name="dw")
    conv = te.compute(
        (batch, out_channel_chunks, out_height, out_width, out_channel_block),
        lambda b, c, h, w, cb: te.sum(
            temp[
                b, c // out_channel_chunks * (in_channel_chunks) + dcc, h + dh, w + dw, dcb
            ].astype(out_dtype)
            * Filter[
                dcc * in_channel_block + dcb,
                c % out_channel_chunks,
                kernel_h - 1 - dh,
                kernel_w - 1 - dw,
                cb,
            ].astype(out_dtype),
            axis=[dcc, dcb, dh, dw],
        ),
        tag="conv2d_transpose_nchwc",
    )

    if convert_from4d and not autotvm.GLOBAL_SCOPE.in_tuning:
        dummy_cast = te.compute(
            (batch, out_channel_chunks, out_height_orig, out_width_orig, out_channel_block),
            lambda n, fc, y, x, fb: conv[n, fc, y, x, fb].astype(out_dtype),
            tag="dummy_cast",
        )
        return te.compute(
            (batch, out_channels, out_height_orig, out_width_orig),
            lambda n, c, y, x: dummy_cast[n, c // out_channel_block, y, x, c % out_channel_block],
            tag="adreno_conv2d_transpose_latest_op",
        )
    else:
        return te.compute(
            (batch, out_channel_chunks, out_height_orig, out_width_orig, out_channel_block),
            lambda n, ffc, y, x, ffb: conv[n, ffc, y, x, ffb].astype(out_dtype),
            tag="adreno_conv2d_transpose_latest_op",
        )


@autotvm.register_topi_schedule("conv2d_transpose_nchwc.image2d")
def schedule_conv2d_transpose_nchwc(cfg, outs):
    """Create the schedule for conv2d_nchw"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "adreno_conv2d_transpose_latest_op":
            schedule_conv2d_transpose_NCHWc(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def schedule_conv2d_transpose_NCHWc(cfg, s, output):
    """
    schedule optimized for batch size = 1

    Algo:
    1. Split output axis to three parts: global work size, vthread, local worksize.
       The limitations for tuning includes heuristics from some tuned networks to limit
       search space and not pay much time for useles configurations.
    2. In case of 4d convolution schedule copying of the input (and filter) into
      5d tensors
    4. pad should be scheduled separately to create independent opencl kernel. If pad is
       inlined into convolution, this gives 1.5x performance drop
    5. We are using cache_read for intermediate tensors to produce texture and guarantee
       the best performance on the next stage.
       The weights are managed through static texture planning mechanism and guarantied come
       in texture memory scope.
       Thus way we are calling cache_read only for data tensor
    6. For 5d convolution we schedule the latest op with binding 5d axis and vectorize
       for textures
       For 4d tensor we are doing the same for the latest blocked stage, i.e. conversion
       of data type
    7. In case of 4d conv we need to schedule postops as well
    """
    latest = s.outputs[0].output(0)
    if len(latest.op.axis) == 4:
        latest_blocked = dummy = output.op.input_tensors[0]
        conv = dummy.op.input_tensors[0]
    else:
        conv = output.op.input_tensors[0]
        latest_blocked = latest

    pad_data, kernel = s[conv].op.input_tensors
    filter_pack_rt = bool(
        isinstance(kernel.op, tvm.te.ComputeOp) and "filter_pack" in kernel.op.tag
    )

    if "pad_temp" in pad_data.op.name:
        input_pad_temp = pad_data.op.input_tensors[0]
    else:
        input_pad_temp = pad_data

    input_pack_rt = bool(
        isinstance(input_pad_temp.op, tvm.te.ComputeOp) and "input_pack" in input_pad_temp.op.tag
    )

    ##### space definition begin #####
    n, fc, y, x, fb = s[conv].op.axis
    rcc, rcb, ry, rx = s[conv].op.reduce_axis

    if conv.shape[1] % 2 == 0:
        min_threads_div = 2
    else:
        min_threads_div = 1
    cfg.define_split(
        "tile_fc",
        fc,
        num_outputs=3,
        filter=lambda entity: entity.size[1] <= 8
        and entity.size[2] >= min_threads_div
        and entity.size[2] < 256,
    )
    cfg.define_split(
        "tile_y",
        y,
        num_outputs=3,
        filter=lambda entity: entity.size[1] <= 8 and entity.size[2] <= 16,
    )
    cfg.define_split(
        "tile_x",
        x,
        num_outputs=3,
        filter=lambda entity: entity.size[1] <= 8 and entity.size[2] <= 16,
    )

    cfg.define_split("tile_rcc", rcc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 64])
    cfg.define_knob("unroll_explicit", [0, 1])
    cfg.multi_filter(
        filter=lambda entity: (  # pylint: disable=chained-comparison
            entity["tile_fc"].size[1] * entity["tile_y"].size[1] * entity["tile_x"].size[1]
        )
        <= 24
        and 32
        <= (entity["tile_fc"].size[2] * entity["tile_y"].size[2] * entity["tile_x"].size[2])
        < 1024
    )
    if cfg.is_fallback:
        get_default_conv2d_config(cfg, conv.shape[1], conv.shape[2], conv.shape[3])
    ##### space definition end #####

    pad_data, kernel = s[conv].op.input_tensors
    # There are several conditions that have to be handled:
    # 1. If we are in the tuning, we always add cache read for data to main conv kernel
    #    to get texture in tuning opencl kernel
    # 2. If we are repacking input in runtime, we should always explicit schedule this one more
    #    stage of data copy from 4d to 5d (referred as pack_data).
    # 3. If we have pad (independently if we have runtime repack or not) we should inline it in the
    #    cache_read("texture")
    if autotvm.GLOBAL_SCOPE.in_tuning or input_pack_rt:
        if autotvm.GLOBAL_SCOPE.in_tuning:
            if "pad_temp" in pad_data.op.name:
                s[pad_data].compute_inline()
        else:
            if "pad_temp" in pad_data.op.name:
                pack_data = pad_data.op.input_tensors[0]
                bind_data_copy(s[pack_data])
                s[pad_data].compute_inline()
            else:
                pack_data = pad_data
                bind_data_copy(s[pack_data])

        AT = s.cache_read(pad_data, get_texture_storage(pad_data.shape), [conv])
        bind_data_copy(s[AT])
    elif "pad_temp" in pad_data.op.name:
        s[pad_data].compute_inline()
        # create cache stage
        AT = s.cache_read(pad_data, get_texture_storage(pad_data.shape), [conv])
        bind_data_copy(s[AT])

    if autotvm.GLOBAL_SCOPE.in_tuning or filter_pack_rt:
        if not autotvm.GLOBAL_SCOPE.in_tuning:
            bind_data_copy(s[kernel])
        if kernel.shape[2] == 1 and kernel.shape[3] == 1:
            WT = s.cache_read(kernel, get_texture_storage(kernel.shape), [conv])
            bind_data_copy(s[WT])

    s[conv].set_scope("local")
    if latest_blocked == latest and output != latest:
        s[output].compute_inline()

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

    rcc, rcb, ry, rx = s[conv].op.reduce_axis
    rco, rci = cfg["tile_rcc"].apply(s, conv, rcc)
    ryo, ryi = cfg["tile_ry"].apply(s, conv, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, conv, rx)

    s[conv].reorder(rco, ryo, rxo, rci, ryi, rxi, rcb, n, fc, y, x, fb)
    s[conv].vectorize(fb)
    s[conv].unroll(rcb)

    # unroll
    s[latest_blocked].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[latest_blocked].pragma(kernel_scope, "unroll_explicit", cfg["unroll_explicit"].val)

    if latest_blocked != latest:
        s[latest].compute_root()
        bind_data_copy(s[latest], 1)
        if latest != output:
            s[output].compute_inline()

    N, OCC, OH, OW, OCB = get_const_tuple(latest_blocked.shape)
    _, IC, KH, KW, _ = get_const_tuple(kernel.shape)
    ICKHKW = IC * KH * KW

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * OCB * ICKHKW)
