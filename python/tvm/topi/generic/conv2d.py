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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""Generic convolution schedules"""
from tvm import te
from tvm import autotvm
from tvm import relay
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from ..utils import get_const_tuple, traverse_inline
from ..nn.utils import get_pad_tuple


def fallback_schedule_cpu_common_int8(cfg, wkl, int32_lanes, num_int8_elements):
    """Fallback schedule for conv2d int8 on cpu.
    Normally the inner most pattern takes two int8/uint8 tensors
    data[num_int8_elements] and kernel[int32_lanes, num_int8_elements],
    produces a dot product int32/uint32 output[int32_lanes].

    Parameters
    ----------
    int32_lanes : int
        How many numbers of int32/uint32 will be produced using intrinsic.
        This is related to output channel.
    num_int8_elements : int
        How many numbers of input int32/uint32 will be multiplied and reduced.
        This is related to input channel.
    """
    pt, pl, pb, pr = wkl.padt, wkl.padl, wkl.padb, wkl.padr
    HSTR, WSTR = wkl.stride_h, wkl.stride_w
    dilated_kernel_w = (wkl.kernel_w - 1) * wkl.dilation_w + 1
    out_width = (wkl.width + pl + pr - dilated_kernel_w) // WSTR + 1

    assert wkl.out_filter % int32_lanes == 0, "wkl.out_filter=%d, int32_lanes=%d" % (
        wkl.out_filter,
        int32_lanes,
    )
    assert wkl.in_filter % num_int8_elements == 0, "wkl.in_filter=%d, num_int8_elements=%d" % (
        wkl.in_filter,
        num_int8_elements,
    )

    oc_bn = int32_lanes if int32_lanes >= num_int8_elements else num_int8_elements
    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def fallback_schedule_cpu_1x1_int8(cfg, wkl, int32_lanes, num_int8_elements):
    """Fallback schedule for 1x1 conv2d int8 on cpu.
    Normally the inner most pattern takes two int8/uint8 tensors
    data[num_int8_elements] and kernel[int32_lanes, num_int8_elements],
    produces a dot product int32/uint32 output[int32_lanes].

    Parameters
    ----------
    int32_lanes : int
        How many numbers of int32/uint32 will be produced using intrinsic.
        This is related to output channel.
    num_int8_elements : int
        How many numbers of input int32/uint32 will be multiplied and reduced.
        This is related to input channel.
    """
    pt, pl, pb, pr = wkl.padt, wkl.padl, wkl.padb, wkl.padr
    HSTR, WSTR = wkl.stride_h, wkl.stride_w
    out_height = (wkl.height + pt + pb - wkl.kernel_h) // HSTR + 1
    out_width = (wkl.width + pl + pr - wkl.kernel_w) // WSTR + 1

    assert wkl.out_filter % int32_lanes == 0, "wkl.out_filter=%d, int32_lanes=%d" % (
        wkl.out_filter,
        int32_lanes,
    )
    assert wkl.in_filter % num_int8_elements == 0, "wkl.in_filter=%d, num_int8_elements=%d" % (
        wkl.in_filter,
        num_int8_elements,
    )

    oc_bn = int32_lanes if int32_lanes >= num_int8_elements else num_int8_elements
    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break

    for ow_factor in range(out_width, 0, -1):
        if out_width % ow_factor == 0:
            for oh_factor in range(out_height, 0, -1):
                if out_height % oh_factor == 0 and ow_factor * oh_factor < 32:
                    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
                    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
                    cfg["tile_oh"] = OtherOptionEntity(oh_factor)
                    cfg["tile_ow"] = SplitEntity([out_width // ow_factor, ow_factor])
                    return
    raise ValueError("cannot decide default schedule for workload: {}".format(wkl))


def schedule_conv_NCHWc_cpu_common_int8(
    s,
    cfg,
    data_vec,
    kernel_vec,
    conv_out,
    last,
    int32_lanes=16,
    int8_elems=4,
    intrin=None,
    inline_fused=True,
):
    """
    Defines the schedule for INT8 for Intel and ARM machines
    Uses the Intel/ARM intrinsics to use INT8 operations
    More details - https://software.intel.com/en-us/articles/
    lower-numerical-precision-deep-learning-inference-and-training
    """
    if isinstance(cfg["tile_ow"], int):
        reg_n = cfg["tile_ow"]
    else:
        reg_n = cfg["tile_ow"].size[-1]

    if isinstance(cfg["unroll_kw"], (int, bool)):
        unroll_kw = cfg["unroll_kw"]
    else:
        unroll_kw = cfg["unroll_kw"].val

    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    # schedule pad
    if isinstance(s[data_vec].op, te.tensor.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # only in autotuning, input data of conv2d_NCHWc will be 4-D.
        # skip this part during tuning to make records accurate.
        # this part will be folded during Relay fold_constant pass.
        if isinstance(data_vec.op, te.tensor.ComputeOp):
            s[data_vec].pragma(s[data_vec].op.axis[0], "debug_skip_region")
        if isinstance(kernel_vec.op, te.tensor.ComputeOp):
            s[kernel_vec].pragma(s[kernel_vec].op.axis[0], "debug_skip_region")
    elif isinstance(kernel_vec.op, te.tensor.ComputeOp) and kernel_vec.name == "kernel_vec":
        # data and kernel are not pre-computed, schedule layout transform here.
        # this should only be used by x86 conv2d_nchw, which is for
        # testing purpose.
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)

        # conv2d_nchwc_int8 has 7D kernel
        oc_chunk, ic_chunk, oh, ow, ic_block, oc_block, _ = s[kernel_vec].op.axis
        s[kernel_vec].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
        oc_bn = cfg["tile_oc"].size[-1]
        if oc_bn > 1:
            s[kernel_vec].vectorize(oc_block)
        parallel_axis = s[kernel_vec].fuse(oc_chunk, oh)
        s[kernel_vec].parallel(parallel_axis)

    # schedule 5-D NCHW[x]c conv
    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[C].fuse(batch, oc_chunk, oh)
    s[C].vectorize(oc_block)
    if C == O:
        s[C].parallel(parallel_axis)

    s[CC].compute_at(s[C], parallel_axis)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)

    assert oc_bn % int32_lanes == 0, f"oc_bn={oc_bn} % int32_lanes={int32_lanes} != 0"
    assert (
        ic_bn % int8_elems == 0
    ), f"ic_bn={ic_bn} % int8_elems={int8_elems} != 0"  # (u)int8 elements in (u)int32

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    if unroll_kw:
        s[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            ic_f_inner,
            kw,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )
        s[CC].unroll(kw)
    else:
        s[CC].reorder(
            oc_chunk,
            oh,
            ow_chunk,
            ic_outer,
            kh,
            kw,
            ic_f_inner,
            ow_block,
            oc_f_inner,
            oc_s_inner,
            ic_s_inner,
        )

    if intrin is not None:
        s[CC].tensorize(oc_s_inner, intrin)
    s[CC].unroll(ow_block)
    s[CC].unroll(oc_f_inner)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
        else:
            raise ValueError("Unsupported output ndim: %s" % out_ndim)
        parallel_axis = s[O].fuse(batch, oc_chunk, oh)
        if inline_fused:
            s[C].compute_at(s[O], ow_block)
        else:
            s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


def schedule_conv_NCHWc_cpu_1x1_int8(
    s,
    cfg,
    data_vec,
    kernel_vec,
    conv_out,
    last,
    int32_lanes=16,
    int8_elems=4,
    intrin=None,
    inline_fused=False,
):
    """
    Defines the 1x1 conv schedule for INT8 for Intel and ARM machines
    Uses the Intel/ARM intrinsics to use INT8 operations
    More details - https://software.intel.com/en-us/articles/
    lower-numerical-precision-deep-learning-inference-and-training
    """
    oh_factor, ow_factor = cfg["tile_oh"].val, cfg["tile_ow"].size[-1]
    _, _, _, _, ic_bn = get_const_tuple(data_vec.shape)
    _, _, _, _, oc_bn = get_const_tuple(conv_out.shape)

    # schedule pad
    if isinstance(s[data_vec].op, te.tensor.ComputeOp) and "pad" in data_vec.op.tag:
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)
        data_vec = data_vec.op.input_tensors[0]

    if autotvm.GLOBAL_SCOPE.in_tuning:
        # only in autotuning, input data of conv2d_NCHWc will be 4-D.
        # skip this part during tuning to make records accurate.
        # this part will be folded during Relay fold_constant pass.
        if isinstance(data_vec.op, te.tensor.ComputeOp):
            s[data_vec].pragma(s[data_vec].op.axis[0], "debug_skip_region")
        if isinstance(kernel_vec.op, te.tensor.ComputeOp):
            s[kernel_vec].pragma(s[kernel_vec].op.axis[0], "debug_skip_region")
    elif isinstance(kernel_vec.op, te.tensor.ComputeOp) and kernel_vec.name == "kernel_vec":
        # data and kernel are not pre-computed, schedule layout transform here.
        # this should only be used by x86 conv2d_nchw, which is for
        # testing purpose.
        batch, ic_chunk, ih, iw, ic_block = s[data_vec].op.axis
        parallel_axis = s[data_vec].fuse(batch, ic_chunk, ih)
        s[data_vec].parallel(parallel_axis)

        # Conv2d int8 schedule has 7D kernel
        oc_chunk, ic_chunk, oh, ow, ic_block, oc_block, _ = s[kernel_vec].op.axis
        s[kernel_vec].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
        oc_bn = cfg["tile_oc"].size[-1]
        if oc_bn > 1:
            s[kernel_vec].vectorize(oc_block)
        parallel_axis = s[kernel_vec].fuse(oc_chunk, oh)
        s[kernel_vec].parallel(parallel_axis)

    C, O = conv_out, last
    CC = s.cache_write(C, "global")

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[C].split(ow, factor=ow_factor)
    s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
    s[C].vectorize(oc_block)

    parallel_axis = s[C].fuse(batch, oc_chunk, oh_outer)
    if C == O:
        s[C].parallel(parallel_axis)
    s[CC].compute_at(s[C], parallel_axis)  # good perf on mobilenet, but not on individuals?

    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

    assert oc_bn % int32_lanes == 0
    assert ic_bn % int8_elems == 0  # (u)int8 elements in (u)int32

    oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=int32_lanes)

    oh_outer, oh_inner = s[CC].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=ow_factor)

    s[CC].reorder(
        oc_chunk,
        oh_outer,
        ow_outer,
        kh,
        kw,
        ic_outer,
        ic_f_inner,
        oh_inner,
        ow_inner,
        oc_f_inner,
        oc_s_inner,
        ic_s_inner,
    )
    s[CC].fuse(oc_chunk, oh_outer)

    if intrin is not None:
        s[CC].tensorize(oc_s_inner, intrin)
    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if C != O:
        out_ndim = len(s[O].op.axis)
        if out_ndim == 5:
            batch, oc_chunk, oh, ow, oc_block = s[O].op.axis
            oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
            ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
        elif out_ndim == 4:
            batch, oc, oh, ow = s[O].op.axis
            oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
            oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
            ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
        else:
            raise ValueError("Unsupported output ndim: %s" % out_ndim)

        s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
        parallel_axis = s[O].fuse(batch, oc_chunk, oh_outer)
        if inline_fused:
            s[C].compute_at(s[O], ow_inner)
        else:
            s[C].compute_at(s[O], parallel_axis)
        s[O].vectorize(oc_block)
        s[O].parallel(parallel_axis)

    return s


def schedule_depthwise_conv2d_nhwc(outs):
    """Create schedule for depthwise conv2d in NHWC layout.
    Parameters
    ----------
    outs : list[te.tensor.Tensor]
            The output tensors.
    Returns
    -------
    s : tvm.te.schedule.Schedule
        The computation schedule for depthwise conv2d.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        """Traverse operators from computation graph"""
        if "depthwise_conv2d_nhwc" in op.tag:
            out = outs[0]
            depthwise_conv2d_out = op.output(0)
            data_pad = depthwise_conv2d_out.op.input_tensors[0]
            s[data_pad].compute_inline()
            if depthwise_conv2d_out != out:
                s[depthwise_conv2d_out].compute_at(s[out], s[out].op.axis[3])
            s[out].fuse(*s[out].op.axis)

    traverse_inline(s, outs[0].op, _callback)
    return s


def conv2d_alter_int8_common(
    data,
    data_tensor,
    kernel,
    kernel_tensor,
    output_tensor,
    attrs,
    data_dtype: str,
    in_channel_vector_length: int,
    out_channel_vector_length: int,
):
    """
    Convert TE inputs/outputs so that they are suitable for fast Int8 instructions.

    Int8 instructions require input channels and output channels to be a
    multiple of the vector length. For input channels, we pad both the inputs
    and weights channels. For output channels, we pad the weight and
    stride_slice the output.

    Arguments
    ---------
    data: Expr
        Data Expr
    data_tensor: Tensor
        Data tensor
    kernel: Expr
        Kernel Expr
    kernel_tensor: Tensor
        Kernel tensor
    output_tensor: Tensor
        Output tensor
    attrs: Conv2dAttrs
        Attributes of the computation
    data_dtype: "int8" or "uint8"
        Desired dtype of data. Data will be converted to this dtype before the main computation.
    in_channel_vector_length: int
        Length of vector units on target hardware. Input channels are padded to this length.
    out_channel_vector_length: int
        Output size of vector instruction. Output channels are padded to this length.

    Returns
    -------
    out : Tensor
        Conv2d computation with inputs in the correct order for tensorization.
    """
    # Dilation not supported yet. Return None if dilation is not (1, 1)
    dilation = attrs.get_int_tuple("dilation")
    if not (dilation[0] == 1 and dilation[1] == 1):
        return None

    # No legalization for depthwise convolutions yet.
    groups = attrs.get_int("groups")
    if groups != 1:
        return None

    # Get the conv attrs
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    padding = attrs.get_int_tuple("padding")
    kh, kw = attrs.get_int_tuple("kernel_size")
    pt, pl, pb, pr = get_pad_tuple(padding, (kh, kw))

    if data_tensor.dtype != data_dtype:
        # How to convert data to uint8
        # Original --> C = A (conv) B
        # A and B are int8
        #   C = (A + 128 - 128) (conv) B
        #   C = (A' conv B) - 128 (conv) B
        # where A' = A + 128
        # and 128 (conv) B is basically a reduce on CRS axis for weights.
        #
        # How to convert data to int8
        #   C = (A - 128 + 128) (conv) B
        #   C = (A' conv B) + 128 (conv) B
        # where A' = A - 128
        if data_dtype == "uint8":
            # shift data to uint8
            before_shift = relay.add
            after_shift = relay.subtract
            pad_value = 128
        else:
            # shift data to int8
            before_shift = relay.subtract
            after_shift = relay.add
            pad_value = -128

        if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
            adjust_shift = relay.sum(relay.cast(kernel, dtype="int32"), axis=(0, 1, 2))
            pad_width = ((0, 0), (pt, pb), (pl, pr), (0, 0))
        elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
            pad_width = ((0, 0), (0, 0), (pt, pb), (pl, pr))
            adjust_shift = relay.sum(relay.cast(kernel, dtype="int32"), axis=(1, 2, 3))
            adjust_shift = relay.expand_dims(adjust_shift, axis=1, num_newaxis=2)
        else:
            return None

        data = relay.cast(data, "int32")
        data = before_shift(data, relay.const(128, "int32"))
        data = relay.cast(data, data_dtype)

        # Do external padding as pad value has to be 128.
        if any(padding):
            data = relay.nn.pad(data, pad_width=pad_width, pad_value=pad_value)

        new_attrs["padding"] = (0, 0)

        # Multiply 128 to adjust shift.
        adjust_shift = relay.multiply(adjust_shift, relay.const(128, "int32"))

    # Flags to remember if the expr is modified
    ic_modified = False
    oc_modified = False

    # Find the value of input and output channel.
    in_channel = -1
    out_channel = -1
    if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
        in_channel = data_tensor.shape[3].value
        out_channel = kernel_tensor.shape[3].value
    elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
        in_channel = data_tensor.shape[1].value
        out_channel = kernel_tensor.shape[0].value
    else:
        return None

    if in_channel % in_channel_vector_length != 0:
        new_in_channel = (
            (in_channel + in_channel_vector_length) // in_channel_vector_length
        ) * in_channel_vector_length
        diff = new_in_channel - in_channel
        if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
            data = relay.nn.pad(data, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
            kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, diff), (0, 0)))
            ic_modified = True
        elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
            pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
            data = relay.nn.pad(data, pad_width=pad_width)
            kernel = relay.nn.pad(kernel, pad_width=pad_width)
            ic_modified = True
        else:
            return None

    new_out_channel = out_channel
    if out_channel % out_channel_vector_length != 0:
        new_out_channel = (
            (out_channel + out_channel_vector_length) // out_channel_vector_length
        ) * out_channel_vector_length
        diff = new_out_channel - out_channel
        if attrs["data_layout"] == "NHWC" and attrs["kernel_layout"] == "HWIO":
            kernel = relay.nn.pad(kernel, pad_width=((0, 0), (0, 0), (0, 0), (0, diff)))
            oc_modified = True
        elif attrs["data_layout"] == "NCHW" and attrs["kernel_layout"] == "OIHW":
            kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
            oc_modified = True
        else:
            return None

    if oc_modified:
        new_attrs["channels"] = new_out_channel
        out = relay.nn.conv2d(data, kernel, **new_attrs)
        original_out_shape = [x.value for x in output_tensor.shape]
        out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
    else:
        out = relay.nn.conv2d(data, kernel, **new_attrs)

    if data_tensor.dtype != data_dtype:
        out = after_shift(out, adjust_shift)

    return out
