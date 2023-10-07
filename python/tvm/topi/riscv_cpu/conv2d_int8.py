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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-member
"""Conv2D int8 schedule on RISCV"""
from tvm import te, target, autotvm
from ..utils import traverse_inline, get_const_tuple
from ..generic import conv2d as conv2d_generic
from .. import nn
from ..nn.conv2d import _get_workload as _get_conv2d_workload, unpack_NCHWc_to_nchw
from ..x86.conv2d_int8 import _pack_data
from ..nn.utils import get_pad_tuple
from .tensor_intrin import dot_int8_int8_int32, int8_conv2d_impl


def _get_default_config(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """
    Get default int8 schedule config for the workload
    """
    wkl = _get_conv2d_workload(data, kernel, strides, padding, dilation, out_dtype)
    is_kernel_1x1 = wkl.kernel_h == 1 and wkl.kernel_w == 1
    if is_kernel_1x1:
        conv2d_generic.fallback_schedule_cpu_1x1_int8(cfg, wkl, int32_lanes=4, num_int8_elements=4)
    else:
        conv2d_generic.fallback_schedule_cpu_common_int8(
            cfg, wkl, int32_lanes=4, num_int8_elements=4
        )


def is_int8_hw_support(data_dtype, kernel_dtype):
    """
    Checks to ensure that we can use int8 on riscv_cpu.
    1) The datatypes are correct.
    2) The vector extension "V" is used.
    """
    # 1) Check datatypes.
    is_dtype_support = data_dtype == "uint8" and kernel_dtype == "int8"

    # 2) Check target.
    current_target = target.Target.current(allow_none=False)
    has_attr = "+v" in current_target.mattr
    is_arch_support = "v" in current_target.arch[2:]
    if not is_arch_support and "march" in current_target.attrs:
        is_arch_support = "v" in current_target.attrs["march"]
    is_target_support = has_attr or is_arch_support

    return is_dtype_support and is_target_support


@autotvm.register_topi_compute("conv2d_NCHWc_int8.riscv_cpu")
def conv2d_NCHWc_int8(cfg, data, kernel, strides, padding, dilation, layout, out_layout, out_dtype):
    """Compute conv2d int8 with NCHWc layout"""
    # layout and out_layout are not used here,
    # we keep them for debug convenience when dumping autotvm workload

    if len(data.shape) == 5:  # data is in nchwc
        n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
        in_channel = ic_chunk * ic_bn

        oc_chunk, ic_chunk, kh, kw, ic_bn, oc_bn, _ = get_const_tuple(kernel.shape)
        num_filter = oc_chunk * oc_bn
    else:
        # data is nchw, implicitly treat it as nchw1c
        n, in_channel, ih, iw = get_const_tuple(data.shape)
        num_filter, _, kh, kw = get_const_tuple(kernel.shape)

    # Define autotvm tuning space
    is_kernel_1x1 = kh == 1 and kw == 1
    pt, pl, pb, pr = get_pad_tuple(padding, (kh, kw))
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    dh, dw = dilation if isinstance(dilation, (tuple, list)) else (dilation, dilation)
    dilated_kernel_h = (kh - 1) * dh + 1
    dilated_kernel_w = (kw - 1) * dw + 1
    oh = (ih - dilated_kernel_h + pt + pb) // sh + 1
    ow = (iw - dilated_kernel_w + pl + pr) // sw + 1

    # input and output should be a multiple of 8 (intrinsics are 8 lanes)
    cfg.define_split(
        "tile_ic", in_channel, num_outputs=2, filter=lambda y: y.size[-1] % min(8, in_channel) == 0
    )
    cfg.define_split(
        "tile_oc", num_filter, num_outputs=2, filter=lambda y: y.size[-1] % min(8, num_filter) == 0
    )
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    # If no config was set, we can fallback to NCHW config.
    if cfg.is_fallback:
        _get_default_config(
            cfg,
            te.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
            te.placeholder((num_filter, in_channel, kh, kw), dtype=kernel.dtype),
            strides,
            padding,
            dilation,
            out_dtype,
        )
    # Pack data if raw 4-D data is provided.
    # This can only happen when autotuning.
    if len(data.shape) == 4:
        data, kernel = _pack_data(cfg, data, kernel)

    n_elems = int(kernel.shape[-1])

    return nn.conv2d_NCHWc_int8(
        data, kernel, strides, padding, dilation, layout, out_layout, out_dtype, n_elems=n_elems
    )


@autotvm.register_topi_schedule("conv2d_NCHWc_int8.riscv_cpu")
def schedule_conv2d_NCHWc_int8(cfg, outs):
    """Create schedule for tensors"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_NCHWc_int8" in op.tag:
            inline_fused = False
            conv_out = op.output(0)
            kernel_vec = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = (
                data_vec.op.input_tensors[0]
                if isinstance(data_vec.op, te.tensor.ComputeOp) and "pad" not in data_vec.op.tag
                else data_vec
            )
            if isinstance(data.op, te.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, n_elems = get_const_tuple(kernel_vec.shape)

            assert n_elems == 4

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
                batch, ic_chunk, ih, _, ic_block = s[data_vec].op.axis
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
            C, O = conv_out, outs[0]
            CC = s.cache_write(C, "global")

            batch, oc_chunk, oh, ow, oc_block = s[C].op.axis

            if kh == 1 and kw == 1:
                oh_factor, ow_factor = cfg["tile_oh"].val, cfg["tile_ow"].size[-1]
                oh_outer, oh_inner = s[C].split(oh, factor=oh_factor)
                ow_outer, ow_inner = s[C].split(ow, factor=ow_factor)
                s[C].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)
                s[C].vectorize(oc_block)

                parallel_axis = s[C].fuse(batch, oc_chunk, oh_outer)
                if C == O:
                    s[C].parallel(parallel_axis)
                s[CC].compute_at(s[C], parallel_axis)

                _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
                kh, kw, ic_outer, ic_f_inner, ic_s_inner = s[CC].op.reduce_axis

                oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=4)

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

                s[CC].tensorize(oc_s_inner, dot_int8_int8_int32())
                s[CC].pragma(oc_f_inner, "import_c", int8_conv2d_impl())

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
            else:
                if isinstance(cfg["tile_ow"], int):
                    reg_n = cfg["tile_ow"]
                else:
                    reg_n = cfg["tile_ow"].size[-1]

                if isinstance(cfg["unroll_kw"], (int, bool)):
                    unroll_kw = cfg["unroll_kw"]
                else:
                    unroll_kw = cfg["unroll_kw"].val

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

                assert oc_bn % 4 == 0, f"oc_bn={oc_bn} % int32_lanes={4} != 0"
                assert (
                    ic_bn % 4 == 0
                ), f"ic_bn={ic_bn} % int8_elems={4} != 0"  # (u)int8 elements in (u)int32

                oc_f_inner, oc_s_inner = s[CC].split(oc_block, factor=4)

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

                s[CC].tensorize(oc_s_inner, dot_int8_int8_int32())
                s[CC].pragma(oc_f_inner, "import_c", int8_conv2d_impl())

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

    traverse_inline(s, outs[0].op, _callback)
    return s


def conv2d_nchw_int8(data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with NCHW layout and int8 dtype"""
    layout = "NCHW"
    # pylint: disable=no-value-for-parameter
    packed_out = conv2d_NCHWc_int8(
        data, kernel, strides, padding, dilation, layout, layout, out_dtype
    )
    return unpack_NCHWc_to_nchw(packed_out, out_dtype)


def schedule_conv2d_nchw_int8(outs):
    """Create the schedule for conv2d_nchw_int8"""
    # pylint: disable=no-value-for-parameter
    return schedule_conv2d_NCHWc_int8(outs)
