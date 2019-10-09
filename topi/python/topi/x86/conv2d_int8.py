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
"""Conv2D int8 schedule on x86"""

import re
import tvm
from tvm import autotvm
from tvm.autotvm.task import get_config
from tvm.autotvm.task.topi_integration import deserialize_args
from ..nn.conv2d import _get_workload as _get_conv2d_workload
from .. import generic, tag
from ..generic import conv2d as conv2d_generic
from ..util import get_const_tuple
from ..nn.conv2d import conv2d_NCHWc_int8
from .. import nn
from . import conv2d_avx_1x1, conv2d_avx_common

def _get_default_config_int8(cfg, data, kernel, strides, padding, out_dtype, is_depthwise=False,
                             layout='NCHW'):
    """
    Get default schedule config for the workload
    """
    assert not is_depthwise, "Depthwise Int8 not supported"
    wkl = _get_conv2d_workload(data, kernel, strides, padding, out_dtype, layout)
    is_kernel_1x1 = wkl.hkernel == 1 and wkl.wkernel == 1
    if is_kernel_1x1:
        conv2d_generic.fallback_schedule_cpu_1x1_int8(
            cfg, wkl, int32_lanes=16, num_int8_elements=4)
    else:
        conv2d_generic.fallback_schedule_cpu_common_int8(
            cfg, wkl, int32_lanes=16, num_int8_elements=4)


def _is_int8_hw_support(data_dtype, kernel_dtype):
    """
    Checks to ensure that we can use Intel DLBoost instructions
    1) The datatypes are correct.
    2) LLVM version has support for the instructions.
    3) Target is skylake and above.
    """
    # 1) Check datatypes
    is_dtype_support = data_dtype == 'uint8' and kernel_dtype == 'int8'

    # 2) Check LLVM support
    llvm_intrin_fast_int8 = "llvm.x86.avx512.pmaddubs.w.512"
    llvm_id = tvm.codegen.llvm_lookup_intrinsic_id(llvm_intrin_fast_int8)
    is_llvm_support = llvm_id != 0

    # 3) Check target
    target = tvm.target.current_target()
    is_target_support = False
    for opt in target.options:
        if opt == '-mcpu=skylake-avx512':
            is_target_support = True

    return is_dtype_support and is_llvm_support and is_target_support


def _create_tuning_space_int8(cfg, data, kernel, strides, padding, dilation, layout):
    """Create schedule configuration from input arguments"""
    dshape = get_const_tuple(data.shape)
    kshape = get_const_tuple(kernel.shape)
    pat = re.compile(r'NCHW.+(\d+)c')
    if layout == 'NCHW':
        n, ic, h, w = dshape
        oc, _, kh, kw = kshape
    elif layout == 'NHWC':
        n, h, w, ic = dshape
        kh, kw, oc, _ = kshape
    elif pat.match(layout) is not None:
        n, ic_chunk, h, w, ic_bn = dshape
        target = tvm.target.current_target(allow_none=False)
        oc_chunk, k_ic, kh, kw, k_ic_f, oc_bn, k_ic_s = kshape
        ic = ic_chunk * ic_bn
        assert ic == k_ic * k_ic_f * k_ic_s
        oc = oc_chunk*oc_bn
    else:
        raise ValueError("Not support this layout {} with "
                         "schedule template.".format(layout))

    is_kernel_1x1 = kh == 1 and kw == 1
    ph, pw = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    sh, sw = strides if isinstance(strides, (tuple, list)) else (strides, strides)
    oh = (h - kh + 2 * ph) // sh + 1
    ow = (w - kw + 2 * pw) // sw + 1

    # Create schedule config
    cfg.define_split('tile_ic', ic, num_outputs=2, filter=lambda y: y.size[-1] % 4 == 0)
    cfg.define_split('tile_oc', oc, num_outputs=2, filter=lambda y: y.size[-1] % 16 == 0)
    cfg.define_split("tile_ow", ow, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if oh > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])


# Define template function for autotvm task
# We define schedule template in this function instead of
# declaration function since actual input arguments need
# to be altered by the schedule selected.
@autotvm.task.register("topi_x86_conv2d_NCHWc_int8")
def _topi_nn_conv2d_NCHWc_int8(*args, **kwargs):
    assert not kwargs, "Do not support kwargs in template function call"
    args = deserialize_args(args)

    if len(args) == 7:
        data, kernel, strides, padding, dilation, origin_layout, dtype = args
    else:
        assert len(args) == 8
        data, kernel, strides, padding, dilation, origin_layout, out_layout, dtype = args

    raw_data_shape = get_const_tuple(data.shape)
    raw_kernel_shape = get_const_tuple(kernel.shape)

    # get config here
    cfg = get_config()
    _create_tuning_space_int8(cfg, data, kernel, strides, padding, dilation, origin_layout)

    # change shape with the value in config
    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                           cfg["tile_ow"].size[-1])

    data_layout = "NCHW%dc" % ic_bn
    out_layout = "NCHW%dc" % oc_bn

    # Set up the new shape for data and kernel
    new_data_shape = (raw_data_shape[0], raw_data_shape[1] // ic_bn,
                      raw_data_shape[2], raw_data_shape[3], ic_bn)
    n_elems = 4
    new_kernel_shape = (raw_kernel_shape[0] // oc_bn,
                        raw_kernel_shape[1] // ic_bn,
                        raw_kernel_shape[2],
                        raw_kernel_shape[3],
                        ic_bn // n_elems,
                        oc_bn,
                        n_elems)

    new_data = tvm.placeholder(new_data_shape, data.dtype)
    new_kernel = tvm.placeholder(new_kernel_shape, kernel.dtype)

    C = _declaration_conv_NCHWc_int8(cfg, new_data, new_kernel, strides, padding, dilation,
                                     data_layout, out_layout, dtype)
    s = _schedule_conv2d_NCHWc_int8(cfg, [C])
    return s, [new_data, new_kernel, C]


@autotvm.register_topi_compute(conv2d_NCHWc_int8, 'cpu', 'direct')
def _declaration_conv_NCHWc_int8(cfg, data, kernel, strides,
                                 padding, dilation, layout, out_layout, out_dtype):
    return nn.conv2d_NCHWc_int8_compute(data,
                                        kernel,
                                        strides,
                                        padding,
                                        dilation,
                                        layout,
                                        out_layout,
                                        out_dtype)


@autotvm.register_topi_schedule(generic.schedule_conv2d_NCHWc_int8, 'cpu', ['direct'])
def _schedule_conv2d_NCHWc_int8(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_NCHWc_int8' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, cfg, data_vec, conv_out, outs[0]]
            target = tvm.target.current_target(allow_none=False)
            # int8 conv kernel is 7-dim
            _, _, kh, kw, _, _, _ = get_const_tuple(kernel.shape)
            if kh == 1 and kw == 1:
                conv2d_avx_1x1._schedule_conv_NCHWc_int8(*args)
            else:
                conv2d_avx_common._schedule_conv_NCHWc_int8(*args)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

@autotvm.register_topi_schedule(generic.schedule_conv2d_nhwc_pack, 'cpu', ['direct'])
def schedule_conv2d_nhwc_pack(cfg, outs):
    """Create schedule for tensors"""
    s = tvm.create_schedule([x.op for x in outs])
    output_op = outs[0].op
    scheduled_ops = []

    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else: # inject custom schedule
                if len(op.axis) == 4: # schedule bias + bn + relu
                    n, h, w, c = op.axis
                    fused = s[op].fuse(n, h, w)
                    s[op].parallel(fused)
                    s[op].vectorize(c)
            for tensor in op.input_tensors:
                if isinstance(tensor.op, tvm.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_nhwc_pack_int8' in op.tag:
            conv_out = op.output(0)
            kernel = conv_out.op.input_tensors[1]
            data_vec = conv_out.op.input_tensors[0]
            data = data_vec.op.input_tensors[0] \
                if isinstance(data_vec.op, tvm.tensor.ComputeOp) and "pad" not in data_vec.op.tag \
                else data_vec
            if isinstance(data.op, tvm.tensor.ComputeOp) and "pad" in data.op.tag:
                data_pad = data
                data = data_pad.op.input_tensors[0]

            args = [s, cfg, data_vec, conv_out, outs[0]]
            if data.dtype == 'uint8':
                kh, kw, _, _, _ = get_const_tuple(kernel.shape)
                if kh == 1 and kw == 1:
                    conv2d_avx_1x1._schedule_conv_nhwc_pack_int8(*args)
                else:
                    raise ValueError("Only support 1x1 kernel with "
                                     "schedule_conv2d_nhwc_pack.")
            else:
                raise ValueError("Not support this data type {} with "
                                 "schedule_conv2d_nhwc_pack. Only support int8".format(data.dtype))

        scheduled_ops.append(op)
    traverse(output_op)
    return s
