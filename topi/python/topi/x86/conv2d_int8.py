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

import tvm
from tvm import autotvm
from .. import generic, tag
from ..util import get_const_tuple
from ..nn.conv2d import conv2d_NCHWc_int8
from .. import nn
from .conv2d import _get_default_config
from . import conv2d_avx_1x1, conv2d_avx_common

@autotvm.register_topi_compute(conv2d_NCHWc_int8, 'cpu', 'direct')
def _declaration_conv_NCHWc_int8(cfg, data, kernel, strides,
                                 padding, dilation, layout, out_layout, out_dtype):
    n, ic_chunk, ih, iw, ic_bn = get_const_tuple(data.shape)
    in_channel = ic_chunk * ic_bn
    oc_chunk, _, kernel_height, kernel_width, _, oc_bn, _ = \
            get_const_tuple(kernel.shape)
    num_filter = oc_chunk * oc_bn

    # If config is not set, we can reuse the default config for NCHW.
    if cfg.is_fallback:
        _get_default_config(cfg, tvm.placeholder((n, in_channel, ih, iw), dtype=data.dtype),
                            tvm.placeholder((num_filter, in_channel, kernel_height, kernel_width),
                                            dtype=kernel.dtype),
                            strides, padding, out_dtype)
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
