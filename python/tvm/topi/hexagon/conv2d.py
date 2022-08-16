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

"""Schedule for conv2d"""

import tvm
from tvm import te
from ..utils import traverse_inline

from tvm.topi.hexagon.utils import get_layout_transform_fn

def schedule_conv2d_nhwc(outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    output_op = outs[0].op

    def _callback(op):
        if "conv2d_nhwc" in op.tag:
            conv = op.output(0)
            kernel = op.input_tensors[1]
            data = op.input_tensors[0]
            C = conv

            activations = data
            weights = kernel
            output = C
            transform_activation_layout = "nhwc-8h2w32c2w-1d"
            transform_weights_layout = "iohw-16i32o2i-1d"
            transform_output_layout = "nhwc-8h2w32c2w-1d"
            reduce_channel, reduce_height, reduce_width = s[output].op.reduce_axis
            s[activations].transform_layout(get_layout_transform_fn(transform_activation_layout))
            s[weights].transform_layout(get_layout_transform_fn(transform_weights_layout))
            transformed_axis = s[output].transform_layout(
                get_layout_transform_fn(transform_output_layout)
            )
            fused_out_axis = s[output].fuse(transformed_axis[-1], transformed_axis[-2])
            s[output].reorder(
                *[*transformed_axis[:-2], reduce_height, reduce_width, reduce_channel, fused_out_axis]
            )

        elif "elemwise" in op.tag:
            O = op.output(0)
            I = op.input_tensors[0]
            input_layout = "nhwc-8h2w32c2w-2d"
            output_layout = "nhwc-8h2w32c2w-2d"
            input_layout = get_layout_transform_fn(input_layout)
            output_layout = get_layout_transform_fn(output_layout)
            # s[I].transform_layout(input_layout)
            # def fourd_layout(n, h, w, c, ih, iow, ic, iiw):
            #     return [n, h * 8 + ih, w * 4 + iow*2 + iiw, c * 32 + ic]
            # s[O].transform_layout(fourd_layout)
            # s[O].transform_layout(fourd_layout)

            # s[output]

    traverse_inline(s, output_op, _callback)
    return s

def schedule_conv2d_nchw(outs):
    return schedule_conv2d_nhwc(outs)


def schedule_conv2d(outs, layout="NHWC"):
    layout_uncase = layout.casefold()
    if layout_uncase == "NHWC".casefold():
        return schedule_conv2d_nhwc(outs)
    if layout_uncase == "NCHW".casefold():
        return schedule_conv2d_nchw(outs)

    raise ValueError(f"Unexpected layout={layout}")


def schedule_depthwise_conv2d_nchw(outs):
    return schedule_conv2d_nchw(outs)


def schedule_depthwise_conv2d_nhwc(out):
    return schedule_conv2d_nhwc(out)


def schedule_conv2d_transpose_nchw(outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, tvm.te.tensor.Tensor) else outs
    s = schedule_conv2d_nchw(outs)

    def _callback(op):
        if "unpack_nchwc" in op.tag:
            conv_out = op.input_tensors[0]
            # retrieve data
            data_vec = conv_out.op.input_tensors[0]
            if isinstance(data_vec, tvm.te.ComputeOp):
                data_pad = data_vec.op.input_tensors[0]
                data_dilate = data_pad.op.input_tensors[0]
                s[data_dilate].compute_inline()
                s[data_pad].compute_inline()
            # retrieve kernel
            kernel_vec = conv_out.op.input_tensors[1]
            if isinstance(kernel_vec, tvm.te.ComputeOp):
                kernel_transform = kernel_vec.op.input_tensors[0]
                s[kernel_transform].compute_inline()

    traverse_inline(s, outs[0].op, _callback)
    return s
