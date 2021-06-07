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

# pylint: disable=invalid-name,unused-variable,unused-argument
"""depthwise_conv2d schedule on ARM Mali GPU"""

from __future__ import absolute_import as _abs
import tvm
from tvm import te

from .. import utils
from .. import tag


def schedule_depthwise_conv2d_nchw(outs):
    """Schedule for depthwise_conv2d nchw forward.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of depthwise_conv2d
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for depthwise_conv2d nchw.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(pad_data, kernel, conv):
        raw_data = s[pad_data].op.input_tensors[0]

        if conv.op not in s.outputs:  # has bias or relu
            output = outs[0]
        else:  # no bias or relu
            output = conv

        def tile_and_bind3d(tensor, z, y, x, z_factor=2, y_factor=None, x_factor=None):
            """tile and bind 3d"""
            y_factor = y_factor or z_factor
            x_factor = x_factor or y_factor
            zo, zi = s[tensor].split(z, z_factor)
            yo, yi = s[tensor].split(y, y_factor)
            xo, xi = s[tensor].split(x, x_factor)
            s[tensor].bind(zo, te.thread_axis("blockIdx.z"))
            s[tensor].bind(zi, te.thread_axis("threadIdx.z"))
            s[tensor].bind(yo, te.thread_axis("blockIdx.y"))
            s[tensor].bind(yi, te.thread_axis("threadIdx.y"))
            s[tensor].bind(xo, te.thread_axis("blockIdx.x"))
            s[tensor].bind(xi, te.thread_axis("threadIdx.x"))
            return zo, zi, yo, yi, xo, xi

        # set tunable parameters
        VH = 1
        VW = 1
        num_thread = 4
        while utils.get_const_int(conv.shape[3]) % (VW * 2) == 0 and VW * 2 <= 4:
            VW = VW * 2
        while utils.get_const_int(conv.shape[2]) % (VH * 2) == 0 and VH * 2 <= 2:
            VH = VH * 2
        if raw_data.dtype == "float16":
            if utils.get_const_int(conv.shape[3]) % (VW * 2) == 0:
                VW *= 2
                num_thread *= 2
            else:
                num_thread *= 2

        # schedule padding
        _, c, y, x = s[pad_data].op.axis
        tile_and_bind3d(pad_data, c, y, x, num_thread, 1, 1)

        # schedule conv
        di, dj = s[conv].op.reduce_axis
        s[conv].unroll(di)
        s[conv].unroll(dj)

        _, c, y, x = s[output].op.axis
        y, x, yi, xi = s[output].tile(y, x, VH, VW)
        s[output].unroll(yi)
        s[output].vectorize(xi)

        _, _, _, _, _, ji = tile_and_bind3d(output, c, y, x, num_thread, 1, 1)

        if conv.op not in s.outputs:
            _, c, y, x = s[conv].op.axis
            y, x, yi, xi = s[conv].tile(y, x, VH, VW)
            s[conv].unroll(yi)
            s[conv].vectorize(xi)
            s[conv].compute_at(s[output], ji)

    def traverse(op):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

        # schedule depthwise_conv2d
        if op.tag == "depthwise_conv2d_nchw":
            pad_data = op.input_tensors[0]
            kernel = op.input_tensors[1]
            if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
                s[kernel].compute_inline()
            conv = op.output(0)
            _schedule(pad_data, kernel, conv)

    traverse(outs[0].op)
    return s
