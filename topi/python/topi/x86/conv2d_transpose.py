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
"""Conv2D Transpose schedule on x86"""

import tvm
from tvm import autotvm
from .. import generic, tag
from ..nn.conv2d_transpose import conv2d_transpose_nchw
from ..nn.dilate import dilate
from ..nn.pad import pad
from ..nn.util import get_pad_tuple
from ..util import simplify

@autotvm.register_topi_compute(conv2d_transpose_nchw, 'cpu', ['direct'])
def _declaration_conv2d_transpose(cfg, data, kernel, strides, padding, out_dtype):
    return _declaration_conv2d_transpose_impl(cfg, data, kernel, strides, padding, out_dtype)

def _declaration_conv2d_transpose_impl(cfg, data, kernel, strides, padding, out_dtype):
    batch, in_c, in_h, in_w = data.shape
    _, out_c, filter_h, filter_w = kernel.shape
    stride_h, stride_w = strides
    # dilate stage
    DilatedInput = dilate(data, [1, 1, stride_h, stride_w], name='DilatedInput')
    # padding stage
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(padding, (filter_h, filter_w))
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right
    PaddedInput = pad(DilatedInput, \
                        [0, 0, bpad_top, bpad_left], \
                        [0, 0, bpad_bottom, bpad_right], \
                        name='PaddedInput')
    # convolution stage
    out_c = simplify(out_c)
    out_h = simplify((in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h)
    out_w = simplify((in_w - 1) * stride_w - fpad_left - fpad_right + filter_w)
    dc = tvm.reduce_axis((0, in_c), name='dc')
    dh = tvm.reduce_axis((0, filter_h), name='dh')
    dw = tvm.reduce_axis((0, filter_w), name='dw')

    Output = tvm.compute(
        (batch, out_c, out_h, out_w),
        lambda b, c, h, w: tvm.sum(
            PaddedInput[b, dc, h+dh, w+dw].astype(out_dtype) *
            kernel[dc, c, filter_h-1-dh, filter_w-1-dw].astype(out_dtype),
            axis=[dc, dh, dw]), tag="conv2d_transpose_nchw")

    return Output

@autotvm.register_topi_schedule(generic.schedule_conv2d_transpose_nchw, 'cpu', ['direct'])
def schedule_conv2d_transpose(cfg, outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    
    def traverse(op):
        """Traverse operators from computation graph"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(op.tag) or tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        if 'conv2d_transpose_nchw' in op.tag:
            C = op.output(0)

            N, OC, OH, OW = C.op.axis
            rc, ry, rx = C.op.reduce_axis

            OH, oh = s[C].split(OH, factor=2)
            OC, oc = s[C].split(OC, factor=32)
            IC, ic = s[C].split(rc, factor=32)

            s[C].reorder(N, OC, OH, OW, oc, IC, ry, rx, ic)
            N = s[C].fuse(N, OC)
            s[C].vectorize(oc)
            s[C].parallel(N)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s
