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
# pylint: disable=no-value-for-parameter

"""Conv3D Transpose schedule on x86"""
from tvm import te
from ..utils import traverse_inline
from .. import nn
from .conv3d import conv3d_ncdhw, schedule_conv3d_ncdhw


def conv3d_transpose_ncdhw(data, kernel, strides, padding, out_dtype, output_padding):
    data_pad, kernel_transform = nn.conv3d_transpose_ncdhw_preprocess(
        data, kernel, strides, padding, out_dtype, output_padding
    )

    # reuse conv3d_ncdhw implementation
    return conv3d_ncdhw(data_pad, kernel_transform, (1, 1, 1), (0, 0, 0), (1, 1, 1), out_dtype)


def schedule_conv3d_transpose_ncdhw(outs):
    """Create schedule for tensors"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = schedule_conv3d_ncdhw(outs)

    def _callback(op):
        if "unpack_ncdhwc" in op.tag:
            conv_out = op.input_tensors[0]
            # retrieve data
            data_vec = conv_out.op.input_tensors[0]
            data_pad = data_vec.op.input_tensors[0]
            data_dilate = data_pad.op.input_tensors[0]
            s[data_dilate].compute_inline()
            s[data_pad].compute_inline()
            # retrieve kernel
            kernel_vec = conv_out.op.input_tensors[1]
            kernel_transform = kernel_vec.op.input_tensors[0]
            s[kernel_transform].compute_inline()

    traverse_inline(s, outs[0].op, _callback)
    return s
