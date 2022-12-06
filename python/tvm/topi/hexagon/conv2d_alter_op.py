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
"""Conv2d alter op functions for Hexagon"""

from tvm import relay
from ..utils import get_const_tuple
from .. import nn
from ..nn import conv2d_alter_layout
from ..generic.conv2d import conv2d_alter_int8_common


@conv2d_alter_layout.register("hexagon")
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    """Convert nn.conv2d into nn.contrib_conv2d_nchwc if vrmpy is applicable."""
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data_tensor, kernel_tensor = tinfos
    out_channel, in_channel, _, _ = get_const_tuple(kernel_tensor.shape)

    if (
        "int8" in data_tensor.dtype
        and "int8" in kernel_tensor.dtype
        and out_channel % 32 == 0
        and in_channel % 4 == 0
        and data_layout == "NCHW"
        and kernel_layout == "OIHW"
    ):
        out_channel, in_channel, _, _ = get_const_tuple(kernel_tensor.shape)

        n_elems = 4
        oc_bn = 32
        ic_bn = min(in_channel, 32)

        new_attrs = {k: attrs[k] for k in attrs.keys()}

        new_attrs["channels"] = out_channel
        new_attrs["data_layout"] = "NCHW%dc" % ic_bn
        new_attrs["kernel_layout"] = "OIHW{:n}i{:n}o{:n}i".format(ic_bn // n_elems, oc_bn, n_elems)
        new_attrs["out_layout"] = "NCHW%dc" % oc_bn

        return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)

    return None


@nn.conv2d_legalize.register("hexagon")
def _conv2d_legalize(attrs, inputs, arg_types):
    """Legalize conv2d op for vrmpy tensorization.

    If the inputs are signed or unsigned int8, the input and output channels are padded to be
    a multiple of 4 and 32 respectively.

    If the input data types are (int8, int8), they are converted to (uint8, int8) and
    the vector-by-vector variant of vrmpy is applied.
    If the input data types are (uint8, uint8), the more efficient vector-by-scalar variant of vrmpy
    is applied.

    Unlike the nn.dense case (see dense_alter_op.py), we do not convert (uint8, int8) to
    (uint8, uint8). That would introduce another convolution by a constant (128 or 1) filter,
    to compensate for the dtype legalization. In the nn.dense case, such compensation factor is
    just a sum over the K axis.
    """
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]

    output_tensor = arg_types[2]

    data, kernel = inputs

    if data_layout != "NCHW" or kernel_layout != "OIHW":
        return None

    data_tensor, kernel_tensor = arg_types[0], arg_types[1]

    if "int8" in data_tensor.dtype and "int8" in data_tensor.dtype:
        output_tensor = arg_types[2]
        data, kernel = inputs
        desired_data_dtype = "uint8"
        in_channel_vector_length = 4
        out_channel_vector_length = 32

        return conv2d_alter_int8_common(
            data,
            data_tensor,
            kernel,
            kernel_tensor,
            output_tensor,
            attrs,
            desired_data_dtype,
            in_channel_vector_length,
            out_channel_vector_length,
        )

    return None
