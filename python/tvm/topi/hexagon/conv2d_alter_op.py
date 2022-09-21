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
"""Dense alter op functions for ARM"""

import tvm
from tvm import relay
from tvm import autotvm
from ..utils import get_const_tuple
from .. import nn
from ..nn.utils import get_pad_tuple
from ..nn import conv2d_legalize, conv2d_alter_layout
from ..generic.conv2d import conv2d_alter_int8_common


def check_vrmpy_applicable(x, y):
    out_channel, in_channel, _, _ = get_const_tuple(y.shape)
    return (
        "int8" in x.dtype and "int8" in y.dtype and out_channel % 32 == 0 and in_channel % 4 == 0
    )


@conv2d_alter_layout.register("hexagon")
def _alter_conv2d_layout(attrs, inputs, tinfos, out_type):
    target = tvm.target.Target.current(allow_none=False)
    dispatch_ctx = autotvm.task.DispatchContext.current
    new_attrs = {k: attrs[k] for k in attrs.keys()}

    # Parse the attributes.
    padding = attrs.get_int_tuple("padding")
    strides = attrs.get_int_tuple("strides")
    dilation = attrs.get_int_tuple("dilation")
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]
    data_tensor, kernel_tensor = tinfos
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype
    out_dtype = out_type.dtype

    if not check_vrmpy_applicable(data_tensor, kernel_tensor) or data_layout != "NCHW" or kernel_layout != "OIHW":
        return None

    batch_size, in_channel, height, width = get_const_tuple(data_tensor.shape)
    out_channel, _, kh, kw = get_const_tuple(kernel_tensor.shape)
    data_dtype = data_tensor.dtype
    kernel_dtype = kernel_tensor.dtype

    n_elems = 4
    ic_bn, oc_bn = 32, 32

    if ic_bn > in_channel:
        assert in_channel == 4
        ic_bn = in_channel

    new_attrs = {k: attrs[k] for k in attrs.keys()}

    new_attrs["channels"] = out_channel
    new_attrs["data_layout"] = "NCHW%dc" % ic_bn
    new_attrs["kernel_layout"] = "OIHW{:n}i{:n}o{:n}i".format(ic_bn // n_elems, oc_bn, n_elems)
    new_attrs["out_layout"] = "NCHW%dc" % oc_bn

    return relay.nn.contrib_conv2d_nchwc(*inputs, **new_attrs)


@nn.conv2d_legalize.register("hexagon")
def _conv2d_legalize(attrs, inputs, arg_types):
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]

    output_tensor = arg_types[2]

    # Collect the input exprs.
    data, kernel = inputs

    if data_layout != "NCHW" or kernel_layout != "OIHW":
        return None

    # Collect the input tensors.
    data_tensor, kernel_tensor = arg_types[0], arg_types[1]
    out_channel = kernel_tensor.shape[0]

    if "int8" in data_tensor.dtype and "int8" in data_tensor.dtype and out_channel % 32 == 0:
        data_dtype = data_tensor.dtype

        # Collect the output tensor.
        output_tensor = arg_types[2]

        # Collect the input exprs.
        data, kernel = inputs

        data_dtype = "uint8"

        return conv2d_alter_int8_common(
            data, data_tensor, kernel, kernel_tensor, output_tensor, attrs, data_dtype, 4, 32
        )

    return None
