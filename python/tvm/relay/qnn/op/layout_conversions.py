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
# pylint: disable=invalid-name, unused-argument
"""Convert layout related registration"""
from __future__ import absolute_import

from tvm.relay.op import op as reg

from ...op.strategy.generic import is_depthwise_conv2d


@reg.register_convert_op_layout("qnn.conv2d")
def convert_qnn_conv2d(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for QNN conv2d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    assert len(desired_layouts) == 2, "A desired layout is expected for both of qnn.conv2d's inputs"
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"

    new_attrs = dict(attrs)
    new_attrs["data_layout"] = desired_data_layout

    if desired_kernel_layout != "default":
        new_attrs["kernel_layout"] = desired_kernel_layout
        return relay.qnn.op.conv2d(*inputs, **new_attrs)

    if desired_data_layout == "NCHW":
        new_attrs["kernel_layout"] = "OIHW"
        return relay.qnn.op.conv2d(*inputs, **new_attrs)
    if desired_data_layout == "NHWC":
        # Check for depthwise convolution.
        data_info = tinfos[0]
        weight_info = tinfos[1]
        if is_depthwise_conv2d(
            data_info.shape,
            attrs["data_layout"],
            weight_info.shape,
            attrs["kernel_layout"],
            attrs["groups"],
        ):
            new_attrs["kernel_layout"] = "HWOI"
        else:
            new_attrs["kernel_layout"] = "HWIO"
        return relay.qnn.op.conv2d(*inputs, **new_attrs)

    raise ValueError("Layout %s is not yet supported" % desired_data_layout)


@reg.register_convert_op_layout("qnn.conv2d_transpose")
def convert_qnn_conv2d_transpose(attrs, inputs, tinfos, desired_layouts):
    """Convert Layout pass registration for QNN conv2d_transpose op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    desired_layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs respectively.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay

    assert (
        len(desired_layouts) == 2
    ), "A desired layout is expected for both of qnn.conv2d_transpose's inputs"
    desired_data_layout, desired_kernel_layout = map(str, desired_layouts)
    assert desired_data_layout != "default", "Data layout cannot be default"

    new_attrs = dict(attrs)
    new_attrs["data_layout"] = desired_data_layout

    if desired_kernel_layout != "default":
        new_attrs["kernel_layout"] = desired_kernel_layout
        return relay.qnn.op.conv2d_transpose(*inputs, **new_attrs)

    # Handle default kernel layouts
    if desired_data_layout == "NCHW":
        new_attrs["kernel_layout"] = "IOHW"
        return relay.qnn.op.conv2d_transpose(*inputs, **new_attrs)
    if desired_data_layout == "NHWC":
        new_attrs["kernel_layout"] = "HWIO"
        return relay.qnn.op.conv2d_transpose(*inputs, **new_attrs)

    raise ValueError("Layout %s is not yet supported" % desired_data_layout)
