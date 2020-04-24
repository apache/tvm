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


@reg.register_convert_op_layout("qnn.conv2d")
def convert_qnn_conv2d(attrs, inputs, tinfos, layouts):
    """Convert Layout pass registration for QNN conv2d op.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    tinfos : list of types
        List of input and output types
    layouts : list of layout strings
        List of layouts defining our desired
        layout for the data and kernel inputs.

    Returns
    -------
    result : tvm.relay.Expr
        The transformed expr
    """
    # pylint: disable=import-outside-toplevel
    from tvm import relay
    desired_layout = str(layouts[0])
    assert desired_layout == 'NCHW', \
            "Currently only transformation to NCHW layout is supported."
    if desired_layout == 'NCHW':
        new_attrs = dict(attrs)
        new_attrs['data_layout'] = desired_layout

        desired_kernel_layout = "default"
        if len(layouts) > 1:
            desired_kernel_layout = str(layouts[1])

        if desired_kernel_layout != "default":
            new_attrs['kernel_layout'] = desired_kernel_layout
        else:
            new_attrs['kernel_layout'] = 'OIHW'

        return relay.qnn.op.conv2d(*inputs, **new_attrs)
    return None
