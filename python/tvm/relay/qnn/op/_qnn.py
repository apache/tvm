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
"""Backend QNN related feature registration"""
from __future__ import absolute_import

import tvm
from tvm import relay
from tvm.api import min_value, max_value
from tvm.relay.qnn.op import register_qnn_legalize
from .. import op as reg
from topi.util import get_const_int

@tvm.target.generic_func
def qnn_conv2d_legalize(attrs, inputs, types):
    """Default legalization is None."""
    return None

@qnn_conv2d_legalize.register('cpu')
def _qnn_conv2d_legalize(attrs, inputs, types):
    """Legalizes QNN conv2d op. VNNI supports u8 x i8 fast conv/MM. If the dtypes are already good,
    we dont transform. Else, we shift the tensor values and zero points to change the dtype.

    Converting from int8 to uint8 can be done in following manner.

    Original equation
      scale * (QA - zp_a)
      scale * (QA + 128 - 128 - zp_a)
      scale * ( (QA + 128) - (zp_a + 128))

    Replacing QA + 128 with QA' and (zp_a + 128) with zp_a'
    We get our new uint8 tensor - scale * (QA' - zp_a')

    Similarly we can convert from int8 to uint8.

    Parameters
    ----------
    attrs : tvm.attrs.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """

    def _shift_quantized_tensor(data, shift, out_dtype):
        """Shifts (add/subtracts) the qnn tensor with +/-128)"""
        data_modified = relay.cast(data, 'int32')
        data_modified = relay.add(data_modified, relay.const(shift, 'int32'))
        data_modified = relay.clip(data_modified,
                                   a_min=min_value(out_dtype).value,
                                   a_max=max_value(out_dtype).value)
        data_modified = relay.cast(data_modified, out_dtype)
        return data_modified

    channels_expr = attrs['channels']
    if isinstance(channels_expr, tvm.expr.IntImm):
        channels = channels_expr.value
        if channels == 1001:
            return None

    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    # Collect the input exprs.
    data, kernel = inputs

    # VNNI supports u8 x i8 fast conv/MM.
    if data_dtype == 'uint8' and kernel_dtype == 'int8':
        return None

    # Shift input if necessary.
    input_zp = attrs['input_zero_point']
    if data_dtype == 'int8':
        # Compute (QA + 128) and (zp_a + 128)
        data = _shift_quantized_tensor(data, 128, 'uint8')
        input_zp = input_zp + 128

    # Shift kernel if necessary.
    kernel_zp = attrs['kernel_zero_point']
    if kernel_dtype == 'uint8':
        # Compute (QA - 128) and (zp_a - 128)
        kernel = _shift_quantized_tensor(kernel, -128, 'int8')
        kernel_zp = kernel_zp - 128

    # Call qnn.conv2d with modified inputs and zero points.
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['input_zero_point'] = input_zp
    new_attrs['kernel_zero_point'] = kernel_zp
    return relay.qnn.op.conv2d(data, kernel, **new_attrs)

@reg.register_qnn_legalize("qnn.conv2d")
def legalize_qnn_conv2d(attrs, inputs, types):
    """Legalizes QNN conv2d op.

    Parameters
    ----------
    attrs : tvm.attrs.Attrs
        Attributes of current convolution
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types

    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    return qnn_conv2d_legalize(attrs, inputs, types)
