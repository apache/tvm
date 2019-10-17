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
from .. import op as reg

# Registering QNN Conv2D legalization function.
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

# Generic QNN Conv2D legalization function.
@tvm.target.generic_func
def qnn_conv2d_legalize(attrs, inputs, types):
    """Default legalization is None."""
    return None

# Intel x86 QNN Conv2D legalization function.
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
    We get our new quantized uint8 tensor - scale * (QA' - zp_a')

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

    def _shift(data, out_dtype):
        """Shifts (add/subtracts) the qnn tensor with +/-128)"""
        if out_dtype == 'uint8':
            shift = 128
        elif out_dtype == 'int8':
            shift = -128
        else:
            raise ValueError("Unsupport out dtype.")
        data_modified = relay.cast(data, 'int32')
        data_modified = relay.add(data_modified, relay.const(shift, 'int32'))
        data_modified = relay.cast(data_modified, out_dtype)
        return data_modified

    def _is_int8_hw_support(target):
        """
        Checks to ensure that we can use Intel DLBoost instructions - Check if the target is skylake
        and above.
        """
        supported_arches = {'-mcpu=skylake-avx512', '-mcpu=cascadelake'}
        return supported_arches.intersection(set(target.options))

    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    # Collect the input exprs.
    data, kernel = inputs

    # The VNNI transformations are applicable only Skylake and above.g
    target = tvm.target.current_target(allow_none=False)
    if not _is_int8_hw_support(target):
        return None

    # VNNI supports u8 x i8 fast conv/MM. Don't do anything if it is already satisfied.
    if data_dtype == 'uint8' and kernel_dtype == 'int8':
        return None

    # Shift input if necessary.
    input_zp = attrs['input_zero_point']
    if data_dtype == 'int8':
        # Compute (QA + 128) and (zp_a + 128)
        data = _shift(data, 'uint8')
        input_zp = input_zp + 128

    # Shift kernel if necessary.
    kernel_zp = attrs['kernel_zero_point']
    if kernel_dtype == 'uint8':
        # Compute (QA - 128) and (zp_a - 128)
        kernel = _shift(kernel, 'int8')
        kernel_zp = kernel_zp - 128

    # Call qnn.conv2d with modified inputs and zero points.
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['input_zero_point'] = input_zp
    new_attrs['kernel_zero_point'] = kernel_zp
    return relay.qnn.op.conv2d(data, kernel, **new_attrs)
