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

#################################################
# Register the functions for different operators.
#################################################

# Registering QNN Conv2D legalization function.
@reg.register_qnn_legalize("qnn.conv2d")
def legalize_qnn_conv2d(attrs, inputs, types):
    return qnn_conv2d_legalize(attrs, inputs, types)

# Registering QNN dense legalization function.
@reg.register_qnn_legalize("qnn.dense")
def legalize_qnn_dense(attrs, inputs, types):
    return qnn_dense_legalize(attrs, inputs, types)

# Default to None. If overridden by target, this will not be run.
# Generic QNN Conv2D legalization function.
@tvm.target.generic_func
def qnn_conv2d_legalize(attrs, inputs, types):
    """Default legalization is None."""
    return None

# Generic QNN Conv2D legalization function.
@tvm.target.generic_func
def qnn_dense_legalize(attrs, inputs, types):
    """Default legalization is None."""
    return None

###################
# Helper functions.
###################

# Helper function for lowering in the abscence of fast Int8 arithmetic units.
def helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay_op):
    """ Converts QNN operators into a sequence of Relay operators that are friendly to HW that do
    not have fast Int8 arithmetic. For example, for ARM, LLVM utilizes the assembly instructions
    much more efficiently if the convolution or dense operator input datatypes are int16 instead of
    int8. More details are present at https://github.com/apache/incubator-tvm/pull/4277.

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

    # Collect the input exprs.
    data, kernel = inputs

    input_zp = attrs['input_zero_point']
    kernel_zp = attrs['kernel_zero_point']

    shift_data = relay.subtract(relay.cast(data, dtype='int16'),
                                relay.const(input_zp, 'int16'))
    shift_kernel = relay.subtract(relay.cast(kernel, dtype='int16'),
                                  relay.const(kernel_zp, 'int16'))
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    del new_attrs['kernel_zero_point']
    del new_attrs['input_zero_point']
    del new_attrs['input_scale']
    del new_attrs['kernel_scale']
    return relay_op(shift_data, shift_kernel, **new_attrs)

# Helper function to change dtypes to uint8 x int8. Intel VNNI instructions prefer this setting.
def helper_change_dtypes_to_uint8_int8(attrs, inputs, types, relay_op):
    """Legalizes QNN conv2d/dense op for Intel HW. VNNI supports u8 x i8 fast conv/MM. If the dtypes
    are already good, we dont transform. Else, we shift the tensor values and zero points to change
    the dtype.

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

    def _shift(data, zero_point, out_dtype):
        """Shifts (add/subtracts) the qnn tensor with +/-128)"""
        if out_dtype == 'uint8':
            shift = 128
        elif out_dtype == 'int8':
            shift = -128
        else:
            raise ValueError("Unsupported out dtype.")
        data_modified = relay.cast(data, 'int32')
        data_modified = relay.add(data_modified, relay.const(shift, 'int32'))
        data_modified = relay.cast(data_modified, out_dtype)
        return (data_modified, zero_point + shift)

    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    # Collect the input exprs.
    data, kernel = inputs

    # VNNI supports u8 x i8 fast conv/MM. Don't do anything if it is already satisfied.
    if data_dtype == 'uint8' and kernel_dtype == 'int8':
        return None

    # Shift input if necessary.
    input_zp = attrs['input_zero_point']
    if data_dtype == 'int8':
        # Compute (QA + 128) and (zp_a + 128)
        data, input_zp = _shift(data, input_zp, 'uint8')

    # Shift kernel if necessary.
    kernel_zp = attrs['kernel_zero_point']
    if kernel_dtype == 'uint8':
        # Compute (QA - 128) and (zp_a - 128)
        kernel, kernel_zp = _shift(kernel, kernel_zp, 'int8')

    # Call qnn.conv2d with modified inputs and zero points.
    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['input_zero_point'] = input_zp
    new_attrs['kernel_zero_point'] = kernel_zp
    return relay_op(data, kernel, **new_attrs)

# Helper function to change dtypes to be same. ARM dotprod instructions prefer this setting.
def helper_change_dtypes_to_be_same(attrs, inputs, types, relay_op):
    """ Sometimes MxNet + MLDNN can lead to uint8 x int8 datatypes for the conv inputs. However,
    many devices like ARM prefer the datatypes to be same for the HW units. This helper transforms
    conv2d/dense such that both the dtypes are same.

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

    def _shift(data, zero_point, out_dtype):
        """Shifts (adds/subtracts) the qnn tensor by 128)"""
        if out_dtype == 'uint8':
            shift = 128
        elif out_dtype == 'int8':
            shift = -128
        else:
            raise ValueError("Unsupported out dtype.")
        data_modified = relay.cast(data, 'int32')
        data_modified = relay.add(data_modified, relay.const(shift, 'int32'))
        data_modified = relay.cast(data_modified, out_dtype)
        return (data_modified, zero_point + shift)

    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    if data_dtype == kernel_dtype:
        return None

    # Collect the input exprs.
    data, kernel = inputs

    assert 'int8' in data_dtype and 'int8' in kernel_dtype, \
            "Qnn Conv2D/Dense only accepts uint8 or int8 inputs"

    # Shift input if necessary.
    input_zp = attrs['input_zero_point']
    data, input_zp = _shift(data, input_zp, kernel_dtype)

    new_attrs = {k : attrs[k] for k in attrs.keys()}
    new_attrs['input_zero_point'] = input_zp
    return relay_op(data, kernel, **new_attrs)

def is_fast_int8_on_intel():
    """ Checks whether the hardware has support for fast Int8 arithmetic operations. """
    target = tvm.target.current_target(allow_none=False)
    intel_supported_arches = {'-mcpu=skylake-avx512', '-mcpu=cascadelake'}
    return intel_supported_arches.intersection(set(target.options))

def is_fast_int8_on_arm():
    """ Checks whether the hardware has support for fast Int8 arithmetic operations. """
    target = tvm.target.current_target(allow_none=False)
    return '+v8.2a,+dotprod' in ' '.join(target.options)

########################
# ARM CPU legalizations.
########################

@qnn_conv2d_legalize.register('arm_cpu')
def _qnn_conv2d_legalize_arm_cpu(attrs, inputs, types):
    # ARM prefers the dtypes to be same.
    if is_fast_int8_on_arm():
        return helper_change_dtypes_to_be_same(attrs, inputs, types, relay.qnn.op.conv2d)
    return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.conv2d)

@qnn_dense_legalize.register('arm_cpu')
def _qnn_dense_legalize_arm_cpu(attrs, inputs, types):
    # ARM prefers the dtypes to be same.
    if is_fast_int8_on_arm():
        return helper_change_dtypes_to_be_same(attrs, inputs, types, relay.qnn.op.dense)
    return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.dense)

##########################
# Intel CPU legalizations.
##########################

@qnn_conv2d_legalize.register('cpu')
def _qnn_conv2d_legalize_intel_cpu(attrs, inputs, types):
    # The VNNI transformations prefer uint8 x int8 datatypes.
    if is_fast_int8_on_intel():
        return helper_change_dtypes_to_uint8_int8(attrs, inputs, types, relay.qnn.op.conv2d)
    return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.conv2d)

@qnn_dense_legalize.register('cpu')
def _qnn_dense_legalize_intel_cpu(attrs, inputs, types):
    # The VNNI transformations prefer uint8 x int8 datatypes.
    if is_fast_int8_on_intel():
        return helper_change_dtypes_to_uint8_int8(attrs, inputs, types, relay.qnn.op.dense)
    return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.dense)
