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
import numpy as np

import tvm
from tvm import relay
from tvm._ffi.base import TVMError
from tvm.relay.qnn.op.canonicalizations import create_integer_lookup_op

from ....target.x86 import target_has_features
from ....topi.utils import is_target
from .. import op as reg

#################################################
# Register the functions for different operators.
#################################################

# Registering QNN Conv2D legalization function.


@reg.register_qnn_legalize("qnn.conv2d")
def legalize_qnn_conv2d(attrs, inputs, types):
    return qnn_conv2d_legalize(attrs, inputs, types)


# Registering QNN Conv2DTranspose legalization function.
@reg.register_qnn_legalize("qnn.conv2d_transpose")
def legalize_qnn_conv2d_transpose(attrs, inputs, types):
    return qnn_conv2d_transpose_legalize(attrs, inputs, types)


# Registering QNN dense legalization function.
@reg.register_qnn_legalize("qnn.dense")
def legalize_qnn_dense(attrs, inputs, types):
    return qnn_dense_legalize(attrs, inputs, types)


def register_qnn_unary_op_legalize(op_name, floating_point_func):
    """Register unary qnn op for legalization via table lookup op."""

    def legalize_qnn_unary_op(attrs, inputs, types):
        return create_integer_lookup_op(
            input_arg=inputs[0],
            floating_point_func=floating_point_func,
            in_scale=inputs[1],
            in_zero_point=inputs[2],
            out_scale=inputs[3],
            out_zero_point=inputs[4],
            in_dtype=types[0].dtype,
            out_dtype=types[0].dtype,
        )

    return reg.register_qnn_legalize(op_name, legalize_qnn_unary_op)


def hardswish_func(x):
    x2 = x + 3.0
    x2 = np.clip(x2, 0.0, 6.0)
    return x * x2 / 6.0


register_qnn_unary_op_legalize("qnn.sqrt", np.sqrt)
register_qnn_unary_op_legalize("qnn.rsqrt", lambda arr: 1 / np.sqrt(arr))
register_qnn_unary_op_legalize("qnn.exp", np.exp)
register_qnn_unary_op_legalize("qnn.sigmoid", lambda arr: 1 / (1 + np.exp(-arr)))
register_qnn_unary_op_legalize("qnn.hardswish", hardswish_func)
register_qnn_unary_op_legalize("qnn.tanh", np.tanh)
register_qnn_unary_op_legalize("qnn.log", np.log)
register_qnn_unary_op_legalize("qnn.abs", np.abs)


@reg.register_qnn_legalize("qnn.erf")
def _legalize_qnn_erf(attrs, inputs, types):
    from scipy import special  # pylint: disable=import-outside-toplevel

    return create_integer_lookup_op(
        input_arg=inputs[0],
        floating_point_func=special.erf,
        in_scale=inputs[1],
        in_zero_point=inputs[2],
        out_scale=inputs[3],
        out_zero_point=inputs[4],
        in_dtype=types[0].dtype,
        out_dtype=types[0].dtype,
    )


# Default to None. If overridden by target, this will not be run.
# Generic QNN Conv2D legalization function.
@tvm.target.generic_func
def qnn_conv2d_legalize(attrs, inputs, types):
    """Default legalization is None."""
    return None


# Generic QNN Conv2DTranspose legalization function.
@tvm.target.generic_func
def qnn_conv2d_transpose_legalize(attrs, inputs, types):
    """Convert kernel and data to int16, subtract offsets upfront
    and calls into relay.nn.conv2d_transpose."""

    # Collect the input exprs.
    data, kernel, input_zero_point, kernel_zero_point, _, _ = inputs

    # If input zero point is a scalar, we can directly subtract it.
    if len(types[2].shape) == 0:
        shift_data = relay.subtract(
            relay.cast(data, dtype="int16"), relay.cast(input_zero_point, "int16")
        )
    # Otherwise it needs to be broadcast.
    else:
        shift_data = relay.nn.bias_add(
            relay.cast(data, dtype="int16"),
            -relay.cast(input_zero_point, dtype="int16"),
        )

    # If kernel zero point is a scalar, we can directly subtract it.
    if len(types[3].shape) == 0:
        shift_kernel = relay.subtract(
            relay.cast(kernel, dtype="int16"), relay.cast(kernel_zero_point, "int16")
        )
    # Otherwise it needs to be broadcast.
    else:
        shift_kernel = relay.nn.bias_add(
            relay.cast(kernel, dtype="int16"),
            -relay.cast(kernel_zero_point, dtype="int16"),
        )

    return relay.nn.conv2d_transpose(shift_data, shift_kernel, **attrs)


# Generic QNN Conv2D legalization function.
@tvm.target.generic_func
def qnn_dense_legalize(attrs, inputs, types):
    """Default legalization is None."""
    return None


###################
# Helper functions.
###################


def get_scalar_from_constant(expr):
    """Returns scalar value from Relay constant scalar."""
    assert (
        isinstance(expr, relay.Constant) and not expr.data.shape
    ), "Expr is not a constant scalar."
    value = expr.data.numpy()
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(
        np.float32
    ), "value must be float32/int32"
    return value.item(0)


def _shift(data, zero_point, out_dtype):
    """Shifts (add/subtracts) the qnn tensor with +/-128)"""
    if out_dtype == "uint8":
        shift = 128
    elif out_dtype == "int8":
        shift = -128
    else:
        raise ValueError("Unsupported out dtype.")
    data_modified = relay.cast(data, "int32")
    data_modified = relay.add(data_modified, relay.const(shift, "int32"))
    data_modified = relay.cast(data_modified, out_dtype)
    if isinstance(zero_point, relay.Constant):
        zero_point_val = get_scalar_from_constant(zero_point)
        zero_point_modified = relay.const(zero_point_val + shift, "int32")
    else:
        zero_point_modified = zero_point + relay.const(shift, "int32")
    return (data_modified, zero_point_modified)


# Helper function for lowering in the abscence of fast Int8 arithmetic units.
def helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay_op):
    """Converts QNN operators into a sequence of Relay operators that are friendly to HW that do
    not have fast Int8 arithmetic. For example, for ARM, LLVM utilizes the assembly instructions
    much more efficiently if the convolution or dense operator input datatypes are int16 instead of
    int8. More details are present at https://github.com/apache/tvm/pull/4277.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
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
    data, kernel, input_zero_point, kernel_zero_point, _, _ = inputs

    shift_data = relay.subtract(
        relay.cast(data, dtype="int16"), relay.cast(input_zero_point, dtype="int16")
    )
    # If kernel zero point is a scalar we can directly subtract it.
    if len(types[3].shape) == 0:
        shift_kernel = relay.subtract(
            relay.cast(kernel, dtype="int16"), relay.cast(kernel_zero_point, dtype="int16")
        )
    # Otherwise it needs to be broadcast.
    else:
        # Determine output axis of kernel for spatial operations.
        if hasattr(attrs, "kernel_layout"):
            output_axis = tvm.tir.layout(attrs["kernel_layout"]).index_of("O")
        # For dense operations, broadcast to [N, K] layout.
        elif isinstance(attrs, relay.op.op_attrs.DenseAttrs):
            output_axis = 0
        # For matrix multiplication instead expand to [K, N] layout.
        elif isinstance(attrs, relay.op.op_attrs.MatmulAttrs):
            output_axis = 1
        else:
            raise TVMError(
                "Legalization of %s is not yet supported with per channel parameters"
                % str(type(attrs))
            )

        shift_kernel = relay.nn.bias_add(
            relay.cast(kernel, dtype="int16"),
            -relay.cast(kernel_zero_point, dtype="int16"),
            output_axis,
        )
    new_attrs = {k: attrs[k] for k in attrs.keys()}
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

    Similarly we can convert from uint8 to int8.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
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
    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    # Collect the input exprs.
    data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale = inputs

    # VNNI supports u8 x i8 fast conv/MM. Don't do anything if it is already satisfied.
    if data_dtype == "uint8" and kernel_dtype == "int8":
        return None

    # Shift input if necessary.
    if data_dtype == "int8":
        # Compute (QA + 128) and (zp_a + 128)
        data, input_zero_point = _shift(data, input_zero_point, "uint8")

    # Shift kernel if necessary.
    if kernel_dtype == "uint8":
        # Compute (QA - 128) and (zp_a - 128)
        kernel, kernel_zero_point = _shift(kernel, kernel_zero_point, "int8")

    # Call qnn.conv2d with modified inputs and zero points.
    new_attrs = {k: attrs[k] for k in attrs.keys()}
    return relay_op(
        data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale, **new_attrs
    )


# Helper function to change dtypes to int8 x int8. Cuda dp4a instructions prefer this setting.
def helper_change_dtypes_to_int8(attrs, inputs, types, relay_op):
    """Legalizes QNN conv2d/dense op for Nvidia HW. dp4a supports i8 x i8 fast conv/MM. If the
    dtypes are already good, we dont transform. Else, we shift the tensor values and zero points
    to change the dtype.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
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
    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    # Collect the input exprs.
    data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale = inputs

    # dp4a supports i8 x i8 fast conv/MM. Don't do anything if it is already satisfied.
    if data_dtype == "int8" and kernel_dtype == "int8":
        return None

    # Shift input if necessary.
    if data_dtype == "uint8":
        # Compute (QA + 128) and (zp_a + 128)
        data, input_zero_point = _shift(data, input_zero_point, "int8")

    # Shift kernel if necessary.
    if kernel_dtype == "uint8":
        # Compute (QA - 128) and (zp_a - 128)
        kernel, kernel_zero_point = _shift(kernel, kernel_zero_point, "int8")

    # Call qnn.conv2d with modified inputs and zero points.
    new_attrs = {k: attrs[k] for k in attrs.keys()}
    return relay_op(
        data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale, **new_attrs
    )


def helper_change_dtypes_to_uint8(attrs, inputs, types, relay_op):
    """Helper function to change dtypes to uint8 x uint8.
    Legalizes QNN dense op for Hexagon DSP. It supports fast u8 x u8 vrmpy instruction.

    Converting from int8 to uint8 can be done in following manner:

    Original equation
      scale * (QA - zp_a)
      scale * (QA + 128 - 128 - zp_a)
      scale * ( (QA + 128) - (zp_a + 128))

    Replacing QA + 128 with QA' and (zp_a + 128) with zp_a'
    We get our new quantized uint8 tensor - scale * (QA' - zp_a')

    Parameters
    ----------
    attrs : tvm.ir.Attrs
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
    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    # Do nothing since it is already uint8.
    if data_dtype == "uint8" and kernel_dtype == "uint8":
        return None

    # Collect the input exprs.
    data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale = inputs

    # Shift input if necessary.
    if data_dtype == "int8":
        # Compute (QA + 128) and (zp_a + 128)
        data, input_zero_point = _shift(data, input_zero_point, "uint8")

    # Shift kernel if necessary.
    if kernel_dtype == "int8":
        # Compute (QA + 128) and (zp_a + 128)
        kernel, kernel_zero_point = _shift(kernel, kernel_zero_point, "uint8")

    # Call qnn.conv2d/qnn.dense with modified inputs and zero points.
    new_attrs = dict(attrs)
    return relay_op(
        data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale, **new_attrs
    )


# Helper function to change dtypes to be same. ARM dotprod instructions prefer this setting.
def helper_change_dtypes_to_be_same(attrs, inputs, types, relay_op):
    """Sometimes MxNet + MLDNN can lead to uint8 x int8 datatypes for the conv inputs. However,
    many devices like ARM prefer the datatypes to be same for the HW units. This helper transforms
    conv2d/dense such that both the dtypes are same.

    Parameters
    ----------
    attrs : tvm.ir.Attrs
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
        if out_dtype == "uint8":
            shift = 128
        elif out_dtype == "int8":
            shift = -128
        else:
            raise ValueError("Unsupported out dtype.")
        data_modified = relay.cast(data, "int32")
        data_modified = relay.add(data_modified, relay.const(shift, "int32"))
        data_modified = relay.cast(data_modified, out_dtype)
        zero_point_val = get_scalar_from_constant(zero_point)
        zero_point_modified = relay.const(zero_point_val + shift, "int32")
        return (data_modified, zero_point_modified)

    # Collect the dtypes.
    data_dtype = types[0].dtype
    kernel_dtype = types[1].dtype

    if data_dtype == kernel_dtype:
        return None

    # Collect the input exprs.
    data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale = inputs

    assert (
        "int8" in data_dtype and "int8" in kernel_dtype
    ), "Qnn Conv2D/Dense only accepts uint8 or int8 inputs"

    # Shift input if necessary.
    data, input_zero_point = _shift(data, input_zero_point, kernel_dtype)

    new_attrs = {k: attrs[k] for k in attrs.keys()}
    return relay_op(
        data, kernel, input_zero_point, kernel_zero_point, input_scale, kernel_scale, **new_attrs
    )


def is_fast_int8_on_intel():
    """Checks whether the hardware has support for fast Int8 arithmetic operations."""
    return target_has_features("sse4.2")


# Helper function to align up given value.
def helper_align_up(value, aligner):
    return ((value + aligner) // aligner) * aligner


########################
# ARM CPU legalizations.
########################


@qnn_conv2d_legalize.register("arm_cpu")
def _qnn_conv2d_legalize_arm_cpu(attrs, inputs, types):
    target = tvm.target.Target.current(allow_none=False)
    is_depthwise = relay.op.strategy.is_depthwise_conv2d(
        types[0].shape,
        attrs["data_layout"],
        types[1].shape,
        attrs["kernel_layout"],
        attrs["groups"],
    )
    use_int8_on_arm = (not is_depthwise) and attrs["data_layout"] == "NHWC"
    other_options = use_int8_on_arm or target.features.has_dotprod
    if target.features.has_asimd and not other_options:
        return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.conv2d)
    # ARM prefers the dtypes to be same.
    return helper_change_dtypes_to_be_same(attrs, inputs, types, relay.qnn.op.conv2d)


@qnn_dense_legalize.register("arm_cpu")
def _qnn_dense_legalize_arm_cpu(attrs, inputs, types):
    target = tvm.target.Target.current(allow_none=False)
    if target.features.has_asimd and not target.features.has_dotprod:
        return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.dense)
    # ARM prefers the dtypes to be same.
    return helper_change_dtypes_to_be_same(attrs, inputs, types, relay.qnn.op.dense)


##########################
# Intel CPU legalizations.
##########################


@qnn_conv2d_legalize.register("cpu")
def _qnn_conv2d_legalize_intel_cpu(attrs, inputs, types):
    # TODO(vvchernov): not only VNNI
    # The VNNI transformations prefer uint8 x int8 datatypes.
    if is_fast_int8_on_intel():
        return helper_change_dtypes_to_uint8_int8(attrs, inputs, types, relay.qnn.op.conv2d)
    return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.conv2d)


@qnn_dense_legalize.register("cpu")
def _qnn_dense_legalize_intel_cpu(attrs, inputs, types):
    # TODO(vvchernov): not only VNNI
    # The VNNI transformations prefer uint8 x int8 datatypes.
    if is_fast_int8_on_intel():
        return helper_change_dtypes_to_uint8_int8(attrs, inputs, types, relay.qnn.op.dense)
    return helper_no_fast_int8_hw_legalization(attrs, inputs, types, relay.nn.dense)


#####################
# CUDA and vulkan legalizations.
#####################


@qnn_conv2d_legalize.register(["cuda", "gpu"])
def _qnn_conv2d_legalize_cuda(attrs, inputs, types):
    if is_target("vulkan"):
        # prefers the dtypes to be same. Mixed type is not yet supported.
        return helper_change_dtypes_to_be_same(attrs, inputs, types, relay.qnn.op.conv2d)
    if is_target(["cuda", "rocm"]):
        # CUDA prefers both datatypes to be int8.
        return helper_change_dtypes_to_int8(attrs, inputs, types, relay.qnn.op.conv2d)
    return None


@qnn_dense_legalize.register(["cuda", "gpu"])
def _qnn_dense_legalize_cuda(attrs, inputs, types):
    if is_target("vulkan"):
        # prefers the dtypes to be same. Mixed type is not yet supported.
        return helper_change_dtypes_to_be_same(attrs, inputs, types, relay.qnn.op.dense)
    if is_target(["cuda", "rocm"]):
        # CUDA prefers both datatypes to be the int8.
        return helper_change_dtypes_to_int8(attrs, inputs, types, relay.qnn.op.dense)
    return None


########################
# Hexagon legalizations.
########################

IN_CHANNEL_VECTOR_LENGTH = 4
OUT_CHANNEL_VECTOR_LENGTH = 32


@qnn_conv2d_legalize.register("hexagon")
def _qnn_conv2d_legalize_hexagon(attrs, inputs, types):
    """Legalize qnn.conv2d op for vrmpy tensorization.

    If the inputs are signed or unsigned int8 and data/kernel layouts are NCHW/OIHW, then the input
    and output channels are padded to be a multiple of 4 and 32 respectively.
    """
    data_layout = attrs["data_layout"]
    kernel_layout = attrs["kernel_layout"]

    if data_layout != "NCHW" or kernel_layout != "OIHW":
        return None

    data_tensor, kernel_tensor = types[0], types[1]

    if "int8" in data_tensor.dtype and "int8" in kernel_tensor.dtype:
        in_channel = data_tensor.shape[1].value
        out_channel = kernel_tensor.shape[0].value
        ic_modified = False
        oc_modified = False
        data, kernel, data_zp, kernel_zp, data_scale, kernel_scale = inputs

        if in_channel % IN_CHANNEL_VECTOR_LENGTH != 0:
            new_in_channel = helper_align_up(in_channel, IN_CHANNEL_VECTOR_LENGTH)
            diff = new_in_channel - in_channel
            pad_width = ((0, 0), (0, diff), (0, 0), (0, 0))
            data = relay.nn.pad(data, pad_width=pad_width)
            kernel = relay.nn.pad(kernel, pad_width=pad_width)
            ic_modified = True

        new_out_channel = out_channel
        if out_channel % OUT_CHANNEL_VECTOR_LENGTH != 0:
            new_out_channel = helper_align_up(out_channel, OUT_CHANNEL_VECTOR_LENGTH)
            diff = new_out_channel - out_channel
            kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0), (0, 0), (0, 0)))
            oc_modified = True

            # Pad kernel zero point by 'diff' elements of 0 if it is not scalar
            kernel_zp_tensor = types[3]
            if len(kernel_zp_tensor.shape) != 0:
                assert isinstance(kernel_zp, relay.Constant)
                padded_kernel_zp_np = np.append(kernel_zp.data.numpy(), [0] * diff)
                kernel_zp = relay.const(padded_kernel_zp_np)

            # Pad kernel scale by 'diff' elements of 1.0 if it is not scalar
            kernel_scale_tensor = types[5]
            if len(kernel_scale_tensor.shape) != 0:
                assert isinstance(kernel_scale, relay.Constant)
                padded_kernel_scale_np = np.append(kernel_scale.data.numpy(), [1.0] * diff)
                kernel_scale = relay.const(padded_kernel_scale_np)

        if ic_modified is True or oc_modified is True:
            new_attrs = dict(attrs)
            if oc_modified:
                new_attrs["channels"] = new_out_channel
                out = relay.qnn.op.conv2d(
                    data, kernel, data_zp, kernel_zp, data_scale, kernel_scale, **new_attrs
                )
                output_tensor = types[6]
                original_out_shape = list(output_tensor.shape)
                out = relay.strided_slice(out, begin=[0, 0, 0, 0], end=original_out_shape)
            else:
                out = relay.qnn.op.conv2d(
                    data, kernel, data_zp, kernel_zp, data_scale, kernel_scale, **new_attrs
                )

            return out

    return None


@qnn_dense_legalize.register("hexagon")
def _qnn_dense_legalize_hexagon(attrs, inputs, types):
    """Legalize qnn.dense op for vrmpy tensorization.

    N dimension of weights should be aligned on vector length. If not, then N dimension is padded to
    be a multiple of 32.
    """
    assert len(types) == 7
    assert len(inputs) == 6

    data_tensor, kernel_tensor = types[0], types[1]
    if "int8" not in data_tensor.dtype or "int8" not in kernel_tensor.dtype:
        return None

    N, _ = kernel_tensor.shape

    if N % OUT_CHANNEL_VECTOR_LENGTH != 0:
        N_padded = helper_align_up(N, OUT_CHANNEL_VECTOR_LENGTH)
        diff = N_padded - N

        data, kernel, data_zp, kernel_zp, data_scale, kernel_scale = inputs

        # Pad weights by 'diff'
        padded_kernel = relay.nn.pad(kernel, pad_width=((0, diff), (0, 0)))

        kernel_zp_tensor, kernel_scale_tensor = types[3], types[5]

        # Pad kernel zero point by 'diff' elements of 0 if it is not scalar
        if len(kernel_zp_tensor.shape) != 0:
            assert isinstance(kernel_zp, relay.Constant)
            assert isinstance(diff, tvm.tir.IntImm)
            padded_kernel_zp_np = np.append(kernel_zp.data.numpy(), [0] * diff.value)
            kernel_zp = relay.const(padded_kernel_zp_np)

        # Pad kernel scale by 'diff' elements of 1.0 if it is not scalar
        if len(kernel_scale_tensor.shape) != 0:
            assert isinstance(kernel_scale, relay.Constant)
            assert isinstance(diff, tvm.tir.IntImm)
            padded_kernel_scale_np = np.append(kernel_scale.data.numpy(), [1.0] * diff.value)
            kernel_scale = relay.const(padded_kernel_scale_np)

        # If units is explicitly specified, it is used to compute the output shape.
        # We need to update units after padding to prevent a type error.
        new_attrs = dict(attrs)
        if attrs["units"] is not None:
            new_attrs["units"] = N + diff

        new_inputs = (data, padded_kernel, data_zp, kernel_zp, data_scale, kernel_scale)

        out = relay.qnn.op.dense(*new_inputs, **new_attrs)

        output_tensor = types[6]
        out = relay.strided_slice(out, begin=[0, 0], end=list(output_tensor.shape))
        return out

    return None
