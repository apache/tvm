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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""ONNX: Open Neural Network Exchange frontend for Relay."""
import copy
import math
import warnings
from typing import Optional

import numpy as np
import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import loops as _loops
from .. import op as _op
from .. import qnn as _qnn
from .. import random as _random
from .. import ty as _ty
from .. import vision as _vision
from .common import (
    AttrCvt,
    Renamer,
    autopad,
    ensure_scalar_shape,
    fold_constant,
    get_name,
    get_relay_op,
    gru_cell,
    infer_channels,
    infer_shape,
    infer_type,
    infer_value,
    lstm_cell,
    new_var,
    shape_of,
    try_resolve_var_to_const,
    unbind,
)

__all__ = ["from_onnx"]

# The default configurations of Relay ONNX frontend.
ONNX_DEFAULT_CONFIGS = {
    # By default, TVM converts qualified onnx `matmul` to `transpose(weight) + nn.batch_matmul_NT`.
    # Change this flag to False to directly convert to `nn.batch_matmul`.
    # Note that `nn.batch_matmul` with format other than NT is in experimental, it may have some
    # performance issues.
    "use_nt_batch_matmul": True,
}


class onnx_input(list):
    """A helper extension to list that returns None for out of bound indices."""

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.stop is None:
                stop = len(self)
            else:
                stop = item.stop
            indices = list(range(stop)[item])
            return [self[i] for i in indices]
        if isinstance(item, int):
            return list(self)[item] if item < len(self) else None
        raise TypeError("list indices must be integers or slices, not %s" % type(item).__name__)


def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)


def get_type(elem_type):
    """Converts onnx integer datatype to numpy datatype"""
    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))

    try:
        from onnx import TensorProto
    except ImportError as e:
        raise ImportError("Unable to import TensorProto from onnx {}".format(e))

    # Onnx mapping converts bfloat16 to float16 because
    # numpy does not have a bfloat16 data type. However,
    # tvm has one, so we force the return type to be bfloat16
    if elem_type == int(TensorProto.BFLOAT16):
        return "bfloat16"
    return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])


def get_info(info_proto):
    """Extract the shape from a ValueInfoProto."""
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            value = _ty.Any()
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    if info_proto.type.tensor_type.elem_type:
        dtype = get_type(info_proto.type.tensor_type.elem_type)
    else:
        dtype = None
    return name, shape, dtype, shape_name


def dimension_picker(prefix, suffix=""):
    """Check that dimensions are supported."""

    def _impl(attr):
        kernel = attr["kernel_shape"]
        if len(kernel) == 1:
            return prefix + "1d" + suffix
        if len(kernel) == 2:
            return prefix + "2d" + suffix
        if len(kernel) == 3:
            return prefix + "3d" + suffix
        msg = "Only 1D, 2D, and 3D kernels are supported for operator {}."
        op_name = prefix + "1d/2d/3d"
        raise tvm.error.OpAttributeInvalid(msg.format(op_name))

    return _impl


def revert_caffe2_pad(pads):
    """Caffe2 requires two times the normal padding."""
    if len(pads) == 4:
        pads = pads[:2]
    elif len(pads) == 2:
        pass
    else:
        raise tvm.error.OpAttributeInvalid("Number of pads must be either 2 or 4.")
    return pads


def get_pad_pair(input1d, kernel1d, stride1d, mode):
    """infer pad size"""
    if input1d % stride1d == 0:
        pad = max(kernel1d - stride1d, 0)
    else:
        pad = max(kernel1d - (input1d % stride1d), 0)
    pad_before = pad // 2
    pad_after = pad - pad_before
    if "LOWER" in mode:
        return [pad_after, pad_before]
    return [pad_before, pad_after]


def onnx_default_layout(dims, op_name):
    if dims == 1:
        return "NCW"
    if dims == 2:
        return "NCHW"
    if dims == 3:
        return "NCDHW"

    msg = "Only 1D, 2D and 3D layouts are currently supported for operator {}."
    raise tvm.error.OpAttributeInvalid(msg.format(op_name))


def onnx_storage_order2layout(storage_order, dims, op_name):
    """converter of onnx storage order parameter to tvm storage order format"""
    if storage_order not in (0, 1):
        raise tvm.error.OpAttributeInvalid("Mode of storage_order must be either 0 or 1")

    if dims == 1:
        return "NCW" if storage_order == 0 else "NWC"
    if dims == 2:
        return "NCHW" if storage_order == 0 else "NHWC"
    if dims == 3:
        return "NCDHW" if storage_order == 0 else "NDHWC"

    msg = "Only 1D, 2D and 3D layouts are currently supported for operator {}."
    raise tvm.error.OpAttributeInvalid(msg.format(op_name))


def dimension_constraint():
    def _dim_check(attrs):
        if len(attrs["kernel_shape"]) in [1, 2, 3]:
            return True
        return False

    return _dim_check, "Only 1d, 2d and 3d kernel supported."


def get_scalar(x, params, dtype="float32"):
    """Helper to get a scalar value for Quantized operators."""
    if isinstance(x, _expr.Var) and x.name_hint in params:
        return _op.const(params[x.name_hint].numpy(), dtype)
    rank = len(infer_shape(x))
    assert rank <= 1, "scale and zero_point input must be scalars"
    if rank == 1:
        x = _op.squeeze(x, [0])
    return _op.cast(x, dtype)


def get_scalar_or_1d_tensor(x, params, dtype="float32"):
    """Helper to get a scalar value or 1D tensor for Quantized operators."""
    if isinstance(x, _expr.Var) and x.name_hint in params:
        return _op.const(params[x.name_hint].numpy(), dtype)
    rank = len(infer_shape(x))
    assert rank <= 1, "scale and zero_point input must be scalars or 1D tensors"
    return _op.cast(x, dtype)


def matmul_out_dtype(inputs, out_dtype):
    """Common function to handle MatMul and MatMulInteger16"""
    a_shape = shape_of(inputs[0])
    a_rank = infer_shape(a_shape)[0]
    b_shape = shape_of(inputs[1])
    b_rank = infer_shape(b_shape)[0]
    if a_rank > 2 or b_rank > 2:

        def flatten_to_nd(x, x_shape, nd=3):
            ndims = infer_shape(x_shape)[0]
            if ndims == nd:
                return x
            newshape = _op.concatenate(
                [
                    _expr.const([-1], dtype=infer_type(x_shape).checked_type.dtype),
                    _op.strided_slice(x_shape, [ndims - nd + 1], [ndims]),
                ],
                0,
            )
            out = _op.reshape(x, fold_constant(newshape))
            return out

        # Determine the output batch dimension.
        new_a_shape = a_shape
        new_b_shape = b_shape
        if a_rank > b_rank:
            rank_diff = a_rank - b_rank
            new_b_shape = _op.concatenate(
                [
                    _expr.const([1] * rank_diff, dtype=infer_type(b_shape).checked_type.dtype),
                    b_shape,
                ],
                0,
            )
        elif a_rank < b_rank:
            rank_diff = b_rank - a_rank
            new_a_shape = _op.concatenate(
                [
                    _expr.const([1] * rank_diff, dtype=infer_type(a_shape).checked_type.dtype),
                    a_shape,
                ],
                0,
            )
        else:
            pass

        out_batch = _op.concatenate(
            [
                _op.maximum(
                    _op.strided_slice(new_b_shape, [i], [i + 1]),
                    _op.strided_slice(new_a_shape, [i], [i + 1]),
                )
                for i in range(max(a_rank, b_rank) - 2)
            ],
            0,
        )

        b_type = infer_type(inputs[1])
        # Convert to dense if the second matrix is 2d and non-dynamic
        if b_rank == 2 and not _ty.is_dynamic(b_type.checked_type):
            a = flatten_to_nd(inputs[0], a_shape, 2)
            b = _op.transpose(inputs[1])
            output = _op.nn.dense(a, b, out_dtype=out_dtype)
        else:
            a = inputs[0]
            b = inputs[1]
            # broadcast a and b
            a_broadcasted_shape = fold_constant(
                _op.concatenate(
                    [
                        out_batch,
                        _op.strided_slice(a_shape, [a_rank - 2], [a_rank]),
                    ],
                    0,
                )
            )
            b_broadcasted_shape = fold_constant(
                _op.concatenate(
                    [
                        out_batch,
                        _op.strided_slice(b_shape, [b_rank - 2], [b_rank]),
                    ],
                    0,
                )
            )
            if not tvm.ir.structural_equal(a_shape, a_broadcasted_shape):
                a = _op.transform.broadcast_to(a, a_broadcasted_shape)
            if not tvm.ir.structural_equal(b_shape, b_broadcasted_shape):
                b = _op.transform.broadcast_to(b, b_broadcasted_shape)
            # Convert a and b into 3 dimensional tensors.
            a = flatten_to_nd(a, shape_of(a), 3)
            b = flatten_to_nd(b, shape_of(b), 3)
            if ONNX_DEFAULT_CONFIGS["use_nt_batch_matmul"]:
                # Transpose matrix dimensions of b.
                bt = _op.transpose(b, [0, 2, 1])
                # Perform a NT batch matmul.
                output = _op.nn.batch_matmul(a, bt, out_dtype=out_dtype)
            else:
                # Perform a NN batch matmul.
                output = _op.nn.batch_matmul(a, b, out_dtype=out_dtype, transpose_b=False)
        # Reshape output to original dimensions.
        final_shape = _op.concatenate(
            [
                out_batch,
                _op.strided_slice(
                    a_shape, [infer_shape(a_shape)[0] - 2], [infer_shape(a_shape)[0] - 1]
                ),
                _op.strided_slice(
                    b_shape, [infer_shape(b_shape)[0] - 1], [infer_shape(b_shape)[0]]
                ),
            ],
            0,
        )
        return _op.reshape(output, fold_constant(final_shape))

    if a_rank == 1:
        return _op.squeeze(_op.nn.matmul(_op.expand_dims(inputs[0], axis=0), inputs[1]), axis=[0])

    # Otherwise a simple dense op will get the job done.
    input_1_t = _op.transpose(inputs[1], axes=(1, 0))
    return _op.nn.dense(inputs[0], input_1_t, out_dtype=out_dtype)


def layer_norm(x, eps, gamma, beta):
    """Common function to handle layer norm"""
    eps_dtype = infer_type(x).checked_type.dtype

    u, s = _op.mean_variance(x, axis=-1, keepdims=True)
    output = _op.divide(
        _op.subtract(x, u),
        _op.sqrt(_op.add(s, _op.const(eps, dtype=eps_dtype))),
    )
    output = _op.multiply(output, gamma)
    if beta is not None:
        output = _op.add(output, beta)

    return output


class OnnxOpConverter(object):
    """A helper class for holding onnx op converters."""

    @classmethod
    def get_converter(cls, opset):
        """Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        versions = [int(d.replace("_impl_v", "")) for d in dir(cls) if "_impl_v" in d]
        versions = sorted(versions + [opset])
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError(
            "opset version {} of {} not implemented".format(version, cls.__name__)
        )


class Unary(OnnxOpConverter):
    """A helper class for unary op converters."""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 1, "Unary math op {} takes 1 input, {} given".format(
            cls.name, len(inputs)
        )
        op_name = cls.name
        return get_relay_op(op_name)(*inputs)


class Elemwise(OnnxOpConverter):
    """A helper class for elemwise op converters."""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Math op {} take 2 inputs, {} given".format(cls.name, len(inputs))
        op_name = cls.name
        conv_ops = ["conv2d", "conv2d_transpose"]
        if attr.get("broadcast", 0) and any(x in str(inputs[0]) for x in conv_ops):
            # TODO(zhreshold): remove hard coded infershape
            axis = int(attr.get("axis", 0))
            inputs[1] = _op.expand_dims(inputs[1], axis=axis, num_newaxis=2)
        return get_relay_op(op_name)(*inputs)


class Pool(OnnxOpConverter):
    """A helper class for pool op converters."""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        input_shape = infer_shape(data)
        ndim = len(input_shape)

        attr_cvt, data = cls._run_calculation(inputs, attr, params)
        out = attr_cvt([data], attr, params)

        if ndim - len(attr["kernel_shape"]) == 1:
            out = _op.squeeze(out, axis=[0])
        return out

    @classmethod
    def _run_calculation(cls, inputs, attr, params):
        """Helper method to return the processed input data and AttrCvt object"""

        data = inputs[0]
        input_shape = infer_shape(data)
        input_dtype = infer_type(data).checked_type.dtype
        ndim = len(input_shape)
        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                if cls.name == "avg_pool":
                    pad_tuple = []
                    for axis in range(len(input_shape) - 2):
                        axis_shape = input_shape[2 + axis]
                        stride = attr.get("strides", [1] * ndim)[axis]
                        kernel = attr["kernel_shape"][axis]
                        pad = get_pad_pair(axis_shape, kernel, stride, attr["auto_pad"])
                        pad_tuple.append(pad)
                    pad_tuple = tuple([val for pair in zip(*pad_tuple) for val in pair])
                    attr["pads"] = pad_tuple
                else:
                    # Warning: Pool does not yet support dynamic shapes,
                    # one will need to run dynamic_to_static on this model after import
                    if "int" in input_dtype:
                        pad_val = np.iinfo(np.dtype(input_dtype)).min
                    else:
                        pad_val = np.finfo(np.dtype(input_dtype)).min
                    data = autopad(
                        data,
                        attr.get("strides", [1] * (ndim - 2)),
                        attr["kernel_shape"],
                        [1] * ndim,
                        pad_value=pad_val,
                        mode=attr["auto_pad"],
                    )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator {} is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"], cls.name))
            attr.pop("auto_pad")

        if "storage_order" in attr:
            attr["layout"] = onnx_storage_order2layout(
                attr["storage_order"], dims=(len(input_shape) - 2), op_name=cls.name
            )
        else:
            if ndim - len(attr["kernel_shape"]) == 1:
                data = _op.expand_dims(data, axis=0)
                input_shape = [1] + list(input_shape)

            attr["layout"] = onnx_default_layout(dims=(len(input_shape) - 2), op_name=cls.name)

        return (
            AttrCvt(
                op_name=dimension_picker(cls.name),
                transforms={
                    "kernel_shape": "pool_size",
                    "pads": ("padding", 0),
                    "dilations": ("dilation", 1),
                },
                ignores=["storage_order"],
                custom_check=dimension_constraint(),
            ),
            data,
        )


class Absolute(Unary):
    """Operator converter for Absolute."""

    name = "abs"


class Add(Elemwise):
    """Operator converter for Add."""

    name = "add"


class AveragePool(Pool):
    """Operator converter for AveragePool."""

    name = "avg_pool"


class QLinearAveragePool(Pool):
    """Operator converter for QLinearAveragePool from Microsoft onnxruntime contrib opset."""

    name = "avg_pool"

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        x_scale = get_scalar(inputs[1], params)
        x_zero_point = get_scalar(inputs[2], params, dtype="int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        y_zero_point = get_scalar(inputs[4], params, dtype="int32")

        attr_cvt, data = cls._run_calculation(inputs, attr, params)

        input_dtype = infer_type(data).checked_type.dtype
        # Onnxruntime doesn't actually do this op in integer, they dequantize to fp32
        # and then requantize afer (according to documentation below)
        # https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.QLinearAveragePool
        float_node = _qnn.op.dequantize(data, x_scale, x_zero_point)
        out = attr_cvt([float_node], attr, params)
        return _qnn.op.quantize(out, y_scale, y_zero_point, out_dtype=input_dtype)


class BatchNorm(OnnxOpConverter):
    """Operator converter for BatchNorm."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # TODO(zhreshold): 'spatial' is not properly handled here.
        # TODO(vvchernov): 'training_mode' (onnx tag) is not correctly handled, ignore for now
        out = AttrCvt(
            op_name="batch_norm",
            ignores=["spatial", "is_test", "consumed_inputs", "momentum", "training_mode"],
        )(inputs, attr, params)
        # We only support test mode, so we return data, moving_mean, moving_var,
        # and then moving_mean and moving_var again as placeholders for
        # the expected "saved_mean", "saved_var".
        return _expr.TupleWrapper(_expr.Tuple((*out, out[1], out[2])), 5)


class InstanceNorm(OnnxOpConverter):
    """Operator converter for BatchNorm."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return AttrCvt(op_name="instance_norm")(inputs, attr, params)


class Conv(OnnxOpConverter):
    """Operator converter for Conv."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Use shape of input to determine convolution type.
        data = inputs[0]
        kernel = inputs[1]
        input_shape = infer_shape(data)
        ndim = len(input_shape)

        kernel_type = infer_type(inputs[1])
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]

        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shapes[0][2:]

        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                data = autopad(
                    data,
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    attr.get("dilations", [1] * (ndim - 2)),
                    mode=attr["auto_pad"],
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = [0 for i in range(ndim - 2)]
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"]))
            attr.pop("auto_pad")

        attr["channels"] = kernel_shapes[0][0]
        out = AttrCvt(
            op_name=dimension_picker("conv"),
            transforms={
                "kernel_shape": "kernel_size",
                "dilations": ("dilation", 1),
                "pads": ("padding", 0),
                "group": ("groups", 1),
            },
            custom_check=dimension_constraint(),
        )([data, kernel], attr, params)

        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out


class ConvTranspose(OnnxOpConverter):
    """Operator converter for ConvTranspose."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # get number of channels
        out_type = infer_type(inputs[1])
        kernel_shape = [get_const_tuple(out_type.checked_type.shape)]
        out_channels = kernel_shape[0][1] * attr.get("group", 1)
        attr["channels"] = out_channels
        groups = attr.get("group", 1)

        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shape[0][2:]

        attr["groups"] = groups
        # infer pads for auto_pad
        data = inputs[0]
        input_shape = infer_shape(data)
        ndim = len(input_shape)
        if "auto_pad" in attr or "output_shape" in attr:
            if "auto_pad" in attr:
                attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if "output_shape" in attr or attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                kernel_shape = attr["kernel_shape"]
                kndim = len(kernel_shape)
                dilations = attr.get("dilations", [1] * kndim)
                output_padding = attr.get("output_padding", [0] * kndim)
                strides = attr["strides"]
                total_pad = [0] * kndim
                # https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
                if "output_shape" in attr:
                    for i in range(kndim):
                        total_pad[i] = (
                            strides[i] * (input_shape[ndim - kndim + i] - 1)
                            + output_padding[i]
                            + ((kernel_shape[i] - 1) * dilations[i] + 1)
                            - attr["output_shape"][i]
                        )
                    left = [p // 2 for p in total_pad]
                    right = [total_pad[i] - left[i] for i in range(kndim)]
                    if "output_shape" in attr and "auto_pad" not in attr:
                        pad = right + left
                    elif "LOWER" in attr["auto_pad"]:
                        pad = left + right
                    else:
                        pad = right + left
                    attr["pads"] = pad
                else:
                    data = autopad(
                        data,
                        attr.get("strides", [1] * (ndim - 2)),
                        attr["kernel_shape"],
                        attr.get("dilations", [1] * (ndim - 2)),
                        deconv=True,
                        mode=attr["auto_pad"],
                    )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"]))
            if "auto_pad" in attr:
                attr.pop("auto_pad")

        out = AttrCvt(
            op_name=dimension_picker("conv", "_transpose"),
            transforms={
                "kernel_shape": "kernel_size",
                "dilations": ("dilation", 1),
                "pads": ("padding", 0),
                "group": ("groups", 1),
            },
            disables=["output_shape"],
            custom_check=dimension_constraint(),
        )([data, inputs[1]], attr, params)
        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        # get number of channels
        out_type = infer_type(inputs[1])
        kernel_shape = [get_const_tuple(out_type.checked_type.shape)]
        out_channels = kernel_shape[0][1] * attr.get("group", 1)
        attr["channels"] = out_channels
        groups = attr.get("group", 1)

        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shape[0][2:]

        attr["groups"] = groups
        # infer pads for auto_pad
        data = inputs[0]
        input_shape = infer_shape(data)
        ndim = len(input_shape)
        if "auto_pad" in attr or "output_shape" in attr:
            if "auto_pad" in attr:
                attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if "output_shape" in attr or attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                kernel_shape = attr["kernel_shape"]
                kndim = len(kernel_shape)
                dilations = attr.get("dilations", [1] * kndim)
                output_padding = attr.get("output_padding", [0] * kndim)
                strides = attr["strides"]
                total_pad = [0] * kndim
                # https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConvTranspose
                if "output_shape" in attr:
                    for i in range(kndim):
                        total_pad[i] = (
                            strides[i] * (input_shape[ndim - kndim + i] - 1)
                            + output_padding[i]
                            + ((kernel_shape[i] - 1) * dilations[i] + 1)
                            - attr["output_shape"][i]
                        )
                else:
                    for i in range(kndim):
                        total_pad[i] = (
                            output_padding[i]
                            + ((kernel_shape[i] - 1) * dilations[i] + 1)
                            - strides[i]
                        )
                left = [p // 2 for p in total_pad]
                right = [total_pad[i] - left[i] for i in range(kndim)]
                if "output_shape" in attr and "auto_pad" not in attr:
                    pad = right + left
                elif "LOWER" in attr["auto_pad"]:
                    pad = left + right
                else:
                    pad = right + left
                attr["pads"] = pad
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"]))
            if "auto_pad" in attr:
                attr.pop("auto_pad")

        out = AttrCvt(
            op_name=dimension_picker("conv", "_transpose"),
            transforms={
                "kernel_shape": "kernel_size",
                "dilations": ("dilation", 1),
                "pads": ("padding", 0),
                "group": ("groups", 1),
            },
            disables=["output_shape"],
            custom_check=dimension_constraint(),
        )([data, inputs[1]], attr, params)
        use_bias = len(inputs) == 3
        if use_bias:
            out = _op.nn.bias_add(out, inputs[2])
        return out


class GlobalAveragePool(OnnxOpConverter):
    """Operator converter for GlobalAveragePool"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        rank = len(infer_shape(inputs[0]))
        if rank == 3:
            return _op.nn.global_avg_pool1d(inputs[0])
        if rank == 4:
            return _op.nn.global_avg_pool2d(inputs[0])
        if rank == 5:
            return _op.nn.global_avg_pool3d(inputs[0])
        raise NotImplementedError(
            "Global average pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
            % (rank - 2),
        )


class QLinearGlobalAveragePool(OnnxOpConverter):
    "Operator converter for QLinearGlobalAveragePool from Microsoft onnxruntime contrib opset."

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        rank = len(infer_shape(inputs[0]))

        x_scale = get_scalar(inputs[1], params)
        x_zero_point = get_scalar(inputs[2], params, dtype="int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        y_zero_point = get_scalar(inputs[4], params, dtype="int32")

        input_dtype = infer_type(inputs[0]).checked_type.dtype

        # Onnxruntime documentation does not mention that this global avg_pool should follow the
        # sequence dequantize -> float op -> quantize, but that is how QLinearAveragePool is done.
        #
        # This op also follows the same pattern since qnn op is not available right now.
        # TODO: Generate QNN op to perform quantized operation instead of dequant -> op -> quant
        x = _qnn.op.dequantize(inputs[0], x_scale, x_zero_point)
        if rank == 3:
            out = _op.nn.global_avg_pool1d(x)
        elif rank == 4:
            out = _op.nn.global_avg_pool2d(x)
        elif rank == 5:
            out = _op.nn.global_avg_pool3d(x)
        else:
            raise NotImplementedError(
                "Global average pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
                % (rank - 2),
            )
        return _qnn.op.quantize(out, y_scale, y_zero_point, out_dtype=input_dtype)


class GlobalMaxPool(OnnxOpConverter):
    """Operator converter for GlobalMaxPool"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        rank = len(infer_shape(inputs[0]))
        if rank == 3:
            return _op.nn.global_max_pool1d(inputs[0])
        if rank == 4:
            return _op.nn.global_max_pool2d(inputs[0])
        if rank == 5:
            return _op.nn.global_max_pool3d(inputs[0])
        raise NotImplementedError(
            "Global max pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
            % (rank - 2),
        )


class Div(Elemwise):
    """Operator converter for Divide."""

    name = "divide"


class Elu(OnnxOpConverter):
    """Operator converter for Elu."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get("alpha", 1.0))
        return _expr.const(-alpha) * _op.nn.relu(
            _expr.const(1.0) - _op.exp(inputs[0])
        ) + _op.nn.relu(inputs[0])


class Gelu(OnnxOpConverter):
    """Operator converter for Gelu from Microsoft onnxruntime contrib opset.

    gelu(x) = 0.5x(1 + erf(x/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        x = inputs[0]

        # Declare consts
        const_dtype = infer_type(x).checked_type.dtype
        half = _expr.const(0.5, dtype=const_dtype)
        one = _expr.const(1.0, dtype=const_dtype)
        sqrt2 = _expr.const(math.sqrt(2), dtype=const_dtype)

        # Compute gelu
        term1 = _op.multiply(half, x)
        erf = _op.erf(_op.divide(x, sqrt2))
        term2 = _op.add(one, erf)
        return _op.multiply(term1, term2)


class BiasGelu(OnnxOpConverter):
    """Operator converter for BiasGelu from Microsoft onnxruntime contrib opset.

    bias_gelu(x, b) = 0.5(x, b)(1 + erf((x + b)/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        x = inputs[0]
        b = inputs[1]

        b_shape = infer_shape(b)
        assert len(b_shape) == 1, "BiasGelu bias term must be a 1D tensor"

        inp = _op.add(x, b)
        return Gelu._impl_v1([inp], attr, params)


class EmbedLayerNormalization(OnnxOpConverter):
    """Operator converter for EmbedLayerNormalization from Microsoft onnxruntime contrib opset.

    This layer embeds the input tokens, sums them, and applies layer normalization.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        input_ids = inputs[0]
        segment_ids = inputs[1]
        word_emb = inputs[2]
        pos_emb = inputs[3]
        segment_emb = inputs[4]
        gamma = inputs[5]
        beta = inputs[6]

        mask = inputs[7]
        pos_ids = inputs[8]

        eps = attr.get("epsilon", 1e-12)

        (batch_size, seq_len) = infer_shape(input_ids)

        if segment_ids:
            assert segment_emb

        if pos_ids is None:
            pos_ids = _op.const([list(range(seq_len))] * batch_size, dtype="int32")

        word_vec = _op.take(word_emb, input_ids, axis=0)
        segment_vec = _op.take(segment_emb, segment_ids, axis=0)
        pos_vec = _op.take(pos_emb, pos_ids, axis=0)

        vec_sum = _op.add(word_vec, pos_vec)
        if segment_ids:
            vec_sum = _op.add(vec_sum, segment_vec)

        ln = layer_norm(vec_sum, eps, gamma, beta)

        mask_index = _op.const(np.zeros((batch_size,), dtype="int32"))
        if mask:
            # calculate number of words per sentence
            mask_index = _op.sum(mask, axis=1)

        # TODO(@anwang2009): onnxruntime v1.10.0 requires a third output of vec_sum
        return _expr.TupleWrapper(_expr.Tuple([ln, mask_index]), 2)


class SkipLayerNormalization(OnnxOpConverter):
    """Operator converter for SkipLayerNormalization from Microsoft onnxruntime contrib opset.

    This layer sums the two input tensors (along with optional bias), and applies layer
    normalization.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        skip = inputs[1]
        gamma = inputs[2]
        beta = inputs[3]
        bias = inputs[4]

        assert (
            beta is not None and bias is not None
        ), "SkipLayerNormalization import currently only supports required beta and bias"

        eps = attr.get("epsilon", 1e-12)

        x = _op.add(data, skip)
        if bias is not None:
            x = _op.add(x, bias)

        output = layer_norm(x, eps, gamma, beta)

        # onnxruntime doesn't compute the other outputs, despite the documentation
        placeholder = _op.const(0, dtype="float32")

        return _expr.TupleWrapper(_expr.Tuple([output, placeholder, placeholder]), 3)


class Attention(OnnxOpConverter):
    """Operator converter for Attention from Microsoft onnxruntime contrib opset.

    This is the self-attention mechanism used in transformer models.
    """

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        num_heads = attr["num_heads"]
        assert (
            "qkv_hidden_sizes" not in attr
        ), "different hidden sizes for Q, K, V are not currently supported"
        assert "unidirectional" not in attr, "unidirectional attention not current supported"

        # (batch, seq, in_hidden)
        input_emb = inputs[0]

        # (in_hidden, 3 * out_hidden), where out_hidden = num_heads * head_size
        weight = inputs[1]

        # (3 * out_hidden,)
        bias = inputs[2]

        # 1. (    batch,              1,        max_seq, max_seq)
        # 2. (    batch, past_seq + seq,)
        # 3. (    batch,            seq, past_seq + seq,)
        # 4. (    batch,)
        # 5. (2 * batch,)
        # For now, we only support case 2.
        mask_index = inputs[3]

        # (2, batch, num_heads, past_seq, head_size)
        past = inputs[4]

        # (batch, num_heads, seq, seq)
        extra_add = inputs[5]

        (batch_size, seq_len, _) = infer_shape(input_emb)
        (out_hidden_x3,) = infer_shape(bias)
        assert out_hidden_x3 % 3 == 0, "bias shape should be divisible by 3"
        out_hidden = out_hidden_x3 // 3
        assert (
            out_hidden % num_heads == 0
        ), "output hidden size should be divisible by number of attention heads"
        head_size = out_hidden // num_heads

        assert (
            mask_index is not None
        ), "Attention import currently only supports required mask_index"
        mask_index_shape = infer_shape(mask_index)
        assert (
            len(mask_index_shape) == 2
            and mask_index_shape[0] == batch_size
            and mask_index_shape[1] == seq_len
        ), "currently only support (batch_size, sequence_length) mask index"

        assert past is None, "past K, V state is not currently supported"
        assert extra_add is None, "extra add to QxK not currently supported"

        # split weight and biases and do the matmuls
        w_Q, w_K, w_V = _op.split(weight, 3, axis=1)
        b_Q, b_K, b_V = _op.split(bias, 3, axis=0)
        # need to merge batch dimensions since TVM matmul is 2D
        input_emb = _op.reverse_reshape(input_emb, (-1, 0))
        Q = _op.add(_op.nn.matmul(input_emb, w_Q), b_Q)
        K = _op.add(_op.nn.matmul(input_emb, w_K), b_K)
        V = _op.add(_op.nn.matmul(input_emb, w_V), b_V)

        # massage tensors in preparation for batched matmul
        def massage(tensor):
            tensor = _op.reshape(tensor, (batch_size, seq_len, num_heads, head_size))

            # (batch_size, num_heads, seq_len, head_size)
            tensor = _op.transpose(tensor, axes=[0, 2, 1, 3])

            # (batch_size * num_heads, seq_len, head_size)
            return _op.reverse_reshape(tensor, (-1, 0, 0))

        Q = massage(Q)
        K = massage(K)
        V = massage(V)

        K_present = _op.reshape(K, (batch_size, num_heads, seq_len, head_size))
        V_present = _op.reshape(V, (batch_size, num_heads, seq_len, head_size))
        present = _op.stack([K_present, V_present], axis=0)

        att_scores = _op.nn.batch_matmul(Q, K, transpose_a=False, transpose_b=True)
        score_dtype = infer_type(att_scores).checked_type.dtype
        att_scores = _op.divide(
            att_scores,
            _op.const(np.sqrt(head_size), dtype=infer_type(att_scores).checked_type.dtype),
        )
        att_scores = _op.reshape(att_scores, (batch_size, num_heads, seq_len, seq_len))

        # build the attention mask
        att_mask = _op.cast(mask_index, score_dtype)
        att_mask = _op.expand_dims(att_mask, 1, num_newaxis=2)
        att_mask = _op.subtract(_op.const(1, dtype=score_dtype), att_mask)
        att_mask = _op.multiply(att_mask, _op.const(-10000, dtype=score_dtype))

        # apply the mask
        att_scores = _op.add(att_scores, att_mask)
        att_scores = _op.reshape(att_scores, (batch_size * num_heads, seq_len, seq_len))

        att_probs = _op.nn.softmax(att_scores, axis=-1)

        output = _op.nn.batch_matmul(att_probs, V, transpose_a=False, transpose_b=False)
        output = _op.reverse_reshape(output, (-1, num_heads, 0, 0))
        output = _op.transpose(output, axes=[0, 2, 1, 3])
        output = _op.reshape(output, (0, 0, out_hidden))

        return _expr.TupleWrapper(_expr.Tuple([output, present]), 2)


class Gemm(OnnxOpConverter):
    """Operator converter for Gemm."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 3 or len(inputs) == 2, "Gemm op take 2 or 3 inputs, {} given".format(
            len(inputs)
        )
        input0_state = infer_type(inputs[0])
        dtype = input0_state.checked_type.dtype
        # Y = alpha * A * B + beta * C
        alpha = float(attr.get("alpha", 1.0))
        beta = float(attr.get("beta", 1.0))
        transA = int(attr.get("transA", 0))
        transB = int(attr.get("transB", 0))
        # get number of channels
        channels = infer_channels(inputs[1], not transB)
        if transA:
            inputs[0] = _op.transpose(inputs[0], axes=(1, 0))
        if not transB:
            inputs[1] = _op.transpose(inputs[1], axes=(1, 0))
        if len(input0_state.checked_type.shape) != 2:
            inputs[0] = _op.nn.batch_flatten(inputs[0])
        if alpha != 1.0:
            inputs[0] *= _expr.const(alpha, dtype=dtype)
        out = _op.nn.dense(inputs[0], inputs[1], units=channels)
        if len(inputs) == 3:
            out = out + _expr.const(beta, dtype=dtype) * inputs[2]
        return out


class MatMul(OnnxOpConverter):
    """Operator converter for MatMul."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "MatMul op take 2 inputs, {} given".format(len(inputs))
        # Need to check input shape as batch matmul must be supported.
        return matmul_out_dtype(inputs, out_dtype=infer_type(inputs[0]).checked_type.dtype)


class MatMulInteger16(OnnxOpConverter):
    """Operator converter for MatMulInteger16 from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        assert len(inputs) == 2, "MatMulInteger16 op take 2 inputs, {} given".format(len(inputs))
        a_dtype = infer_type(inputs[0]).checked_type.dtype
        b_dtype = infer_type(inputs[1]).checked_type.dtype
        # Check input data types
        assert a_dtype in ("int16", "uint16"), "MatMulInteger16: invalid dtype for first input"
        assert b_dtype in ("int16", "uint16"), "MatMulInteger16: invalid dtype for second input"
        out_dtype = "int32"
        if a_dtype == "uint16" and b_dtype == "uint16":
            out_dtype = "uint32"
        return matmul_out_dtype(inputs, out_dtype)


class Mod(OnnxOpConverter):
    """Operator converter for Mod."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Mod op take 2 inputs, {} given".format(len(inputs))

        # Note: attr['fmod'] determines whether the operator should behave like np.fmod or np.mod.
        # attr['fmod'] == 0 will behave as np.mod and attr['fmod'] == 1 will force fmod treatment.
        # The relay equivalent of np.fmod is relay.mod and np.mod is relay.floor_mod
        if attr.get("fmod", 0) == 0:
            op_name = "floor_mod"
        else:
            op_name = "mod"

        return AttrCvt(op_name)(inputs, {}, params)


class MaxPool(Pool):
    """Operator converter for MaxPool"""

    name = "max_pool"


class MaxUnpool(OnnxOpConverter):
    """Operator converter for MaxUnpool"""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        # Unpack inputs and attributes
        data = inputs[0]
        data_type = infer_type(data).checked_type.dtype
        indices = inputs[1]
        output_shape = inputs[2]
        kernel_shape = attr.get("kernel_shape")
        pads = attr.get("pads", None)
        strides = attr.get("strides", [1] * len(kernel_shape))

        # Compute the proper output shape before padding.
        multiplier = _op.concatenate(
            [_expr.const([1, 1], dtype="int64"), _expr.const(list(strides), dtype="int64")], axis=0
        )
        total_output_shape = multiplier * shape_of(data, dtype="int64")
        # Add extra dimensions from kernel size and stride mismatch
        total_output_shape += _op.concatenate(
            [_expr.const([0, 0], "int64"), _expr.const(list(kernel_shape), "int64")], axis=0
        ) - _op.concatenate(
            [_expr.const([0, 0], "int64"), _expr.const(list(strides), "int64")], axis=0
        )

        # Compute padding amount if output shape is specified.
        if output_shape is not None:
            total_output_shape = output_shape

        elif pads is not None:
            # Get pads in the proper format for relay.
            pads = _op.concatenate(
                [_expr.const([0, 0, 0, 0], "int64"), _expr.const(list(pads), "int64")], axis=0
            )
            pads = _op.reshape(pads, [-1, 2])
            # Compute the total padding per axis.
            total_pad = _op.sum(pads, axis=-1)
            # Reversing maxpool means that padding actually makes our output smaller.
            total_output_shape = total_output_shape - total_pad

        # Create a tensor of zeros then scatter our data through it.
        zeros_tensor = _op.zeros(total_output_shape, data_type)
        # We need to flatten all our tensors before scattering.
        flat_tensor = _op.scatter(
            _op.reshape(zeros_tensor, [-1]),
            _op.reshape(indices, [-1]),
            _op.reshape(data, [-1]),
            axis=0,
        )
        # Now reshape back to prepadded shape.
        output_tensor = _op.reshape(flat_tensor, total_output_shape)

        return output_tensor


class LpPool(OnnxOpConverter):
    """A helper class for lppool op converters."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        dtype = infer_type(inputs[0]).checked_type.dtype
        data = inputs[0]
        input_shape = infer_shape(data)
        ndim = len(input_shape)
        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: LpPool does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                data = autopad(
                    data,
                    attr["strides"],
                    attr["kernel_shape"],
                    [1] * ndim,
                    mode=attr["auto_pad"],
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator {} is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"], "LpPool"))
            attr.pop("auto_pad")

        if "storage_order" in attr:
            attr["layout"] = onnx_storage_order2layout(
                attr["storage_order"], dims=(len(input_shape) - 2), op_name="LpPool"
            )
        else:
            attr["layout"] = onnx_default_layout(dims=(len(input_shape) - 2), op_name="LpPool")

        p_value = attr.get("p", 2)
        p = _expr.const(p_value, dtype)
        reci_p = _expr.const(1.0 / p_value, dtype)
        data = _op.power(data, p)

        out = AttrCvt(
            op_name=dimension_picker("avg_pool"),
            transforms={"kernel_shape": "pool_size", "pads": ("padding", 0)},
            extras={"count_include_pad": True},
            ignores=["p"],
            custom_check=dimension_constraint(),
        )([data], attr, params)
        kernels = attr["kernel_shape"]
        out = _op.abs(out) * _expr.const(np.prod(kernels).astype(dtype))
        return _op.power(out, reci_p)


class GlobalLpPool(OnnxOpConverter):
    """Operator converter for GlobalLpPool."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # TODO: GlobalLpPool does not yet support dynamic shapes
        in_shape = infer_shape(inputs[0])
        attr["kernel_shape"] = in_shape[2:]

        return LpPool._impl_v1(inputs, attr, params)


class Mul(Elemwise):
    """Operator converter for Multiply."""

    name = "multiply"


class Pad(OnnxOpConverter):
    """Operator converter for Pad."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        pad_width = []
        pads = attr.pop("paddings")
        dims = int(len(pads) / 2)
        for i in range(dims):
            pad_width.append((pads[i], pads[i + dims]))
        attr["pad_width"] = pad_width
        pad_mode = attr.get("mode", b"constant").decode("utf-8")
        if pad_mode in ["constant", "edge", "reflect"]:
            attr["pad_mode"] = pad_mode
            attr.pop("mode", None)
        else:
            raise tvm.error.OpAttributeInvalid(
                "Value " + pad_mode + ' in attribute "mode" is invalid for operator Pad.'
            )

        return AttrCvt(
            _op.nn.pad,
            transforms={
                "value": "pad_value",
            },
        )(inputs, attr, params)

    @classmethod
    def _impl_v2(cls, inputs, attr, params):
        pad_width = []
        pads = attr.pop("pads")
        dims = int(len(pads) / 2)
        for i in range(dims):
            pad_width.append((pads[i], pads[i + dims]))
        attr["pad_width"] = pad_width
        pad_mode = attr.get("mode", b"constant").decode("utf-8")
        if pad_mode in ["constant", "edge", "reflect"]:
            attr["pad_mode"] = pad_mode
            attr.pop("mode", None)
        else:
            raise tvm.error.OpAttributeInvalid(
                "Value " + pad_mode + ' in attribute "mode" is invalid for operator Pad.'
            )

        return AttrCvt(
            "pad",
            transforms={
                "value": "pad_value",
            },
        )(inputs, attr, params)

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        pads = inputs[1]
        if len(inputs) == 3:
            value = fold_constant(_op.take(inputs[2], _op.const(0)))
        else:
            value = 0.0

        pad_width_expr = fold_constant(_op.transpose(_op.reshape(pads, (2, -1))))
        pad_mode = attr.get("mode", b"constant").decode("utf-8")
        if not pad_mode in ["constant", "edge", "reflect"]:
            raise tvm.error.OpAttributeInvalid(
                "Value " + pad_mode + ' in attribute "mode" is invalid for operator Pad.'
            )

        return _op.nn.pad(inputs[0], pad_width_expr, value, pad_mode=pad_mode)


class ParametricSoftPlus(OnnxOpConverter):
    """Operator converter for ParametricSoftPlus."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = _expr.const(float(attr.get("alpha", 1.0)))
        beta = _expr.const(float(attr.get("beta", 1.0)))
        return _op.log(_op.exp(beta * inputs[0]) + _expr.const(1.0)) * alpha


class Pow(OnnxOpConverter):
    """Operator converter for Pow."""

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        x = inputs[0]
        y = inputs[1]

        x_type = infer_type(x).checked_type.dtype
        output_type = x_type
        y_type = infer_type(y).checked_type.dtype

        if not x_type.startswith("float"):
            x_type = "float32"
            x = _op.cast(x, x_type)

        if x_type != y_type:
            y = _op.cast(y, x_type)

        # TODO: come up with good default integer pow() func for common backends
        result = _op.power(x, y)
        if x_type != output_type:
            return _op.cast(result, output_type)
        return result


class Prelu(OnnxOpConverter):
    """Operator converter for Prelu."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "Prelu need 2 inputs, {} given".format(len(inputs))
        input_shape = shape_of(inputs[0])
        alpha = _op.broadcast_to_like(inputs[1], inputs[0])
        alpha = _op.reshape(alpha, [-1])
        output = _op.nn.prelu(_op.reshape(inputs[0], [-1]), alpha, axis=0)
        return _op.reshape(output, input_shape)


class Reciprocal(OnnxOpConverter):
    """Operator converter for Reciprocal."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        dtype = infer_type(inputs[0]).checked_type.dtype
        return _expr.const(1.0, dtype=dtype) / inputs[0]


class Flatten(OnnxOpConverter):
    """Operator converter for Flatten."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axis", 1)
        ishape = shape_of(inputs[0])
        ndim = infer_shape(ishape)[0]
        if axis < 0:
            axis = axis + ndim

        if axis == 1:
            out = _op.nn.batch_flatten(inputs[0])
        else:
            pre_shape = _op.prod(_op.strided_slice(ishape, [0], [axis], [1]), keepdims=True)
            post_shape = _op.prod(_op.strided_slice(ishape, [axis], [ndim], [1]), keepdims=True)
            newshape = fold_constant(_op.concatenate([pre_shape, post_shape], axis=0))
            out = _op.reshape(inputs[0], newshape)
        return out


class Reshape(OnnxOpConverter):
    """Operator converter for Reshape."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.reshape(inputs[0], attr["shape"])

    @classmethod
    def _impl_v5(cls, inputs, attr, params):
        allowzero = attr.get("allowzero", False)
        if get_name(inputs[1]) in params:
            shape = tuple(params[inputs[1].name_hint].numpy().astype("int32"))
            out = _op.reshape(inputs[0], shape, allowzero=allowzero)
        else:
            out = _op.reshape(*inputs, allowzero=allowzero)
        return out


class DepthToSpace(OnnxOpConverter):
    """Operator converter for DepthToSpace."""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        block_size = int(attr["blocksize"])
        mode = attr.get("mode", b"DCR").decode("utf-8")
        return _op.nn.depth_to_space(inputs[0], block_size, mode=mode)


class SpaceToDepth(OnnxOpConverter):
    """Operator converter for SpaceToDepth."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):

        block_size = int(attr["blocksize"])
        return _op.nn.space_to_depth(inputs[0], block_size)


class Concat(OnnxOpConverter):
    """Operator converter for Concat."""

    @classmethod
    def _impl_v1(cls, inputs, args, params):
        return AttrCvt(op_name="concatenate")((inputs,), args)


class Scale(OnnxOpConverter):
    """Operator converter for Scale."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        scale = float(attr.get("scale", 1.0))
        return inputs[0] * _expr.const(scale)


class Selu(OnnxOpConverter):
    """Operator converter for Selu."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get("alpha", 1.67326319217681884765625))
        gamma = float(attr.get("gamma", 1.05070102214813232421875))
        return _expr.const(gamma) * (
            _expr.const(-alpha) * _op.nn.relu(_expr.const(1.0) - _op.exp(inputs[0]))
            + _op.nn.relu(inputs[0])
        )


class ScaledTanh(OnnxOpConverter):
    """Operator converter for ScaledTanh."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get("alpha", 1.0))
        beta = float(attr.get("beta", 1.0))
        return _op.tanh(_expr.const(beta) * inputs[0]) * _expr.const(alpha)


class Shrink(OnnxOpConverter):
    """Operator converter for Shrink."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        x = inputs[0]
        dtype = infer_type(x).checked_type.dtype
        lambd = _op.const(attr.get("lambd", 0.5), dtype=dtype)
        bias = _op.const(attr.get("bias", 0.0), dtype=dtype)

        zeros = _op.zeros_like(x)
        return _op.where(x < -lambd, x + bias, zeros) + _op.where(x > lambd, x - bias, zeros)


class Softsign(OnnxOpConverter):
    """Operator converter for Softsign."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return inputs[0] / (_expr.const(1.0) + Absolute.get_converter(1)(inputs, attr, params))


class Sub(Elemwise):
    """Operator converter for Subtract."""

    name = "subtract"


class Sum(OnnxOpConverter):
    """Operator converter for Sum."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # Onnx Sum Operator
        for in_index in range(len(inputs) - 1):
            inputs[in_index + 1] = _op.add(inputs[in_index], inputs[in_index + 1])

        return inputs[len(inputs) - 1]


class Affine(OnnxOpConverter):
    """Operator converter for Affine transformation."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = _expr.const(attr.get("alpha", 1.0))
        beta = _expr.const(attr.get("beta", 0.0))
        return (alpha * inputs[0]) + beta


class ThresholdedRelu(OnnxOpConverter):
    """Operator converter for ThresholdedRelu."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = float(attr.get("alpha", 1.0))
        alpha_tensor = _op.full_like(inputs[0], fill_value=_expr.const(alpha))
        mask = _op.greater(inputs[0], alpha_tensor).astype("float32")
        return inputs[0] * mask


def _broadcast_constraint():
    def _broadcast_check(attrs):
        if attrs.get("axis", None):
            return False
        return True

    return _broadcast_check, "Specifying broadcast axis not allowed."


def _fully_connected(opset):
    def _impl(inputs, attr, params):
        # get number of channels
        channels = infer_channels(inputs[1], params)
        attr["units"] = channels
        return AttrCvt("dense", ignores=["axis", "axis_w"])(inputs, attr)

    return _impl


class Upsample(OnnxOpConverter):
    """Operator converter for Upsample (nearest mode)."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        scales = attr.get("scales")

        input_shape = infer_shape(inputs[0])
        dims = len(input_shape)

        if not scales:
            # Here we are going to higher OPSET version.
            assert len(inputs) == 2, "Upsample op takes 2 inputs, {} given".format(len(inputs))

            if get_name(inputs[1]) in params:
                scales = params[inputs[1].name_hint].numpy()
            else:
                scales = inputs[1]
        if isinstance(scales, _expr.Constant):
            scales = list(scales.data.numpy())
        if not isinstance(scales, _expr.Expr):
            assert scales[0] == 1.0 and scales[1] == 1.0

        mode = attr.get("mode")
        if mode == b"nearest":
            method = "nearest_neighbor"
        elif mode == b"linear":
            method = "trilinear" if dims == 5 else "bilinear"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "mode" of operator Upsample is not valid.'.format(mode)
            )

        # in 3d case, we use the purely static op
        if dims == 5:
            if isinstance(scales, _expr.Expr):
                scale_h = _op.take(scales, _op.const(3))
                scale_w = _op.take(scales, _op.const(4))
                scale_d = _op.take(scales, _op.const(1))
            else:
                assert len(scales) == 5
                scale_h = scales[-2]
                scale_w = scales[-1]
                scale_d = scales[-3]

            layout = "NCDHW"
            out = _op.nn.upsampling3d(
                inputs[0],
                scale_d,
                scale_h,
                scale_w,
                layout=layout,
                method=method,
                coordinate_transformation_mode="asymmetric",
            )
        # in 2d case, use dynamic op
        else:
            if isinstance(scales, _expr.Expr):
                scale_h = _op.take(scales, _op.const(3))
                scale_w = _op.take(scales, _op.const(4))
            else:
                assert len(scales) == 4
                scale_h = scales[-2]
                scale_w = scales[-1]
            layout = "NCHW"

            out = _op.nn.upsampling(
                inputs[0],
                scale_h,
                scale_w,
                layout=layout,
                method=method,
                align_corners=False,
            )
        return out


class Shape(OnnxOpConverter):
    """Operator converter for Shape."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return shape_of(inputs[0], "int64")

    @classmethod
    def _impl_v15(cls, inputs, attr, params):
        start = attr.get("start")
        end = attr.get("end")
        return shape_of(inputs[0], dtype="int64", start=start, end=end)


class CumSum(OnnxOpConverter):
    """Operator converter for CumSum."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        dim = inputs[1]

        if dim is not None:
            dim = int(infer_value(dim, params).numpy())

        exclusive = attr.get("exclusive", 0)
        reverse = attr.get("reverse", 0)

        if reverse != 0:
            out = _op.reverse(data, axis=dim)
            out = _op.cumsum(out, axis=dim, exclusive=exclusive)
            return _op.reverse(out, axis=dim)

        return _op.cumsum(data, axis=dim, exclusive=exclusive)


class Cast(OnnxOpConverter):
    """Operator converter for Cast."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return AttrCvt(op_name="cast", transforms={"to": "dtype"})(inputs, attr)

    @classmethod
    def _impl_v5(cls, inputs, attr, params):
        try:
            from onnx import TensorProto
        except ImportError as e:
            raise ImportError("Unable to import TensorProto from onnx {}".format(e))

        # If onnx mapping is used, bfloat16 gets converted to float16
        # which is not the desired behavior
        if attr["to"] == int(TensorProto.BFLOAT16):
            attr["to"] = "bfloat16"
        else:
            try:
                from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

                attr["to"] = str(TENSOR_TYPE_TO_NP_TYPE[attr["to"]])
            except ImportError as e:
                raise ImportError("Unable to import onnx.mapping which is required {}".format(e))
        return AttrCvt(op_name="cast", transforms={"to": "dtype"})(inputs, attr)


class Unsqueeze(OnnxOpConverter):
    """Operator converter for Unsqueeze."""

    @classmethod
    def run_calculation(cls, tensor, axes):
        axes = sorted(axes)
        for axis in axes:
            tensor = _op.expand_dims(tensor, axis=axis, num_newaxis=1)
        return tensor

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return cls.run_calculation(inputs[0], attr["axes"])

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        if isinstance(inputs[1], _expr.Constant):
            constant_axes = list(inputs[1].data.numpy())
            constant_axes = list(map(int, constant_axes))
            return cls.run_calculation(inputs[0], constant_axes)

        rank_input = len(infer_type(inputs[0]).checked_type.shape)
        num_new_axis = int(infer_type(inputs[1]).checked_type.shape[0])
        axes = relay.sort(inputs[1])
        axes = relay.split(axes, num_new_axis).astuple()
        result = inputs[0]

        # TODO (AndrewZhaoLuo): investigate performance issues with consecutive
        # dynamic expand_dims on non-llvm targets.
        for i in range(num_new_axis):
            axis = relay.TupleGetItem(axes, i)
            # Unpack scalar
            axis = relay.reshape(axis, [])
            axis = relay.where(
                axis >= relay.const(0, "int64"), axis, axis + relay.const(rank_input, "int64")
            )
            result = _op.expand_dims(result, axis)
        return result


class Squeeze(OnnxOpConverter):
    """Operator converter for Squeeze."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axes", None)
        return _op.squeeze(inputs[0], axis)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        ishape = infer_shape(inputs[0])
        axis = inputs[1]

        if axis is None:
            # If axes is not provided, all the single dimensions will be removed from the shape.
            if not ishape:  # scalar
                return inputs[0]

            axis = [i for i in range(len(ishape)) if ishape[i] == 1]
            axis = _op.const(axis)

        dtype = infer_type(axis).checked_type.dtype

        if isinstance(axis, _expr.Constant):
            constant_axes = list(axis.data.numpy())
            constant_axes = list(map(int, constant_axes))
            return _op.squeeze(inputs[0], constant_axes)

        rank = _op.shape_of(_op.shape_of(inputs[0], dtype), dtype)
        axis = _op.where(axis < _op.const(0, dtype), axis + rank, axis)
        return _op.squeeze(inputs[0], fold_constant(axis))


class Split(OnnxOpConverter):
    """Operator converter for Split."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        splits = attr.get("split", None)
        if splits is not None and len(splits) > 1:
            indices = []
            index = 0
            for i in splits[:-1]:
                index += i
                indices.append(index)
        # When splits isnt specified divide evenly over axis.
        else:
            indices = attr["tvm_custom"]["num_outputs"]
        output = _op.split(inputs[0], indices, attr.get("axis", 0))
        # If the output of split is a single value, unpack if from the TupleWrapper
        if len(output) == 1:
            output = output[0]
        return output

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        splits = inputs[1]
        splits_rank = None
        if splits is not None:
            splits_rank = len(infer_shape(splits))
        if splits is not None and splits_rank > 0:
            if isinstance(splits, _expr.Constant):
                splits = splits.data.asnumpy()
                indices = []
                index = 0
                for i in splits[:-1]:
                    index += i
                    indices.append(index)
            else:
                raise ValueError("Dynamic Split not yet supported")
        # When splits isnt specified divide evenly over axis.
        else:
            indices = attr["tvm_custom"]["num_outputs"]
        output = _op.split(inputs[0], indices, attr.get("axis", 0))
        # If the output of split is a single value, unpack if from the TupleWrapper
        if len(output) == 1:
            output = output[0]
        return output


class Slice(OnnxOpConverter):
    """Operator converter for Slice."""

    @classmethod
    def _common(cls, starts, ends, axes):
        N = max(axes) + 1
        new_axes = list(range(N))
        new_starts = [0] * N
        new_ends = [np.iinfo(np.int32).max] * N
        for i, axis in enumerate(axes):
            new_starts[axis] = starts[i]
            new_ends[axis] = ends[i]
        return new_starts, new_ends, new_axes

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if isinstance(attr["starts"], int):
            attr["starts"] = (attr["starts"],)
            attr["ends"] = (attr["ends"],)

        try:
            # Update the starts and ends according to axes if required.
            if isinstance(attr["axes"], int):
                attr["axes"] = (attr["axes"],)
            new_starts, new_ends, new_axes = cls._common(attr["starts"], attr["ends"], attr["axes"])
            attr["axes"] = new_axes
            attr["starts"] = new_starts
            attr["ends"] = new_ends
        except KeyError:
            pass
        begin = list(attr["starts"])
        end = list(attr["ends"])

        return _op.strided_slice(inputs[0], begin=begin, end=end)

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        starts = inputs[1]
        ends = inputs[2]
        axes = inputs[3]
        steps = inputs[4]

        ishape = infer_shape(inputs[0])
        data_rank = len(ishape)

        if axes is not None:
            # Normalize for negative axes
            axes_dtype = infer_type(axes).checked_type.dtype
            axes = fold_constant(
                _op.where(
                    axes < _op.const(0, axes_dtype), axes + _op.const(data_rank, axes_dtype), axes
                )
            )

        def has_static_axes():
            return (
                isinstance(axes, _expr.Constant)
                and isinstance(starts, _expr.Constant)
                and isinstance(ends, _expr.Constant)
                and (steps is None or isinstance(steps, _expr.Constant))
            )

        if axes is not None and has_static_axes():
            axes_np = axes.data.numpy().astype("int64")
            begin_np = starts.data.numpy().astype("int64")
            end_np = ends.data.numpy().astype("int64")
            if steps is None:
                strides_np = np.ones_like(begin_np).astype("int64")
            else:
                strides_np = steps.data.numpy().astype("int64")
            if all([isinstance(ishape[i], int) for i in axes_np]):
                return _op.strided_slice(
                    inputs[0], list(begin_np), list(end_np), list(strides_np), axes=list(axes_np)
                )

        # Update the starts and ends according to axes if required.
        if axes is not None:
            data_shape = shape_of(inputs[0], dtype=infer_type(ends).checked_type.dtype)
            starts = _op.scatter(
                _op.const([0] * data_rank, dtype=infer_type(starts).checked_type.dtype),
                axes,
                starts,
                axis=0,
            )
            ends = _op.scatter(data_shape, axes, ends, axis=0)
            if steps is not None:
                steps = _op.scatter(
                    _op.const([1] * data_rank, dtype=infer_type(steps).checked_type.dtype),
                    axes,
                    steps,
                    axis=0,
                )

        if steps is None:
            steps = _op.const([1] * data_rank, dtype=infer_type(starts).checked_type.dtype)

        return _op.strided_slice(
            inputs[0], fold_constant(starts), fold_constant(ends), fold_constant(steps)
        )


def normalize_gather_indices(data, indices, axis):
    """Make sure gather indicies aren't negative"""
    ind_dtype = infer_type(indices).checked_type.dtype
    # Normalize the indices to a positive range
    s = _op.take(_op.shape_of(data, dtype=ind_dtype), _op.const(axis, dtype="int64"))
    cond = fold_constant(indices < _op.const(0, ind_dtype))
    if isinstance(cond, _expr.Constant):
        val = cond.data.numpy()
        if val.size == 1:
            cond = val.item()
            if cond:
                indices = indices + s
            return indices
    indices = _op.where(cond, indices + s, indices)
    return indices


class Gather(OnnxOpConverter):
    """Operator converter for Gather."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axis", 0)
        data = inputs[0]
        indices = inputs[1]
        indices = normalize_gather_indices(data, indices, axis)
        return _op.take(data, indices, axis)


class GatherElements(OnnxOpConverter):
    """Operator converter for GatherElements."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        indices = inputs[1]
        axis = attr.get("axis", 0)
        indices = normalize_gather_indices(data, indices, axis)
        return _op.gather(data, axis, indices)


class GatherND(OnnxOpConverter):
    """Operator converter for GatherND."""

    @classmethod
    def _impl_common(cls, data, indices, batch_dims=0):
        indices_dims = len(infer_shape(indices))
        indices_shape = infer_shape(indices)
        indices = _op.transpose(indices, axes=[-1] + list(range(indices_dims - 1)))
        index_rank = indices_shape[-1]
        return _op.gather_nd(
            data,
            indices,
            batch_dims=batch_dims,
            index_rank=index_rank,
        )

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return cls._impl_common(inputs[0], inputs[1])

    @classmethod
    def _impl_v12(cls, inputs, attr, params):
        batch_dims = attr.get("batch_dims", 0)
        return cls._impl_common(inputs[0], inputs[1], batch_dims)


class Compress(OnnxOpConverter):
    """Operator converter for compress"""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        input_tensor, condition_tensor = inputs

        axis = attr.get("axis", None)

        # Change one hot tensor to indices e.g. [0, 1, 1, 0, 1] -> [1, 2, 4]
        condition_tensor = _op.reshape(_op.argwhere(condition_tensor), (-1,))

        if axis is not None:
            return _op.take(input_tensor, condition_tensor, axis=axis)

        # if axis is None, flatten input tensor before selection
        input_tensor = _op.reshape(input_tensor, (-1,))
        return _op.take(input_tensor, condition_tensor, axis=0)


class Scatter(OnnxOpConverter):
    """Operator converter for Scatter."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axis", 0)
        return _op.scatter(inputs[0], inputs[1], inputs[2], axis)


class ScatterND(OnnxOpConverter):
    """Operator converter for ScatterND."""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        indices_dim = len(infer_shape(inputs[1]))
        axes = list(range(indices_dim))
        return _op.scatter_nd(
            inputs[0], _op.transpose(inputs[1], axes[-1:] + axes[:-1]), inputs[2], "update"
        )


class EyeLike(OnnxOpConverter):
    """Operator converter for EyeLike."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        in_checked_type = infer_type(inputs[0]).checked_type
        in_dtype = in_checked_type.dtype
        in_shape = list(get_const_tuple(in_checked_type.shape))
        dtype = attr.get("dtype", None)
        if dtype is None:
            dtype = in_dtype
        else:
            dtype = get_type(dtype)
        zeros = _op.zeros(in_shape, dtype)
        dim = in_shape[0]
        indices = _op.arange(_op.const(0), _op.const(dim), dtype="int32")
        ones = _op.full(_op.const(1), (dim,), dtype=dtype)
        k = _op.const(attr.get("k", 0), dtype="int32")
        return _op.scatter_nd(zeros, _op.stack([indices, indices + k], axis=0), ones, "update")


class LRN(OnnxOpConverter):
    """Operator converter for Local Response Normalization."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        """LRN support only NCHW format
        https://github.com/onnx/onnx/blob/main/docs/Operators.md#LRN
        """
        axis = 1
        alpha = attr.get("alpha", 0.0001)
        beta = attr.get("beta", 0.75)
        bias = attr.get("bias", 1.0)
        nsize = attr.get("size")
        attr = {"size": nsize, "axis": axis, "alpha": alpha, "beta": beta, "bias": bias}
        return AttrCvt("lrn")(inputs, attr)


class Maximum(OnnxOpConverter):
    """Operator converter for Maximum."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) == 1:
            return inputs[0]
        _max = inputs[0]
        for i in range(1, len(inputs)):
            _max = AttrCvt("maximum")([_max, inputs[i]], {})
        return _max


class Minimum(OnnxOpConverter):
    """Operator converter for Minimum."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) == 1:
            return inputs[0]
        _min = inputs[0]
        for i in range(1, len(inputs)):
            _min = AttrCvt("minimum")([_min, inputs[i]], {})
        return _min


class Mean(OnnxOpConverter):
    """Operator converter for Mean."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) == 1:
            return inputs[0]
        # avoid overflow
        concat = _op.concatenate([_op.expand_dims(x, axis=0) for x in inputs], axis=0)
        return _op.mean(concat, axis=0, keepdims=False)


class HardSigmoid(OnnxOpConverter):
    """Operator converter for HardSigmoid."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = attr.get("alpha", 0.2)
        beta = attr.get("beta", 0.5)
        transformX = (inputs[0] * _expr.const(alpha)) + _expr.const(beta)
        attr = {"a_min": 0, "a_max": 1}
        return AttrCvt("clip")([transformX], attr)


class HardSwish(OnnxOpConverter):
    """Operator converter for HardSwish."""

    @classmethod
    def _impl_v14(cls, inputs, attr, params):
        alpha = attr.get("alpha", 1 / 6)
        beta = attr.get("beta", 0.5)
        transformX = inputs[0] * _expr.const(alpha) + _expr.const(beta)
        attr = {"a_min": 0, "a_max": 1}
        return inputs[0] * AttrCvt("clip")([transformX], attr)


class Reduce(OnnxOpConverter):
    """Operator converter for reduce ops."""

    name = ""

    @classmethod
    def run_calculation(cls, inputs, axis, keepdims):
        attr = {"axis": axis, "keepdims": keepdims}
        return AttrCvt(cls.name)(inputs, attr)

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not infer_shape(inputs[0]):  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)

        if "axes" in attr:
            axis = attr.get("axes", 0)
        else:
            axis_len = len(infer_shape(inputs[0]))
            axis = list(range(axis_len))

        return cls.run_calculation(inputs, axis, attr.get("keepdims", True))

    @classmethod
    def _impl_v12(cls, inputs, attr, params):
        if not infer_shape(inputs[0]):  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)

        if len(inputs) == 2:
            if isinstance(inputs[1], _expr.Constant):
                # Get axis and unpack scalar
                constant_axis = int(inputs[1].data.numpy()[0])
                return cls.run_calculation([inputs[0]], constant_axis, attr.get("keepdims", True))

            raise ValueError("Dynamic Reduce is not supported yet!")

        return cls._impl_v1(inputs, attr, params)


class ReduceMax(Reduce):
    """Operator converter for ReduceMax."""

    name = "max"


class ReduceMin(Reduce):
    """Operator converter for ReduceMin."""

    name = "min"


class ReduceSum(Reduce):
    """Operator converter for ReduceSum."""

    name = "sum"


class ReduceMean(Reduce):
    """Operator converter for ReduceMean."""

    name = "mean"


class ReduceProd(Reduce):
    """Operator converter for ReduceProd."""

    name = "prod"


class ReduceLogSumExp(Reduce):
    """Operator converter for ReduceLogSumExp."""

    name = "logsumexp"


class ReduceSumSquare(OnnxOpConverter):
    """Operator converter for ReduceSumSquare."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not infer_shape(inputs[0]):  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)

        if "axes" in attr:
            axis = attr.get("axes", 0)
        else:
            axis_len = len(infer_shape(inputs[0]))
            axis = list(range(axis_len))
        attr = {"axis": axis, "keepdims": attr.get("keepdims", True)}
        inputs[0] = inputs[0] * inputs[0]

        return AttrCvt("sum")(inputs, attr)


class ReduceL1(OnnxOpConverter):
    """Operator converter for ReduceL1."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not infer_shape(inputs[0]):  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)

        if "axes" in attr:
            axis = attr.get("axes", 0)
        else:
            axis_len = len(infer_shape(inputs[0]))
            axis = list(range(axis_len))
        attr = {"axis": axis, "keepdims": attr.get("keepdims", True)}
        inputs[0] = _op.abs(inputs[0])

        return AttrCvt("sum")(inputs, attr)


class ReduceL2(OnnxOpConverter):
    """Operator converter for ReduceL2."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not infer_shape(inputs[0]):  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)

        if "axes" in attr:
            axis = attr.get("axes", 0)
        else:
            axis_len = len(infer_shape(inputs[0]))
            axis = list(range(axis_len))
        attr = {"axis": axis, "keepdims": attr.get("keepdims", True)}
        inputs[0] = inputs[0] * inputs[0]
        out = AttrCvt("sum")(inputs, attr)

        return _op.sqrt(out)


class ReduceLogSum(OnnxOpConverter):
    """Operator converter for ReduceLogSum."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if not infer_shape(inputs[0]):  # promote scalar to 1-D tensor
            inputs[0] = _op.expand_dims(inputs[0], axis=0)

        if "axes" in attr:
            axis = attr.get("axes", 0)
        else:
            axis_len = len(infer_shape(inputs[0]))
            axis = list(range(axis_len))
        attr = {"axis": axis, "keepdims": attr.get("keepdims", True)}
        out = AttrCvt("sum")(inputs, attr)

        return _op.log(out)


class ArgMax(OnnxOpConverter):
    """Operator converter for ArgMax."""

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        axis = attr.get("axis", 0)
        keepdims = attr.get("keepdims", True)
        select_last_index = attr.get("select_last_index", False)
        attr = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}
        return _op.cast(AttrCvt("argmax")(inputs, attr), "int64")


class ArgMin(OnnxOpConverter):
    """Operator converter for ArgMin."""

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        axis = attr.get("axis", 0)
        keepdims = attr.get("keepdims", True)
        select_last_index = attr.get("select_last_index", False)
        attr = {"axis": axis, "keepdims": keepdims, "select_last_index": select_last_index}
        return _op.cast(AttrCvt("argmin")(inputs, attr), "int64")


class Softmax(OnnxOpConverter):
    """Operator converter for Softmax."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axis", 1)
        in_shape = infer_shape(inputs[0])
        ndim = len(in_shape)
        if axis < 0:
            axis += ndim
        if axis == 0:
            reshape_shape = [-1]
        else:
            axis_val = [in_shape[i] for i in range(axis)]
            reshape_shape = [np.prod(axis_val)] + [-1]
        data_reshape = _op.reshape(inputs[0], newshape=reshape_shape)
        out = _op.nn.softmax(data_reshape, axis=-1)
        out = _op.reshape(out, newshape=in_shape)
        return out

    @classmethod
    def _impl_v13(cls, inputs, attr, _):
        axis = attr.get("axis", -1)
        ndim = len(infer_shape(inputs[0]))
        if axis < 0:
            axis += ndim
        return _op.nn.softmax(inputs[0], axis=axis)


class LogSoftmax(OnnxOpConverter):
    """Operator converter for Softmax."""

    @classmethod
    def run_calculation(cls, inputs, attr, params, opset):
        """Run the calculation for Log Softmax calculation."""
        res = Softmax.get_converter(opset)(inputs, attr, params)
        return _op.log(res)

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return cls.run_calculation(inputs, attr, params, opset=1)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        return cls.run_calculation(inputs, attr, params, opset=13)


class Hardmax(OnnxOpConverter):
    """Operator converter for Hardmax."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        axis = attr.get("axis", 1)
        ndim = len(infer_shape(inputs[0]))
        if axis < 0:
            axis += ndim
        dtype = infer_type(inputs[0]).checked_type.dtype

        if axis == 0:
            pre = _op.const([1], "int64")
        else:
            pre = _op.prod(
                _op.strided_slice(shape_of(inputs[0]), [0], [axis], [1]), axis=0, keepdims=True
            )
        post = _op.prod(
            _op.strided_slice(shape_of(inputs[0]), [axis], [2147483647], [1]), axis=0, keepdims=True
        )
        newshape = _op.concatenate([pre, post], axis=0)
        x = _op.reshape(inputs[0], fold_constant(newshape))
        argmax = _op.argmax(x, axis=1)
        onehot = _op.one_hot(
            argmax,
            _op.const(1.0, dtype),
            _op.const(0.0, dtype),
            fold_constant(_op.take(shape_of(x), _op.const([1], "int64"))),
            1,
            dtype,
        )
        return _op.reshape(onehot, shape_of(inputs[0]))

    @classmethod
    def _impl_v13(cls, inputs, attr, params) -> relay.Expr:
        inferred_type = infer_type(inputs[0])
        dtype = inferred_type.checked_type.dtype
        ndim = len(inferred_type.checked_type.shape)
        axis = attr.get("axis", -1) % ndim

        argmax = _op.argmax(inputs[0], axis=axis)
        return _op.one_hot(
            argmax,
            _op.const(1.0, dtype),
            _op.const(0.0, dtype),
            fold_constant(_op.take(shape_of(inputs[0]), _op.const([axis], "int64"))),
            axis,
            dtype,
        )


class OneHot(OnnxOpConverter):
    """Operator converter for OneHot."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        # Extract relay one_hot inputs.
        indices, depth, values = inputs
        ndim = len(infer_shape(indices))
        # Split onnx on off values into two separate expressions.
        off_value, on_value = _op.take(values, _op.const(0)), _op.take(values, _op.const(1))
        # Extract the datatype of the output from on_value.
        dtype = infer_type(on_value).checked_type.dtype
        ind_dtype = infer_type(indices).checked_type.dtype
        # Normalize the indices to a positive range
        indices = _op.where(
            indices < _op.const(0, ind_dtype), indices + _op.cast(depth, ind_dtype), indices
        )
        # set default value when axis is not set in the model
        if "axis" not in attr:
            attr["axis"] = -1
        axis = attr["axis"]
        if axis < 0:
            axis += ndim + 1

        return _op.one_hot(indices, on_value, off_value, depth, axis, dtype=dtype)


class ConstantOfShape(OnnxOpConverter):
    """Operator converter for ConstantOfShape."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        if "value" in attr:
            np_value = get_numpy(attr.pop("value"))[0]
            value = _expr.const(np_value)
            dtype = np_value.dtype.name
        else:
            value = _expr.const(0)
            dtype = "float32"
        output = _op.full(value, inputs[0], dtype=dtype)
        return output


class Constant(OnnxOpConverter):
    """Operator converter for ConstantOfShape."""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        if "value" not in attr:
            raise tvm.errors.OpAttributeRequired("no value in Constant")
        value = attr.pop("value")
        # Constants may rarely have string types. These are likely exported
        # from other frameworks and not actually used in TVM. We'll just use
        # a zero valued constant for compatibility.
        if isinstance(value, bytes):
            np_value = np.asarray([0]).astype("int64")
        else:
            np_value = get_numpy(value)
        dtype = np_value.dtype.name
        value = _expr.const(np_value, dtype)
        return value


class Sign(OnnxOpConverter):
    """Operator converter for Sign."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.sign(inputs[0])


class Equal(Elemwise):
    """Operator converter for Equal."""

    name = "equal"


class Not(Elemwise):
    """Operator converter for Not."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.logical_not(inputs[0])


class And(Elemwise):
    """Operator converter for And."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.logical_and(inputs[0], inputs[1])


class Tile(Elemwise):
    """Operator converter for Tile"""

    @classmethod
    def _impl_v6(cls, inputs, attr, params):
        return _op.tile(inputs[0], inputs[1])


class Erf(OnnxOpConverter):
    """Operator converter for Erf"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        return _op.erf(inputs[0])


class Where(OnnxOpConverter):
    """Operator converter for Where"""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        return _op.where(*inputs)


class Or(Elemwise):
    """Operator converter for Or."""

    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        return _op.logical_or(inputs[0], inputs[1])


class Expand(OnnxOpConverter):
    """Operator converter for Expand."""

    @classmethod
    def _impl_v8(cls, inputs, attr, params):
        dtype = infer_type(inputs[1]).checked_type.dtype
        in_shape = shape_of(inputs[0], dtype=dtype)
        shape = inputs[1]

        # Currently 'op.broadcast_to' expect the rank of the given 'shape'
        # (the 2nd input) is always higher than that of the given 'input' (the 1st input)
        # However, ONNX Expand supports multi-directional broadcasting, which allows
        # above pattern and also some extent of 'shape' can be smaller than the corresponding
        # extent of 'input'. In this case, the extent of 'shape' must be 1.
        # https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
        # In above cases, we cannot directorly apply 'op.broadcast_to' instead of 'expand'
        # so, here we solved this problem by expanding the given 'shape' itself.
        def expand_shape(in_shape, shape):
            """A function expands the shape when the rank is lower than that of the given
            intput. Also it replaces the extent of the shape with the corresponding extent
            of the intput when it is 1.
            """
            in_dims = infer_shape(in_shape)[0]
            new_dims = infer_shape(shape)[0]

            if in_dims < new_dims:
                in_shape = _op.concatenate(
                    [
                        _expr.const(
                            [
                                1,
                            ]
                            * (new_dims - in_dims),
                            dtype=dtype,
                        ),
                        in_shape,
                    ],
                    axis=0,
                )
            elif new_dims < in_dims:
                shape = _op.concatenate(
                    [
                        _expr.const(
                            [
                                1,
                            ]
                            * (in_dims - new_dims),
                            dtype=dtype,
                        ),
                        shape,
                    ],
                    axis=0,
                )
            new_shape = _op.maximum(in_shape, shape)
            return new_shape

        shape = fold_constant(expand_shape(in_shape, shape))
        return _op.broadcast_to(inputs[0], shape=shape)


class RNN(OnnxOpConverter):
    """Operator converter for RNNs such as LSTM and GRU."""

    @classmethod
    def _activation_helper(cls, activation, alpha, beta):
        convert_map = _get_convert_map(1)
        attrs = {}
        if alpha is not None:
            attrs["alpha"] = alpha
        if beta is not None:
            attrs["beta"] = beta
        return lambda x: convert_map[activation.decode("utf-8")]([x], attrs, {})

    @classmethod
    def _activation_needs_alpha(cls, activation):
        needs_alpha = [
            "Affine",
            "LeakyRelu",
            "ThresholdedRelu",
            "ScaledTanh",
            "HardSigmoid",
            "Elu",
        ]
        return activation.decode("utf-8") in needs_alpha

    @classmethod
    def _activation_needs_beta(cls, activation):
        needs_beta = [
            "Affine",
            "ScaledTanh",
            "HardSigmoid",
        ]
        return activation.decode("utf-8") in needs_beta


class LSTM(RNN):
    """Operator converter for LSTM"""

    @classmethod
    def bidir_lstm_cell(
        cls,
        input_seqs,
        weight_dicts,
        acts,
    ):
        """
        Bidirectional LSTM cell
        """
        seq_len = len(input_seqs)
        forward_outputs, fw_H_t, fw_C_t = lstm_cell(
            input_seqs,
            **weight_dicts[0],
            f_act=acts[0],
            g_act=acts[1],
            h_act=acts[2],
        )

        reverse_outputs, rev_H_t, rev_C_t = lstm_cell(
            input_seqs,
            **weight_dicts[1],
            f_act=acts[3],
            g_act=acts[4],
            h_act=acts[5],
            backwards=True,
        )

        final_outputs = []
        for i in range(seq_len):
            final_outputs.append(
                _op.stack([forward_outputs[i], reverse_outputs[seq_len - 1 - i]], axis=0)
            )

        return (
            _op.stack(final_outputs, axis=0),
            _op.stack([fw_H_t, rev_H_t], axis=0),
            _op.stack([fw_C_t, rev_C_t], axis=0),
        )

    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        # Unpack inputs, note that if optional and not provided then value will be None.
        X = inputs[0]
        Wp = inputs[1]
        Rp = inputs[2]
        Bp = inputs[3]
        # Sequence length currently unused as it can be inferred from shapes.
        # sequence_lens = inputs['sequence_lens']
        Hp_0 = inputs[5]
        Cp_0 = inputs[6]
        Pp = inputs[7]

        num_directions = infer_shape(Wp)[0]
        W_dtype = infer_type(Wp).checked_type.dtype

        if num_directions not in [1, 2]:
            raise ValueError("num_directions must be either 1 or 2!")

        X_shape = infer_shape(X)
        hidden_size = infer_shape(Rp)[-1]
        batch_size = X_shape[1]

        # Initialize state if not provided.
        # Otherwise remove bidirectional axis.
        if Hp_0 is None:
            Hp_0 = _op.zeros((num_directions, batch_size, hidden_size), W_dtype)
        if Cp_0 is None:
            Cp_0 = _op.zeros((num_directions, batch_size, hidden_size), W_dtype)

        if "activations" in attr:
            activations = attr["activations"]
            if len(activations) != 3 * num_directions:
                raise NotImplementedError(
                    f"LSTM assumes 3 * num_directions activation functions are provided"
                )
            alpha_loc = 0
            alphas = attr.get("activation_alpha", [])
            if isinstance(alphas, float):
                alphas = [alphas]
            beta_loc = 0
            betas = attr.get("activation_beta", [])
            if isinstance(betas, float):
                betas = [betas]
            acts = []
            for i in range(3 * num_directions):
                alpha = None
                beta = None
                activation = activations[i]
                if cls._activation_needs_alpha(activation) and len(alphas) > alpha_loc:
                    alpha = alphas[alpha_loc]
                    alpha_loc += 1
                if cls._activation_needs_beta(activation) and len(betas) > beta_loc:
                    beta = betas[beta_loc]
                    beta_loc += 1
                acts.append(cls._activation_helper(activation, alpha, beta))
        else:
            acts = [_op.sigmoid, _op.tanh, _op.tanh] * num_directions

        # TODO (vvchernov): It can be replaced by _op.split if issue #8412 is resolved
        X_steps = unbind(X, axis=0)

        H_ts = _op.split(Hp_0, num_directions)
        C_ts = _op.split(Cp_0, num_directions)
        Ws = _op.split(Wp, num_directions)
        Rs = _op.split(Rp, num_directions)

        if Bp is not None:
            Bs = _op.split(Bp, num_directions)
        if Pp is not None:
            p_i, p_o, p_f = _op.split(Pp, 3, axis=1)

            p_is = _op.split(p_i, num_directions)
            p_fs = _op.split(p_f, num_directions)
            p_os = _op.split(p_o, num_directions)

        weights_dicts = []
        for i in range(num_directions):
            weights_dict = {}

            weights_dict["hidden_state"] = _op.squeeze(H_ts[i], axis=[0])
            weights_dict["cell_state"] = _op.squeeze(C_ts[i], axis=[0])

            # Weights permutation: onnx format i-o-f-c, lstm cell format i-f-c-o
            mati, mato, matf, matc = _op.split(_op.squeeze(Ws[i], axis=[0]), 4)
            weights_dict["w_inp"] = _op.concatenate([mati, matf, matc, mato], axis=0)
            mati, mato, matf, matc = _op.split(_op.squeeze(Rs[i], axis=[0]), 4)
            weights_dict["w_hid"] = _op.concatenate([mati, matf, matc, mato], axis=0)
            if Bp is not None:
                Bi, Bh = _op.split(Bs[i], 2, -1)
                mati, mato, matf, matc = _op.split(_op.squeeze(Bi, axis=[0]), 4)
                weights_dict["b_inp"] = _op.concatenate([mati, matf, matc, mato], axis=0)
                mati, mato, matf, matc = _op.split(_op.squeeze(Bh, axis=[0]), 4)
                weights_dict["b_hid"] = _op.concatenate([mati, matf, matc, mato], axis=0)
            if Pp is not None:
                weights_dict["p_i"] = _op.squeeze(p_is[i], axis=[0])
                weights_dict["p_f"] = _op.squeeze(p_fs[i], axis=[0])
                weights_dict["p_o"] = _op.squeeze(p_os[i], axis=[0])
            weights_dicts.append(weights_dict)

        if num_directions == 2:
            output, H, C = LSTM.bidir_lstm_cell(
                input_seqs=X_steps,
                weight_dicts=weights_dicts,
                acts=acts,
            )
        else:
            # outputs shape = [seqs_num, (batch_size, hidden_size)]
            outputs, H, C = lstm_cell(
                input_seqs=X_steps,
                **weights_dicts[0],
                f_act=acts[0],
                g_act=acts[1],
                h_act=acts[2],
            )

            # output shape = (seqs_num, num_directions, batch_size, hidden_size)
            output = _op.expand_dims(_op.stack(outputs, axis=0), axis=1)
            H = _op.expand_dims(H, axis=0)
            C = _op.expand_dims(C, axis=0)

        return _expr.TupleWrapper(_expr.Tuple((output, H, C)), 3)


class GRU(RNN):
    """Operator convert for GRU"""

    @classmethod
    def bidir_gru_cell(
        cls,
        input_seqs,
        weight_dicts,
        acts,
    ):
        """
        Bidirectional GRU cell
        """
        seq_len = len(input_seqs)
        forward_outputs, fw_H_t = gru_cell(
            input_seqs,
            **weight_dicts[0],
            rz_act=acts[0],
            n_act=acts[1],
        )

        reverse_outputs, rev_H_t = gru_cell(
            input_seqs,
            **weight_dicts[1],
            rz_act=acts[2],
            n_act=acts[3],
            backwards=True,
        )

        final_outputs = []
        for i in range(seq_len):
            final_outputs.append(
                _op.stack([forward_outputs[i], reverse_outputs[seq_len - 1 - i]], axis=0)
            )

        return (
            _op.stack(final_outputs, axis=0),
            _op.stack([fw_H_t, rev_H_t], axis=0),
        )

    @classmethod
    def _impl_v7(cls, inputs, attr, params):
        # Unpack inputs, note that if optional and not provided then value will be None.
        X = inputs[0]
        Wp = inputs[1]
        Rp = inputs[2]
        Bp = inputs[3]
        # Sequence length currently unused as it can be inferred from shapes.
        # sequence_lens = inputs['sequence_lens']
        Hp_0 = inputs[5]
        linear_before_reset = attr.get("linear_before_reset", 0)

        num_directions = infer_shape(Wp)[0]
        W_dtype = infer_type(Wp).checked_type.dtype

        if num_directions not in [1, 2]:
            raise ValueError("num_directions must be either 1 or 2!")

        X_shape = infer_shape(X)
        hidden_size = infer_shape(Rp)[-1]
        batch_size = X_shape[1]

        if Hp_0 is None:
            Hp_0 = _op.zeros((num_directions, batch_size, hidden_size), W_dtype)

        if "activations" in attr:
            activations = attr["activations"]
            if len(activations) != 2 * num_directions:
                raise NotImplementedError(
                    "GRU assumes 2 * num_directions activation functions are provided"
                )
            alpha_loc = 0
            alphas = attr.get("activation_alpha", [])
            if isinstance(alphas, float):
                alphas = [alphas]
            beta_loc = 0
            betas = attr.get("activation_beta", [])
            if isinstance(betas, float):
                betas = [betas]
            acts = []
            for i in range(2 * num_directions):
                alpha = None
                beta = None
                activation = activations[i]
                if cls._activation_needs_alpha(activation) and len(alphas) > alpha_loc:
                    alpha = alphas[alpha_loc]
                    alpha_loc += 1
                if cls._activation_needs_beta(activation) and len(betas) > beta_loc:
                    beta = betas[beta_loc]
                    beta_loc += 1
                acts.append(cls._activation_helper(activation, alpha, beta))
        else:
            acts = [_op.sigmoid, _op.tanh] * 2

        # TODO (vvchernov): It can be replaced by _op.split if issue #8412 is resolved
        X_steps = unbind(X, axis=0)

        H_ts = _op.split(Hp_0, num_directions)
        Ws = _op.split(Wp, num_directions)
        Rs = _op.split(Rp, num_directions)

        if Bp is not None:
            Bs = _op.split(Bp, num_directions)

        weights_dicts = []
        for i in range(num_directions):
            weights_dict = {}

            weights_dict["hidden_state"] = _op.squeeze(H_ts[i], axis=[0])
            weights_dict["linear_before_reset"] = linear_before_reset

            # Weights permutation: onnx format i-o-f-c, lstm cell format i-f-c-o
            matz, matr, matn = _op.split(_op.squeeze(Ws[i], axis=[0]), 3)
            weights_dict["w_inp"] = _op.concatenate([matr, matz, matn], axis=0)
            matz, matr, matn = _op.split(_op.squeeze(Rs[i], axis=[0]), 3)
            weights_dict["w_hid"] = _op.concatenate([matr, matz, matn], axis=0)
            if Bp is not None:
                Bi, Bh = _op.split(Bs[i], 2, -1)
                matz, matr, matn = _op.split(_op.squeeze(Bi, axis=[0]), 3)
                weights_dict["b_inp"] = _op.concatenate([matr, matz, matn], axis=0)
                matz, matr, matn = _op.split(_op.squeeze(Bh, axis=[0]), 3)
                weights_dict["b_hid"] = _op.concatenate([matr, matz, matn], axis=0)
            weights_dicts.append(weights_dict)

        if num_directions == 2:
            output, H = GRU.bidir_gru_cell(
                input_seqs=X_steps,
                weight_dicts=weights_dicts,
                acts=acts,
            )
        else:
            # outputs shape = [seqs_num, (batch_size, hidden_size)]
            outputs, H = gru_cell(
                input_seqs=X_steps,
                **weights_dicts[0],
                rz_act=acts[0],
                n_act=acts[1],
            )

            # output shape = (seqs_num, num_directions, batch_size, hidden_size)
            output = _op.expand_dims(_op.stack(outputs, axis=0), axis=1)
            H = _op.expand_dims(H, axis=0)

        return _expr.TupleWrapper(_expr.Tuple((output, H)), 2)


class Resize(OnnxOpConverter):
    """Operator converter for Resize"""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        mode = attr.get("mode").decode("ascii")
        if mode == "nearest":
            method = "nearest_neighbor"
        elif mode == "linear":
            method = "linear"
        elif mode == "cubic":
            method = "cubic"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "mode" of operator Resize is not valid.'.format(mode)
            )

        scale = inputs[1]
        size = _op.cast(shape_of(inputs[0]), infer_type(scale).checked_type.dtype) * scale
        ndims = len(infer_shape(inputs[0]))
        out = None
        if ndims == 3:
            out_size = fold_constant(_op.strided_slice(size, [2], [3]))
            out = _op.image.resize1d(inputs[0], out_size, None, "NCW", method, "asymmetric")
        elif ndims == 4:
            out_size = fold_constant(_op.strided_slice(size, [2], [4]))
            out = _op.image.resize2d(inputs[0], out_size, None, "NCHW", method, "asymmetric")
        elif ndims == 5:
            out_size = fold_constant(_op.strided_slice(size, [2], [5]))
            out = _op.image.resize3d(inputs[0], out_size, None, "NCDHW", method, "asymmetric")
        else:
            raise NotImplementedError("Resize only supports 3, 4, or 5 dims")
        return out

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        scale = inputs[2]
        scale_shape = infer_shape(scale)
        if len(inputs) == 4:
            assert (
                len(scale_shape) == 0 or scale_shape[0] == 0
            ), "One of scale or size should be passed, not both."
            size = inputs[3]
        else:
            assert len(scale_shape) != 0, "One of scale or size should be passed."
            size = _op.cast(shape_of(inputs[0]), infer_type(scale).checked_type.dtype) * scale
        return cls.v11_13_common(inputs, size, attr, params)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        scale = inputs[2]
        size = inputs[3]

        # Some versions of onnx exporters produce an opset 13 model with the opset 11
        # resize op, handle that edge case
        if scale is not None and size is not None:
            return cls._impl_v11(inputs, attr, params)

        if size is not None:
            assert scale is None, "One of scale or size should be passed, not both."
        else:
            scale_type = infer_type(scale)
            scale_shape = scale_type.checked_type.shape
            scale_dtype = scale_type.checked_type.dtype
            assert len(scale_shape) != 0, "One of scale or size should be passed."
            size = _op.cast(shape_of(inputs[0]), scale_dtype) * scale

        return cls.v11_13_common(inputs, size, attr, params)

    @classmethod
    def v11_13_common(cls, inputs, size, attr, params):
        """
        Resize v11 and Resize v13 are identical except in how
        they handle the passing of scale and size. This utility
        provides the implementation for both
        """
        roi = inputs[1]
        if roi is not None and infer_shape(roi)[0] == 0:
            roi = None
        ndims = len(infer_shape(inputs[0]))
        mode = attr.get("mode").decode("ascii")
        if mode == "nearest":
            method = "nearest_neighbor"
        elif mode == "linear":
            method = "linear"
        elif mode == "cubic":
            method = "cubic"
        else:
            raise tvm.error.OpAttributeInvalid(
                'Value {} in attribute "mode" of operator Resize is not valid.'.format(mode)
            )

        coord_trans = attr.get("coordinate_transformation_mode", b"half_pixel").decode("ascii")
        nearest_mode = attr.get("nearest_mode", b"round_prefer_floor").decode("ascii")
        alpha = attr.get("cubic_coeff_a", -0.75)
        exclude = attr.get("exclude_outside", 0)
        extrapolation_value = attr.get("extrapolation_value", 0.0)

        if roi is not None:
            roi = fold_constant(
                _op.concatenate(
                    [
                        _op.strided_slice(roi, [2], [ndims]),
                        _op.strided_slice(roi, [ndims + 2], [2 * ndims]),
                    ],
                    axis=0,
                )
            )

        out_size = fold_constant(_op.strided_slice(size, [2], [ndims]))

        out = None
        if ndims == 3:
            out = _op.image.resize1d(
                inputs[0],
                out_size,
                roi,
                "NCW",
                method,
                coord_trans,
                nearest_mode,
                alpha,
                exclude,
                extrapolation_value,
            )
        elif ndims == 4:
            out = _op.image.resize2d(
                inputs[0],
                out_size,
                roi,
                "NCHW",
                method,
                coord_trans,
                nearest_mode,
                alpha,
                exclude,
                extrapolation_value,
            )
        elif ndims == 5:
            out = _op.image.resize3d(
                inputs[0],
                out_size,
                roi,
                "NCDHW",
                method,
                coord_trans,
                nearest_mode,
                alpha,
                exclude,
                extrapolation_value,
            )
        else:
            raise NotImplementedError("Resize only supports 3, 4, or 5 dims")

        return out


class NonZero(OnnxOpConverter):
    """Operator converter for NonZero"""

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        if len(inputs) > 1:
            raise ValueError("Expect 1 input only")

        output = AttrCvt(op_name="argwhere")(inputs, attr, params)
        # ONNX NonZero always outputs int64
        output = _op.cast(output, "int64")
        return _op.transpose(output, axes=(1, 0))


class ReverseSequence(OnnxOpConverter):
    """Operator converter for ReverseSequence"""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):

        return _op.reverse_sequence(inputs[0], inputs[1], attr["time_axis"], attr["batch_axis"])


class TopK(OnnxOpConverter):
    """Operator converter for TopK"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) != 2:
            raise ValueError("Expect 2 input only")
        axis = attr.get("axis", -1)
        largest = attr.get("largest", 1)

        if largest == 0:
            # TODO(mbrookhart): optimize this by adding a smallest attribute to topi if this
            # ever becomes a bottleneck
            ndim = len(infer_shape(inputs[0]))
            if axis < 0:
                axis += ndim
            sort = _op.sort(inputs[0], axis=axis)
            argsort = _op.argsort(inputs[0], axis=axis, dtype="int64")
            begin = [0] * ndim
            stride = [1] * ndim
            end = _op.concatenate(
                [
                    _op.const([np.iinfo(np.int64).max] * axis, dtype="int64"),
                    inputs[1],
                    _op.const([np.iinfo(np.int64).max] * (ndim - axis - 1), dtype="int64"),
                ],
                axis=0,
            )
            return _expr.TupleWrapper(
                _expr.Tuple(
                    [
                        _op.strided_slice(sort, begin, end, stride),
                        _op.strided_slice(argsort, begin, end, stride),
                    ]
                ),
                2,
            )

        return _op.topk(inputs[0], inputs[1], axis=axis, dtype="int64")


class Range(OnnxOpConverter):
    """Operator converter for Range"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) != 3:
            raise ValueError("Expect 3 input only")

        return _op.arange(
            inputs[0], inputs[1], inputs[2], dtype=infer_type(inputs[0]).checked_type.dtype
        )


class IsInf(OnnxOpConverter):
    """Operator converter for IsInf"""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        detect_negative = attr.get("detect_negative", 1)
        detect_positive = attr.get("detect_positive", 1)
        dtype = infer_type(inputs[0]).checked_type.dtype
        isinf = _op.isinf(inputs[0])
        if not detect_negative:
            isinf = isinf * (inputs[0] > _op.const(0, dtype))
        if not detect_positive:
            isinf = isinf * (inputs[0] < _op.const(0, dtype))
        return isinf


class Celu(OnnxOpConverter):
    """Operator convereter for celu"""

    @classmethod
    def _impl_v12(cls, inputs, attr, params):
        x = inputs[0]
        dtype = infer_type(x).checked_type.dtype
        alpha = _op.const(attr.get("alpha", 1.0), dtype)
        zero = _op.const(0, dtype)
        one = _op.const(1, dtype)
        out = _op.maximum(zero, x) + _op.minimum(zero, alpha * (_op.exp(x / alpha) - one))
        return out


class MaxRoiPool(OnnxOpConverter):
    """Operator converter for MaxRoiPool."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        assert len(inputs) == 2, "MMaxRoiPool op take 2 inputs, {} given".format(len(inputs))

        data = inputs[0]
        rois = inputs[1]
        pooled_shape = attr.get("pooled_shape")
        spatial_scale = attr.get("spatial_scale", 1.0)

        return _vision.roi_pool(data, rois, pooled_shape, spatial_scale)


class RoiAlign(OnnxOpConverter):
    """Operator converter for RoiAlign."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if len(inputs) != 3:
            raise ValueError("Expect 3 inputs only")
        x = inputs[0]
        rois = inputs[1]
        batch_indices = inputs[2]
        mode = attr.get("mode", b"avg")
        if mode not in (b"avg", b"max"):
            raise NotImplementedError("RoiAlign in Relay only uses avg and max modes")
        output_height = attr.get("output_height", 1)
        output_width = attr.get("output_width", 1)

        sampling_ratio = attr.get("sampling_ratio", 0)
        spatial_scale = attr.get("spatial_scale", 1.0)

        batch_indices = _op.expand_dims(batch_indices, axis=1, num_newaxis=1)
        batch_indices = _op.cast(batch_indices, infer_type(rois).checked_type.dtype)
        rois = _op.concatenate([batch_indices, rois], 1)

        return _vision.roi_align(
            x, rois, [output_height, output_width], spatial_scale, sampling_ratio, mode=mode
        )


class Clip(OnnxOpConverter):
    """Operator converter for Clip."""

    @staticmethod
    def convert_attributes(inputs, attr, params):
        convert = AttrCvt("clip", transforms={"min": "a_min", "max": "a_max"})
        return convert(inputs, attr, params)

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        if "min" not in attr:
            attr["min"] = -np.inf
        if "max" not in attr:
            attr["max"] = np.inf
        return Clip.convert_attributes(inputs, attr, params)

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        if len(inputs) == 3 and isinstance(inputs[2], _expr.Constant):
            attr["max"] = inputs[2].data.numpy().item()
            inputs = inputs[0:2]
        if len(inputs) >= 2 and isinstance(inputs[1], _expr.Constant):
            attr["min"] = inputs[1].data.numpy().item()
            inputs = inputs[0:1]
        if "min" in attr and "max" in attr:
            return Clip.convert_attributes(inputs, attr, params)

        assert len(inputs) <= 3, "Clip-11 takes up to 3 inputs, input, min, max"
        result = inputs[0]
        for i, op in enumerate([_op.tensor.maximum, _op.tensor.minimum]):
            if i < len(inputs) - 1:
                if inputs[i + 1] is not None:
                    result = op(result, inputs[i + 1])
        return result


class Softplus(OnnxOpConverter):
    """Operator converter for Softplus."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        data_dtype = infer_type(data).checked_type.dtype
        data = _op.exp(data) + _expr.const(1, dtype=data_dtype)
        return _op.log(data)


class Loop(OnnxOpConverter):
    """Operator converter for Loop"""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        max_loop_count = inputs[0]
        cond = inputs[1]
        loop_deps = inputs[2:]
        num_deps = len(loop_deps)
        # Create a copy of the body function to prevent the original
        # from being modified.
        body = copy.copy(attr["body"])
        iter_dtype = infer_type(max_loop_count).checked_type.dtype

        # Determine what condition mode we're in.
        assert cond is not None or max_loop_count is not None
        is_for_loop = max_loop_count is not None and cond is None
        is_condition_for_loop = cond is not None and max_loop_count is not None

        # Loop inputs will be packed as
        # [iter_count, max_count, condition, loop_deps, scan_outputs]
        def cond_fn(*loop_inputs):
            i = loop_inputs[0]
            max_count = loop_inputs[1]
            w = loop_inputs[2]

            if cond is not None:
                out_while = _op.equal(w, _expr.const(True, "bool"))
            if max_loop_count is not None:
                out_loop = _op.less(i, max_count)

            if is_condition_for_loop:
                return _op.logical_and(out_while, out_loop)
            if is_for_loop:
                return out_loop
            return out_while

        # Get the current graph proto and create a clone for the subgraph
        graph_scope = GraphProto.current
        subgraph_scope = GraphProto(
            graph_scope._shape, graph_scope._dtype, graph_scope._freeze_params
        )
        # Load nodes from outer graph into inner graph.
        subgraph_scope._nodes = graph_scope._nodes.copy()

        # Create a list of variables for each value updated in the loop.
        def get_var(name, val, scan=False):
            checked_type = infer_type(val)
            if hasattr(checked_type, "type_annotation"):
                checked_type = checked_type.type_annotation
            if hasattr(checked_type, "checked_type"):
                checked_type = checked_type.checked_type
            shape = get_const_tuple(checked_type.shape)
            actual_shape = []
            for dim in shape:
                if isinstance(dim, int) and dim == 0:
                    actual_shape.append(_ty.Any())
                else:
                    actual_shape.append(dim)
            if scan:
                return _expr.var(name, shape=[_ty.Any()] + actual_shape, dtype=checked_type.dtype)

            return _expr.var(name, shape=actual_shape, dtype=checked_type.dtype)

        loop_vars = [
            _expr.var(body.input[0].name, shape=(), dtype=iter_dtype),  # iteration count
            _expr.var("max_count", shape=(), dtype=iter_dtype),  # iteration count
            get_var(body.input[1].name, cond),  # exit condition
        ]
        loop_vars += [get_var(body.input[i + 2].name, v) for i, v in enumerate(loop_deps)]
        loop_var_names = [v.name_hint for v in loop_vars]

        num_scan_outputs = len(body.output) - (1 + num_deps)

        # Construct variables and initial empty tensors for any scan outputs.
        # To do this, we'll figure out the output shapes of the body subgraph by importing
        # it and doing type inference.
        scan_output_vars = []
        scan_output_init = []
        if num_scan_outputs > 0:
            with subgraph_scope:
                loop_outputs = subgraph_scope.from_onnx(
                    body, graph_scope.opset, get_output_expr=True
                )
            loop_outputs = _expr.TupleWrapper(loop_outputs, len(body.output))

        for i in range(num_scan_outputs):
            name, _, _, _ = get_info(body.output[i + 1 + num_deps])
            output_node = infer_type(loop_outputs[i + 1 + num_deps])
            shape = get_const_tuple(output_node.checked_type.shape)
            dtype = output_node.checked_type.dtype
            scan_output_vars.append(
                _expr.var(name, shape=([_ty.Any()] * (len(shape) + 1)), dtype=dtype)
            )
            scan_output_init.append(
                _op.reshape(_expr.const(np.array([]).astype(dtype)), [0] + [1] * len(shape))
            )

        # Now we can remove loop iter variables from our inner loop's inputs.
        # This is kind of a hack since we have graph inputs that we don't
        # want to treat as actual inputs.
        while len(body.input) != 0:
            body.input.pop(0)

        # Define the loop body, in this function we need to unpack loop inputs,
        # convert the loop subgraph, and pack outputs for the next iteration.
        def body_fn(*loop_inputs):
            # Unpack inputs
            loop_count = loop_inputs[0]
            max_count = loop_inputs[1]
            cond = loop_inputs[2]
            current_vars = list(loop_inputs[3 : (3 + num_deps)])
            scan_outputs = loop_inputs[(3 + num_deps) :]

            # Prepare body inputs by adding them to node dictionary.
            new_inputs = [loop_count, max_count, cond] + current_vars
            for i, inp in enumerate(new_inputs):
                subgraph_scope._nodes[loop_var_names[i]] = inp

            # Get the output of the current loop using the updated inputs.
            with subgraph_scope:
                loop_outputs = subgraph_scope.from_onnx(
                    body, graph_scope.opset, get_output_expr=True
                )
            # Unpack the body outputs and prepare variables for next iteration.
            new_cond = loop_outputs[0]
            new_loop_vars = [loop_outputs[i] for i in range(1, 1 + num_deps)]
            new_scan_outputs = [loop_outputs[i] for i in range(1 + num_deps, len(loop_outputs))]

            # Add new scan outputs to tracking
            combined_scan_outputs = []
            for i, scan in enumerate(scan_outputs):
                rank = len(infer_shape(scan)) - 1
                new_scan = new_scan_outputs[i]
                expand_scan = _op.expand_dims(new_scan, axis=0)
                # For non scalar outputs we need to broadcast the initial value.
                if rank > 0:
                    new_scan_shape = shape_of(new_scan, dtype=iter_dtype)
                    scan_broadcast = _op.concatenate(
                        [_op.reshape(loop_count, [1]), new_scan_shape], axis=0
                    )
                    scan = _op.broadcast_to(scan, scan_broadcast)
                combined_scan = _op.concatenate([scan, expand_scan], axis=0)
                combined_scan_outputs.append(combined_scan)

            # Increment counter.
            if max_loop_count is not None:
                incr = _expr.const(1, dtype=iter_dtype)
                loop_count = loop_count + incr

            # Pack loop outputs for next iteration
            # [iter_count, cond, loop_deps, loop_scans]
            return [loop_count, max_count, new_cond] + new_loop_vars + combined_scan_outputs

        # Create the loop function.
        loop = fold_constant(_loops.while_loop(cond_fn, loop_vars + scan_output_vars, body_fn))

        # Now need to run initial values through the graph.
        init_count = _expr.const(0, dtype=iter_dtype)
        loop_vals = loop(init_count, max_loop_count, cond, *loop_deps, *scan_output_init)

        # Extract final iteration outputs.
        if num_deps + num_scan_outputs == 1:
            outputs = _expr.TupleGetItem(loop_vals, 3)
        else:
            outputs = _expr.TupleWrapper(
                _expr.Tuple(
                    [
                        _expr.TupleGetItem(loop_vals, i + 3)
                        for i in range(num_deps + num_scan_outputs)
                    ]
                ),
                num_deps + num_scan_outputs,
            )

        # Update outer graph with constants found in the subgraph.
        free_vars = analysis.free_vars(loop)
        graph_scope._params.update(subgraph_scope._params)
        graph_scope._nodes.update(subgraph_scope._nodes)
        for var in free_vars:
            graph_scope._nodes.update({var.name_hint: var})
        return outputs


class If(OnnxOpConverter):
    """Operator converter for If"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        cond = inputs[0]
        # Convert array to bool if needed.
        if len(infer_shape(cond)) > 0:
            cond = _op.take(cond, _expr.const(0, dtype="int64"))
        then_branch = attr.get("then_branch", None)
        else_branch = attr.get("else_branch", None)
        assert then_branch is not None and else_branch is not None

        # Create graph converters for both branches.
        graph_scope = GraphProto.current
        then_graph = GraphProto(graph_scope._shape, graph_scope._dtype, graph_scope._freeze_params)
        then_graph._nodes = graph_scope._nodes.copy()
        else_graph = GraphProto(graph_scope._shape, graph_scope._dtype, graph_scope._freeze_params)
        else_graph._nodes = graph_scope._nodes.copy()

        # Convert each branch to a relay expression.
        with then_graph:
            then_expr = then_graph.from_onnx(then_branch, graph_scope.opset, get_output_expr=True)
        with else_graph:
            else_expr = else_graph.from_onnx(else_branch, graph_scope.opset, get_output_expr=True)

        # Add constants from both branches to parent graph.
        graph_scope._params.update(then_graph._params)
        graph_scope._nodes.update(then_graph._nodes)
        then_free_vars = analysis.free_vars(then_expr)
        for var in then_free_vars:
            graph_scope._nodes.update({var.name_hint: var})
        graph_scope._params.update(else_graph._params)
        graph_scope._nodes.update(else_graph._nodes)
        else_free_vars = analysis.free_vars(else_expr)
        for var in else_free_vars:
            graph_scope._nodes.update({var.name_hint: var})

        # Now we can construct the relay if statement and return.
        ret = _expr.If(cond, then_expr, else_expr)
        if len(then_branch.output) > 1:
            ret = _expr.TupleWrapper(ret, len(then_branch.output))
        return ret


class Scan(OnnxOpConverter):
    """Operator converter for Scan"""

    @classmethod
    def _impl_v8(cls, inputs, attr, params):
        new_inputs = inputs[1:]
        batch_num = infer_shape(inputs[1])[0]
        out = []
        for i in range(batch_num):
            v9_inputs = [
                _op.take(new_inputs[j], _expr.const(i), axis=0) for j in range(len(new_inputs))
            ]
            results = cls._impl_v9(v9_inputs, attr, params)
            results = [_op.expand_dims(results[j], axis=0) for j in range(len(results))]
            if i == 0:
                out = results
            else:
                out = [_op.concatenate([out[j], results[j]], axis=0) for j in range(len(results))]

        out = _expr.TupleWrapper(_expr.Tuple(out), len(out))
        return out

    @classmethod
    def _impl_v9(cls, inputs, attr, params):
        body = attr.get("body")
        num_scan_inputs = attr.get("num_scan_inputs")
        num_all_inputs = len(inputs)
        num_state_inputs = len(body.input) - num_scan_inputs
        num_state_outputs = num_state_inputs
        num_all_outputs = len(body.output)
        num_scan_outputs = num_all_outputs - num_state_outputs
        scan_input_axes = attr.get("scan_input_axes", [0] * num_scan_inputs)
        scan_input_directions = attr.get("scan_input_directions", [0] * num_scan_inputs)
        scan_output_axes = list(attr.get("scan_output_axes", [0] * num_scan_outputs))
        scan_output_directions = attr.get("scan_output_directions", [0] * num_scan_outputs)
        # loop count are the same for all scan inputs, so get loop count by first input scan
        # strided_slice not support dynamic axes, so assume input shape are static
        max_loop_count = infer_shape(inputs[num_state_inputs])[scan_input_axes[0]]

        # Create a copy of the body function to prevent the original
        # from being modified.
        body = copy.copy(attr["body"])

        # Loop inputs will be packed as
        # [iter_count, loop_deps, scan_outputs]
        def cond_fn(*loop_inputs):
            i = loop_inputs[0]
            return _op.less(i, relay.const(max_loop_count, "int32"))

        # Get the current graph proto and create a clone for the subgraph
        graph_scope = GraphProto.current
        subgraph_scope = GraphProto(
            graph_scope._shape, graph_scope._dtype, graph_scope._freeze_params
        )
        # Load nodes from outer graph into inner graph.
        subgraph_scope._nodes = graph_scope._nodes.copy()

        # Create a list of variables for each value updated in the loop.
        def get_var(name, val, scan=False):
            checked_type = infer_type(val)
            if hasattr(checked_type, "type_annotation"):
                checked_type = checked_type.type_annotation
            if hasattr(checked_type, "checked_type"):
                checked_type = checked_type.checked_type
            shape = get_const_tuple(checked_type.shape)
            actual_shape = []
            for dim in shape:
                if isinstance(dim, int) and dim == 0:
                    actual_shape.append(_ty.Any())
                else:
                    actual_shape.append(dim)
            if scan:
                return _expr.var(name, shape=[_ty.Any()] + actual_shape, dtype=checked_type.dtype)

            return _expr.var(name, shape=actual_shape, dtype=checked_type.dtype)

        # Construct variables and initial empty tensors for any scan outputs.
        # To do this, we'll figure out the output shapes of the body subgraph by importing
        # it and doing type inference.
        scan_output_vars = []
        scan_output_init = []
        if num_scan_outputs > 0:
            with subgraph_scope:
                loop_outputs = subgraph_scope.from_onnx(
                    body, graph_scope.opset, get_output_expr=True
                )
            loop_outputs = _expr.TupleWrapper(loop_outputs, len(body.output))

        for i in range(num_scan_outputs):
            name, _, _, _ = get_info(body.output[i + num_state_outputs])
            output_node = infer_type(loop_outputs[i + num_state_outputs])
            shape = list(get_const_tuple(output_node.checked_type.shape))
            if scan_output_axes[i] < 0:
                scan_output_axes[i] = len(shape) + scan_output_axes[i] + 1
            shape.insert(scan_output_axes[i], max_loop_count)
            dtype = output_node.checked_type.dtype
            scan_output_vars.append(_expr.var(name, shape=shape, dtype=dtype))
            scan_output_init.append(_op.zeros(shape, dtype))

        # loop vars = [iter_count, scan_state, scan_out]
        loop_vars = [
            _expr.var("iter", shape=(), dtype="int32"),  # iteration count
        ]
        loop_vars += [
            get_var(body.input[i].name, v) for i, v in enumerate(inputs) if i < num_state_inputs
        ]
        loop_vars += scan_output_vars
        body_input_var_names = ["iter"] + [body.input[i].name for i in range(len(body.input))]

        # # Now we can remove loop iter variables from our inner loop's inputs.
        # # This is kind of a hack since we have graph inputs that we don't
        # # want to treat as actual inputs.
        while len(body.input) != 0:
            body.input.pop(0)

        # Define the loop body, in this function we need to unpack loop inputs,
        # convert the loop subgraph, and pack outputs for the next iteration.
        def body_fn(*loop_inputs):
            # Unpack inputs
            loop_count = loop_inputs[0]
            state_vars = list(loop_inputs[1 : 1 + num_state_inputs])
            scan_vars = list(loop_inputs[1 + num_state_inputs :])
            # body take scan graph scan inputs as original input
            input_scan_exprs = []
            for i in range(num_state_inputs, num_all_inputs):
                if scan_input_directions[i - num_state_inputs] != 0:
                    input_scan_exprs.append(
                        relay.take(
                            inputs[i],
                            relay.const(max_loop_count - 1, "int32") - loop_count,
                            axis=scan_input_axes[i - num_state_inputs],
                        )
                    )
                else:
                    input_scan_exprs.append(
                        relay.take(
                            inputs[i],
                            loop_count,
                            axis=scan_input_axes[i - num_state_inputs],
                        )
                    )

            # Prepare body inputs by adding them to node dictionary.
            body_inputs = [loop_count] + state_vars + input_scan_exprs
            for i, inp in enumerate(body_inputs):
                subgraph_scope._nodes[body_input_var_names[i]] = inp

            # Get the output of the current loop using the updated inputs.
            with subgraph_scope:
                loop_outputs = subgraph_scope.from_onnx(
                    body, graph_scope.opset, get_output_expr=True
                )
            # Unpack the body outputs and prepare variables for next iteration.
            new_state_vars = [loop_outputs[i] for i in range(num_state_outputs)]
            new_scan_vars = [loop_outputs[i] for i in range(num_state_outputs, num_all_outputs)]

            # Add new scan outputs to tracking
            combined_scan_outputs = []
            for i in range(num_scan_outputs):
                if scan_output_directions[i] == 0:
                    # append new scan output
                    combined_scan = _op.concatenate(
                        [scan_vars[i], _op.expand_dims(new_scan_vars[i], axis=scan_output_axes[i])],
                        axis=scan_output_axes[i],
                    )
                    # pop head scan output
                    combined_scan = _op.strided_slice(
                        combined_scan,
                        begin=[1],
                        end=[max_loop_count + 1],
                        strides=[1],
                        axes=[scan_output_axes[i]],
                    )
                else:
                    # prepend new scan output
                    combined_scan = _op.concatenate(
                        [_op.expand_dims(new_scan_vars[i], axis=scan_output_axes[i]), scan_vars[i]],
                        axis=scan_output_axes[i],
                    )
                    # pop tail scan output
                    combined_scan = _op.strided_slice(
                        combined_scan,
                        begin=[0],
                        end=[max_loop_count],
                        strides=[1],
                        axes=[scan_output_axes[i]],
                    )
                combined_scan_outputs.append(combined_scan)

            incr = _expr.const(1, dtype="int32")
            loop_count = loop_count + incr

            # Pack loop outputs for next iteration
            # [iter_count, state_var, scan_var]
            return [loop_count] + new_state_vars + combined_scan_outputs

        # Create the loop function.
        loop = fold_constant(_loops.while_loop(cond_fn, loop_vars, body_fn))

        # Now need to run initial values through the graph.
        init_count = _expr.const(0, dtype="int32")

        input_states = [inputs[i] for i in range(num_state_inputs)]
        loop_vals = loop(init_count, *input_states, *scan_output_init)

        outputs = _expr.TupleWrapper(
            _expr.Tuple([_expr.TupleGetItem(loop_vals, i + 1) for i in range(num_all_outputs)]),
            num_all_outputs,
        )

        # Update outer graph with constants found in the subgraph.
        free_vars = analysis.free_vars(loop)
        graph_scope._params.update(subgraph_scope._params)
        graph_scope._nodes.update(subgraph_scope._nodes)
        for var in free_vars:
            graph_scope._nodes.update({var.name_hint: var})
        return outputs


class LinearRegressor(OnnxOpConverter):
    """Operator converter for LinearRegressor."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        data = inputs[0]
        coefficients = attr.get("coefficients", 0)
        data_shape = infer_shape(data)
        targets = attr.get("targets", 1)
        coefficients = _expr.const(list(coefficients), dtype="float32")
        coefficients_shape = infer_shape(coefficients)

        coefficients = _op.reshape(coefficients, (targets, coefficients_shape[0] // targets))
        if coefficients_shape[0] // targets < data_shape[-1]:
            data = _op.split(data, [coefficients_shape[0] // targets], -1)[0]

        mm_out = _op.nn.dense(data, coefficients)

        if "intercepts" in attr:
            intercepts = attr.get("intercepts", 0)
            intercepts = _expr.const(list(intercepts), dtype="float32")

            if targets == 1:
                return _op.nn.bias_add(mm_out, intercepts, axis=-1)
            return get_relay_op("add")(mm_out, intercepts)

        return mm_out


class NonMaxSuppression(OnnxOpConverter):
    """Operator converter for NonMaxSuppression."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        # Get parameter values
        boxes = inputs[0]
        scores = inputs[1]
        max_output_boxes_per_class = inputs[2]
        iou_threshold = inputs[3]
        score_threshold = inputs[4]

        boxes_dtype = infer_type(boxes).checked_type.dtype

        if attr.get("center_point_box", 0) != 0:
            xc, yc, w, h = _op.split(boxes, 4, axis=2)
            half_w = w / _expr.const(2.0, boxes_dtype)
            half_h = h / _expr.const(2.0, boxes_dtype)
            x1 = xc - half_w
            x2 = xc + half_w
            y1 = yc - half_h
            y2 = yc + half_h
            boxes = _op.concatenate([y1, x1, y2, x2], axis=2)

        if iou_threshold is None:
            iou_threshold = _expr.const(0.0, dtype="float32")
        if score_threshold is None:
            score_threshold = _expr.const(0.0, dtype="float32")

        def conditionally_squeeze_scalar(x):
            rank = len(infer_shape(x))
            assert rank <= 1, "nms thresholds must be scalars"
            if rank == 1:
                return _op.squeeze(x, [0])
            return x

        max_output_boxes_per_class = conditionally_squeeze_scalar(max_output_boxes_per_class)
        iou_threshold = conditionally_squeeze_scalar(iou_threshold)
        score_threshold = conditionally_squeeze_scalar(score_threshold)

        nms_out = _op.vision.all_class_non_max_suppression(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold
        )

        return _op.strided_slice(nms_out[0], _op.const([0], dtype="int64"), nms_out[1])


class ATen(OnnxOpConverter):
    """Operator converter for Pytorch ATen ops."""

    @classmethod
    def _op_dispatch(cls, operator, inputs, attr, params):
        op_map = {
            "size": cls._size,
            "arange": cls._arange,
            "index_put": cls._index_put,
            "reshape": cls._reshape,
            "embedding_bag": cls._embedding_bag,
        }
        assert operator in op_map, "Operator %s is not supported." % operator
        return op_map[operator](inputs, attr, params)

    @classmethod
    def _size(cls, inputs, attr, params):
        return _op.take(
            _op.shape_of(inputs[0], dtype="int64"),
            _expr.const(-1, dtype="int64"),
            axis=0,
            mode="wrap",
        )

    @classmethod
    def _arange(cls, inputs, attr, params):
        return _op.arange(inputs[0], inputs[1], inputs[2], dtype="int64")

    @classmethod
    def _check_index(cls, indices, values):
        def unfolding_indices(indices, values):
            n = len(indices)
            flatten_indices = []
            slices_size = []
            for index in indices:
                flatten_indices.append(_op.reshape(index, _op.const([-1])))
                slices_size.append(infer_shape(flatten_indices[-1])[0])
            repeat_size = [1]
            tile_size = [1]
            for i in range(1, n):
                repeat_size.append(slices_size[-i] * repeat_size[-1])
                tile_size.append(slices_size[i - 1] * tile_size[-1])
            repeat_size.reverse()
            unflod_slices = []
            for i in range(n):
                unflod_slices.append(
                    fold_constant(
                        _op.repeat(_op.tile(flatten_indices[i], (tile_size[i],)), repeat_size[i], 0)
                    )
                )
            return unflod_slices, _op.reshape(values, _op.const([-1]))

        values_shape = infer_shape(values)
        if len(values_shape) != 1:
            return unfolding_indices(indices, values)
        return indices, values

    @classmethod
    def _index_put(cls, inputs, attr, params):
        in_tensor = inputs[0]
        indices, values = cls._check_index(inputs[1 : len(inputs) - 2], inputs[len(inputs) - 2])
        accumulate = inputs[len(inputs) - 1].data.asnumpy() != 0
        if not accumulate:
            mode = "update"
        else:
            mode = "add"
        index_tensor = _op.stack(indices, axis=0)
        return _op.transform.scatter_nd(in_tensor, index_tensor, values, mode)

    @classmethod
    def _reshape(cls, inputs, attr, params):
        return _op.reshape(inputs[0], inputs[1])

    @classmethod
    def _embedding_bag(cls, inputs, attr, params):
        mode_map = {0: _op.sum, 1: _op.mean, 2: _op.max}

        mode = attr.get("mode", 1)
        reduction_fn = mode_map[mode]
        weights, indices, offsets = inputs[0], inputs[1], inputs[2]
        offsets_shape = _op.shape_of(offsets, dtype="int64")
        indices_shape = _op.stack(
            [
                _op.take(offsets_shape, _expr.const(0, dtype="int64")),
                _expr.const(-1, dtype="int64"),
            ],
            axis=0,
        )
        indices = _op.reshape(indices, indices_shape)
        embedding = _op.take(weights, indices.astype("int64"), axis=0)
        rembedding = reduction_fn(embedding, axis=1)
        # EmbeddingBag has 4 outputs for some reason despite only one ever being used.
        # Fill the rest with 0s.
        unused_output = _expr.const(0, dtype="float32")
        return _expr.TupleWrapper(
            _expr.Tuple((rembedding, unused_output, unused_output, unused_output)), 4
        )

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        operator = attr.get("operator", None).decode("utf-8")
        assert operator, "ATen Operator not found"
        return cls._op_dispatch(operator, inputs, attr, params)


class QuantizeLinear(OnnxOpConverter):
    """Operator converter for QuantizeLinear."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        data, scale, zp = inputs
        out_dtype = infer_type(zp).checked_type.dtype
        return _qnn.op.quantize(data, scale, _op.cast(zp, "int32"), 0, out_dtype)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        data, scale, zp = inputs
        out_dtype = infer_type(zp).checked_type.dtype
        axis = attr.get("axis", 1)
        if len(infer_shape(data)) < 2:
            axis = 0
        return _qnn.op.quantize(data, scale, _op.cast(zp, "int32"), axis, out_dtype)


class DequantizeLinear(OnnxOpConverter):
    """Operator converter for QuantizeLinear."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        data, scale, zp = inputs
        return _qnn.op.dequantize(data, scale, _op.cast(zp, "int32"), 0)

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        data, scale, zp = inputs
        axis = attr.get("axis", 1)
        if len(infer_shape(data)) <= 1:
            axis = 0
        return _qnn.op.dequantize(data, scale, _op.cast(zp, "int32"), axis)


class DynamicQuantizeLinear(OnnxOpConverter):
    """Operator converter for QuantizeLinear."""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        """This op is deprecated an only supports uint8"""
        data = inputs[0]
        data_dtype = infer_type(data).checked_type.dtype
        zero = _op.const(0, dtype=data_dtype)
        maximum = _op.maximum(zero, _op.max(data))
        minimum = _op.minimum(zero, _op.min(data))
        scale = (maximum - minimum) / _op.const(255, dtype=data_dtype)
        zp = zero - _op.min(data) / scale
        zp = _op.cast(_op.round(_op.clip(zp, 0, 255)), "uint8")
        return _expr.TupleWrapper(
            _expr.Tuple(
                [_qnn.op.quantize(data, scale, _op.cast(zp, "int32"), 0, "uint8"), scale, zp]
            ),
            size=3,
        )


class QLinearConv(OnnxOpConverter):
    """Operator converter for QLinearConv."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        data = inputs[0]
        x_scale = get_scalar(inputs[1], params)
        x_zero_point = get_scalar(inputs[2], params, "int32")
        weight = inputs[3]
        w_scale = get_scalar_or_1d_tensor(inputs[4], params)
        w_zero_point = get_scalar_or_1d_tensor(inputs[5], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[6], params))
        y_zero_point = get_scalar(inputs[7], params, "int32")

        # Check shapes for per channel quantization
        w_scale_shape = infer_shape(w_scale)
        w_zero_point_shape = infer_shape(w_zero_point)
        if len(w_scale_shape) == 1 or len(w_zero_point_shape) == 1:
            m = infer_shape(weight)[0]
            if m != w_scale_shape[0] or m != w_zero_point_shape[0]:
                raise tvm.error.OpAttributeInvalid(
                    "The number of elements should be equal to the number of output channels"
                )

        input_shape = infer_shape(data)

        ndim = len(input_shape)
        kernel_type = infer_type(weight)
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]
        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shapes[0][2:]

        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                zp = fold_constant(x_zero_point)
                assert isinstance(zp, relay.Constant), "Zero point expected to be a constant"
                data = autopad(
                    data,
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    attr.get("dilations", [1] * (ndim - 2)),
                    pad_value=zp.data,
                    mode=attr["auto_pad"],
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"]))
            attr.pop("auto_pad")

        out_channels = kernel_shapes[0][0]
        dilation = attr.get("dilations", [1] * (ndim - 2))
        strides = attr.get("strides", [1] * (ndim - 2))
        padding = attr["pads"] if "pads" in attr else 0
        groups = attr["group"] if "group" in attr else 1

        if ndim != 4:
            raise tvm.error.OpAttributeInvalid(
                "Only 2D kernels are supported for operator QLinearConv."
            )

        out = _qnn.op.conv2d(
            data,
            weight,
            x_zero_point,
            w_zero_point,
            x_scale,
            w_scale,
            kernel_size=attr["kernel_shape"],
            channels=out_channels,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        use_bias = len(inputs) == 9
        if use_bias:
            out = _op.nn.bias_add(out, inputs[8])

        out_dtype = infer_type(inputs[7]).checked_type.dtype
        requantize_scale = _op.multiply(x_scale, w_scale)

        # requantize requires y_scale to be constant,
        # if y_scale is not constant, doing dequantize -> quantize
        if isinstance(y_scale, _expr.Constant):
            out = _qnn.op.requantize(
                out,
                requantize_scale,
                _op.const(0, dtype="int32"),
                y_scale,
                y_zero_point,
                out_dtype=out_dtype,
                axis=1,
            )
        else:
            out = _qnn.op.dequantize(out, requantize_scale, _op.const(0, dtype="int32"), axis=1)
            out = _qnn.op.quantize(out, y_scale, y_zero_point, axis=1, out_dtype=out_dtype)
        return out


class QLinearAdd(OnnxOpConverter):
    """Operator converter for QLinearAdd from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        a = inputs[0]
        a_scale = get_scalar(inputs[1], params)
        a_zero_point = get_scalar(inputs[2], params, "int32")
        b = inputs[3]
        b_scale = get_scalar(inputs[4], params)
        b_zero_point = get_scalar(inputs[5], params, "int32")
        c_scale = get_scalar(inputs[6], params)
        c_zero_point = get_scalar(inputs[7], params, "int32")

        dtype = infer_type(a).checked_type.dtype

        ## Onnxruntime doesn't actually do this op in integer, they dequantize to fp32
        ## and then requantize afer
        ## https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/mlas/lib/qladd.cpp
        a = _qnn.op.dequantize(
            inputs[0], a_scale, a_zero_point
        )  # , c_scale, c_zero_point, out_dtype = dtype)
        b = _qnn.op.dequantize(
            inputs[3], b_scale, b_zero_point
        )  # , c_scale, c_zero_point, out_dtype = dtype)
        out = _op.add(a, b)
        return _qnn.op.quantize(out, c_scale, c_zero_point, out_dtype=dtype)


class QLinearMatMul(OnnxOpConverter):
    """
    Operator converter for QLinearMatMul from Microsoft onnxruntime contrib opset.

    Limitations:
    - Only supports 2D input tensors.
    - Not guaranteed to meet the integer-overflow behavior stipulated in the
      ONNX documentation for this operator.

    The QLinearMatMul converter is re-used for MatMulInteger and is adapted for
    the latter with the optional `expected_out_dtypes` argument.
    """

    @classmethod
    def _impl_v10(cls, inputs, attr, params, expected_out_dtypes=None):
        if expected_out_dtypes is None:
            # The default QLinearMatMul converter is expected to have one of
            # these output dtypes.
            expected_out_dtypes = ["int8", "uint8"]

        # Some of the ops used below take scalar-like inputs, and may require either
        # of the following:
        #
        # - the input is Const node (not merely an expression that *could* be reduced
        #   to a single Const at graph-compilation time)
        #
        # - the input has a specific dtype
        #
        # This function attempts to present 'x' in a form that meets both of those
        # requirements.
        def try_resolve_to_const(x, dtype_override=None):
            x2 = try_resolve_var_to_const(x, params)
            num_elem = np.prod(infer_shape(x))
            if num_elem == 1:
                x2 = ensure_scalar_shape(x2)
            x_dtype = infer_type(x).checked_type.dtype
            if (dtype_override is not None) and (dtype_override != x_dtype):
                x2 = _op.cast(x2, dtype_override)
            x3 = fold_constant(x2)
            return x3

        # Unpack the inputs and obtain some type info...
        a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp = inputs

        a_type = infer_type(a).checked_type  # 'T1' in ONNX doc for this op
        a_scale_type = infer_type(a_scale).checked_type
        a_zp_type = infer_type(a_zp).checked_type

        b_type = infer_type(b).checked_type  # 'T2' in ONNX doc for this op
        b_scale_type = infer_type(b_scale).checked_type
        b_zp_type = infer_type(b_zp).checked_type

        y_scale_type = infer_type(y_scale).checked_type
        y_zp_type = infer_type(y_zp).checked_type  # 'T3' in ONNX doc for this op

        a_shape = infer_shape(a)
        b_shape = infer_shape(b)

        # Verify type assumptions, based on the ONNX doc for this op...
        assert a_type.dtype in ["int8", "uint8"]
        assert a_scale_type.dtype == "float32"
        assert a_zp_type.dtype == a_type.dtype

        assert b_type.dtype in ["int8", "uint8"]
        assert b_scale_type.dtype == "float32"
        assert b_zp_type.dtype == b_type.dtype

        assert y_scale_type.dtype == "float32"
        assert y_zp_type.dtype in expected_out_dtypes

        # TODO: relax this limitation in a future version of this importer.
        a_rank = len(a_shape)
        b_rank = len(b_shape)
        assert (a_rank == 2) and (b_rank == 2), (
            "QLinearMatMul importer currently requires both 'a' and 'b' tensors to be 2D, but"
            " rank(a)={}, rank(b)={}".format(a_rank, b_rank)
        )

        # _qnn.op.dense requires the zero-point values to have dtype int32.
        a_scale_scalar = try_resolve_to_const(a_scale)
        a_zp_scalar = try_resolve_to_const(a_zp, "int32")

        b_scale_scalar = try_resolve_to_const(b_scale)
        b_zp_scalar = try_resolve_to_const(b_zp, "int32")

        y_scale_scalar = try_resolve_to_const(y_scale)
        y_zp_scalar = try_resolve_to_const(y_zp, "int32")

        # TODO: Confirm that we're using 'num_hidden_units' correctly / as intended with
        # the '_qnn.op.dense' instance below.
        num_hidden_units = infer_shape(b)[-1]

        # - Specify the matmul result dtype as int32, so that hopefully the matmul will use
        #   a 32-bit accumulator as seems to be required by the ONNX op's documentation.
        #
        # TL;DR:
        # The ONNX documentation for this op is clear about acceptable overflow
        # behavior during the matmul operation:
        #   - The scalar multiplication ops MAY NOT overflow.
        #   - The scalar addition ops, which sum the results of the scalar multiplication,
        #     MAY overflow, but if they do so, it must behave as one would expect during
        #     32-bit integer-addition overflow.
        # As of this writing, Relay's qnn.op.dense operator doesn't expose a way for us to
        # express these constraints.
        #
        # TODO: Extend TVM / Relay / TIR / etc. to allow this kind of constraint to be
        # expressed in a Relay graph. And then update this importer and various TVM
        # backends accordingly.
        matmul_result_dtype = "int32"

        matmul_result = _qnn.op.dense(
            a,
            _op.transpose(b),
            a_zp_scalar,
            b_zp_scalar,
            a_scale_scalar,
            b_scale_scalar,
            num_hidden_units,
            matmul_result_dtype,
        )

        # This information might only be found in the C++ code-comments for the
        # dense.matmul op, but the quantized tensor returned by _qnn.op.dense
        # has scale==(a_scale_scalar * b_scale_scalar), and zero_point==0.
        #
        # 'matmul_result_zp_scalar' has type 'int32' to satisfy input requirements
        # of the [de/re]quantize ops below.
        matmul_result_scale_scalar = fold_constant(_op.multiply(a_scale_scalar, b_scale_scalar))
        matmul_result_zp_scalar = _op.const(0, dtype="int32")

        if "int32" in expected_out_dtypes:
            # This is the adaptation of the QLinearMatMul converter for MatMulInteger,
            # in the MatMulInteger case we skip the unnecessary requantization step.
            return matmul_result

        # requantize requires y_scale to be constant,
        # if y_scale is not constant, doing dequantize -> quantize
        if isinstance(y_scale_scalar, _expr.Constant):
            y = _qnn.op.requantize(
                matmul_result,
                matmul_result_scale_scalar,
                matmul_result_zp_scalar,
                y_scale_scalar,
                y_zp_scalar,
                axis=-1,
                rounding="TONEAREST",
                out_dtype=y_zp_type.dtype,
            )
        else:
            matmul_result_deq = _qnn.op.dequantize(
                matmul_result, matmul_result_scale_scalar, matmul_result_zp_scalar, axis=0
            )

            y = _qnn.op.quantize(
                matmul_result_deq, y_scale_scalar, y_zp_scalar, axis=0, out_dtype=y_zp_type.dtype
            )

        return y


class MatMulInteger(OnnxOpConverter):
    """Operator converter for MatMulInteger."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        a = inputs[0]
        b = inputs[1]

        a_dtype = infer_type(a).checked_type.dtype
        b_dtype = infer_type(b).checked_type.dtype

        assert a_dtype in ("int8", "uint8"), "MatMulInteger: invalid dtype for first input"
        assert b_dtype in ("int8", "uint8"), "MatMulInteger: invalid dtype for second input"

        assert a_dtype == b_dtype, "MatMulInteger: input dtypes must match"

        a_scale = _op.const(1.0, dtype="float32")
        b_scale = _op.const(1.0, dtype="float32")
        out_scale = _op.const(1.0, dtype="float32")

        a_zero_point = _op.const(0.0, dtype=a_dtype)
        b_zero_point = _op.const(0.0, dtype=b_dtype)
        out_zero_point = _op.const(0.0, dtype="int32")

        if len(inputs) == 4:
            a_zero_point = inputs[2]
            b_zero_point = inputs[3]

            a_zp_dtype = infer_type(a_zero_point).checked_type.dtype
            b_zp_dtype = infer_type(b_zero_point).checked_type.dtype
            assert (
                a_zp_dtype == a_dtype and b_zp_dtype == b_dtype
            ), "MatMulInteger: input dtype doesn't match zero point dtype"
        elif len(inputs) != 2:
            raise AssertionError(
                "MatMulInteger op takes 2 or 4 inputs, {} given".format(len(inputs))
            )

        inputs = [
            a,
            a_scale,
            a_zero_point,
            b,
            b_scale,
            b_zero_point,
            out_scale,
            out_zero_point,
        ]

        return QLinearMatMul.get_converter(10)(inputs, attr, params, expected_out_dtypes=["int32"])


class QLinearMul(OnnxOpConverter):
    """Operator converter for QLinearMul from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        a = inputs[0]
        a_scale = get_scalar(inputs[1], params)
        a_zero_point = get_scalar(inputs[2], params, "int32")
        b = inputs[3]
        b_scale = get_scalar(inputs[4], params)
        b_zero_point = get_scalar(inputs[5], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[6], params))
        y_zero_point = get_scalar(inputs[7], params, "int32")

        dtype = infer_type(a).checked_type.dtype

        ## Onnxruntime doesn't actually do this op in integer, they dequantize to fp32
        ## and then requantize afer
        ## https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/mlas/lib/qlmul.cpp
        a = _qnn.op.dequantize(inputs[0], a_scale, a_zero_point)
        b = _qnn.op.dequantize(inputs[3], b_scale, b_zero_point)
        out = _op.multiply(a, b)
        return _qnn.op.quantize(out, y_scale, y_zero_point, out_dtype=dtype)


class QLinearLeakyRelu(OnnxOpConverter):
    """Operator converter for QLinearLeakyRelu from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):

        a_scale = get_scalar(inputs[1], params)
        a_zero_point = get_scalar(inputs[2], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        y_zero_point = get_scalar(inputs[4], params, "int32")
        alpha = float(attr.get("alpha", 1.0))

        dtype = infer_type(inputs[0]).checked_type.dtype

        # Onnxruntime doesn't actually do this op in integer, they dequantize to fp32
        # and then requantize afer (according to documentation below)
        # https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.QLinearLeakyRelu
        a = _qnn.op.dequantize(inputs[0], a_scale, a_zero_point)
        out = _op.nn.leaky_relu(a, alpha)
        return _qnn.op.quantize(out, y_scale, y_zero_point, out_dtype=dtype)


class QLinearSigmoid(OnnxOpConverter):
    """Operator converter for QLinearSigmoid from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        x = inputs[0]
        x_scale = get_scalar(inputs[1], params)
        x_zero_point = get_scalar(inputs[2], params, "int32")
        y_scale = fold_constant(get_scalar(inputs[3], params))
        y_zero_point = get_scalar(inputs[4], params, "int32")

        dtype = infer_type(x).checked_type.dtype

        ## Apparently, onnxruntime doesn't do this op in integer, they dequantize to fp32
        ## and then requantize after:
        ## https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/
        ## providers/dml/DmlExecutionProvider/src/GraphTransformer.cpp#L245
        x = _qnn.op.dequantize(x, x_scale, x_zero_point)
        out = _op.sigmoid(x)
        return _qnn.op.quantize(out, y_scale, y_zero_point, out_dtype=dtype)


class QLinearConcat(OnnxOpConverter):
    """Operator converter for QLinearConcat from Microsoft onnxruntime contrib opset."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        # which axis to concat on
        axis = attr["axis"]

        y_scale = fold_constant(get_scalar(inputs[0], params))
        y_zero_point = get_scalar(inputs[1], params, "int32")

        # input tensors, scales, zero_points
        assert (
            len(inputs) % 3 == 2
        ), "Additional input count must be a multiple of 3 -- tensor/scale/zero_point tuples"
        tensors = []
        scales = []
        zero_points = []
        for i in range(2, len(inputs), 3):
            tensors.append(inputs[i])
            scales.append(get_scalar(inputs[i + 1], params))
            zero_points.append(get_scalar(inputs[i + 2], params, "int32"))

        return _qnn.op.concatenate(tensors, scales, zero_points, y_scale, y_zero_point, axis)


class ConvInteger(OnnxOpConverter):
    """Operator converter for ConvInteger."""

    @classmethod
    def _impl_v10(cls, inputs, attr, params):
        data = inputs[0]
        weight = inputs[1]
        data_zp = inputs[2]
        weight_zp = inputs[3]
        if data_zp is None:
            data_zp = _expr.const(0, "int32")
        if weight_zp is None:
            weight_zp = _expr.const(0, "int32")

        input_type = infer_type(data)
        input_shape = get_const_tuple(input_type.checked_type.shape)

        ndim = len(input_shape)
        kernel_type = infer_type(weight)
        kernel_shape = get_const_tuple(kernel_type.checked_type.shape)
        if "kernel_shape" not in attr:
            attr["kernel_shape"] = kernel_shape[2:]

        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                # Warning: Convolution does not yet support dynamic shapes,
                # one will need to run dynamic_to_static on this model after import
                data = autopad(
                    data,
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    attr.get("dilations", [1] * (ndim - 2)),
                    pad_value=data_zp,
                    mode=attr["auto_pad"],
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = tuple([0 for i in range(ndim - 2)])
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = 'Value {} in attribute "auto_pad" of operator Conv is invalid.'
                raise tvm.error.OpAttributeInvalid(msg.format(attr["auto_pad"]))
            attr.pop("auto_pad")

        out_channels = kernel_shape[0]
        dilation = attr.get("dilations", [1] * (ndim - 2))
        strides = attr.get("strides", [1] * (ndim - 2))
        padding = attr["pads"] if "pads" in attr else 0
        groups = attr["group"] if "group" in attr else 1

        if ndim != 4:
            raise tvm.error.OpAttributeInvalid(
                "Only 2D kernels are supported for operator ConvInteger."
            )

        return _qnn.op.conv2d(
            data,
            weight,
            _op.cast(data_zp, "int32"),
            _op.cast(weight_zp, "int32"),
            _expr.const(1.0, "float32"),
            _expr.const(1.0, "float32"),
            kernel_size=attr["kernel_shape"],
            channels=out_channels,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )


class BitShift(OnnxOpConverter):
    """Operator converter for NonZero"""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        if len(inputs) != 2:
            raise ValueError("Bitshift expects 2 inputs")

        direction = attr.get("direction", "LEFT").decode("ascii")
        if direction == "LEFT":
            out = _op.left_shift(*inputs)
        elif direction == "RIGHT":
            out = _op.right_shift(*inputs)
        else:
            raise ValueError("Unsupported Shift Direction: " + direction)
        return out


class Unique(OnnxOpConverter):
    """Operator converter for unique"""

    @classmethod
    def _impl_v11(cls, inputs, attr, params):
        if len(inputs) != 1:
            raise ValueError("Unique expects 1 input")

        data = inputs[0]
        axis = attr.get("axis", None)
        if axis is None:  # If axis is None, flatten the input before calling unique
            data = _op.reshape(data, _op.const([-1]))
        else:
            data_shape = infer_shape(data)
            if len(data_shape) != 1:
                raise ValueError("TVM only supports 1D Unique operator.")
        is_sorted = attr.get("sorted", 1)  # sorted is 0 or 1, 1 by default

        # ONNX documentation lists return_counts as optional but there is no input to specify
        # whether it is returned. Therefore we'll just always return it.
        unique = _op.unique(data, is_sorted=(is_sorted == 1), return_counts=True)
        num_unique = unique[3]

        trim_unique_lambda = lambda input: _op.strided_slice(input, _op.const([0]), num_unique)

        unique_vals = trim_unique_lambda(unique[0])
        indices = _op.cast(trim_unique_lambda(unique[1]), "int64")  # ONNX always returns int64
        inverse_indices = _op.cast(unique[2], "int64")  # ONNX always returns int64
        counts = _op.cast(trim_unique_lambda(unique[4]), "int64")  # ONNX always returns int64
        # ONNX unique returns unique, indices, inverse_indices, (optional) counts
        return _expr.TupleWrapper(_expr.Tuple([unique_vals, indices, inverse_indices, counts]), 4)


class Einsum(OnnxOpConverter):
    """Operator converter for Einsum"""

    @classmethod
    def _impl_v12(cls, inputs, attr, params):
        equation = attr["equation"].decode("utf-8")
        return _op.einsum(inputs, equation)


class RandomNormal(OnnxOpConverter):
    """Operator converter for random_normal"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        dtype = get_type(attr.get("dtype", 1))
        mean = attr.get("mean", 0.0)
        scale = attr.get("scale", 1.0)
        seed = attr.get("seed", None)
        shape = attr["shape"]

        assert dtype in [
            "float32",
            "float64",
        ], "Only float random value generation is currently supported."

        if seed is None:
            seed = np.random.randint(1e6)
        else:
            seed = int(seed)
        key = _random.threefry_key(seed)
        output = _op.random.normal(key, shape, dtype=dtype, mean=mean, scale=scale)
        _, vals = _expr.TupleWrapper(output, 2)
        return vals


class RandomNormalLike(OnnxOpConverter):
    """Operator converter for random_normal_like"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        dtype = attr.get("dtype", None)
        scale = attr.get("scale", 1.0)
        mean = attr.get("mean", 0.0)
        seed = attr.get("seed", None)
        shape = infer_shape(inputs[0])
        if dtype is None:
            dtype = infer_type(inputs[0]).checked_type.dtype
        else:
            dtype = get_type(dtype)

        assert dtype in [
            "float32",
            "float64",
        ], "Only float random value generation is currently supported."

        if seed is None:
            seed = np.random.randint(1e6)
        else:
            seed = int(seed)
        key = _random.threefry_key(seed)
        output = _op.random.normal(key, shape, dtype=dtype, mean=mean, scale=scale)
        _, vals = _expr.TupleWrapper(output, 2)
        return vals


class RandomUniform(OnnxOpConverter):
    """Operator converter for random_uniform"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        dtype = get_type(attr.get("dtype", 1))
        high = attr.get("high", 1.0)
        low = attr.get("low", 0.0)
        seed = attr.get("seed", None)
        shape = attr["shape"]

        assert dtype in [
            "float32",
            "float64",
        ], "Only float random value generation is currently supported."

        if seed is None:
            seed = np.random.randint(1e6)
        else:
            seed = int(seed)
        key = _random.threefry_key(seed)
        output = _op.random.uniform(key, shape, dtype=dtype, low=low, high=high)
        _, vals = _expr.TupleWrapper(output, 2)
        return vals


class RandomUniformLike(OnnxOpConverter):
    """Operator converter for random_uniform_like"""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        dtype = attr.get("dtype", None)
        high = attr.get("high", 1.0)
        low = attr.get("low", 0.0)
        seed = attr.get("seed", None)
        shape = infer_shape(inputs[0])
        if dtype is None:
            dtype = infer_type(inputs[0]).checked_type.dtype
        else:
            dtype = get_type(dtype)

        assert dtype in [
            "float32",
            "float64",
        ], "Only float random value generation is currently supported."

        if seed is None:
            seed = np.random.randint(1e6)
        else:
            seed = int(seed)
        key = _random.threefry_key(seed)
        output = _op.random.uniform(key, shape, dtype=dtype, low=low, high=high)
        _, vals = _expr.TupleWrapper(output, 2)
        return vals


class NegativeLogLikelihoodLoss(OnnxOpConverter):
    """Operator converter for NegativeLogLikehoodLoss"""

    VALID_REDUCTIONS = {"mean", "sum", "none"}

    @classmethod
    def run_calculation(
        cls: "NegativeLogLikelihoodLoss",
        input_tensor: relay.Expr,
        target_tensor: relay.Expr,
        weight_tensor: Optional[relay.Expr],
        ignore_index: int,
    ):
        """Run calculation for NegativeLogLikelihood, returning output tensor and
        weight tensor used for mean-style reductions.
        """
        # Convert negative indices --> positive indices for gather ops, note we have to
        # use the original target tensor to interact with ignore_index to have proper behavior.
        normalized_target_tensor = normalize_gather_indices(input_tensor, target_tensor, 1)

        if weight_tensor is None:
            channels = infer_shape(input_tensor)[1]
            weight_tensor = relay.ones(
                [channels],
                dtype=infer_type(input_tensor).checked_type.dtype,
            )

        loss = -relay.gather(
            input_tensor,
            axis=1,
            indices=relay.expand_dims(normalized_target_tensor, 1),
        )
        loss = relay.squeeze(loss, axis=[1])

        expanded_normalized_target_tensor = relay.expand_dims(normalized_target_tensor, 0)
        expanded_normalized_target_tensor = relay.nn.batch_flatten(
            expanded_normalized_target_tensor
        )
        flattened_weights = relay.gather_nd(weight_tensor, expanded_normalized_target_tensor)
        select_weights = relay.reshape_like(flattened_weights, loss)
        loss *= select_weights

        if ignore_index is not None:
            # "Ignore" values whose target is the ignore_index
            mask_tensor = relay.equal(
                target_tensor, relay.const(ignore_index, dtype=target_tensor.type_annotation.dtype)
            )
            mask_tensor = relay.const(1, dtype="int8") - relay.cast(mask_tensor, "int8")
            loss = relay.where(
                mask_tensor, loss, relay.const(0, infer_type(loss).checked_type.dtype)
            )

            # This is not explained super clearly in the onnx spec, but masked values don't
            # contribute toward the final value in reduction
            select_weights *= relay.cast_like(mask_tensor, select_weights)

        weight_total = relay.sum(select_weights)
        return loss, weight_total

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        ignore_index = attr.get("ignore_index", None)
        reduction = attr.get("reduction", b"mean").decode("utf-8")

        if reduction not in cls.VALID_REDUCTIONS:
            raise ValueError(
                f"Unknown reduction type {reduction}, choices are {cls.VALID_REDUCTIONS}"
            )

        input_tensor, target_tensor = inputs[0], inputs[1]
        if len(inputs) == 3:
            weight_tensor = inputs[2]
        else:
            weight_tensor = None

        loss, weight_total = cls.run_calculation(
            input_tensor,
            target_tensor,
            weight_tensor=weight_tensor,
            ignore_index=ignore_index,
        )
        if reduction == "mean":
            return relay.sum(loss) / weight_total
        if reduction == "sum":
            return relay.sum(loss)
        # Case reduction == 'none'
        return loss


class SoftmaxCrossEntropyLoss(OnnxOpConverter):
    """Operator converter for SCE_loss"""

    @classmethod
    def _impl_v13(cls, inputs, attr, params):
        ignore_index = attr.get("ignore_index", None)
        reduction = attr.get("reduction", b"mean").decode("utf-8")
        input_tensor, target_tensor = inputs[0], inputs[1]
        if len(inputs) == 3:
            weight_tensor = inputs[2]
        else:
            weight_tensor = None

        get_log_prob = attr["tvm_custom"]["num_outputs"] == 2
        log_softmax_attr = {"axis": 1}
        log_softmax_tensor = LogSoftmax.get_converter(13)([input_tensor], log_softmax_attr, None)

        loss, weight_total = NegativeLogLikelihoodLoss.run_calculation(
            log_softmax_tensor,
            target_tensor,
            weight_tensor,
            ignore_index=ignore_index,
        )

        if reduction == "mean":
            loss = relay.sum(loss) / weight_total
        elif reduction == "sum":
            loss = relay.sum(loss)

        if get_log_prob:
            return relay.TupleWrapper(relay.Tuple((loss, log_softmax_tensor)), 2)
        return loss


class Adagrad(OnnxOpConverter):
    """Operator converter for adagrad op."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        decay_factor = attr.get("decay_factor", 0.0)
        epsilon = attr.get("epsilon", 0.0)
        norm_coefficient = attr.get("norm_coefficient", 0.0)

        R = inputs[0]
        T = inputs[1]

        # convert attributes to constants, proper types
        dtype_inputs = infer_type(inputs[3]).checked_type.dtype
        decay_factor = relay.const(decay_factor, dtype=dtype_inputs)
        epsilon = relay.const(epsilon, dtype=dtype_inputs)
        norm_coefficient = relay.const(norm_coefficient, dtype=dtype_inputs)
        T = relay.cast_like(T, inputs[3])

        assert (
            len(inputs) - 2
        ) % 3 == 0, f"Expect triplets for remaining inputs, found {len(inputs) - 2}"

        # Remaining inputs are:
        # [x_1, x_2 ..., x_1_gradient, x_2_gradient, ... x_1_sq_g, x_2_sq_g...]
        num_input_tensors = (len(inputs) - 2) // 3
        output_tensors = []
        output_accumulated_squared_gradients = []
        for i in range(num_input_tensors):
            x = inputs[i + 2]
            gradient = inputs[i + 2 + num_input_tensors]
            accumulated_squared_gradient = inputs[i + 2 + 2 * num_input_tensors]

            r = R / (relay.const(1.0, dtype=dtype_inputs) + T * decay_factor)
            g_regularized = norm_coefficient * x + gradient
            new_accumulated_squared_gradient = (
                accumulated_squared_gradient + g_regularized * g_regularized
            )
            h_adaptive = relay.sqrt(new_accumulated_squared_gradient) + epsilon

            x_new = x - r * g_regularized / h_adaptive

            output_tensors.append(x_new)
            output_accumulated_squared_gradients.append(new_accumulated_squared_gradient)

        # append lists together, momentums come after result tensors
        result = output_tensors + output_accumulated_squared_gradients
        return _expr.TupleWrapper(_expr.Tuple(result), len(result))


class Adam(OnnxOpConverter):
    """Operator converter for Adam op."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = attr.get("alpha", 0.9)
        beta = attr.get("beta", 0.999)

        # Note in the docs epsilon default is 0.0 but in the tests it is set to 1e-2:
        # https://git.io/Ju5C4
        epsilon = attr.get("epsilon", 1e-2)
        norm_coefficient = attr.get("norm_coefficient", 0.0)
        norm_coefficient_post = attr.get("norm_coefficient_post", 0.0)

        R = inputs[0]
        T = inputs[1]

        assert (
            len(inputs) - 2
        ) % 4 == 0, f"Expect 4-lets for remaining inputs, found {len(inputs) - 2}"

        # convert attributes to constants, proper types
        dtype_inputs = infer_type(inputs[3]).checked_type.dtype
        inverse_alpha = relay.const(1 - alpha, dtype=dtype_inputs)
        alpha = relay.const(alpha, dtype=dtype_inputs)
        inverse_beta = relay.const(1 - beta, dtype=dtype_inputs)
        beta = relay.const(beta, dtype=dtype_inputs)
        epsilon = relay.const(epsilon, dtype=dtype_inputs)
        norm_coefficient = relay.const(norm_coefficient, dtype=dtype_inputs)
        norm_coefficient_post = relay.const(norm_coefficient_post, dtype=dtype_inputs)
        one = relay.const(1, dtype=dtype_inputs)
        T = relay.cast_like(T, inputs[3])

        # Remaining inputs are:
        # [x_1, x_2 ..., x_1_grad, x_2_grad, ... x_1_g_accum, x_2_g_accum..., x_1_g_sq_accum, ...]
        num_input_tensors = (len(inputs) - 2) // 4
        output_tensors = []
        output_accumulated_gradients = []
        output_accumulated_squared_gradients = []
        for i in range(num_input_tensors):
            x = inputs[i + 2]
            g = inputs[i + 2 + num_input_tensors]
            v = inputs[i + 2 + 2 * num_input_tensors]
            h = inputs[i + 2 + 3 * num_input_tensors]

            g_regularized = norm_coefficient * x + g
            v_new = alpha * v + inverse_alpha * g_regularized
            h_new = beta * h + inverse_beta * g_regularized * g_regularized
            h_sqrt = relay.sqrt(h_new) + epsilon

            true_branch = R * relay.sqrt(one - relay.power(beta, T)) / (one - relay.power(alpha, T))
            R_adjusted = relay.If(T > relay.const(0, dtype=dtype_inputs), true_branch, R)

            x_new = x - R_adjusted * (v_new / h_sqrt)
            x_result = (one - norm_coefficient_post) * x_new

            output_tensors.append(x_result)
            output_accumulated_gradients.append(v_new)
            output_accumulated_squared_gradients.append(h_new)

        # append lists together to get final result
        result = (
            output_tensors + output_accumulated_gradients + output_accumulated_squared_gradients
        )
        return _expr.TupleWrapper(_expr.Tuple(result), len(result))


class Momentum(OnnxOpConverter):
    """Operator converter for Momentum op."""

    @classmethod
    def _impl_v1(cls, inputs, attr, params):
        alpha = attr["alpha"]
        beta = attr["beta"]
        mode = attr["mode"].decode("utf-8")
        norm_coefficient = attr["norm_coefficient"]

        assert mode in ["nesterov", "standard"], f"Unknown momentum mode {mode}"
        R = inputs[0]
        T = inputs[1]

        assert (
            len(inputs) - 2
        ) % 3 == 0, f"Expect triplets for remaining inputs, found {len(inputs) - 2}"
        # Remaining inputs are:
        # [x_1, x_2 ..., x_1_gradient, x_2_gradient, ... x_1_momentum, x_2_momentum...]
        num_input_tensors = (len(inputs) - 2) // 3

        # convert attributes to constants
        dtype_inputs = infer_type(inputs[3]).checked_type.dtype
        alpha = relay.const(alpha, dtype=dtype_inputs)
        beta = relay.const(beta, dtype=dtype_inputs)
        norm_coefficient = relay.const(norm_coefficient, dtype=dtype_inputs)
        default_beta = relay.const(1.0, dtype=dtype_inputs)

        # Calculate updated values for every input
        output_tensors = []
        output_momentums = []
        for i in range(num_input_tensors):
            x = inputs[i + 2]
            gradient = inputs[i + 2 + num_input_tensors]
            momentum = inputs[i + 2 + 2 * num_input_tensors]
            g_regularized = norm_coefficient * x + gradient
            beta_adjusted = relay.If(T > relay.const(0, dtype="int64"), beta, default_beta)
            new_momentum = alpha * momentum + beta_adjusted * g_regularized

            if mode == "standard":
                x_output = x - R * new_momentum
            else:
                # mode == 'nesterov'
                x_output = x - R * (g_regularized + alpha * new_momentum)

            output_tensors.append(x_output)
            output_momentums.append(new_momentum)

        # append lists together, momentums come after result tensors
        result = output_tensors + output_momentums
        return _expr.TupleWrapper(_expr.Tuple(result), len(result))


# compatible operators that do NOT require any conversion.
_identity_list = []


# _convert_map defines maps of name to converter functor(callable)
# for 1 to 1 mapping, use Renamer if nothing but name is different
# use AttrCvt if attributes need to be converted
# for 1 to N mapping(composed), use custom callable functions
# for N to 1 mapping, currently not supported(?)
def _get_convert_map(opset):
    return {
        # defs/experimental
        "Identity": Renamer("copy"),
        "Affine": Affine.get_converter(opset),
        "BitShift": BitShift.get_converter(opset),
        "ThresholdedRelu": ThresholdedRelu.get_converter(opset),
        "ScaledTanh": ScaledTanh.get_converter(opset),
        "ParametricSoftplus": ParametricSoftPlus.get_converter(opset),
        "Constant": Constant.get_converter(opset),
        "ConstantOfShape": ConstantOfShape.get_converter(opset),
        # 'GivenTensorFill'
        "FC": AttrCvt("dense", ignores=["axis", "axis_w"]),
        "Scale": Scale.get_converter(opset),
        # 'GRUUnit'
        # 'ATen'
        # 'ImageScaler'
        # 'MeanVarianceNormalization'
        # 'Crop'
        # 'Embedding'
        "Upsample": Upsample.get_converter(opset),
        "SpatialBN": BatchNorm.get_converter(opset),
        # defs/generator
        # 'Constant' # Implemented
        # 'RandomUniform'
        # 'RandomNormal'
        # 'RandomUniformLike'
        # 'RandomNormalLike'
        # defs/logical
        # defs/math
        "Add": Add.get_converter(opset),
        "Sub": Sub.get_converter(opset),
        "Mul": Mul.get_converter(opset),
        "Div": Div.get_converter(opset),
        "Neg": Renamer("negative"),
        "Abs": Absolute.get_converter(opset),
        "Reciprocal": Reciprocal.get_converter(opset),
        "Floor": Renamer("floor"),
        "Ceil": Renamer("ceil"),
        "Round": Renamer("round"),
        "IsInf": IsInf.get_converter(opset),
        "IsNaN": Renamer("isnan"),
        "Sqrt": Renamer("sqrt"),
        "Relu": Renamer("relu"),
        "Celu": Celu.get_converter(opset),
        "LeakyRelu": Renamer("leaky_relu"),
        "Selu": Selu.get_converter(opset),
        "Elu": Elu.get_converter(opset),
        "Gelu": Gelu.get_converter(opset),
        "BiasGelu": BiasGelu.get_converter(opset),
        # TODO: We need a better way to handle different domains, in case
        # of name collisions. EmbedLayerNormalization, SkipLayerNormalization, and Attention
        # are in the `com.microsoft` domain.
        "EmbedLayerNormalization": EmbedLayerNormalization.get_converter(opset),
        "SkipLayerNormalization": SkipLayerNormalization.get_converter(opset),
        "Attention": Attention.get_converter(opset),
        "Exp": Renamer("exp"),
        "Greater": Renamer("greater"),
        "GreaterOrEqual": Renamer("greater_equal"),
        "Less": Renamer("less"),
        "LessOrEqual": Renamer("less_equal"),
        "Log": Renamer("log"),
        "Acos": Renamer("acos"),
        "Acosh": Renamer("acosh"),
        "Asin": Renamer("asin"),
        "Asinh": Renamer("asinh"),
        "Atan": Renamer("atan"),
        "Atanh": Renamer("atanh"),
        "Cos": Renamer("cos"),
        "Cosh": Renamer("cosh"),
        "Sin": Renamer("sin"),
        "Sinh": Renamer("sinh"),
        "Tan": Renamer("tan"),
        "Tanh": Renamer("tanh"),
        "Pow": Pow.get_converter(opset),
        "PRelu": Prelu.get_converter(opset),
        "Sigmoid": Renamer("sigmoid"),
        "HardSigmoid": HardSigmoid.get_converter(opset),
        "HardSwish": HardSwish.get_converter(opset),
        "Max": Maximum.get_converter(opset),
        "Min": Minimum.get_converter(opset),
        "Sum": Sum.get_converter(opset),
        "Mean": Mean.get_converter(opset),
        "Clip": Clip.get_converter(opset),
        "Softplus": Softplus.get_converter(opset),
        # softmax default axis is different in onnx
        "Softmax": Softmax.get_converter(opset),
        "LogSoftmax": LogSoftmax.get_converter(opset),
        "OneHot": OneHot.get_converter(opset),
        "Hardmax": Hardmax.get_converter(opset),
        "Shrink": Shrink.get_converter(opset),
        "Softsign": Softsign.get_converter(opset),
        "Gemm": Gemm.get_converter(opset),
        "MatMul": MatMul.get_converter(opset),
        "MatMulInteger": MatMulInteger.get_converter(opset),
        "MatMulInteger16": MatMulInteger16.get_converter(opset),
        "Mod": Mod.get_converter(opset),
        "Xor": Renamer("logical_xor"),
        # defs/nn
        "AveragePool": AveragePool.get_converter(opset),
        "LpPool": LpPool.get_converter(opset),
        "GlobalLpPool": GlobalLpPool.get_converter(opset),
        "MaxPool": MaxPool.get_converter(opset),
        "MaxUnpool": MaxUnpool.get_converter(opset),
        "Conv": Conv.get_converter(opset),
        "ConvTranspose": ConvTranspose.get_converter(opset),
        "GlobalAveragePool": GlobalAveragePool.get_converter(opset),
        "GlobalMaxPool": GlobalMaxPool.get_converter(opset),
        "BatchNormalization": BatchNorm.get_converter(opset),
        "InstanceNormalization": InstanceNorm.get_converter(opset),
        # 'LpNormalization'
        "Dropout": AttrCvt("dropout", {"ratio": "rate"}, ignores=["is_test"]),
        "Flatten": Flatten.get_converter(opset),
        "LRN": LRN.get_converter(opset),
        # Recurrent Layers
        "LSTM": LSTM.get_converter(opset),
        "GRU": GRU.get_converter(opset),
        # defs/vision
        "MaxRoiPool": MaxRoiPool.get_converter(opset),
        "RoiAlign": RoiAlign.get_converter(opset),
        "NonMaxSuppression": NonMaxSuppression.get_converter(opset),
        # defs/reduction
        "ReduceMax": ReduceMax.get_converter(opset),
        "ReduceMin": ReduceMin.get_converter(opset),
        "ReduceSum": ReduceSum.get_converter(opset),
        "ReduceMean": ReduceMean.get_converter(opset),
        "ReduceProd": ReduceProd.get_converter(opset),
        "ReduceLogSumExp": ReduceLogSumExp.get_converter(opset),
        "ReduceLogSum": ReduceLogSum.get_converter(opset),
        "ReduceSumSquare": ReduceSumSquare.get_converter(opset),
        "ReduceL1": ReduceL1.get_converter(opset),
        "ReduceL2": ReduceL2.get_converter(opset),
        # defs/sorting
        "ArgMax": ArgMax.get_converter(opset),
        "ArgMin": ArgMin.get_converter(opset),
        "TopK": TopK.get_converter(opset),
        # defs/tensor
        "Cast": Cast.get_converter(opset),
        "Reshape": Reshape.get_converter(opset),
        "Expand": Expand.get_converter(opset),
        "Concat": Concat.get_converter(opset),
        "Split": Split.get_converter(opset),
        "Slice": Slice.get_converter(opset),
        "Transpose": AttrCvt("transpose", {"perm": "axes"}),
        "DepthToSpace": DepthToSpace.get_converter(opset),
        "SpaceToDepth": SpaceToDepth.get_converter(opset),
        "Gather": Gather.get_converter(opset),
        "GatherElements": GatherElements.get_converter(opset),
        "GatherND": GatherND.get_converter(opset),
        "Compress": Compress.get_converter(opset),
        "Size": AttrCvt("ndarray_size", extras={"dtype": "int64"}),
        "Scatter": Scatter.get_converter(opset),
        "ScatterElements": Scatter.get_converter(opset),
        "ScatterND": ScatterND.get_converter(opset),
        "EyeLike": EyeLike.get_converter(opset),
        "Squeeze": Squeeze.get_converter(opset),
        "Unsqueeze": Unsqueeze.get_converter(opset),
        "Pad": Pad.get_converter(opset),
        "Shape": Shape.get_converter(opset),
        "Sign": Sign.get_converter(opset),
        "Equal": Equal.get_converter(opset),
        "Not": Not.get_converter(opset),
        "And": And.get_converter(opset),
        "Tile": Tile.get_converter(opset),
        "Erf": Erf.get_converter(opset),
        "Where": Where.get_converter(opset),
        "Or": Or.get_converter(opset),
        "Resize": Resize.get_converter(opset),
        "NonZero": NonZero.get_converter(opset),
        "Range": Range.get_converter(opset),
        "CumSum": CumSum.get_converter(opset),
        "Unique": Unique.get_converter(opset),
        "Einsum": Einsum.get_converter(opset),
        # defs/control_flow
        "Loop": Loop.get_converter(opset),
        "If": If.get_converter(opset),
        # Torch ATen Dispatcher.
        "ATen": ATen.get_converter(opset),
        # Quantization
        "QuantizeLinear": QuantizeLinear.get_converter(opset),
        "DequantizeLinear": DequantizeLinear.get_converter(opset),
        "DynamicQuantizeLinear": DynamicQuantizeLinear.get_converter(opset),
        "ReverseSequence": ReverseSequence.get_converter(opset),
        "QLinearConv": QLinearConv.get_converter(opset),
        "QLinearConcat": QLinearConcat.get_converter(opset),
        "QLinearAdd": QLinearAdd.get_converter(opset),
        "QLinearMatMul": QLinearMatMul.get_converter(opset),
        "QLinearMul": QLinearMul.get_converter(opset),
        "QLinearSigmoid": QLinearSigmoid.get_converter(opset),
        "ConvInteger": ConvInteger.get_converter(opset),
        "QLinearAveragePool": QLinearAveragePool.get_converter(opset),
        "QLinearGlobalAveragePool": QLinearGlobalAveragePool.get_converter(opset),
        "QLinearLeakyRelu": QLinearLeakyRelu.get_converter(opset),
        # Random number generation.
        "RandomNormal": RandomNormal.get_converter(opset),
        "RandomNormalLike": RandomNormalLike.get_converter(opset),
        "RandomUniform": RandomUniform.get_converter(opset),
        "RandomUniformLike": RandomUniformLike.get_converter(opset),
        # Loss functions / training
        "NegativeLogLikelihoodLoss": NegativeLogLikelihoodLoss.get_converter(opset),
        "SoftmaxCrossEntropyLoss": SoftmaxCrossEntropyLoss.get_converter(opset),
        "Adagrad": Adagrad.get_converter(opset),
        "Adam": Adam.get_converter(opset),
        "Momentum": Momentum.get_converter(opset),
        "Scan": Scan.get_converter(opset),
        # ML
        "LinearRegressor": LinearRegressor.get_converter(opset),
    }


class GraphProto:
    """A helper class for handling Relay expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

        Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.

    """

    current = None

    def __init__(self, shape, dtype, freeze_params=False):
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._renames = {}
        self._num_input = 0
        self._num_param = 0
        self._shape = shape.copy() if shape else {}
        self._input_names = []
        self._dtype = dtype
        self.opset = None
        self._freeze_params = freeze_params

    def __enter__(self):
        self._old_manager = GraphProto.current
        GraphProto.current = self
        return self

    def __exit__(self, ptype, value, trace):
        GraphProto.current = self._old_manager

    def freeze(self, func, params):
        bind_map = {}
        for name in params.keys():
            if name in self._nodes.keys():
                bind_map[self._nodes[name]] = _expr.const(params[name])
        body = _expr.bind(func.body, bind_map)
        fn = _function.Function(analysis.free_vars(body), body)
        return fn, {}

    def from_onnx(self, graph, opset, get_output_expr=False):
        """Construct Relay expression from ONNX graph.

        Onnx graph is a python protobuf object.
        The companion parameters will be handled automatically.
        However, the input names from onnx graph is vague, mixing inputs and
        network weights/bias such as "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        opset : opset version

        get_output_expr: bool
            If set to true, this conversion will return each output expression rather
            than a packaged module. This can be useful when converting subgraphs to
            relay.

        Returns
        -------
        mod : tvm.IRModule
            The returned relay module

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        self.opset = opset
        self._parse_graph_initializers(graph)
        self._parse_graph_input(graph)
        self._check_user_inputs_in_outermost_graph_scope()
        self._check_for_unsupported_ops(graph)
        self._construct_nodes(graph)

        # now return the outputs
        outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        # If requested, directly return the converted expressions.
        if get_output_expr:
            return outputs
        ## Maintain the order of inputs and parameters from the ONNX graph, but only include
        ## those parameters that are needed to execute the relay graph
        free_vars = analysis.free_vars(outputs)
        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params:
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]
        # Create a function from our output expression and all input variables.
        func = _function.Function([v for k, v in self._inputs.items()], outputs)
        return IRModule.from_expr(func), self._params

    def _parse_graph_initializers(self, graph):
        """Parse network inputs to relay, aka parameters."""
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            array = self._parse_array(init_tensor)
            if self._freeze_params:
                self._nodes[init_tensor.name] = _expr.const(array)
            else:
                self._params[init_tensor.name] = array
                self._nodes[init_tensor.name] = new_var(
                    init_tensor.name,
                    shape=self._params[init_tensor.name].shape,
                    dtype=self._params[init_tensor.name].dtype,
                )

    def _parse_graph_input(self, graph):
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name in self._params:
                # i is a param instead of input
                self._num_param += 1
                self._nodes[i_name] = new_var(
                    i_name, shape=self._params[i_name].shape, dtype=self._params[i_name].dtype
                )
            elif i_name in self._nodes:
                continue
            else:
                self._num_input += 1
                self._input_names.append(i_name)
                if i_name in self._shape:
                    i_shape = self._shape[i_name]
                else:
                    if "?" in str(i_shape):
                        warning_msg = (
                            "Input %s has unknown dimension shapes: %s. "
                            "Specifying static values may improve performance"
                            % (i_name, str(i_shape_name))
                        )
                        warnings.warn(warning_msg)
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=i_shape, dtype=dtype)
            self._inputs[i_name] = self._nodes[i_name]

    def _check_user_inputs_in_outermost_graph_scope(self):
        """Only check user inputs in the outer-most graph scope."""
        if self._old_manager is None:
            assert all(
                [name in self._input_names for name in self._shape.keys()]
            ), "User specified the shape for inputs that weren't found in the graph: " + str(
                self._shape
            )

    def _check_for_unsupported_ops(self, graph):
        convert_map = _get_convert_map(self.opset)
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = "The following operators are not supported for frontend ONNX: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

    def _construct_nodes(self, graph):
        """Nodes are stored as directed acyclic graph."""
        for node in graph.node:
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            # Create and populate input list.
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    inputs.append(self._nodes[self._renames.get(i, i)])
                else:
                    inputs.append(None)
            i_name = self._parse_value_proto(node)
            node_output = self._fix_outputs(op_name, node.output)
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(node_output)

            op = self._convert_operator(op_name, inputs, attr, self.opset)
            if not isinstance(op, _expr.TupleWrapper):
                outputs_num = 1
            else:
                outputs_num = len(op)

            if outputs_num == 1:
                op = fold_constant(op)
            else:
                op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))

            if outputs_num > 1:
                # ONNX supports optional outputs for some nodes.
                # This block searches for missing outputs in the ONNX graph
                # and removes any unneeded ops
                valid_outputs = [False] * outputs_num
                for i, output in enumerate(node_output):
                    if output != "":
                        valid_outputs[i] = True
                # If we have outputs ONNX isn't expecting, we need to drop them
                if not all(valid_outputs):
                    tup = op.astuple()
                    # TupleWrapper can also wrap ops with TupleType outputs
                    if isinstance(tup, _expr.Tuple):
                        # For tuples, we extract the fields instead of using GetTupleItem
                        outputs = [tup.fields[i] for i, valid in enumerate(valid_outputs) if valid]
                    else:
                        # For call nodes, we need to GetTupleItem
                        outputs = [op[i] for i, valid in enumerate(valid_outputs) if valid]
                    # Create the new op with valid outputs
                    if len(outputs) == 1:
                        op = outputs[0]
                    elif len(outputs) != outputs_num:
                        op = _expr.TupleWrapper(_expr.Tuple(outputs), len(outputs))
                    # Drop invalid outputs for the onnx node
                    outputs_num = len(outputs)
                    node_output = [output for output in node_output if output != ""]
            assert (
                len(node_output) == outputs_num
            ), "Number of output mismatch {} vs {} in {}.".format(
                len(node_output), outputs_num, op_name
            )

            if outputs_num == 1:
                self._nodes[node_output[0]] = op
            else:
                for k, i in zip(list(node_output), range(len(node_output))):
                    self._nodes[k] = op[i]

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

    def _parse_array(self, tensor_proto):
        np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
        return _nd.array(np_array)

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ["f", "i", "s", "g"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["floats", "ints", "strings"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["t"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["tensors"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["graphs"]:
                if list(getattr(a, f)):
                    raise NotImplementedError("Field {} is not supported in relay.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _convert_operator(self, op_name, inputs, attrs, opset):
        """Convert ONNX operator into a Relay operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relay.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        convert_map = _get_convert_map(opset)
        if op_name in _identity_list:
            sym = get_relay_op(op_name)(*inputs, **attrs)
        elif op_name in convert_map:
            sym = convert_map[op_name](inputs, attrs, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym

    def _fix_outputs(self, op_name, outputs):
        """A hack to handle dropout or similar operator that have more than one out
        in ONNX.
        """
        if op_name == "Dropout":
            if len(outputs) == 1:
                return outputs
            # TODO(zhreshold): support dropout mask?
            outputs = outputs[:-1]
        return outputs


def from_onnx(
    model, shape=None, dtype="float32", opset=None, freeze_params=True, convert_config=None
):
    """Convert a ONNX model into an equivalent Relay Function.

    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...

    By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
    retains that dynamism upon import, and the compiler attempts to convert the
    model into a static shapes at compile time. If this fails, there may still
    be dynamic operations in the model. Not all TVM kernels currently support
    dynamic shapes, please file an issue on discuss.tvm.apache.org
    if you hit an error with dynamic kernels.

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0

    shape : dict of str to tuple, optional
        The input shape to the graph

    dtype : str or dict of str to str
        The input types to the graph

    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.

    freeze_params: bool
        If this parameter is true, the importer will take any provided
        onnx input values (weights, shapes, etc) and embed them into the relay model
        as Constants instead of variables. This allows more aggressive optimizations
        at compile time and helps in making models static if certain inputs represent
        attributes relay would traditionally consider compile-time constants.

    convert_config : Optional[Dict[str, Any]]
        Default config:
            use_nt_batch_matmul : bool = True
                True to convert qualified onnx `matmul` to `nn.batch_matmul` strict to NT format
                (transpose_a=False, transpose_b=True).

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    global ONNX_DEFAULT_CONFIGS
    if convert_config is not None:
        ONNX_DEFAULT_CONFIGS.update(convert_config)

    try:
        import onnx

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except Exception as e:  # pylint: disable=c-extension-no-member, broad-except
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass
    g = GraphProto(shape, dtype, freeze_params)
    graph = model.graph

    try:
        opset_in_model = model.opset_import[0].version if model.opset_import else 1
    except AttributeError:
        opset_in_model = 1

    if opset is None:
        opset = opset_in_model
    elif opset < opset_in_model:
        warnings.warn(
            ""
            f"You are overwritting original opset ver = {opset_in_model} by lower ver = {opset}. "
            f"That might cause model conversion errors."
        )

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    with g:
        mod, params = g.from_onnx(graph, opset)

    if freeze_params:
        mod = relay.transform.DynamicToStatic()(mod)

    return mod, params
