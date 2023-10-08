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
# pylint: disable=import-outside-toplevel, used-before-assignment, use-implicit-booleaness-not-comparison
"""OneFlow: OneFlow is a performance-centered and open-source deep learning framework."""

import os
import re
import copy
from collections import OrderedDict

import numpy as np
import tvm
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import ty as _ty
from .common import AttrCvt, Renamer, fold_constant, get_relay_op, infer_shape, infer_type, new_var

__all__ = ["from_oneflow"]

FLOW_2_STR_DTYPE = {
    2: "float32",
    3: "float64",
    6: "int64",
    5: "int32",
    4: "int8",
    7: "uint8",
    9: "float16",
}


def is_input_op(node):
    """Return true when the node is the input of the graph."""
    return node.WhichOneof("op_type") == "input_conf"


def is_user_op(node):
    """Return true when the node is the intermediate variables of graph."""
    return node.WhichOneof("op_type") == "user_conf"


def is_output_op(node):
    """Return true when the node is the output of the graph."""
    return node.WhichOneof("op_type") == "output_conf"


def is_param_op(node):
    """Return true when the node is the intermediate variables of model(saved)."""
    return node.WhichOneof("op_type") == "variable_conf"


def get_node_info(node):
    """
    Get basic information about nodes: shape, data_type
    """
    # list->tuple
    shape = tuple(node.input_conf.blob_conf.shape.dim)
    # get data type
    dtype = node.input_conf.blob_conf.data_type
    if dtype in list(FLOW_2_STR_DTYPE.keys()):
        data_type = FLOW_2_STR_DTYPE[dtype]
    else:
        raise IndexError(f"Please check the data type of your node: {node.name}")

    return shape, data_type


def _dtype_shape_promotion(inputs):
    """Promote data type and shape for list of tensors."""

    dtype_order = ["bool", "int8", "int16", "int32", "int64", "float32", "float64"]
    ranks = [len(infer_shape(x)) for x in inputs]
    if set(ranks) == set([1, 0]):
        for i, r in enumerate(ranks):
            if r == 0:
                inputs[i] = _op.expand_dims(inputs[i], axis=0)

    dtypes = set(dtype_order.index(infer_type(x).checked_type.dtype) for x in inputs)
    if len(dtypes) == 1:
        return inputs
    max_dtype = dtype_order[max(dtypes)]
    for i, input_op in enumerate(inputs):
        if infer_type(input_op).checked_type.dtype != max_dtype:
            inputs[i] = input_op.astype(max_dtype)
    return inputs


def parse_attr(attr):
    """Parse attribute of user op in oneflow."""
    attrs = {}
    for a in attr:
        attr_str = str(attr[a])

        if attr_str[0:7] == "at_list":
            attr_str_ = attr_str.split(" ")[0]

            if attr_str_ == "at_list_float":
                attrs[a] = tuple(attr[a].at_list_float.val)
            elif attr_str_ == "at_list_int32":
                attrs[a] = tuple(attr[a].at_list_int32.val)
            elif attr_str_ == "at_list_int64":
                attrs[a] = tuple(attr[a].at_list_int64.val)

        elif attr_str.split(":")[0] == "at_string":
            attrs[a] = attr[a].at_string

        elif attr_str.split(" ")[0] == "at_shape":
            attrs[a] = tuple(list(attr[a].at_shape.dim))

        else:
            attr_str_ = attr_str.split(":")[0]
            if attr_str_ == "at_bool":
                attrs[a] = attr[a].at_bool
            elif attr_str_ == "at_double":
                attrs[a] = attr[a].at_double
            elif attr_str_ == "at_float":
                attrs[a] = attr[a].at_float
            elif attr_str_ == "at_int32":
                attrs[a] = attr[a].at_int32
            elif attr_str_ == "at_int64":
                attrs[a] = attr[a].at_int64

    return attrs


def shape_of(x, dtype="int64"):
    ttype = infer_type(x).checked_type
    if not _ty.is_dynamic(ttype):
        shape = list(ttype.shape)
        return _expr.const(shape, dtype)

    return _op.shape_of(x, dtype)


def dimension_constraint():
    def _dim_check(attrs):
        if len(attrs["kernel_size"]) in [1, 2, 3]:
            return True
        return False

    return _dim_check, "Only 1d, 2d and 3d kernel supported."


class OneFlowOpConverter(object):
    """A helper class for holding oneflow op converters."""

    @classmethod
    def get_converter(cls):
        """
        Get converter matches given opset.
        Parameters
        ----------
        None

        Returns
        -------
        converter, which should be `_impl_vx`.
        """
        version = 1
        if hasattr(cls, f"_impl_v{version}"):
            return getattr(cls, f"_impl_v{version}")
        raise NotImplementedError(f"version {version} of {cls.__name__} not implemented")


class Pool(OneFlowOpConverter):
    """A helper class for pool op converters."""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        data = inputs[0]
        attrs.pop("data_format")
        out = AttrCvt(
            op_name=cls.name,
            transforms={
                "kernel_size": "pool_size",
                "stride": "strides",
                "dilations": ("dilation", 1),
            },
            ignores=["return_indices", "divisor_override"],
            custom_check=dimension_constraint(),
        )([data], attrs, params)

        return out


class AdaptiveAvgPool2d(OneFlowOpConverter):
    """Operator converter for AdaptiveAvgPool2d"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.nn.adaptive_avg_pool2d(inputs[0], output_size=attrs["output_size"])


class AdaptiveMaxPool2d(OneFlowOpConverter):
    """Operator converter for AdaptiveMaxPool2d"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.nn.adaptive_max_pool2d(inputs[0], output_size=attrs["output_size"])


class GlobalAveragePool(OneFlowOpConverter):
    """Operator converter for GlobalAveragePool"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        rank = len(infer_shape(inputs[0]))
        if rank == 3:
            return _op.nn.global_avg_pool1d(inputs[0])
        if rank == 4:
            return _op.nn.global_avg_pool2d(inputs[0])
        if rank == 5:
            return _op.nn.global_avg_pool3d(inputs[0])
        raise NotImplementedError(
            "Global average pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
            % (rank - 2)
        )


class GlobalMaxPool(OneFlowOpConverter):
    """Operator converter for GlobalMaxPool"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        rank = len(infer_shape(inputs[0]))
        if rank == 3:
            return _op.nn.global_max_pool1d(inputs[0])
        if rank == 4:
            return _op.nn.global_max_pool2d(inputs[0])
        if rank == 5:
            return _op.nn.global_max_pool3d(inputs[0])
        raise NotImplementedError(
            "Global max pooling is only implemented for 1D, 2D, and 3D kernels, got %dD."
            % (rank - 2)
        )


class Conv(OneFlowOpConverter):
    """A helper class for conv op converters."""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        # The kernel is imported from model_dir_path, without the ".weight" logo, etc.
        # The data is obtained through the graph, its op contains "_input."
        in_names = ["_input."]
        kernel_names = [".weight"]
        for i in inputs:
            IN_NAMES = any(x in str(i) for x in in_names)
            KERNEL_NAMES = any(x in str(i) for x in kernel_names)
            if IN_NAMES:
                data = i
            elif KERNEL_NAMES:
                kernel = i
            else:
                data = i

        # Use shape of input to determine convolution type.
        kernel_type = infer_type(kernel)
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]

        if "kernel_size" not in attrs:
            attrs["kernel_size"] = kernel_shapes[0][2:]
        if "dilation_rate" in attrs:
            attrs["dilation"] = list(attrs["dilation_rate"])
            attrs.pop("dilation_rate")

        pad_v = attrs.get("padding_before", [0, 0])
        attrs["padding"] = [pad_v[0], pad_v[1], pad_v[0], pad_v[1]]

        group_conv1d = False
        if cls.name == "conv1d" and attrs.get("groups") != 1:
            group_conv1d = True
            # Expand input from NCW to NCHW
            data = _op.expand_dims(data, axis=2)
            # Expand kernel from OIW to OIHW
            kernel = _op.expand_dims(kernel, axis=2)
            # Add new value to kernel_shape, strices, dilation, pads, if needed
            attrs["kernel_size"] = [1] + list(attrs["kernel_size"])
            if "strides" in attrs:
                attrs["strides"] = [1] + list(attrs["strides"])
            if "dilations" in attrs:
                attrs["dilation"] = [1] + list(attrs["dilations"])

        out = AttrCvt(
            op_name=cls.name,
            transforms={"group": ("groups", 1)},
            ignores=["data_format", "filters", "padding_after", "padding_before"],
            custom_check=dimension_constraint(),
        )([data, kernel], attrs, params)

        # If this was a group_conv1d, squish output back to NCW.
        if group_conv1d:
            out = _op.squeeze(out, axis=[2])

        return out


class ConvTranspose(OneFlowOpConverter):
    """Operator converter for ConvTranspose."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        in_names = ["_input."]
        kernel_names = [".weight"]
        for i in inputs:
            IN_NAMES = any(x in str(i) for x in in_names)
            KERNEL_NAMES = any(x in str(i) for x in kernel_names)
            if IN_NAMES:
                data = i
            elif KERNEL_NAMES:
                kernel = i
            else:
                data = i

        # get number of channels
        attrs["channels"] = attrs.get("filters", 1)
        attrs["groups"] = attrs.get("group", 1)

        kernel_type = infer_type(kernel)
        kernel_shapes = [get_const_tuple(kernel_type.checked_type.shape)]

        if "kernel_size" not in attrs:
            attrs["kernel_size"] = kernel_shapes[0][2:]

        if "dilation_rate" in attrs:
            attrs["dilation"] = list(attrs["dilation_rate"])
            attrs.pop("dilation_rate")

        pad_v = attrs.get("padding_before", [0, 0])
        attrs["padding"] = [pad_v[0], pad_v[1], pad_v[0], pad_v[1]]

        out = AttrCvt(
            op_name=cls.name,
            transforms={"group": ("groups", 1)},
            disables=["filters", "data_format", "padding_before"],
            custom_check=dimension_constraint(),
        )([data, kernel], attrs, params)

        return out


class Upsample(OneFlowOpConverter):
    """A helper class for upsample op converters"""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        data = inputs[0]
        input_shape = infer_shape(data)
        dims = len(input_shape)

        width_scale = attrs.get("width_scale", 1.0)
        height_scale = attrs.get("height_scale", 1.0)
        align_corners = attrs.get("align_corners", False)

        if "nearest" in cls.name:
            method = "nearest_neighbor"
        elif "trilinear" in cls.name:
            method = "trilinear"
        elif "bilinear" in cls.name:
            method = "bilinear"

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
                data,
                scale_d,
                scale_h,
                scale_w,
                layout=layout,
                method=method,
                coordinate_transformation_mode="asymmetric",
            )
        # in 2d case, use dynamic op
        else:
            if isinstance(height_scale, _expr.Expr):
                height_scale = _op.take(height_scale, _op.const(3))
                width_scale = _op.take(width_scale, _op.const(4))
            layout = "NCHW"

            out = _op.nn.upsampling(
                inputs[0],
                height_scale,
                width_scale,
                layout=layout,
                method=method,
                align_corners=align_corners,
            )
        return out


class UpsampleNearest(Upsample):
    """Operator converter for Upsample Nearest"""

    name = "upsample_nearest"


class UpsampleBiLinear(Upsample):
    """Operator converter for Upsample Bilinear"""

    name = "upsample_bilinear"


class Conv2d(Conv):
    """Operator converter for Conv2d"""

    name = "conv2d"


class ConvTranspose2d(ConvTranspose):
    """Operator converter for ConvTranspose2d"""

    name = "conv2d_transpose"


class BatchNorm(OneFlowOpConverter):
    """Operator converter for BatchNorm"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        # sort the inputs
        sorted_inputs = copy.deepcopy(inputs)
        for i in inputs:
            IN_NAMES = "_input." in str(i)
            if IN_NAMES:
                sorted_inputs[0] = i
            elif "weight" in str(i) and not IN_NAMES:
                sorted_inputs[1] = i
            elif "bias" in str(i) and not IN_NAMES:
                sorted_inputs[2] = i
            elif "mean" in str(i) and not IN_NAMES:
                sorted_inputs[3] = i
            elif "var" in str(i) and not IN_NAMES:
                sorted_inputs[4] = i

        if "data_format" in attrs:
            if attrs["data_format"] == "channel_first":
                attrs["axis"] = 1

        out = AttrCvt(op_name="batch_norm", ignores=["training"], disables=["momentum"])(
            sorted_inputs, attrs, params
        )
        return out[0]


class Flatten(OneFlowOpConverter):
    """Operator converter for Flatten"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        x = inputs[0]
        input_shape = list(infer_shape(x))

        start = attrs["start_dim"]
        end = attrs["end_dim"]
        ndim = len(input_shape)
        if end < 0:
            end += ndim
        new_shape = [0] * start

        new_shape.append(-1)
        squeeze_axes = []
        for i in range(start + 1, end + 1):
            new_shape.append(1)
            squeeze_axes.append(i)
        for _ in range(end + 1, ndim):
            new_shape.append(0)
        out = _op.reshape(x, new_shape)
        if squeeze_axes:
            out = _op.squeeze(out, axis=squeeze_axes)
        return out


class MatMul(OneFlowOpConverter):
    """Operator converter for MatMul"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 2, f"MatMul op take 2 inputs, {len(inputs)} given"

        dtype = infer_type(inputs[0]).checked_type.dtype
        # Y = alpha * A * B
        alpha = float(attrs.get("alpha", 1.0))
        transA = bool(attrs.get("transpose_a", False))
        transB = bool(attrs.get("transpose_b", False))

        a_shape = infer_shape(inputs[0])
        b_shape = infer_shape(inputs[1])
        if (
            (transA and transB and a_shape[-2] != b_shape[-1])
            or (transA and not transB and a_shape[-2] != b_shape[-2])
            or (transB and not transA and a_shape[-1] != b_shape[-1])
            or (not transB and not transA and a_shape[-1] != b_shape[-2])
        ):
            matmul_a = inputs[1]
            matmul_b = inputs[0]
        else:
            matmul_a = inputs[0]
            matmul_b = inputs[1]

        if transA:
            perm = list(range(len(a_shape)))
            perm[-2] = len(a_shape) - 1
            perm[-1] = len(a_shape) - 2
            matmul_a = _op.transpose(matmul_a, axes=perm)
        if transB:
            perm = list(range(len(b_shape)))
            perm[-2] = len(b_shape) - 1
            perm[-1] = len(b_shape) - 2
            matmul_b = _op.transpose(matmul_b, axes=perm)

        # This implemention almost keeps same with ONNX
        # Need to check input shape as batch matmul must be supported.
        a_shape = shape_of(matmul_a, dtype="int32")
        a_rank = infer_shape(a_shape)[0]
        b_shape = shape_of(matmul_b, dtype="int32")
        b_rank = infer_shape(b_shape)[0]
        # When performing a batch matmul, we need to properly handle N-dim shapes.
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

            b_type = infer_type(matmul_b)
            # Convert to dense if the second matrix is 2d and non-dynamic
            if b_rank == 2 and not _ty.is_dynamic(b_type.checked_type):
                a = flatten_to_nd(matmul_a, a_shape, 2)
                b = _op.transpose(matmul_b)
                output = _op.nn.dense(a, b)
            else:
                # Convert a and b into 3 dimensional tensors.
                a = flatten_to_nd(matmul_a, a_shape, 3)
                b = flatten_to_nd(matmul_b, b_shape, 3)
                # Transpose matrix dimensions of b.
                b = _op.transpose(b, [0, 2, 1])
                # Perform a batch matmul.
                output = _op.nn.batch_matmul(a, b)
            # Determine the output batch dimension.
            if a_rank > b_rank:
                out_batch = _op.strided_slice(a_shape, [0], [a_rank - 2])
            elif a_rank < b_rank:
                out_batch = _op.strided_slice(b_shape, [0], [b_rank - 2])
            # If its unclear how broadcasting should be applied, the output
            # shape is determined by choosing the maximum value from each input.
            else:
                out_batch = _op.concatenate(
                    [
                        _op.maximum(
                            _op.strided_slice(a_shape, [i], [i + 1]),
                            _op.strided_slice(b_shape, [i], [i + 1]),
                        )
                        for i in range(a_rank - 2)
                    ],
                    0,
                )
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
            out = _op.reshape(output, fold_constant(final_shape))
        else:
            if b_rank == 1:
                matmul_b = _op.expand_dims(matmul_b, 1, 1)
            # Otherwise a simple dense op will get the job done.
            input_1_t = _op.transpose(matmul_b, axes=(1, 0))
            out = _op.nn.dense(matmul_a, input_1_t)
            if b_rank == 1:
                out = _op.squeeze(out, axis=[-1])
        if not np.isclose(alpha, 1.0):
            out = out * _expr.const(alpha, dtype=dtype)
        return out


class Reduce(OneFlowOpConverter):
    """Operator converter for reduce ops"""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        attr = {"axis": attrs.get("axis", 0), "keepdims": attrs.get("keepdims", True)}
        return AttrCvt(cls.name)(inputs, attr)


class ReduceMax(Reduce):
    """Operator converter for ReduceMax"""

    name = "max"


class ReduceMin(Reduce):
    """Operator converter for ReduceMin"""

    name = "min"


class ReduceSum(Reduce):
    """Operator converter for ReduceSum"""

    name = "sum"


class ReduceMean(Reduce):
    """Operator converter for ReduceMean"""

    name = "mean"


class Square(OneFlowOpConverter):
    """Operator converter for square"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 1, f"Square op {cls.name} take 1 inputs, {len(inputs)} given"
        return _op.multiply(inputs[0], inputs[0])


class Add(OneFlowOpConverter):
    """Operator converter for Add"""

    name = "add"

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 2, f"Math op {cls.name} take 2 inputs, {len(inputs)} given"
        axis = int(attrs.get("axis", 0))

        true_names = ["weight", "bias"]
        false_names = ["_input."]

        for i in inputs:
            T_NAMES = any(x in str(i) for x in true_names)
            F_NAMES = any(x in str(i) for x in false_names)
            if T_NAMES and not F_NAMES:
                add_b = i
            else:
                add_a = i

        # fix the shape
        add_shape = infer_shape(add_a)
        if len(add_shape) > 2:
            add_b = _op.expand_dims(add_b, axis=axis, num_newaxis=len(add_shape) - 2)
        add_b_shape = list(infer_shape(add_b))
        add_b_shape.insert(0, add_shape[0])

        add_b = _op.reshape(add_b, tuple(add_b_shape))
        out = get_relay_op(cls.name)(add_a, add_b)

        return out


class Expand(OneFlowOpConverter):
    """Operator converter for Expand"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        data_in = inputs[0]
        shape = list(infer_shape(data_in))

        ndims = len(shape)
        sizes = attrs["logical_expand_shape"]
        out = data_in
        out_dims = len(sizes)
        if ndims < out_dims:
            num_newaxis = out_dims - ndims
            out = _op.expand_dims(out, axis=0, num_newaxis=num_newaxis)
            shape = [1] * num_newaxis + shape

        for i in range(out_dims):
            if sizes[i] != -1 and shape[i] == 1:
                out = _op.repeat(out, sizes[i], axis=i)

        return out


class Transpose(OneFlowOpConverter):
    """Operator converter for transpose."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        perm = attrs["perm"]
        return _op.transpose(inputs[0], axes=perm)


class ExpandDim(OneFlowOpConverter):
    """Operator converter for ExpandDim"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.expand_dims(inputs[0], axis=attrs.get("axis", 0))


class BroadcastMath(OneFlowOpConverter):
    """Operator converter for broadcast math ops"""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 2, f"Math op {cls.name} take 2 inputs, {len(inputs)} given"
        beta_names = ["weight", "bias", "mean", "var", "Constant"]

        for i in inputs:
            T_NAMES = any([x in str(i) for x in beta_names])
            if T_NAMES and "_input." not in str(i):
                input_b = i
            else:
                input_a = i

        if cls.name == "divide":
            length = []
            for i in inputs:
                length.append(len(str(i)))
            for i in inputs:
                if len(str(i)) == max(length):
                    input_a = i
                else:
                    input_b = i
        if cls.name == "subtract":
            length = []
            for i in inputs:
                length.append(len(str(i)))
            for i in inputs:
                if len(str(i)) == max(length):
                    input_b = i
                else:
                    input_a = i
        try:
            return get_relay_op(cls.name)(input_a, input_b)
        except UnboundLocalError:
            return get_relay_op(cls.name)(*inputs)


class BroadcastMul(BroadcastMath):
    """Operator converter for Mul broadcast"""

    name = "multiply"


class BroadcastAdd(BroadcastMath):
    """Operator converter for Add broadcast"""

    name = "add"


class BroadcastSub(BroadcastMath):
    """Operator converter for Sub broadcast"""

    name = "subtract"


class BroadcastDiv(BroadcastMath):
    """Operator converter for Div broadcast"""

    name = "divide"


class LogicalGreater(OneFlowOpConverter):
    """Operator converter for greater"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        res = None
        if attrs.get("has_int_operand", True):
            value = attrs.get("int_operand", 0.0)
            res = _op.greater(inputs[0], _op.full_like(inputs[0], fill_value=_expr.const(value)))
        elif attrs.get("has_float_operand", True):
            value = float(attrs.get("float_operand", 0.0))
            res = _op.greater(
                inputs[0], _op.full_like(inputs[0], fill_value=_expr.const(value)).astype("float32")
            )
        else:
            raise AttributeError(
                "please check if has_int_operand or has_float_operand in your attrs"
            )
        return res


class Log1p(OneFlowOpConverter):
    """Operator converter for Log1p"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.log(inputs[0] + _expr.const(1.0))


class Pow(OneFlowOpConverter):
    """Operator converter for Power"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        inputs = _dtype_shape_promotion(inputs)
        return get_relay_op(cls.name)(inputs[0], inputs[1])


class Expm1(OneFlowOpConverter):
    """Operator converter for Expm1"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.exp(inputs[0]) - _expr.const(1.0)


class Unary(OneFlowOpConverter):
    """A helper class for unary op converters"""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 1, f"Unary math op {cls.name} takes 1 input, {len(inputs)} given"
        return get_relay_op(cls.name)(*inputs)


class Absolute(Unary):
    """Operator converter for Absolute."""

    name = "abs"


class AddN(OneFlowOpConverter):
    """Operator converter for Add_n"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) > 0, "add_n take >=1 inputs, but 0 given."

        res = inputs[0]
        for each in inputs[1:]:
            res = _op.add(res, each)
        return res


class ScalarAdd(OneFlowOpConverter):
    """Operator convert for Add_scalar"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 1, f"add_scalar take == 1 inputs, but {len(inputs)} given."

        if attrs.get("has_int_operand", True):
            res = inputs[0] + _expr.const(attrs["int_operand"])
        elif attrs.get("has_float_operand", True):
            res = inputs[0] + _expr.const(attrs["float_operand"])
        else:
            raise AttributeError(
                "please check if has_int_operand or has_float_operand in your attrs"
            )

        return res


class ScalarMul(OneFlowOpConverter):
    """Operator convert for Mul_scalar"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 1, f"mul_scalar take == 1 inputs, but {len(inputs)} given."

        if attrs.get("has_int_operand", True):
            res = inputs[0] * _expr.const(attrs["int_operand"], dtype="float32")
        elif attrs.get("has_float_operand", True):
            res = inputs[0] * _expr.const(attrs["float_operand"])
        else:
            raise AttributeError(
                "please check if has_int_operand or has_float_operand in your attrs"
            )

        return res


class ScalarDiv(OneFlowOpConverter):
    """Operator convert for Div_scalar"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 1, f"div_scalar take == 1 inputs, but {len(inputs)} given."

        if attrs.get("has_int_operand", True):
            res = inputs[0] / _expr.const(attrs["int_operand"], dtype="float32")
        elif attrs.get("has_float_operand", True):
            res = inputs[0] / _expr.const(attrs["float_operand"])
        else:
            raise AttributeError(
                "please check if has_int_operand or has_float_operand in your attrs"
            )

        return res


class ScalarPow(OneFlowOpConverter):
    """Operator convert for Pow_scalar"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        if attrs.get("has_int_operand", True):
            coeff = _expr.const(attrs["int_operand"])
        elif attrs.get("has_float_operand", True):
            coeff = _expr.const(attrs["float_operand"])
        return _op.power(inputs[0], coeff)


class MaxPool2d(Pool):
    """Operator converter for MaxPool"""

    name = "max_pool2d"


class AveragePool2d(Pool):
    """Operator converter for AveragePool."""

    name = "avg_pool2d"


class Affine(OneFlowOpConverter):
    """Operator converter for Affine transformation."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        alpha = _expr.const(attrs.get("alpha", 1.0))
        beta = _expr.const(attrs.get("beta", 0.0))
        return (alpha * inputs[0]) + beta


class Reshape(OneFlowOpConverter):
    """Operator converter for Reshape."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.reshape(inputs[0], attrs["shape"])


class Softmax(OneFlowOpConverter):
    """Operator converter for Softmax."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        axis = attrs.get("axis", -1)
        data = inputs[0]
        if isinstance(axis, str):
            axis = int(axis)

        return _op.nn.softmax(data, axis=axis)


class LogSoftmax(OneFlowOpConverter):
    """Operator converter for LogSoftmax."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        axis = attrs.get("axis", 1)
        ndim = len(infer_shape(inputs[0]))
        if axis < 0:
            axis += ndim
        axes = list(range(axis, ndim))
        x = inputs[0]
        m = _op.max(x, axes, keepdims=True)
        e = _op.exp(x - m)
        s = _op.sum(e, axes, keepdims=True)
        return x - m - _op.log(s)


class Dropout(OneFlowOpConverter):
    """Operator converter for Dropout."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        out = AttrCvt("dropout", {"ratio": "rate"}, ignores=["is_test"])
        return out


class Threshold(OneFlowOpConverter):
    """Operator converter for Threshold."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        threshold = float(attrs.get("threshold_val", 1.0))
        threshold_tensor = _op.full_like(inputs[0], fill_value=_expr.const(threshold))
        value = float(attrs.get("value"))
        value_tensor = _op.full_like(inputs[0], fill_value=_expr.const(value))
        mask = _op.greater(inputs[0], threshold_tensor)
        return _op.where(mask, inputs[0], value_tensor)


class Elu(OneFlowOpConverter):
    """Operator converter for Elu"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        alpha = float(attrs.get("alpha", 1.0))
        return _expr.const(-alpha) * _op.nn.relu(
            _expr.const(1.0) - _op.exp(inputs[0])
        ) + _op.nn.relu(inputs[0])


class PReLU(OneFlowOpConverter):
    """Operator converter for PReLU"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 2, f"PReLU need 2 inputs, but {len(inputs)} given"
        for i in inputs:
            if "_input." in str(i):
                prelu_a = i
            else:
                prelu_b = i

        input_shape = shape_of(prelu_a)
        alpha = _op.broadcast_to_like(prelu_b, prelu_a)
        alpha = _op.reshape(alpha, [-1])

        output = _op.nn.prelu(_op.reshape(prelu_a, [-1]), alpha, axis=0)
        out = _op.reshape(output, input_shape)
        return out


class Selu(OneFlowOpConverter):
    """Operator converter for Selu"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        alpha = float(attrs.get("alpha", 1.67326319217681884765625))
        gamma = float(attrs.get("gamma", 1.05070102214813232421875))
        return _expr.const(gamma) * (
            _expr.const(-alpha) * _op.nn.relu(_expr.const(1.0) - _op.exp(inputs[0]))
            + _op.nn.relu(inputs[0])
        )


class Silu(OneFlowOpConverter):
    """Operator converter for Silu"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        a = inputs[0]
        b = _op.sigmoid(inputs[0])
        return _op.multiply(a, b)


class Gelu(OneFlowOpConverter):
    """Operator converter for Gelu"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        data = inputs[0]
        return data * (
            _expr.const(0.5) + _op.erf(data * _expr.const(0.5**0.5)) * _expr.const(0.5)
        )


class HardTanh(OneFlowOpConverter):
    """Operator converter for HardTanh"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        tanh_min = attrs.get("min_val", 0.0)
        tanh_max = attrs.get("max_val", 0.0)
        return _op.tensor.clip(inputs[0], tanh_min, tanh_max)


class Softplus(OneFlowOpConverter):
    """Operator converter for Softplus"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        data = inputs[0]
        data_dtype = infer_type(data).checked_type.dtype
        beta = _expr.const(float(attrs.get("beta", 1.0)))
        threshold = float(attrs.get("threshold", 20.0))
        threshold_ = _op.full_like(data, fill_value=_expr.const(threshold))
        softplus_value = _op.log(_op.exp(data * beta) + _expr.const(1.0, dtype=data_dtype)) / beta
        return _op.where(_op.greater(data * beta, threshold_), data, softplus_value)


class Softsign(OneFlowOpConverter):
    """Operator converter for Softsign"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return inputs[0] / (_expr.const(1.0) + Absolute.get_converter()(inputs, attrs, params))


class Variance(OneFlowOpConverter):
    """Operator converter for Variance"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        axis = attrs["dim"]
        keepdims = attrs["keepdim"]
        unbiased = bool(attrs["unbiased"])
        return _op.reduce.variance(inputs[0], axis=axis, keepdims=keepdims, unbiased=unbiased)


class Concat(OneFlowOpConverter):
    """Operator converter for Concat"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        attrs.pop("max_dim_size")
        inputs = _dtype_shape_promotion(inputs)
        return _op.concatenate(inputs, axis=attrs["axis"])


class Clip(OneFlowOpConverter):
    """Operator converter for Clip"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        attr = {}
        dtype = infer_type(inputs[0])

        if "float" in str(dtype):
            attr["a_min"] = attrs["floating_min"]
            attr["a_max"] = attrs["floating_max"]
        elif "int" in str(dtype):
            attr["a_min"] = attrs["integral_min"]
            attr["a_max"] = attrs["integral_max"]
        else:
            attr["a_min"] = -np.inf
            attr["a_max"] = np.inf

        out = AttrCvt("clip")(inputs, attr, params)
        return out


class Slice(OneFlowOpConverter):
    """Operator converter for Slice"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        starts = list(attrs["start"])
        ends = list(attrs["stop"])
        steps = list(attrs["step"])
        return _op.strided_slice(inputs[0], starts, ends, steps)


class Split(OneFlowOpConverter):
    """Operator converter for Split"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        splits = attrs.get("split", None)
        if splits is not None:
            indices = []
            attrs["indices_or_sections"] = []
            index = 0
            for i in splits[:-1]:
                index += i
                indices.append(index)
        output = _op.split(inputs[0], indices, attrs.get("axis", 0))
        # If the output of split is a single value, unpack if from the TupleWrapper
        if len(output) == 1:
            output = output[0]
        return output


class Scatter(OneFlowOpConverter):
    """Operator converter for Scatter"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        axis = attrs.get("axis", 0)
        return _op.scatter_elements(inputs[0], inputs[1], inputs[2], axis)


class Unsqueeze(OneFlowOpConverter):
    """Operator converter for Unsqueeze"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        axes = sorted(attrs["axes"])
        for axis in axes:
            inputs[0] = _op.expand_dims(inputs[0], axis=axis, num_newaxis=1)
        return inputs[0]


class Sign(OneFlowOpConverter):
    """Operator converter for Sign"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.sign(inputs[0])


class Reciprocal(OneFlowOpConverter):
    """Operator converter for Reciprocal"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        dtype = infer_type(inputs[0]).checked_type.dtype
        return _expr.const(1.0, dtype=dtype) / inputs[0]


class Erf(OneFlowOpConverter):
    """Operator converter for Erf"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.erf(inputs[0])


class Erfc(OneFlowOpConverter):
    """Operator converter for Erfs"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _expr.const(1.0) - _op.erf(inputs[0])


class HardSigmoid(OneFlowOpConverter):
    """Operator converter for HardSigmoid"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        alpha = attrs.get("alpha", 0.2)
        beta = attrs.get("beta", 0.5)
        transformX = (inputs[0] * _expr.const(alpha)) + _expr.const(beta)
        attr = {"a_min": 0, "a_max": 1}
        return AttrCvt("clip")([transformX], attr)


class OneHot(OneFlowOpConverter):
    """Operator converter for OneHot"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
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
        axis = attrs.get("axis", -1)
        if axis < 0:
            axis += ndim + 1

        return _op.one_hot(indices, on_value, off_value, depth, axis, dtype=dtype)


class Where(OneFlowOpConverter):
    """Operator converter for Where"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        condition_rank = len(infer_shape(inputs[0]))
        x_rank = len(infer_shape(inputs[1]))
        y_rank = len(infer_shape(inputs[2]))
        ranks = [condition_rank, x_rank, y_rank]

        # If one rank is longer than others, then we can broadcast
        # to that shape.
        max_rank = max(ranks)
        max_rank_idxs = [i for i, x in enumerate(ranks) if x == max_rank]
        broadcast_shape = shape_of(inputs[max_rank_idxs[0]])
        # If two or more inputs have the same rank, compute the broadcast
        # shape by taking the maximum value of each dimensions.
        if len(max_rank_idxs) > 1:
            for idx in max_rank_idxs:
                broadcast_shape = _op.maximum(broadcast_shape, shape_of(inputs[idx]))

        broadcast_shape = fold_constant(broadcast_shape)

        condition = _op.broadcast_to(inputs[0], broadcast_shape)
        x = _op.broadcast_to(inputs[1], broadcast_shape)
        y = _op.broadcast_to(inputs[2], broadcast_shape)
        return _op.where(condition, x, y)


class Constant(OneFlowOpConverter):
    """Operator converter for Constant"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        is_float = attrs.get("is_floating_value", True)
        shape = attrs.get("shape", (1,))
        if is_float:
            dtype = "float32"
            value = attrs.pop("floating_value")
        else:
            dtype = "int8"
            value = attrs.pop("integer_value")
        np_array = np.zeros(shape)
        np_array.fill(value)
        value = _expr.const(np_array, dtype)
        return value


class Range(OneFlowOpConverter):
    """Operator converter for Range"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        if len(inputs) != 0:
            raise ValueError(f"Expect no inputs but get {len(inputs)}")
        start = attrs.get("start", 0.0)
        limit = attrs.get("limit", 1.0)
        delta = attrs.get("delta", 1.0)
        return _op.arange(
            _expr.const(start, dtype="float32"),
            _expr.const(limit, dtype="float32"),
            _expr.const(delta, dtype="float32"),
        )


class Cast(OneFlowOpConverter):
    """Operator converter for Cast"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        attrs["dtype"] = infer_type(inputs[0]).checked_type.dtype
        return AttrCvt(op_name="cast")(inputs, attrs)


def get_convert_map():
    # supported oneflow2relay op
    return {
        # defs/math
        "bias_add": Add.get_converter(),
        "scalar_add": ScalarAdd.get_converter(),
        "scalar_mul": ScalarMul.get_converter(),
        "scalar_div": ScalarDiv.get_converter(),
        "scalar_pow": ScalarPow.get_converter(),
        "reduce_sum": ReduceSum.get_converter(),
        "reduce_max": ReduceMax.get_converter(),
        "reduce_min": ReduceMin.get_converter(),
        "reduce_mean": ReduceMean.get_converter(),
        "broadcast_add": BroadcastAdd.get_converter(),
        "broadcast_mul": BroadcastMul.get_converter(),
        "broadcast_sub": BroadcastSub.get_converter(),
        "broadcast_div": BroadcastDiv.get_converter(),
        "scalar_logical_greater": LogicalGreater.get_converter(),
        "log": Renamer("log"),
        "log1p": Log1p.get_converter(),
        "acos": Renamer("acos"),
        "acosh": Renamer("acosh"),
        "asin": Renamer("asin"),
        "asinh": Renamer("asinh"),
        "atan": Renamer("atan"),
        "atanh": Renamer("atanh"),
        "cos": Renamer("cos"),
        "cosh": Renamer("cosh"),
        "sin": Renamer("sin"),
        "sinh": Renamer("sinh"),
        "tan": Renamer("tan"),
        "tanh": Renamer("tanh"),
        "pow": Pow.get_converter(),
        "exp": Renamer("exp"),
        "expm1": Expm1.get_converter(),
        "floor": Renamer("floor"),
        "ceil": Renamer("ceil"),
        "round": Renamer("round"),
        "add_n": AddN.get_converter(),
        "sqrt": Renamer("sqrt"),
        "rsqrt": Renamer("rsqrt"),
        "square": Square.get_converter(),
        "sign": Sign.get_converter(),
        "erf": Erf.get_converter(),
        "erfc": Erfc.get_converter(),
        "reciprocal": Reciprocal.get_converter(),
        # defs/activation
        "softmax": Softmax.get_converter(),
        "softsign": Softsign.get_converter(),
        "hardtanh": HardTanh.get_converter(),
        "relu": Renamer("relu"),
        "leaky_relu": Renamer("leaky_relu"),
        "prelu": PReLU.get_converter(),
        "threshold": Threshold.get_converter(),
        "selu": Selu.get_converter(),
        "silu": Silu.get_converter(),
        "gelu": Gelu.get_converter(),
        # defs/nn
        "conv2d": Conv2d.get_converter(),
        "deconv2d": ConvTranspose2d.get_converter(),
        "max_pool_2d": MaxPool2d.get_converter(),
        "avg_pool_2d": AveragePool2d.get_converter(),
        "maxpool_2d": MaxPool2d.get_converter(),  # Maintained for oneflow versions <= "0.7.0"
        "avgpool_2d": AveragePool2d.get_converter(),  # Maintained for oneflow versions <= "0.7.0"
        "adaptive_avg_pool2d": AdaptiveAvgPool2d.get_converter(),
        "adaptive_max_pool2d": AdaptiveMaxPool2d.get_converter(),
        "dropout": Dropout.get_converter(),
        "normalization": BatchNorm.get_converter(),
        "upsample_nearest_2d": UpsampleNearest.get_converter(),
        "upsample_bilinear_2d": UpsampleBiLinear.get_converter(),
        # defs/tensor
        "matmul": MatMul.get_converter(),
        "batch_matmul": MatMul.get_converter(),
        "broadcast_matmul": MatMul.get_converter(),
        "concat": Concat.get_converter(),
        "clip_by_scalar": Clip.get_converter(),
        "slice": Slice.get_converter(),
        "expand": Expand.get_converter(),
        "transpose": Transpose.get_converter(),
        "expand_dims": ExpandDim.get_converter(),
        "range": Range.get_converter(),
        "cast": Cast.get_converter(),
        # defs/others
        "reshape": Reshape.get_converter(),
        "constant": Constant.get_converter(),
        "where": Where.get_converter(),
        "flatten": Flatten.get_converter(),
        "sigmoid": Renamer("sigmoid"),
        "sigmoid_v2": Renamer("sigmoid"),
        "hardsigmoid": HardSigmoid.get_converter(),
        "softplus": Softplus.get_converter(),
        "squeeze": AttrCvt("squeeze", {"axes": "axis"}),
        "unsqueeze": Unsqueeze.get_converter(),
        "identity": Renamer("copy"),
        "var": Variance.get_converter(),
    }


class oneflow_input(object):
    """
    Dual purpose list or dictionary access object
    """

    def __init__(self):
        self.input_keys = []
        self.input_dict = {}
        self.n = 0

    def __getitem__(self, item):
        if isinstance(item, int):
            if item > (len(self.input_keys) - 1):
                return None
            return self.input_dict[self.input_keys[item]]
        if isinstance(item, str):
            if item not in self.input_keys:
                return None
            return self.input_dict[item]
        if isinstance(item, slice):
            keys = self.input_keys[item]
            return [self.input_dict[key] for key in keys]

        raise ValueError("Only integer, string, and slice accesses allowed.")

    def __setitem__(self, item, value):
        if isinstance(item, int):
            self.input_dict[self.input_keys[item]] = value
        elif isinstance(item, str):
            self.input_keys.append(item)
            self.input_dict[item] = value
        else:
            raise ValueError("Only integer and string indexed writes allowed.")

    def keys(self):
        return self.input_keys

    def __len__(self):
        return len(self.input_keys)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.input_keys):
            output = self.input_dict[self.input_keys[self.n]]
            self.n += 1
            return output

        raise StopIteration


def deal_with_input_convert(
    node_input, node_input_shape, node_input_dtype, node_path, _nodes, _input_path_2_name
):
    """deal with input convert in oneflow."""
    if node_input not in _nodes:
        if (
            node_path not in _input_path_2_name
            or "_input." in node_input
            or "FreeEagerTensor" in node_input
        ):
            _nodes[node_input] = new_var(node_input, shape=node_input_shape, dtype=node_input_dtype)
        else:
            names = _input_path_2_name[node_path]
            node_replace = None
            for k in names:
                if k in _nodes:
                    node_replace = k
            if node_replace is not None:
                op_replace = copy.deepcopy(_nodes[node_replace])
                _nodes[node_input] = op_replace
            else:
                print(f"{node_input} will not be in _nodes")


def deal_parameter_convert(
    node_input_paths, model_dir_path, _input_path_2_name, _model_array, _params, _nodes
):
    """deal with parameter(weight) convert in oneflow."""
    for node_input_path in node_input_paths:
        node_path = os.path.join(model_dir_path, node_input_path.replace("m.", "", 1))
        node_input_name = node_input_path.split("/")[0]
        _input_path_2_name[node_path] = node_input_name
        for param_name in _model_array:
            node_p = _model_array[param_name]
            if node_path == node_p["path"]:
                node_array = node_p["params"]
                _params[node_input_name] = node_array
                _nodes[node_input_name] = new_var(
                    node_input_name, shape=node_array.shape, dtype=str(node_array.dtype)
                )
                break


class OneflowGraph(object):
    """
    A helper class for handling Relay expression

    Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : dict of str to str
        The input types to the graph

    node name:
    1. param: m.layer4.1.bn1.weight / ...
    2. buffer: m.layer4.1.bn1.running_mean / ...
    3. node inputs: m.layer4.1.bn1_input.0
    4. node outputs: m.layer4.1.bn1_output.0
    """

    def __init__(self, shape, dtype, nodes, model_dir_path):
        self._nodes = {}
        self._params = {}
        self._inputs = {}
        self._num_input = 0
        self._num_param = 0
        self._input_names = []
        self._model_array = {}
        self._input_path_2_name = {}
        self._output_path_2_name = {}
        self._init_variable_node = []
        self._shape = shape
        self._dtype = dtype
        self._identity_list = []
        self._sort_inputs = {}

        import oneflow

        model = oneflow.load(model_dir_path)
        # model_array: keys: layer_name, values: dict('path', 'params')
        for layer_name in model:
            layer = model[layer_name]
            layer_node = {}
            layer_node["path"] = os.path.join(model_dir_path, layer_name, "out")  # get path
            if "System-Train" in layer_name:
                continue
            node_name = "m." + layer_name
            shape = self._shape[node_name]
            dtype = self._dtype[node_name]
            array = layer.detach().cpu().numpy()
            layer_node["params"] = array.reshape(shape)
            self._model_array[layer_name] = layer_node

        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                for input_name in node.user_conf.input:
                    node_input_paths = getattr(node.user_conf.input[input_name], "s")
                    deal_parameter_convert(
                        node_input_paths,
                        model_dir_path,
                        self._input_path_2_name,
                        self._model_array,
                        self._params,
                        self._nodes,
                    )
                for output_name in node.user_conf.output:
                    node_output_paths = getattr(node.user_conf.output[output_name], "s")
                    for node_output_path in node_output_paths:
                        node_path = os.path.join(model_dir_path, node_output_path.replace("m.", ""))
                        node_output_name = node_output_path.split("/")[0]
                        self._output_path_2_name[node_path] = node_output_name
            elif is_output_op(node):
                node_output_path = getattr(node.output_conf, "in")
                output_path = os.path.join(
                    model_dir_path, getattr(node.output_conf, "in").replace("m.", "")
                )
                self._output_path_2_name[output_path] = node_name
            elif is_param_op(node):
                if "FreeEagerTensor" in node.name:
                    shape = tuple(node.variable_conf.shape.dim)
                    dtype = FLOW_2_STR_DTYPE[node.variable_conf.data_type]
                    self._shape[node.name] = shape
                    self._dtype[node.name] = dtype
                    self._init_variable_node.append(node.name)
        if self._init_variable_node != []:
            print(f"{self._init_variable_node} should be defined by user")

    def _parse_input(self, node, model_dir_path):
        input_user_conf_list = []
        for input_name in node.user_conf.input:
            input_user_conf_list.append(input_name)
        input_user_conf_list.sort()
        for input_name in input_user_conf_list:
            node_input_paths = getattr(node.user_conf.input[input_name], "s")
            for i in node_input_paths:
                node_input = i.split("/")[0]
                node_input_shape = self._shape[node_input]
                node_input_dtype = self._dtype[node_input]
                node_path = os.path.join(model_dir_path, i.replace("m.", ""))
                deal_with_input_convert(
                    node_input,
                    node_input_shape,
                    node_input_dtype,
                    node_path,
                    self._nodes,
                    self._input_path_2_name,
                )

    def _parse_output(self, op_name, outputs, cnt_init=0):
        """
        o: m.classifier.1_output.xxx
        new_o: m.classifier.1-conv2d_0
        "_"+new_o_xxx is in self._shape
        """
        for o in outputs:
            if "_output." not in o:
                new_o = o.replace("-" + op_name, "_output")
                new_o = new_o.replace("-" + new_o.split("-")[-1], ".0")
                for k in self._shape.keys():
                    if new_o in k:
                        self._shape[o] = self._shape[k]
                        self._dtype[o] = self._dtype[k]
                        break
            elif len(outputs) > 1:
                outputs.remove(o)
        if op_name.lower() == "dropout":
            if len(outputs) == 1:
                return outputs
            outputs = outputs[:-1]
        elif op_name.lower() == "constant":
            outputs = [self._init_variable_node[cnt_init]]

        if len(outputs) > 1:
            outputs = list(set(outputs))

        return outputs

    def from_oneflow(self, nodes, model_dir_path):
        """
        Implementation of convert the OneFlow model into an equivalent Relay Function.
        """
        # step 1: find out if unsupported ops are used
        convert_map = get_convert_map()
        unsupported_ops = set()
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                # op names, not the layer names
                op_name = node.user_conf.op_type_name
                if (
                    op_name not in convert_map
                    and "constant" not in op_name
                    and op_name not in self._identity_list
                ):
                    unsupported_ops.add(op_name)
        # find out the unsupported op
        if unsupported_ops:
            msg = "The following operators are not supported for frontend OneFlow: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

        # step 2: convert op
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                # If there is a user-defined node, skip the following steps
                if node_name in self._inputs:
                    continue

                op_name = node.user_conf.op_type_name
                op_attr = parse_attr(node.user_conf.attr)

                self._parse_input(node, model_dir_path=model_dir_path)

                node_inputs = oneflow_input()
                input_user_conf_list = []
                for input_name in node.user_conf.input:
                    input_user_conf_list.append(input_name)
                input_user_conf_list.sort()
                for input_name in input_user_conf_list:
                    node_input_paths = getattr(node.user_conf.input[input_name], "s")
                    for i in node_input_paths:
                        node_input = i.split("/")[0]
                        node_inputs[node_input] = self._nodes[node_input]

                node_outputs = []
                for output_name in node.user_conf.output:
                    node_output_paths = getattr(node.user_conf.output[output_name], "s")
                    for i in node_output_paths:
                        node_output_path = os.path.join(model_dir_path, i.replace("m.", ""))
                        if node_output_path in self._input_path_2_name:
                            node_outputs.append(self._input_path_2_name[node_output_path])
                        elif node_output_path in self._output_path_2_name:
                            node_outputs.append(self._output_path_2_name[node_output_path])
                node_outputs = self._parse_output(op_name, node_outputs)

                # convert
                op = self._convert_operator(op_name, node_inputs, op_attr)

                if not isinstance(op, _expr.TupleWrapper):
                    outputs_num = 1
                else:
                    outputs_num = len(op)

                assert (
                    len(node_outputs) == outputs_num
                ), f"Number of output mismatch {len(node_outputs)} vs {outputs_num} in {op_name}."
                if outputs_num == 1:
                    op = fold_constant(op)
                else:
                    op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))

                op_temp = []
                op_temp.append(op)
                for i, _ in enumerate(node_outputs):
                    if isinstance(node_outputs[i], list):
                        for k in node_outputs[i]:
                            self._nodes[k] = op_temp[i]
                    else:
                        self._nodes[node_outputs[i]] = op_temp[i]

        # step 3: get the outputs
        outputs = []
        for node_name, node in nodes.items():
            if is_output_op(node):
                node_name_v2 = getattr(node.output_conf, "in").split("/")[0]
                if node_name in self._nodes:
                    outputs.append(self._nodes[node_name])
                elif node_name_v2 in self._nodes:
                    outputs.append(self._nodes[node_name_v2])
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        # step 4: get the relay IR
        free_vars = analysis.free_vars(outputs)

        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]
        free_vars_inputs = []
        free_vars_parameters = []
        for x in free_vars:
            if "_input.0" in x:
                free_vars_inputs.append(x)
            else:
                free_vars_parameters.append(x)
        free_vars = free_vars_inputs + free_vars_parameters

        # step 5: make sure the '_input.0' is the first in self._inputs
        for free_var in free_vars:
            if free_var not in self._inputs:
                self._inputs[free_var] = self._nodes[free_var]

        input_names = list(self._inputs.keys())
        for input_name in input_names:
            if input_name in self._inputs:
                self._sort_inputs[input_name] = self._inputs[input_name]
            else:
                raise IndexError(f"{input_name} is not in self._inputs")

        # step 6: create a function from our output expression and all input variables.
        func = _function.Function([v for _, v in self._sort_inputs.items()], outputs)

        return IRModule.from_expr(func), self._params

    def _convert_operator(self, op_name, node_inputs, op_attr):
        """
        Parameters
        ----------
        op_name : str
            Operator name, such as conv2d and relu
        node_inputs : list of tvm.relay.function.Function
            List of inputs.
        op_attr : dict
            Dict of operator attributes

        Returns
        -------
        sym : tvm.relay.function.Function
            Converted relay function
        """
        convert_map = get_convert_map()
        if op_name in self._identity_list:
            sym = get_relay_op(op_name)(*node_inputs, **op_attr)
        elif op_name in convert_map:
            sym = convert_map[op_name](node_inputs, op_attr, self._params)
        else:
            raise NotImplementedError(f"Operator {op_name} not implemented.")

        return sym


def from_oneflow(graph, model_dir_path):
    """Convert a OneFlow model into an equivalent Relay Function.

    At present, there are two ways to run models in deep learning framework
    Dynamic Graph and Static Graph, which are also called Eager Mode and Graph
    Mode in OneFlow.

    In general, dynamic graphs are easier to use and static graphs have better performance.
    OneFlow offers nn.Graph, so that users can use the eager-like programming style to build
    static graphs and train the models.

    We utilize the intermediate representation of nn.Graph to convert the OneFlow model to Reley.

    Parameters
    ----------
    nodes : dict, keys: node.name, value: node
        contain the graph
    model_dir_path: str
        The path of weight

    Returns
    -------
    mod : tvm.IRModule
        The returned relay module
    params : dict
        A dict of name: tvm.nd.array pairs, used as pretrained weights
    """
    try:
        import oneflow as flow
    except ImportError:
        raise ImportError("please check that OneFlow is installed")

    # get info of nodes
    shape = {}
    dtype = {}
    graph_str = repr(graph)
    size_where = 2
    if "cuda" in graph_str:
        size_where = 3

    p_size = re.compile(r"size=\(.*?\)", re.S)
    p_type = re.compile(r"dtype=.*?\)", re.S)
    types = ["INPUT", "PARAMETER", "BUFFER", "OUTPUT"]
    for t in types:
        data = re.finditer(t + ":.*", graph_str)
        for i in data:
            attrs = i.group().split(":")
            size_str = re.findall(p_size, attrs[size_where])
            type_str = re.findall(p_type, attrs[size_where])
            assert size_str != [], "size should not be None, please check your repr(graph)"

            size_attr = size_str[0].replace("size=", "")
            if size_attr[-2] == ",":
                size_attr = size_attr.replace(",", "")
            if size_attr == "()":
                data_size = ()
            else:
                data_size = tuple(map(int, size_attr[1:-1].split(", ")))
            node_name = attrs[1]
            shape[node_name] = data_size
            dtype[node_name] = "float32"

            if type_str != []:
                type_attr = type_str[0].replace("dtype=", "").replace(")", "")
                if type_attr[-1] == ",":
                    type_attr = type_attr.replace(",", "")
                dtype[node_name] = type_attr.replace("oneflow.", "")

    # get graph proto, if you don't _compile the graph, the _graph_proto will be None
    graph_input = re.search(r"INPUT:.*", graph_str).group().split(":")
    shape_input = tuple(
        map(
            int,
            re.findall(p_size, graph_input[size_where])[0].replace("size=", "")[1:-1].split(", "),
        )
    )
    if not graph._is_compiled:
        graph._compile(flow.rand(shape_input))
    graph_proto = graph._graph_proto

    # get all nodes
    nodes = OrderedDict()
    for op in graph_proto.net.op:
        nodes[op.name] = op

    g = OneflowGraph(shape, dtype, nodes, model_dir_path)

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    mod, params = g.from_oneflow(nodes=nodes, model_dir_path=model_dir_path)

    return mod, params
