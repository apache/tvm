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
"""OF: OneFlow frontend"""
import os
import re
import copy
import warnings

import numpy as np
import tvm
from tvm.ir import IRModule
from tvm.topi.utils import get_const_tuple

from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import ty as _ty
from .common import (
    AttrCvt,
    Renamer,
    fold_constant,
    get_name,
    get_relay_op,
    infer_channels,
    infer_shape,
    infer_type,
    infer_value,
    new_var,
)

__all__ = ["from_oneflow"]

FLOW_2_STR_DTYPE = {
    2: "float32",
    3: "float64",
    6: "int64",
    5: "int32",
    4: "int8",
    7: "uint8",
    9: "float16"
}

FLOW_2_NP_DTYPE = {
    2: np.float32,
    3: np.float64,
    6: np.int64,
    5: np.int32,
    4: np.int8,
    7: np.uint8,
    9: np.float16
}

_identity_list = []


def is_input_op(node):
    # Determine if the the node is the input of graph
    return node.WhichOneof("op_type") == "input_conf"


def is_user_op(node):
    # Determine if the the node is the intermediate variables of graph
    return node.WhichOneof("op_type") == "user_conf"


def is_output_op(node):
    # Determine if the the node is the output of graph
    return node.WhichOneof("op_type") == "output_conf"


def is_param_op(node):
    # Determine if the the node is the intermediate variables of model(saved)
    return node.WhichOneof("op_type") == "variable_conf"


def get_node_info(node):
    """
    Get basic information about nodes: shape、data_type
    """
    # list->tuple
    shape = tuple(node.input_conf.blob_conf.shape.dim)
    # get data type
    dtype = node.input_conf.blob_conf.data_type
    if dtype in list(FLOW_2_NP_DTYPE.keys()):
        data_type = FLOW_2_NP_DTYPE[dtype]
    else:
        raise IndexError('Please check the data type of your node: %s' % node.name)

    return shape, data_type


def parse_attr(attr):
    # Parse node_attr
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
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError(
            "version {} of {} not implemented".format(version, cls.__name__)
        )


class Pool(OneFlowOpConverter):
    """A helper class for pool op converters."""
    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        data = inputs[0]
        input_shape = infer_shape(data)
        input_dtype = infer_type(data).checked_type.dtype
        ndim = len(input_shape)

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
            % (rank - 2),
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
            % (rank - 2),
        )


class Conv(OneFlowOpConverter):
    """A helper class for conv op converters."""
    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        # The kernel is imported from model_dir_path, without the ".weight" logo, etc.
        # The data is obtained through the graph, its op contains "-input_"
        in_names = ["-input_"]
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
        input_shape = infer_shape(data)
        ndim = len(input_shape)

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
                attrs["dilation"] = [1] + list(attrs["dilation"])

        out = AttrCvt(
            op_name=cls.name,
            transforms={
                "group": ("groups", 1),
            },
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
        in_names = ["-input_"]
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
        out_type = infer_type(kernel)
        out_shapes = [get_const_tuple(out_type.checked_type.shape)]
        attrs["channels"] = attrs.get("filters", 1)
        attrs["groups"] = attrs.get("group", 1)

        input_shape = infer_shape(data)
        ndim = len(input_shape)

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
            op_name="conv2d_transpose",
            transforms={
                "group": ("groups", 1),
            },
            disables=["filters", "data_format", "padding_before"],
            custom_check=dimension_constraint(),
        )([data, kernel], attrs, params)

        return out


class Upsample(OneFlowOpConverter):
    """A helper class for upsample op converters"""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        in_names = ["-input_"]
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


class BatchNorm(OneFlowOpConverter):
    """Operator converter for BatchNorm"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        # sort the inputs
        sorted_inputs = copy.deepcopy(inputs)
        for i in inputs:
            IN_NAMES = "-input_" in str(i)
            if IN_NAMES:
                sorted_inputs[0] = i
            elif 'weight' in str(i) and not IN_NAMES:
                sorted_inputs[1] = i
            elif 'bias' in str(i) and not IN_NAMES:
                sorted_inputs[2] = i
            elif 'mean' in str(i) and not IN_NAMES:
                sorted_inputs[3] = i
            elif 'var' in str(i) and not IN_NAMES:
                sorted_inputs[4] = i

        axis = attrs.get("axis", 3)
        if "data_format" in attrs:
            if attrs["data_format"] == "channel_first":
                attrs["axis"] = 1

        out = AttrCvt(
            op_name="batch_norm",
            ignores=["training"],
            disables=["momentum"]
        )(sorted_inputs, attrs, params)
        return out[0]


class Flatten(OneFlowOpConverter):
    """Operator converter for Flatten"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        axis = attrs.get("axis", 1)
        ishape = _op.shape_of(inputs[0])
        ndim = infer_shape(ishape)[0]
        if axis < 0:
            axis = axis + ndim

        if axis == 1:
            out = _op.nn.batch_flatten(inputs[0])
        else:
            pre_shape = _op.prod(_op.strided_slice(ishape, [0], [axis], [1]), keepdims=True)
            post_shape = _op.prod(_op.strided_slice(ishape, [axis], [ndim], [1]), keepdims=True)
            newshape = _op.concatenate([pre_shape, post_shape], axis=0)
            out = _op.reshape(inputs[0], newshape)
        return out


class MatMul(OneFlowOpConverter):
    """Operator converter for MatMul"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 2, "Gemm op take 2 inputs, {} given".format(
            len(inputs)
        )
        # Similar to 'class Conv'
        true_names = ["weight"]
        false_names = ["-input_"]
        for i in inputs:
            T_NAMES = any(x in str(i) for x in true_names)
            F_NAMES = any(x in str(i) for x in false_names)
            if T_NAMES and not F_NAMES:
                matmul_b = i
            else:
                matmul_a = i

        dtype = infer_type(matmul_a).checked_type.dtype

        # Y = alpha * A * B
        alpha = float(attrs.get("alpha", 1.0))
        transA = bool(attrs.get("transpose_a", False))
        transB = bool(attrs.get("transpose_b", False))

        # get number of channels
        channels = infer_channels(matmul_b, not transB)
        if transA:
            matmul_a = _op.transpose(matmul_a, axes=(1, 0))
        if not transB:
            matmul_b = _op.transpose(matmul_b, axes=(1, 0))
        matmul_a = _op.nn.batch_flatten(matmul_a)
        if alpha != 1.0:
            matmul_a *= _expr.const(alpha, dtype=dtype)

        return _op.nn.dense(matmul_a, matmul_b, units=channels)


class Reduce(OneFlowOpConverter):
    """Operator converter for reduce ops"""

    name = ""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        attr = {
            "axis": attrs.get("axis", 0),
            "keepdims": attrs.get("keepdims", True)
            }
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
        assert len(inputs) == 1, "Square op {} take 1 inputs, {} given".format(
            cls.name, len(inputs)
        )
        return _op.multiply(inputs[0], inputs[0])


class Add(OneFlowOpConverter):
    """Operator converter for Add"""

    name = "add"

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 2, "Math op {} take 2 inputs, {} given".format(cls.name, len(inputs))
        axis = int(attrs.get("axis", 0))

        true_names = ["weight", "bias"]
        false_names = ["-input_"]

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
            add_b = _op.expand_dims(add_b, axis=axis, num_newaxis=len(add_shape)-2)
        add_b_shape = list(infer_shape(add_b))
        add_b_shape.insert(0, add_shape[0])

        add_b = _op.reshape(add_b, tuple(add_b_shape))
        out = get_relay_op(cls.name)(add_a, add_b)

        return out


class Expand(OneFlowOpConverter):
    """Operator converter for Expand"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        input_shape = infer_shape(inputs[0])
        assert input_shape == attrs["in_shape"], "shape wrong"

        new_shape = attrs["out_shape"]
        out = _op.broadcast_to(inputs[0], shape=new_shape)

        return out


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
        assert len(inputs) == 2, "Math op {} take 2 inputs, {} given".format(cls.name, len(inputs))
        beta_names = ["weight", "bias", "mean", "var", "Constant"]

        for i in inputs:
            T_NAMES = any([x in str(i) for x in beta_names])
            if T_NAMES and "-input_" not in str(i):
                input_b = i
            else:
                input_a = i

        # TODO(hujiakui): no info about which is a
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


class Greater(OneFlowOpConverter):
    """Operator converter for greater"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.greater(inputs[0], inputs[1])


class Log1p(OneFlowOpConverter):
    """Operator converter for Log1p"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return _op.log(inputs[0] + _expr.const(1.0))


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
        assert len(inputs) == 1, "Unary math op {} takes 1 input, {} given".format(
            cls.name, len(inputs)
        )
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
        assert len(inputs) == 1, "add_scalar take == 1 inputs, but {} given.".format(len(inputs))

        if attrs.get("has_int_operand", False):
            return inputs[0] + _expr.const(attrs["int_operand"])
        elif attrs.get("has_float_operand", False):
            return inputs[0] + _expr.const(attrs["float_operand"])
        else:
            raise AttributeError(
                "please check if has_int_operand or has_float_operand in your attrs"
            )


class ScalarMul(OneFlowOpConverter):
    """Operator convert for Mul_scalar"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        assert len(inputs) == 1, "add_scalar take == 1 inputs, but {} given.".format(len(inputs))

        if attrs.get("has_int_operand", False):
            return inputs[0] * _expr.const(attrs["int_operand"], dtype="float32")
        elif attrs.get("has_float_operand", False):
            return inputs[0] * _expr.const(attrs["float_operand"])
        else:
            raise AttributeError(
                "please check if has_int_operand or has_float_operand in your attrs"
            )


class ScalarPow(OneFlowOpConverter):
    """Operator convert for Pow_scalar"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        exponent = attrs.get("exponent", 1.0)
        exponent = _expr.const(exponent, dtype="float32")
        return _op.power(inputs[0], exponent)


class Argmax(OneFlowOpConverter):
    """Operator convert for Argmax"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        if "select_last_index" in attrs:
            raise NotImplementedError("select_last_index not supported in ArgMax")
        axis = attrs.get("axis", 0)
        keepdims = attrs.get("keepdims", True)
        attr = {"axis": axis, "keepdims": keepdims}
        return _op.cast(AttrCvt("argmax")(inputs, attr), "int64")


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
        axis = attrs.get("axis", 1)
        ndim = len(infer_shape(inputs[0]))
        if axis < 0:
            axis += ndim
        axes = list(range(axis, ndim))
        x = inputs[0]
        m = _op.max(x, axes, keepdims=True)
        e = _op.exp(x - m)
        return e / _op.sum(e, axes, keepdims=True)


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


class ThresholdedRelu(OneFlowOpConverter):
    """Operator converter for ThresholdedRelu."""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        alpha = float(attrs.get("alpha", 1.0))
        alpha_tensor = _op.full_like(inputs[0], fill_value=_expr.const(alpha))
        mask = _op.greater(inputs[0], alpha_tensor).astype("float32")
        return inputs[0] * mask


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
        assert len(inputs) == 2, "PReLU need 2 inputs, but {} given".format(len(inputs))
        for i in inputs:
            if "-input_" in str(i):
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
            _expr.const(0.5)
            + _op.erf(data * _expr.const(0.5 ** 0.5)) * _expr.const(0.5)
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
        data = _op.exp(data) + _expr.const(1, dtype=data_dtype)
        return _op.log(data)


class Softsign(OneFlowOpConverter):
    """Operator converter for Softsign"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        return inputs[0] / (_expr.const(1.0) + Absolute.get_converter()(inputs, attrs, params))


class Concat(OneFlowOpConverter):
    """Operator converter for Concat"""

    @classmethod
    def _impl_v1(cls, inputs, attrs, params):
        attrs.pop("max_dim_size")
        return AttrCvt(op_name="concatenate")((inputs,), attrs)


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
        # TODO(hujiakui): sort the inputs
        axis = attrs.get("axis", 0)
        return _op.scatter(inputs[0], inputs[1], inputs[2], axis)


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
        shape = attrs.get("shape", (1, ))
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
            raise ValueError("Expect no inputs but get {}".format(len(inputs)))
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
        "scalar_pow": ScalarPow.get_converter(),
        "reduce_sum": ReduceSum.get_converter(),
        "reduce_max": ReduceMax.get_converter(),
        "reduce_min": ReduceMin.get_converter(),
        "reduce_mean": ReduceMean.get_converter(),
        "broadcast_add": BroadcastAdd.get_converter(),
        "broadcast_mul": BroadcastMul.get_converter(),
        "broadcast_sub": BroadcastSub.get_converter(),
        "broadcast_div": BroadcastDiv.get_converter(),
        "broadcast_greater": Greater.get_converter(),
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
        "pow": Renamer("power"),
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
        "reciprocal_no_nan": Reciprocal.get_converter(),
        # defs/activation
        "softmax": Softmax.get_converter(),
        "softsign": Softsign.get_converter(),
        "hardtanh": HardTanh.get_converter(),
        "relu": Renamer("relu"),
        "leaky_relu": Renamer("leaky_relu"),
        "prelu": PReLU.get_converter(),
        "selu": Selu.get_converter(),
        "silu": Silu.get_converter(),
        "gelu": Gelu.get_converter(),
        # defs/nn
        "conv2d": Conv2d.get_converter(),
        "deconv2d": ConvTranspose.get_converter(),
        "maxpool_2d": MaxPool2d.get_converter(),
        "avgpool_2d": AveragePool2d.get_converter(),
        "adaptive_avg_pool2d": AdaptiveAvgPool2d.get_converter(),
        "adaptive_max_pool2d": AdaptiveMaxPool2d.get_converter(),
        "dropout": Dropout.get_converter(),
        "normalization": BatchNorm.get_converter(),
        "upsample_nearest_2d": UpsampleNearest.get_converter(),
        "upsample_bilinear_2d": UpsampleBiLinear.get_converter(),
        # defs/tensor
        "matmul": MatMul.get_converter(),
        "concat": Concat.get_converter(),
        "clip_by_scalar": Clip.get_converter(),
        "slice": Slice.get_converter(),
        "expand": Expand.get_converter(),
        "transpose": AttrCvt("transpose", {"perm": "axes"}),
        "expand_dims": ExpandDim.get_converter(),
        "range": Range.get_converter(),
        "cast": Cast.get_converter(),
        # defs/others
        "reshape": Reshape.get_converter(),
        "constant": Constant.get_converter(),
        # "where": Where.get_converter(),
        "flatten": Flatten.get_converter(),
        "sigmoid": Renamer("sigmoid"),
        "sigmoid_v2": Renamer("sigmoid"),
        "hardgsigmoid": HardSigmoid.get_converter(),
    }


class oneflow_input(object):
    """
    Dual purpose list or dictionary access object
    """
    def __init__(self):
        self.input_keys = []
        self.input_dict = {}

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
    3. node inputs: m.layer4.1.bn1-input_0
    4. node outputs: m.layer4.1.bn1-output_0
    """
    def __init__(self, shape, dtype, nodes, model_dir_path) -> None:
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

        import oneflow

        model = oneflow.load(model_dir_path)
        # model_array: keys: layer_name，values: dict('path', 'params')
        for layer_name in model:
            layer = model[layer_name]
            layer_node = {}
            layer_node['path'] = layer.file_path     # get path
            if layer.has_meta_info_:
                layer_node['params'] = layer.numpy() # get array
            else:
                if "System-Train" in layer_name:
                    continue
                node_name = "m." + layer_name
                shape = self._shape[node_name]
                dtype = self._dtype[node_name]
                array = np.fromfile(layer_node['path'], dtype=dtype)
                layer_node['params'] = array.reshape(shape)
            self._model_array[layer_name] = layer_node

        """
        The names of node_outputs do not appear directly in node.user_conf.input, 
        so the connection between layers will be cut off when building the graph
        steps:
        1. find out the names of node_outputs
        2. match names and node.user_conf.input, see the self._parse_output
        3. If two nodes have the same name after parsing, then both correspond to the same op
        """
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                for input_name in node.user_conf.input:
                    node_input_paths = getattr(node.user_conf.input[input_name], 's')
                    for node_input_path in node_input_paths:
                        node_path = os.path.join(model_dir_path, node_input_path.replace("m.", ""))
                        node_input_name = node_input_path.split("/")[0]
                        self._input_path_2_name[node_path] = node_input_name
                        for param_name in self._model_array:
                            node_p = self._model_array[param_name]
                            if node_path == node_p['path']:
                                node_array = node_p['params']
                                self._params[node_input_name] = node_array
                                self._nodes[node_input_name] = new_var(
                                    node_input_name,
                                    shape=node_array.shape,
                                    dtype=str(node_array.dtype)
                                )
                                break
                for output_name in node.user_conf.output:
                    node_output_paths = getattr(node.user_conf.output[output_name], 's')
                    for node_output_path in node_output_paths:
                        node_path = os.path.join(model_dir_path, node_output_path.replace("m.", ""))
                        node_output_name = node_output_path.split("/")[0]
                        self._output_path_2_name[node_path] = node_output_name
            elif is_output_op(node):
                node_output_path = getattr(node.output_conf, "in")
                output_path = os.path.join(
                    model_dir_path,
                    getattr(node.output_conf, "in").replace("m.", "")
                )
                self._output_path_2_name[output_path] = node_name
            elif is_param_op(node):
                if "FreeEagerTensor" in node.name:
                    shape = tuple(node.variable_conf.shape.dim)
                    dtype = FLOW_2_STR_DTYPE[node.variable_conf.data_type]
                    initializer = node.variable_conf.initializer
                    self._shape[node.name] = shape
                    self._dtype[node.name] = dtype
                    self._init_variable_node.append(node.name)
        if self._init_variable_node != []:
            print("{} should be defined by user".format(self._init_variable_node))

    def _parse_input(self, node, model_dir_path):
        for input_name in node.user_conf.input:
            node_input_paths = getattr(node.user_conf.input[input_name], 's')
            for i in node_input_paths:
                node_input = i.split("/")[0]
                node_input_shape = self._shape[node_input]
                node_input_dtype = self._dtype[node_input]
                node_path = os.path.join(model_dir_path, i.replace("m.", ""))

                if node_input not in self._nodes:
                    if (
                        node_path not in self._input_path_2_name
                        or "-input_" in node_input
                        or "FreeEagerTensor" in node_input
                    ):
                        self._nodes[node_input] = new_var(
                            node_input,
                            shape=node_input_shape,
                            dtype=node_input_dtype,
                        )
                    else:
                        names = self._input_path_2_name[node_path]
                        node_replace = None
                        for k in names:
                            if k in self._nodes:
                                node_replace = k
                        if node_replace is not None:
                            op_replace = copy.deepcopy(self._nodes[node_replace])
                            self._nodes[node_name] = op_replace
                        else:
                            print("{} will not be in self._nodes".format(node_input))


    def _parse_output(self, op_name, outputs, cnt_init=0):
        """
        o: m.classifier.1-output_xxx
        new_o: m.classifier.1-conv2d_0
        "_"+new_o is in self._shape
        """
        for o in outputs:
            if "-output_" not in o:
                new_o = o.replace("-"+op_name, "-output")
                new_o = new_o.replace("_"+new_o.split("_")[-1], "_0")
                self._shape[o] = self._shape["_" + new_o]
                self._dtype[o] = self._dtype["_" + new_o]
            elif len(outputs) > 1:
                outputs.remove(o)
        if op_name.lower() == "dropout":
            if len(output) == 1:
                return outputs
            # TODO(zhreshold): support dropout mask? `form onnx.py`
            outputs = outputs[:-1]
        elif op_name.lower() == "constant":
            outputs = [self._init_variable_node[cnt_init]]
        
        if len(outputs) > 1:
            outputs = list(set(outputs))

        return outputs


    def from_oneflow(self, nodes, model_dir_path, freeze_params=True, user_input=None):
        """
        Parameters
        ----------
        nodes : dict, keys: node.name, value: node
            contain the graph
        model_dir_path: str
            The path of parameter
        freeze_params: bool
            If freeze_params is True,
            the computational graph input is the input of the first layer of the network,
            which cannot be specified by the user, e.g.
            Default input is: %v_ResNetGraph_0-input_0: Tensor[(1, 3, 224, 224), float32]
            User-defined input is: %_0-input_0: Tensor[(1, 3, 640, 480), float32]
            If freeze_params is on, then conv1-in will be the graph input, not Input_0
        user_input: dict
            User-defined input information for the graph
            {
                node1_name:
                {
                    'name':  node1_name,   # str, like "%v_ResNetGraph_0-input_0"
                    'shape': node1_shape,  # tuple
                    'dtype': node1_dtype   # str, like "float32"
                }
                ...
            }
        We recommend that users specify the input by specifying the job function,
        rather than by this function

        Returns
        -------
        mod : tvm.IRModule
            The returned relay module
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        # step 1: get the graph input
        if not freeze_params:
            for node_init_name in user_input:
                if "-input_" not in node_init_name:
                    raise KeyError(
                        "user_input['name'] should contain '-input_' " +
                        "to let program know that this is input node"
                    )
                else:
                    self._nodes[node_init_name] = new_var(
                        node_init_name,
                        shape=user_input[node_init_name]["shape"],
                        dtype=user_input[node_init_name]["dtype"]
                    )
                self._inputs[node_init_name] = self._nodes[node_init_name]

        # step 2: find out if unsupported ops are used
        convert_map = get_convert_map()
        unsupported_ops = set()
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                # op names, not the layer names
                op_name = node.user_conf.op_type_name
                if(
                    op_name not in convert_map
                    and "constant" not in op_name
                    and op_name not in _identity_list
                ):
                    unsupported_ops.add(op_name)
        # find out the unsupported op
        if unsupported_ops:
            msg = "The following operators are not supported for frontend OneFlow: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

        # step 3: convert op
        for node_name in nodes:
            node = nodes[node_name]
            if is_user_op(node):
                # If there is a user-defined node, skip the following steps
                if node_name in self._inputs:
                    continue

                op_name = node.user_conf.op_type_name
                op_attr = parse_attr(node.user_conf.attr)

                self._parse_input(
                    node,
                    model_dir_path=model_dir_path
                )

                node_inputs = oneflow_input()
                for input_name in node.user_conf.input:
                    node_input_paths = getattr(node.user_conf.input[input_name], 's')
                    for i in node_input_paths:
                        node_input = i.split("/")[0]
                        node_inputs[node_input] = self._nodes[node_input]

                node_outputs = []
                for output_name in node.user_conf.output:
                    node_output_paths = getattr(node.user_conf.output[output_name], 's')
                    for i in node_output_paths:
                        node_output_path = os.path.join(
                            model_dir_path, i.replace("m.", "")
                        )
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

                assert (len(node_outputs) == outputs_num), \
                    "Number of output mismatch {} vs {} in {}.".format(
                    len(node_outputs), outputs_num, op_name
                )

                if outputs_num == 1:
                    op = fold_constant(op)
                else:
                    op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))

                op_temp = []
                op_temp.append(op)
                for i in range(len(node_outputs)):
                    if isinstance(node_outputs[i], list):
                        for k in node_outputs[i]:
                            self._nodes[k] = op_temp[i]
                    else:
                        self._nodes[node_outputs[i]] = op_temp[i]

        # step 4: get the outputs
        outputs = []
        for node_name in nodes:
            node = nodes[node_name]
            if is_output_op(node):
                node_name_v2 = getattr(node.output_conf, "in").split("/")[0]
                if node_name in self._nodes:
                    outputs.append(self._nodes[node_name])
                elif node_name_v2 in self._nodes:
                    outputs.append(self._nodes[node_name_v2])
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)

        # step 5: get the relay IR
        free_vars = analysis.free_vars(outputs)

        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]

        # step 6: make sure the '-input_0' is the first in self._inputs
        for free_var in free_vars:
            if free_var not in self._inputs:
                self._inputs[free_var] = self._nodes[free_var]

        input_names = list(self._inputs.keys())
        for i in range(len(input_names)):
            if i != 0 and '-input_0' in input_names[i]:
                str_buffer = copy.deepcopy(input_names[i])
                del input_names[i]
                input_names.insert(0, str_buffer)
                break

        self._sort_inputs = {}
        for input_name in input_names:
            if input_name in self._inputs:
                self._sort_inputs[input_name] = self._inputs[input_name]
            else:
                raise IndexError("{} is not in self._inputs".format(input_name))

        # step 7: create a function from our output expression and all input variables.
        func = _function.Function([v for _, v in self._sort_inputs.items()], outputs)

        return IRModule.from_expr(func), self._params


    def _convert_operator(self, op_name, node_inputs, op_attr):
        """
        Parameters
        ----------
        op_name : str
            Operator name, such as conv2d、relu
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
        if op_name in _identity_list:
            sym = get_relay_op(op_name)(*node_inputs, **op_attr)
        elif op_name in convert_map:
            sym = convert_map[op_name](node_inputs, op_attr, self._params)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))

        return sym


def from_oneflow(graph, model_dir_path, freeze_params=True, user_input=None):
    """
    see OneflowGraph.from_oneflow
    """
    try:
        import oneflow

        if 'snapshot_done' not in os.listdir(model_dir_path):
            raise IndexError(
                "'snapshot_done' is not in the model path, " +
                "please determine whether the model has been trained"
            )

    except ImportError:
        raise ImportError("please check that OneFlow is installed")

    if not freeze_params and user_input is None:
        raise ValueError("if you want to specify graph input, please give the 'user_input'")
    if freeze_params and user_input is not None:
        warnings.warn("'user_input' will not work, please check the 'freeze_params'")

    # get info of nodes
    shape = {}
    dtype = {}
    graph_str = repr(graph)
    DTYPE = 2
    size_where = 2
    if "cuda" in graph_str:
        size_where = 3
    # TODO(hujiakui): prepare for float16 and int8
    # if "float16" in graph_str:
    #     DTYPE = 9
    # elif "int8" in graph_str:
    #     DTYPE = 4

    p1 = re.compile(r"size=\(.*?\)", re.S)
    types = ["INPUT", "PARAMETER", "BUFFER", "OUTPUT"]
    for t in types:
        data = re.finditer(t+":.*", graph_str)
        for i in data:
            attrs = i.group().split(":")
            size_str = re.findall(p1, attrs[size_where])
            assert size_str != [], "size should not be None, please check your inputs dtype"
            size_attr = size_str[0].replace("size=", "")
            if size_attr[-2] == ",":
                size_attr = size_attr.replace(",", "")
            data_size = tuple(map(int, size_attr[1:-1].split(", ")))
            node_name = attrs[1]
            shape[node_name] = data_size
            dtype[node_name] = FLOW_2_STR_DTYPE[DTYPE]

    # get graph proto, if you don't _compile the graph, the _graph_proto will be None
    graph_input = re.search(r"INPUT:.*", graph_str).group().split(":")
    shape_input = tuple(
        map(
            int, re.findall(
                p1, graph_input[size_where]
            )[0].replace("size=", "")[1:-1].split(", ")
        )
    )
    if not graph._is_compiled:
        _ = graph._compile(np.random.rand(shape_input))
    graph_proto = graph._graph_proto

    # get all nodes
    nodes = {}
    for op in graph_proto.net.op:
        nodes[op.name] = op

    g = OneflowGraph(shape, dtype, nodes, model_dir_path)

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    mod, params = g.from_oneflow(
            nodes=nodes, model_dir_path=model_dir_path,
            freeze_params=freeze_params, user_input=user_input
        )

    return mod, params
