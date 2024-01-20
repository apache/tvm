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
"""ONNX: Open Neural Network Exchange importer for Relax.

This module implements the required functionality to read ONNX models
and convert them into equivalent Relax functions. The entry point that encapsulates
this functionality is the function from_onnx.

In order to extend the functionality of the importer, you can add new
operators to the operator registry. The operator registry is a dictionary
that maps operator names to operator converters. The registry is defined
in the _get_converter_map function. To add a new operator, you can define
a new class that inherits from the OnnxOpConverter class and implement
the _impl method.

By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
retains dynamic shapes upon import, and when possible, the compiler attempts to
convert the model to use static shapes at compile time.
If this fails, there may still be dynamic operations in the model.
Not all TVM kernels currently support dynamic shapes, please file an issue on
github.com/apache/tvm/issues if you hit an error with dynamic kernels.
"""
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as _np
import onnx.onnx_ml_pb2

import tvm
from tvm import relax, tir, topi
from tvm.ir import IRModule
from tvm.ir.supply import NameSupply
from tvm.tir.generic import cast


def get_type(elem_type: Union[str, int]) -> str:
    """Converts onnx integer datatype to numpy datatype"""
    # If a string was passed instead of a tensor type, it does not need
    # conversion and can be returned.
    if isinstance(elem_type, str):
        return elem_type

    try:
        from onnx.mapping import (  # pylint: disable=import-outside-toplevel
            TENSOR_TYPE_TO_NP_TYPE,
        )
    except ImportError as exception:
        raise ImportError("Unable to import onnx which is required {}".format(exception))

    return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])


def get_constant(
    var: Union[relax.Constant, relax.Var],
    params: List[Dict[str, relax.Var]],
) -> Union[relax.Constant, relax.Var]:
    """Attempt to convert a variable to a constant if possible.
    This is the primary function meant to interact with params.

    Parameters
    ----------
    var: Union[relax.Constant, relax.Var]
        The input value to try to convert to a constant.
    params: List[Dict[str, relax.Var]]
        The parameters for the graph. Contains both the global registry of nodes
        for the graph and the parameter dictionary. The global registry is updated
        with a constant value if possible.

    Returns
    -------
    var : Union[relax.Constant, relax.Var]
        The input value converted to a constant if possible. If the value
        isn't found in params, the input variable is returned unmodified.
    """
    # Params is actually both the graph nodes and param dictionary, unpack them.
    graph_nodes, params = params
    # Convert if possible
    if isinstance(var, relax.Var) and var.name_hint in params:
        # When converting a parameter to a constant, update references to it as well.
        _, value = params.pop(var.name_hint)
        const_value = relax.const(value)
        graph_nodes[var.name_hint] = const_value
        return const_value
    # Otherwise return variable.
    else:
        return var


def get_info(
    info_proto: onnx.onnx_ml_pb2.ValueInfoProto, value_dict: Dict[str, tvm.tir.SizeVar]
) -> Tuple[str, List, str, List, Dict]:
    """Extract the shape from a ValueInfoProto.

    Parameters
    ----------
    info_proto: onnx.onnx_ml_pb2.ValueInfoProto
        The ValueInfoProto to extract the info from.

    value_dict: Dict
        The Dictionary mapping from the name of ValueInfoProto to SizeVar

    Returns
    -------
    Tuple[str, List, str, List, Dict]
        The name, shape, type, and shape name of the ValueInfoProto, and the
        value_dict.
    """
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            if name not in value_dict or name == "?":
                value_dict[name] = tvm.tir.SizeVar(name, "int64")
            value = value_dict[name]
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    if info_proto.type.tensor_type.elem_type:
        dtype = get_type(info_proto.type.tensor_type.elem_type)
    else:
        dtype = None
    return name, shape, dtype, shape_name, value_dict


def get_numpy(tensor_proto: onnx.onnx_ml_pb2.TensorProto) -> _np.ndarray:
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import (  # pylint: disable=import-outside-toplevel
            to_array,
        )
    except ImportError as exception:
        raise ImportError("Unable to import onnx which is required {}".format(exception))
    return to_array(tensor_proto)


def get_prim_expr_list(
    inputs: Union[relax.Constant, relax.ShapeExpr],
) -> List[Union[int, tir.PrimExpr]]:
    """Attempt to convert a variable to list of PrimExpr if possible.

    Parameters
    ----------
    inputs : Union[relax.Constant, relax.ShapeExpr, relax.PrimValue]
        The input value to try to convert to a list of PrimExpr.

    Returns
    -------
    ret : List[Union[int, tir.PrimExpr]]
        The input value converted to a list of PrimExpr if possible.
    """
    if isinstance(inputs, relax.Constant):
        np_value = inputs.data.numpy()
        if np_value.ndim != 1:
            raise ValueError("Cannot cast {} to list of PrimExpr".format(type(inputs)))
        return np_value.tolist()
    elif isinstance(inputs, relax.ShapeExpr):
        return inputs.values
    elif isinstance(inputs, relax.PrimValue):
        return [inputs.value.value]
    else:
        raise ValueError("Cannot cast {} to list of PrimExpr".format(type(inputs)))


class onnx_input(list):  # pylint: disable=invalid-name
    """A list that returns None when out-of-bounds indices are accessed."""

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


# pylint: disable=invalid-name, len-as-condition, unused-argument, too-many-lines, redefined-builtin
class OnnxOpConverter(object):
    """A helper class for holding the common logic for ONNX op converters.
    Each converter maps to a single ONNX op and defines the equivalent
    functionality using Relax expressions. The converter can define multiple versions
    of the op and the version is selected based on the opset version of the model.
    """

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


class MatMul(OnnxOpConverter):
    """Converts an onnx MatMul node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        return relax.op.matmul(inputs[0], inputs[1])


class Div(OnnxOpConverter):
    """Converts an onnx Div node into an equivalent Relax expression."""

    @classmethod
    def _impl_v14(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = inputs[0].data.numpy() / inputs[1].data.numpy()
            return relax.const(output, inputs[0].struct_info.dtype)
        if any([isinstance(inp, relax.PrimValue) for inp in inputs]):
            x = (
                int(inputs[0].value)
                if isinstance(inputs[0], relax.PrimValue)
                else inputs[0].data.numpy()
            )
            y = (
                int(inputs[1].value)
                if isinstance(inputs[1], relax.PrimValue)
                else inputs[1].data.numpy()
            )
            return relax.PrimValue(int(x / y))

        return relax.op.divide(inputs[0], inputs[1])


class Sigmoid(OnnxOpConverter):
    """Converts an onnx Sigmoid node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        return relax.op.sigmoid(inputs[0])


class Softmax(OnnxOpConverter):
    """Converts an onnx Softmax node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", -1)
        return relax.op.nn.softmax(inputs[0], axis=axis)


class Transpose(OnnxOpConverter):
    """Converts an onnx Transpose node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axes = attr.get("perm", None)
        if isinstance(inputs[0], relax.Constant):
            output = _np.transpose(inputs[0].data.numpy(), axes)
            return relax.const(output, output.dtype)
        return relax.op.permute_dims(inputs[0], axes)


class Unsqueeze(OnnxOpConverter):
    """Converts an onnx Unsqueeze node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        axes = list(attr.get("axes"))
        inputs = inputs + [relax.const(axes, "int64")]
        return cls._impl_v13(bb, inputs, attr, params)

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = get_constant(inputs[1], params)

        # Handle ONNX shape inference
        if isinstance(data, relax.PrimValue) and isinstance(axes, relax.Constant):
            axes = axes.data.numpy().tolist()
            if axes == [0]:
                return relax.ShapeExpr([data.value])
            else:
                raise NotImplementedError(
                    "Unsqueeze with symbolic axes and non-zero axes is not supported."
                )
        # If input is a constant, compute directly
        if isinstance(data, relax.Constant) and isinstance(axes, relax.Constant):
            axes = axes.data.numpy().tolist()
            expanded = data.data.numpy()
            if len(expanded.shape) == 0:
                # Special case implying input is a scalar, wrap it as a list.
                if 0 in axes:
                    axes.remove(0)
                expanded = [expanded]
            for axis in axes:
                expanded = _np.expand_dims(expanded, axis=axis)
            return relax.const(expanded, data.struct_info.dtype)

        if isinstance(axes, relax.Constant):
            constant_axes = list(axes.data.numpy())
            constant_axes = list(map(int, constant_axes))
            constant_axes = sorted(constant_axes)
            for axis in constant_axes:
                data = relax.op.expand_dims(data, axis=axis)
            return data

        raise NotImplementedError("Unsqueeze with dynamic axes is not supported.")


class Concat(OnnxOpConverter):
    """Convert an onnx Concat node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 0)

        def is_shape_like(x: Any) -> bool:
            if isinstance(x, relax.ShapeExpr):
                return True
            elif isinstance(x, relax.Constant):
                return x.struct_info.ndim == 1 and x.struct_info.dtype == "int64"
            else:
                return False

        # If all inputs are shape expr, perform computation directly.
        if all([is_shape_like(inp) for inp in inputs]):
            const_inputs = []
            for inp in inputs:
                if isinstance(inp, relax.ShapeExpr):
                    const_inputs.extend(inp.values)
                elif isinstance(inp, relax.Constant):
                    const_inputs.extend(inp.data.numpy().tolist())
                else:
                    raise NotImplementedError("Unsupported input type: {}".format(type(inp)))
            return relax.ShapeExpr(const_inputs)

        # If all inputs are constant, perform computation directly.
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            const_inputs = []
            for inp in inputs:
                const_inputs.append(inp.data.numpy())
            out = _np.concatenate(const_inputs, axis=axis)
            dtype = inputs[0].struct_info.dtype
            return relax.const(out, dtype)

        return relax.op.concat(inputs, axis=axis)


class Add(OnnxOpConverter):
    """Convert an onnx Add node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = inputs[0].data.numpy() + inputs[1].data.numpy()
            return relax.const(output, output.dtype)
        # If primvalues are involved, handle them directly.
        if any([isinstance(inp, relax.PrimValue) for inp in inputs]):
            x = (
                int(inputs[0].value)
                if isinstance(inputs[0], relax.PrimValue)
                else inputs[0].data.numpy()
            )
            y = (
                int(inputs[1].value)
                if isinstance(inputs[1], relax.PrimValue)
                else inputs[1].data.numpy()
            )
            return relax.PrimValue(int(x + y))
        return relax.op.add(inputs[0], inputs[1])


class Mul(OnnxOpConverter):
    """Convert an onnx Mul node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        # When all inputs are constant, directly multiply.
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = inputs[0].data.numpy() * inputs[1].data.numpy()
            return relax.const(output, output.dtype)
        # If primvalues are involved, handle them directly.
        if any([isinstance(inp, relax.PrimValue) for inp in inputs]):
            x = (
                int(inputs[0].value)
                if isinstance(inputs[0], relax.PrimValue)
                else inputs[0].data.numpy()
            )
            y = (
                int(inputs[1].value)
                if isinstance(inputs[1], relax.PrimValue)
                else inputs[1].data.numpy()
            )
            return relax.PrimValue(int(x * y))

        return relax.op.multiply(inputs[0], inputs[1])


class Cast(OnnxOpConverter):
    """Convert an onnx Cast node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        to_type = get_type(attr["to"])
        if isinstance(inputs[0], relax.Constant):
            output = inputs[0].data.numpy().astype(to_type)
            return relax.const(output, to_type)
        if isinstance(inputs[0], relax.PrimValue):
            return relax.PrimValue(inputs[0].value.astype(to_type))
        return relax.op.astype(inputs[0], to_type)


class Gather(OnnxOpConverter):
    """Convert an onnx Gather node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        # Unpack inputs
        data = inputs[0]
        indices = inputs[1]
        axis = attr.get("axis", 0)

        # If all inputs are constant, we can compute directly.
        if all([isinstance(inp, relax.Constant) for inp in [data, indices]]):
            output = _np.take(data.data.numpy(), indices.data.numpy(), axis=axis)
            return relax.const(output, output.dtype)

        # If input is a shape expression, take a value from that shape and return it as a constant.
        if isinstance(data, relax.ShapeExpr):
            assert isinstance(
                indices, relax.Constant
            ), "Only constant indices supported for shape gather."
            np_index = indices.data.numpy()
            if len(np_index.shape) == 1:
                np_index = np_index[0]
            np_index = int(np_index)
            shape_val = data[np_index]
            return relax.PrimValue(shape_val)

        # TODO(jwfromm) Make relax.take work with other indices shape.
        return bb.emit_te(topi.take, data, indices, axis)


class Gemm(OnnxOpConverter):
    """Convert an onnx Gemm node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        alpha = attr.get("alpha", None)
        beta = attr.get("beta", None)
        transA = attr.get("transA", False)
        transB = attr.get("transB", False)
        A = inputs[0]
        B = inputs[1]
        C = inputs[2]
        dtype = A.checked_type.dtype

        # Compute Y = alpha * A X B + beta * C

        if alpha is not None and alpha != 1.0:
            A = relax.op.multiply(A, relax.const(alpha, dtype=dtype))

        if transA:
            A = relax.op.permute_dims(A, [1, 0])
        if transB:
            B = relax.op.permute_dims(B, [1, 0])
        Y = relax.op.matmul(A, B)

        if C is not None:
            if beta is not None and beta != 1.0:
                C = relax.op.multiply(C, relax.const(beta, dtype=dtype))
            Y = relax.op.add(Y, C)

        return Y


class Reshape(OnnxOpConverter):
    """Convert an onnx Reshape node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        new_shape = get_constant(inputs[1], params)

        if isinstance(data, relax.ShapeExpr) and isinstance(new_shape, relax.Constant):
            new_shape = new_shape.data.numpy().tolist()
            if new_shape != [-1]:
                raise NotImplementedError("Need to fix this case")
            return data

        if isinstance(data, relax.Constant) and isinstance(new_shape, relax.Constant):
            out = _np.reshape(data.data.numpy(), new_shape.data.numpy().tolist())
            return relax.const(out, out.dtype)
        if isinstance(new_shape, relax.Constant):
            new_shape = new_shape.data.numpy().tolist()
        out = relax.op.reshape(data, new_shape)
        return out


class Gelu(OnnxOpConverter):
    """Operator converter for Gelu from Microsoft onnxruntime contrib opset.

    gelu(x) = 0.5x(1 + erf(x/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.nn.gelu(inputs[0])


class BiasGelu(OnnxOpConverter):
    """Operator converter for BiasGelu from Microsoft onnxruntime contrib opset.

    bias_gelu(x, b) = 0.5(x + b)(1 + erf((x + b)/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        inp = relax.op.add(inputs[0], inputs[1])
        return relax.op.nn.gelu(inp)


class Where(OnnxOpConverter):
    """Convert an onnx Where node into an equivalent Relax expression."""

    @classmethod
    def _impl_v16(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            np_inputs = [inp.data.numpy() for inp in inputs]
            output = _np.where(*np_inputs)
            return relax.const(output, output.dtype)
        if all([isinstance(inp, (relax.Constant, relax.ShapeExpr)) for inp in inputs]):
            condition, x, y = [get_prim_expr_list(inp) for inp in inputs]
            if len(condition) != len(x) or len(condition) != len(y):
                raise ValueError("Cannot broadcast condition to x and y")
            output = [x if c else y for c, x, y in zip(condition, x, y)]
            return relax.ShapeExpr(output)
        return relax.op.where(inputs[0], inputs[1], inputs[2])


class Clip(OnnxOpConverter):
    """Converts an onnx Clip node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        results = inputs[0]
        if inputs[1] is not None:
            results = bb.emit_te(topi.maximum, results, inputs[1])
        if inputs[2] is not None:
            results = bb.emit_te(topi.minimum, results, inputs[2])
        return results


class Equal(OnnxOpConverter):
    """Converts an onnx Equal node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = inputs[0].data.numpy() == inputs[1].data.numpy()
            return relax.const(output, output.dtype)
        elif all([isinstance(inp, (relax.Constant, relax.ShapeExpr)) for inp in inputs]):
            lhs = get_prim_expr_list(inputs[0])
            rhs = get_prim_expr_list(inputs[1])
            if len(lhs) != len(rhs):
                raise ValueError("Cannot compare two tensors with different shapes")
            output = [tvm.ir.structural_equal(l, r) for l, r in zip(lhs, rhs)]
            return relax.const(output, "bool")
        return relax.op.equal(inputs[0], inputs[1])


class Shape(OnnxOpConverter):
    """Converts an onnx Equal node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data_info = inputs[0].struct_info

        if isinstance(data_info, relax.ShapeStructInfo):
            if data_info.ndim == -1:
                raise ValueError("The ndim of ShapeExpr is expected to a real number, but got -1.")
            return relax.ShapeExpr([data_info.ndim])

        # If no shape is defined in the struct info, it must be computed at runtime.
        if not data_info.shape:
            data_shape = bb.normalize(relax.op.shape_of(inputs[0]))
            return data_shape

        return data_info.shape


class Tanh(OnnxOpConverter):
    """Converts an onnx Tanh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        return relax.op.tanh(inputs[0])


class Sqrt(OnnxOpConverter):
    """Converts an onnx Sqrt node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        return relax.op.sqrt(inputs[0])


class Trilu(OnnxOpConverter):
    """Given a 2-D matrix or batches of 2-D matrices, returns the upper or
    lower triangular part of the tensor(s)
    """

    @classmethod
    def _impl_v14(cls, bb, inputs, attr, params):
        upper = attr.get("upper", True)
        x = inputs[0]
        k = inputs[1] if len(inputs) > 1 else 0

        if isinstance(k, relax.Var) and k.name_hint in params:
            k = get_constant(k, params)
        elif isinstance(k, relax.Constant):
            k = int(k.data.numpy()[0])
        else:
            k = 0

        if upper:
            return relax.op.triu(x, k)
        else:
            return relax.op.tril(x, k)


class Relu(OnnxOpConverter):
    """Converts an onnx Relu node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        return relax.op.nn.relu(inputs[0])


class Pow(OnnxOpConverter):
    """Converts an onnx Pow node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        return relax.op.power(inputs[0], inputs[1])


class Conv(OnnxOpConverter):
    """Convert an onnx Conv node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        if hasattr(inputs[0].struct_info, "ndim"):
            ndim = inputs[0].struct_info.ndim
        else:
            ndim = len(inputs[0].struct_info.shape)

        if ndim == 3:
            op = relax.op.nn.conv1d
            data_layout = "NCW"
            kernel_layout = "OIW"
        elif ndim == 4:
            op = relax.op.nn.conv2d
            data_layout = "NCHW"
            kernel_layout = "OIHW"
        elif ndim == 5:
            op = relax.op.nn.conv3d
            data_layout = "NCDHW"
            kernel_layout = "OIDHW"
        else:
            raise NotImplementedError("Ndim > 5 not supported for convolution.")

        conv_out = bb.normalize(
            op(
                data=inputs[0],
                weight=inputs[1],
                strides=attr.get("strides", 1),
                padding=attr.get("pads", 0),
                dilation=attr.get("dilation", 1),
                groups=attr.get("group", 1),
                data_layout=data_layout,
                kernel_layout=kernel_layout,
            )
        )
        if inputs[2] is not None:
            bias = relax.op.reshape(
                inputs[2],
                [1, -1]
                + [
                    1,
                ]
                * (ndim - 2),
            )
            conv_out = relax.op.add(conv_out, bias)

        return conv_out


class Erf(OnnxOpConverter):
    """Converts an onnx Erf node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        return relax.op.erf(inputs[0])


class CumSum(OnnxOpConverter):
    """Converts an onnx CumSum node into an equivalent Relax expression."""

    @classmethod
    def _impl_v14(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis = get_constant(inputs[1], params)
        assert not attr.get("exclusive", False), "Exclusive option not yet supported."

        if isinstance(axis, relax.Constant):
            axis = int(axis.data.numpy())
        elif isinstance(axis, relax.Var):
            axis = 0
        data = relax.op.cumsum(data, axis)
        if attr.get("reverse", 0) != 0:
            data = bb.emit_te(topi.flip, data, axis=axis if axis else 0)
        return data


class Squeeze(OnnxOpConverter):
    """Converts an onnx Squeeze node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = get_constant(inputs[1], params)
        if isinstance(axis, relax.Constant):
            axis = [int(x) for x in axis.data.numpy()]
        # If data is constant, perform computation directly.
        if isinstance(inputs[0], relax.Constant):
            out_data = _np.squeeze(inputs[0].data.numpy(), axis)
            return relax.const(out_data, inputs[0].struct_info.dtype)
        return relax.op.squeeze(inputs[0], axis)


class Constant(OnnxOpConverter):
    """Converts an onnx Constant node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if "value" not in attr:
            raise ValueError("no value in Constant")
        value = attr.pop("value")
        # Constants may rarely have string types. These are likely exported
        # from other frameworks and not actually used in TVM. We'll just use
        # a zero valued constant for compatibility.
        if isinstance(value, bytes):
            np_value = _np.asarray([0]).astype("int64")
        else:
            np_value = get_numpy(value)
        dtype = np_value.dtype.name
        value = relax.const(np_value, dtype)
        return value


class ConstantOfShape(OnnxOpConverter):
    """Converts an onnx ConstantOfShape node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        shape = inputs[0]
        value = get_numpy(attr.get("value", 0))
        if isinstance(value, _np.ndarray):
            dtype = str(value.dtype)
        else:
            dtype = "float32"
        # If shape is a constant, treat it as a ShapeExpr.
        if isinstance(shape, relax.Constant):
            shape = relax.ShapeExpr(list(shape.data.numpy()))

        # Special case where requested shape are constant
        if len(shape) == 1 and all([isinstance(x, tir.IntImm) for x in shape]):
            shape = [int(x) for x in shape]
            return relax.const(_np.full(shape, value, dtype), dtype)

        # Convert to shape expression from tensor if needed.
        if not isinstance(shape, relax.ShapeExpr):
            shape = relax.op.tensor_to_shape(shape)

        return relax.op.broadcast_to(relax.const(value, dtype), shape)


class Sub(OnnxOpConverter):
    """Converts an onnx Sub node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = inputs[0].data.numpy() - inputs[1].data.numpy()
            return relax.const(output, output.dtype)
        return relax.op.subtract(inputs[0], inputs[1])


class Sin(OnnxOpConverter):
    """Converts an onnx Sin node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.sin(inputs[0])


class Cos(OnnxOpConverter):
    """Converts an onnx Cos node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.cos(inputs[0])


class Neg(OnnxOpConverter):
    """Converts an onnx Neg node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if isinstance(inputs[0], relax.Constant):
            data_np = inputs[0].data.numpy()
            return relax.const(_np.negative(data_np), inputs[0].struct_info.dtype)
        return relax.op.negative(inputs[0])


class Abs(OnnxOpConverter):
    """Converts an onnx Abs node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if isinstance(inputs[0], relax.Constant):
            output = _np.abs(inputs[0].data.numpy())
            return relax.const(output, output.dtype)
        return relax.op.abs(inputs[0])


class Min(OnnxOpConverter):
    """Converts an onnx Min node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            np_inputs = [inp.data.numpy() for inp in inputs]
            output = _np.minimum(*np_inputs)
            return relax.const(output, output.dtype)

        # Expand inputs, stack them, then perform minimum over the new axis.
        inputs = [bb.normalize(relax.op.expand_dims(i, axis=0)) for i in inputs]
        stacked_tensor = relax.op.concat(inputs, axis=0)
        return relax.op.min(stacked_tensor, axis=0)


class Max(OnnxOpConverter):
    """Converts an onnx Max node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            np_inputs = [inp.data.numpy() for inp in inputs]
            output = _np.maximum(*np_inputs)
            return relax.const(output, output.dtype)

        # Expand inputs, stack them, then perform maximum over the new axis.
        inputs = [bb.normalize(relax.op.expand_dims(i, axis=0)) for i in inputs]
        stacked_tensor = relax.op.concat(inputs, axis=0)
        return relax.op.max(stacked_tensor, axis=0)


class Log(OnnxOpConverter):
    """Converts an onnx Log node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if isinstance(inputs[0], relax.Constant):
            return relax.const(_np.log(inputs[0].data.numpy()), inputs[0].struct_info.dtype)
        return relax.op.log(inputs[0])


class Exp(OnnxOpConverter):
    """Converts an onnx Exp node into an equivalent Relax expression."""

    @classmethod
    def _check_type(cls, dtype, valid_types):
        assert dtype in valid_types, "Types {} are supported only, but {} is given".format(
            valid_types, dtype
        )

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        data = inputs[0]
        valid_types = ["float", "float32", "double", "float64", "float16"]
        cls._check_type(data.checked_type.dtype, valid_types)

        return relax.op.exp(data)

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        valid_types = ["float", "float32", "double", "float64", "float16", "bfloat16"]
        cls._check_type(data.checked_type.dtype, valid_types)

        return relax.op.exp(data)


class Less(OnnxOpConverter):
    """Converts an onnx Less node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = _np.less(inputs[0].data.numpy(), inputs[1].data.numpy())
            return relax.const(output, output.dtype)
        return relax.op.less(inputs[0], inputs[1])


class LessOrEqual(OnnxOpConverter):
    """Converts an onnx LessOrEqual node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = _np.less_equal(inputs[0].data.numpy(), inputs[1].data.numpy())
            return relax.const(output, output.dtype)
        return relax.op.less_equal(inputs[0], inputs[1])


class Split(OnnxOpConverter):
    """Converts an onnx Split node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
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
        return bb.emit_te(topi.split, inputs[0], indices, attr.get("axis", 0))

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        splits = inputs[1]
        splits_rank = None
        if splits is not None:
            splits_rank = splits.checked_type.ndim
        if splits is not None and splits_rank > 0:
            if isinstance(splits, relax.Constant):
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
        return bb.emit_te(topi.split, inputs[0], indices, axis=attr.get("axis", 0))


class Slice(OnnxOpConverter):
    """Converts an onnx Splice node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        # TODO (jwfromm) currently only supports constant parameters.
        data = inputs[0]
        starts = get_constant(inputs[1], params)
        ends = get_constant(inputs[2], params)
        axes = get_constant(inputs[3], params)
        steps = get_constant(inputs[4], params)
        if not all(
            [
                (
                    isinstance(param, (relax.Constant, relax.ShapeExpr, relax.PrimValue))
                    or param is None
                )
                for param in [starts, ends, axes, steps]
            ]
        ):
            raise ValueError("Only constant Slice parameters are currently supported.")
        # Convert parameters to constant lists.
        starts = get_prim_expr_list(starts)
        ends = get_prim_expr_list(ends)
        if axes is not None:
            axes = get_prim_expr_list(axes)
        else:
            axes = list(range(len(starts)))
        # Convert negative axis to positive if needed.
        for i, axis in enumerate(axes):
            if axis < 0:
                axes[i] = axis + len(data.struct_info.shape)
        if steps is not None:
            steps = get_prim_expr_list(steps)
        else:
            steps = [1] * len(axes)
        # If input is a shape tensor, we can directly extract it.
        if isinstance(data, relax.ShapeExpr):
            shape_data = [dim.value for dim in data]
            # Starts, ends, and steps must be 1-d for shape operation.
            assert all(len(i) == 1 for i in [starts, ends, steps])
            sliced_values = shape_data[starts[0] : ends[0] : steps[0]]
            return relax.const(sliced_values, "int64")
        # If all `starts`, `ends`, and `steps` are constant, use strict mode
        # Otherwise, we assume the slice is inbound.
        assume_inbound = not all(
            [isinstance(param, (tir.IntImm, int)) for param in [*starts, *ends, *steps]]
        )
        # return relax.op.strided_slice(data, axes, starts, ends, steps)
        return relax.op.strided_slice(
            data, axes, starts, ends, steps, assume_inbound=assume_inbound
        )


class Pad(OnnxOpConverter):
    """Converts an onnx Pad node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        pads = get_constant(inputs[1], params)
        constant_value = get_constant(inputs[2], params)
        if constant_value is not None:
            constant_value = constant_value.data.numpy().item()
        else:
            constant_value = 0.0

        if isinstance(pads, relax.Constant):
            pad_before, pad_after = _np.split(pads.data.numpy(), 2)
            pad_before = _np.ndarray.tolist(pad_before)
            pad_after = _np.ndarray.tolist(pad_after)
        else:
            raise ValueError("Dynamic pads are not supported yet.")

        pad_mode = attr.get("mode", b"constant").decode("utf-8")
        if not pad_mode in ["constant", "edge", "reflect"]:
            raise tvm.error.OpAttributeInvalid(
                "Value " + pad_mode + ' in attribute "mode" is invalid for operator Pad.'
            )

        if pad_mode == "constant":
            return bb.emit_te(topi.nn.pad, inputs[0], pad_before, pad_after, constant_value)
        elif pad_mode == "reflect":
            return bb.emit_te(topi.nn.mirror_pad, inputs[0], pad_before, pad_after, "REFLECT")
        else:
            # TODO(gigiblender) Support edge mode.
            raise NotImplementedError("Pad mode {} not implemented".format(pad_mode))


class Tile(OnnxOpConverter):
    """Converts an onnx Tile node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        reps = get_constant(inputs[1], params)
        if isinstance(reps, relax.Constant):
            reps = reps.data.numpy().tolist()
        else:
            raise ValueError("Dynamic reps for Tile are supported yet.")
        return bb.emit_te(topi.tile, inputs[0], reps)


class Expand(OnnxOpConverter):
    """Converts an onnx Expand node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        shape = inputs[1]

        if isinstance(shape, relax.ShapeExpr):
            return relax.op.broadcast_to(data, shape)

        # If possible, directly expand to constant shape.
        if isinstance(shape, relax.Constant):
            new_shape = shape.data.numpy().tolist()
            # For some reason, onnx allows target shapes to be smaller than input shapes.
            # We need to go correct it.
            data_shape = [dim.value for dim in data.struct_info.shape]
            # Fix small target shapes.
            for i, s in enumerate(new_shape):
                if i < len(data_shape) and s < data_shape[i]:
                    new_shape[i] = data_shape[i]
            # If the new shape matches the input shape, no transformation is needed.
            if new_shape == data_shape:
                return data
            return relax.op.broadcast_to(data, relax.ShapeExpr(new_shape))

        # Otherwise handle dynamic shapes.
        shape_ndim = [dim.value for dim in shape.struct_info.shape.values][0]
        shape_dataflow_var = bb.emit(
            relax.Call(
                relax.ExternFunc("vm.builtin.tensor_to_shape"),
                [shape],
                sinfo_args=[relax.ShapeStructInfo(ndim=shape_ndim)],
            )
        )

        shape_vars = []
        for i in range(shape_ndim):
            shape_vars.append(tvm.tir.Var("x_%d" % i, "int64"))
        bb.match_cast(shape_dataflow_var, relax.ShapeStructInfo(shape_vars))
        return bb.normalize(relax.op.broadcast_to(data, relax.ShapeExpr(shape_vars)))


class Attention(OnnxOpConverter):
    """Converts an onnx.microsoft Attention node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        num_heads = attr["num_heads"]

        assert "do_rotary" not in attr, "rotary position embedding is not currently supported"
        assert (
            "past_present_share_buffer" not in attr
        ), "past state for key and value is not currently supported"
        assert "scale" not in attr, "custom scale is not currently supported"
        assert "unidirectional" not in attr, "unidirectional attention is not currently supported"

        if "mask_filter_value" in attr:
            mask_filter_value = attr["mask_filter_value"]
        else:
            mask_filter_value = -10000.0

        # (batch_size, sequence_length, input_hidden_size)
        input_emb = bb.normalize(inputs[0])

        # (input_hidden_size, hidden_size + hidden_size + v_hidden_size)
        weight = bb.normalize(inputs[1])

        def optional_input(k: int):
            if inputs[k] is not None:
                return bb.normalize(inputs[k])
            else:
                return None

        # (hidden_size + hidden_size + v_hidden_size)
        bias = optional_input(2)

        # 1. (    batch_size,             1,   max_seq_len, max_seq_len,)
        # 2. (    batch_size, total_seq_len,)
        # 3. (    batch_size,       seq_len, total_seq_len,)
        # 4. (    batch_size,)
        # 5. (2 * batch_size,)
        # For now, we only support case 2 & 3.
        mask_index = optional_input(3)

        # (2, batch_size, num_heads, past_sequence_length, head_size)
        assert inputs[4] is None, "past state for key and value is not currently supported"

        # (batch_size, num_heads, sequence_length, total_sequence_length)
        qk_bias = optional_input(5)

        assert inputs[6] is None, "past_sequence_length is not currently supported"

        (batch_size, seq_len, input_hidden_size) = [
            val.value for val in input_emb.struct_info.shape.values
        ]
        weight_shape = [val.value for val in weight.struct_info.shape.values]

        assert (
            weight_shape[0] == input_hidden_size
        ), "input and weight should share the same input hiden size"

        if "qkv_hidden_sizes" in attr:
            assert (
                attr["qkv_hidden_sizes"][0] == attr["qkv_hidden_sizes"][1]
            ), "Q and K should share the same hidden sizes"
            hidden_size, _, hidden_size_v = attr["qkv_hidden_sizes"]
        else:
            hidden_size = hidden_size_v = weight_shape[1] // 3

        assert (
            hidden_size % num_heads == 0
        ), "hidden size should be divisible by number of attention heads"
        head_size = hidden_size // num_heads
        head_size_v = hidden_size_v // num_heads

        if mask_index is not None:
            mask_index_shape = [val.value for val in mask_index.struct_info.shape.values]
            assert mask_index_shape in (
                [batch_size, seq_len],
                [
                    batch_size,
                    seq_len,
                    seq_len,
                ],
            ), """mask index should be in shape of (batch_size, seq_len),
            or (batch_size, seq_len, seq_len)"""
            mask_bias = relax.op.subtract(
                relax.const(1, dtype=mask_index.struct_info.dtype), mask_index
            )
            mask_bias = relax.op.astype(mask_bias, dtype=input_emb.struct_info.dtype)
            mask_bias = bb.normalize(
                relax.op.multiply(
                    mask_bias,
                    relax.const(mask_filter_value, dtype=input_emb.struct_info.dtype),
                )
            )
            if qk_bias is None:
                qk_bias = mask_bias
            else:
                if len(mask_index_shape) == 2:
                    mask_bias = bb.normalize(
                        relax.op.reshape(mask_bias, [batch_size, 1, 1, seq_len])
                    )
                elif len(mask_index_shape) == 3:
                    mask_bias = bb.normalize(
                        relax.op.reshape(mask_bias, [batch_size, 1, seq_len, seq_len])
                    )
                qk_bias = bb.normalize(relax.op.add(qk_bias, mask_bias))

        QKV = relax.op.matmul(input_emb, weight)

        if bias:
            bias_shape = [val.value for val in bias.struct_info.shape.values]
            assert (
                bias_shape[0] == weight_shape[1]
            ), "bias and weight should share the same hidden size sum"
            QKV = relax.op.add(QKV, bias)

        QKV = relax.op.split(QKV, [hidden_size, hidden_size * 2], 2)
        Q, K, V = QKV[0], QKV[1], QKV[2]

        Q = bb.normalize(relax.op.reshape(Q, (batch_size, seq_len, num_heads, head_size)))
        K = bb.normalize(relax.op.reshape(K, (batch_size, seq_len, num_heads, head_size)))
        V = bb.normalize(relax.op.reshape(V, (batch_size, seq_len, num_heads, head_size_v)))
        output = relax.op.nn.attention(Q, K, V, qk_bias)
        output = bb.normalize(
            relax.op.reshape(output, (batch_size, seq_len, num_heads * head_size_v))
        )
        # add placeholder for optional present state supported in the future
        placeholder = relax.const(0, dtype="float32")
        return relax.Tuple([output, placeholder])


class Identity(OnnxOpConverter):
    """Converts an onnx Identity node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return inputs[0]


class Resize(OnnxOpConverter):
    """Converts an onnx Resize node into an equivalent Relax expression."""

    @classmethod
    def _impl_v18(cls, bb, inputs, attr, params):
        # Extract the many attributes of resize.
        coord_mode = attr.get("coordinate_transformation_mode", b"half_pixel").decode("ascii")
        cubic_coeff_a = attr.get("cubic_coeff_a", -0.75)
        exclude_outside = attr.get("exclude_outside", 0)
        extrapolation_value = attr.get("extrapolation_value", 0.0)
        mode = attr.get("mode", b"nearest").decode("ascii")
        rounding_method = attr.get("nearest_mode", b"round_prefer_floor").decode("ascii")

        # Adapt attributes to fit TVM definition.
        if mode == "nearest":
            mode = "nearest_neighbor"

        # Unpack inputs.
        x = inputs[0]
        roi = get_constant(inputs[1], params)
        scales = get_constant(inputs[2], params)
        sizes = get_constant(inputs[3], params)
        ndims = len(x.struct_info.shape)
        assert ndims == 4, "Only resize2d is currently supported."

        assert (
            scales is None or sizes is None
        ), "Only one of scales and sizes can be provided in Resize."

        # Define relax implementation.
        if roi is not None:
            roi = relax.op.concat(
                [
                    relax.op.strided_slice(roi, axes=[0], begin=[2], end=[ndims]),
                    relax.op.strided_slice(roi, axes=[0], begin=[ndims + 2], end=[2 * ndims]),
                ],
                axis=0,
            )
        else:
            roi = [0.0] * 4

        # Convert scales to sizes if needed.
        if scales is not None:
            assert isinstance(scales, relax.Constant), "Only constant scales currently supported."
            scales = scales.data.numpy()
            sizes = []

            for i, dim in enumerate(x.struct_info.shape):
                sizes.append(cast(scales[i] * dim, "int64"))
            sizes = sizes[2:]
        else:
            assert isinstance(
                sizes, relax.Constant
            ), "Only constant output size currently supported."
            sizes = sizes.data.numpy().astype("int64").tolist()[2:]

        return relax.op.image.resize2d(
            x,
            size=relax.ShapeExpr(sizes),
            roi=roi,
            layout="NCHW",
            method=mode,
            coordinate_transformation_mode=coord_mode,
            rounding_method=rounding_method,
            cubic_alpha=cubic_coeff_a,
            cubic_exclude=exclude_outside,
            extrapolation_value=extrapolation_value,
        )


class Einsum(OnnxOpConverter):
    """Converts an onnx Einsum node into an equivalent Relax expression."""

    @classmethod
    def _impl_v12(cls, bb, inputs, attr, params):
        equation = attr["equation"].decode("utf-8")
        return bb.emit_te(topi.einsum, equation, *inputs)


class Range(OnnxOpConverter):
    """Converts an onnx Range node into an equivalent Relax expression."""

    @classmethod
    def _impl_v12(cls, bb, inputs, attr, params):
        start = get_constant(inputs[0], params)
        limit = get_constant(inputs[1], params)
        delta = get_constant(inputs[2], params)
        out_dtype = start.struct_info.dtype

        if isinstance(start, relax.Constant):
            start = start.data.numpy().tolist()

        if isinstance(limit, relax.Constant):
            limit = limit.data.numpy().tolist()

        assert isinstance(delta, relax.Constant), "Constant delta required for Range."
        step = delta.data.numpy().tolist()

        # If all inputs are constant, compute directly.
        if isinstance(start, int) and isinstance(limit, int):
            out_range = _np.arange(start=start, stop=limit, step=step)
            return relax.const(out_range, out_dtype)

        # Otherwise compute in graph.
        return bb.emit_te(topi.arange, start, limit, step, out_dtype)


class InstanceNormalization(OnnxOpConverter):
    """Converts an onnx InstanceNormalization node into an equivalent Relax expression."""

    @classmethod
    def _impl_v6(cls, bb, inputs, attr, params):
        data = inputs[0]
        scale = inputs[1]
        B = inputs[2]
        epsilon = attr.get("epsilon", 1e-05)
        epsilon = relax.const(epsilon, dtype=data.struct_info.dtype)

        ndim = len(data.struct_info.shape)
        redux_axes = list(range(2, ndim))

        mean = relax.op.mean(data, axis=redux_axes, keepdims=True)
        var = relax.op.variance(data, axis=redux_axes, keepdims=True)
        sqrt = relax.op.sqrt(relax.op.add(var, epsilon))
        out = relax.op.divide(relax.op.subtract(data, mean), sqrt)
        broadcast_shape = [-1] + [
            1,
        ] * (ndim - 2)
        if scale is not None:
            scale = relax.op.reshape(scale, broadcast_shape)
            out = relax.op.multiply(out, scale)
        if B is not None:
            B = relax.op.reshape(B, broadcast_shape)
            out = relax.op.add(out, B)
        return out


class BatchNormalization(OnnxOpConverter):
    """Converts an onnx BatchNormalization node into an equivalent Relax expression."""

    @classmethod
    def _impl_v15(cls, bb, inputs, attr, params):
        # Unpack inputs
        data = inputs[0]
        scale = inputs[1]
        bias = inputs[2]
        mean = inputs[3]
        var = inputs[4]
        epsilon = attr.get("epsilon", 1e-05)
        return relax.op.nn.batch_norm(
            data, gamma=scale, beta=bias, moving_mean=mean, moving_var=var, epsilon=epsilon, axis=1
        )


class MaxPool(OnnxOpConverter):
    """Converts an onnx MaxPool node into an equivalent Relax expression."""

    @classmethod
    def _impl_v12(cls, bb, inputs, attr, params):
        # Unpack inputs and attributes.
        data = inputs[0]
        auto_pad = attr.get("auto_pad", b"NOTSET").decode("utf-8")
        ceil_mode = attr.get("ceil_mode", 0)
        dilations = attr.get("dilations", [1, 1])
        kernel_shape = attr.get("kernel_shape")
        pads = attr.get("pads", 0)
        strides = attr.get("strides", [1, 1])

        assert len(kernel_shape) == 2, "Currently only 2D pooling is supported."
        assert auto_pad in [
            "NOTSET",
            "SAME_UPPER",
            "SAME_LOWER",
            "VALID",
        ], f"Value {auto_pad} in attribute auto_pad is invalid."

        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            input_spatial_shape = cls._get_input_spatial_shape(data)
            output_spatial_shape = [0 for _ in input_spatial_shape]

            pads = _np.array([(0, 0) for _ in range(len(kernel_shape))])

            for i, _ in enumerate(input_spatial_shape):
                if auto_pad == "SAME_UPPER":
                    output_spatial_shape[i] = int(_np.ceil(input_spatial_shape[i] / strides[i]))
                else:
                    output_spatial_shape[i] = int(_np.floor(input_spatial_shape[i] / strides[i]))
                pad_i = (
                    (output_spatial_shape[i] - 1) * strides[i]
                    + ((kernel_shape[i] - 1) * dilations[i] + 1)
                    - input_spatial_shape[i]
                )
                if auto_pad == "SAME_UPPER":
                    pads[i, 0] = pad_i // 2
                    pads[i, 1] = pad_i - pads[i, 0]
                else:
                    pads[i, 1] = pad_i // 2
                    pads[i, 0] = pad_i - pads[i, 1]

            # TODO(agladyshev): for now we support only 2D kernel
            # (top, left, bottom, right)
            flatten_pads = [pads[0][0], pads[1][0], pads[0][1], pads[1][1]]
            pads = tuple(flatten_pads)

        return relax.op.nn.max_pool2d(data, kernel_shape, strides, pads, dilations, ceil_mode)

    @classmethod
    def _get_input_spatial_shape(cls, tensor):
        # shape is (N x C x D1 x D2 ... Dn)
        return _np.array([int(d) for d in tensor.struct_info.shape], dtype="int64")[2:]


class GlobalAveragePool(OnnxOpConverter):
    """Converts an onnx GlobalAveragePool node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.nn.adaptive_avg_pool2d(inputs[0], 1)


class Flatten(OnnxOpConverter):
    """Converts an onnx Flatten node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 1)
        data_shape = [i.value for i in inputs[0].struct_info.shape]
        new_shape = (1, -1) if axis == 0 else (_np.prod(data_shape[0:axis]).astype("int64"), -1)
        return relax.op.reshape(inputs[0], new_shape)


class LayerNormalization(OnnxOpConverter):
    """Converts an onnx LayerNormalization node into an equivalent Relax expression."""

    @classmethod
    def _impl_v17(cls, bb, inputs, attr, params):
        data = inputs[0]
        scale = inputs[1]
        bias = inputs[2]
        axis = attr.get("axis", -1)
        epsilon = attr.get("epsilon", 1e-05)

        output = relax.op.nn.layer_norm(data, scale, bias, axis, epsilon)
        # Onnx layernorm has 3 outputs but only the first is used.
        # We construct two empty constants for this.
        placeholder = relax.const(0, dtype="float32")
        return relax.Tuple([output, placeholder, placeholder])


class ReduceMax(OnnxOpConverter):
    """Converts an onnx ReduceMax node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.max(data, axes, keepdims)


class ReduceMin(OnnxOpConverter):
    """Converts an onnx ReduceMin node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.min(data, axes, keepdims)


class ReduceSum(OnnxOpConverter):
    """Converts an onnx ReduceSum node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.sum(data, axes, keepdims)

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = inputs[1]
        keepdims = attr.get("keepdims", 1)
        assert isinstance(axes, relax.Constant), "Only constant axes currently supported."
        axes = axes.data.numpy().tolist()
        return relax.op.sum(data, axes, keepdims)


class ReduceMean(OnnxOpConverter):
    """Converts an onnx ReduceMean node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.mean(data, axes, keepdims)


class ReduceProd(OnnxOpConverter):
    """Converts an onnx ReduceProd node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.prod(data, axes, keepdims)


class ReduceLogSumExp(OnnxOpConverter):
    """Converts an onnx ReduceLogSumExp node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        x = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        max_x = relax.op.max(x, axes, True)
        exp_x = relax.op.exp(relax.op.subtract(x, max_x))
        sum_x = relax.op.sum(exp_x, axes, True)
        out_x = relax.op.add(relax.op.log(sum_x), max_x)
        if not keepdims:
            out_x = relax.op.squeeze(out_x, axes)
        return out_x


class ReduceLogSum(OnnxOpConverter):
    """Converts an onnx ReduceLogSum node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.log(relax.op.sum(data, axes, keepdims))


class ReduceSumSquare(OnnxOpConverter):
    """Converts an onnx ReduceSumSquare node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.sum(relax.op.multiply(data, data), axes, keepdims)


class ReduceL1(OnnxOpConverter):
    """Converts an onnx ReduceL1 node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.sum(relax.op.abs(data), axes, keepdims)


class ReduceL2(OnnxOpConverter):
    """Converts an onnx ReduceL2 node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        data = inputs[0]
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return relax.op.sqrt(relax.op.sum(relax.op.multiply(data, data), axes, keepdims))


class ArgMax(OnnxOpConverter):
    """Converts an onnx ArgMax node into an equivalent Relax expression."""

    @classmethod
    def _check_attrs(cls, data, attr, shift_axis=True):
        dims_num = len(data.struct_info.shape)
        axis = attr.get("axis", 0)
        if shift_axis and axis < 0:
            axis += dims_num
        assert 0 <= axis < dims_num, "Axis is out of bounds"
        keepdims = attr.get("keepdims", True)
        return axis, keepdims

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis, keepdims = cls._check_attrs(data, attr, False)
        return relax.op.argmax(data, axis, keepdims)

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis, keepdims = cls._check_attrs(data, attr)
        return relax.op.argmax(data, axis, keepdims)

    @classmethod
    def _impl_v12(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis, keepdims = cls._check_attrs(data, attr)
        select_last_index = attr.get("select_last_index", False)
        if select_last_index:
            # TODO(vvchernov): support attr
            raise tvm.error.OpAttributeUnImplemented(
                "'select_last_index' attribute has not been supported yet"
            )
        return relax.op.argmax(data, axis, keepdims)


class ArgMin(OnnxOpConverter):
    """Converts an onnx ArgMin node into an equivalent Relax expression."""

    @classmethod
    def _check_attrs(cls, data, attr, shift_axis=True):
        dims_num = len(data.struct_info.shape)
        axis = attr.get("axis", 0)
        if shift_axis and axis < 0:
            axis += dims_num
        assert 0 <= axis < dims_num, "Axis is out of bounds"
        keepdims = attr.get("keepdims", True)
        return axis, keepdims

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis, keepdims = cls._check_attrs(data, attr, False)
        return relax.op.argmin(data, axis, keepdims)

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis, keepdims = cls._check_attrs(data, attr)
        return relax.op.argmin(data, axis, keepdims)

    @classmethod
    def _impl_v12(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis, keepdims = cls._check_attrs(data, attr)
        select_last_index = attr.get("select_last_index", False)
        if select_last_index:
            # TODO(vvchernov): support attr
            raise tvm.error.OpAttributeUnImplemented(
                "'select_last_index' attribute has not been supported yet"
            )
        return relax.op.argmin(data, axis, keepdims)


class SkipLayerNormalization(OnnxOpConverter):
    """Converts a microsoft contrib SkipLayerNormalization node into a Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        data = inputs[0]
        skip = inputs[1]
        gamma = inputs[2]
        beta = inputs[3]
        bias = inputs[4]

        assert (
            beta is not None and bias is not None
        ), "SkipLayerNormalization import currently only supports required beta and bias"

        epsilon = attr.get("epsilon", 1e-12)

        data = relax.op.add(data, skip)
        if bias is not None:
            data = relax.op.add(data, bias)

        output = relax.op.nn.layer_norm(data, gamma, beta, axes=-1, epsilon=epsilon)

        # Expects three outputs though only the first is used. Construct a placeholder for others.
        placeholder = relax.const(0, dtype="float32")
        return relax.Tuple([output, placeholder, placeholder])


class EmbedLayerNormalization(OnnxOpConverter):
    """Converts a microsoft contrib EmbedLayerNormalization node into a Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        input_ids = inputs[0]
        segment_ids = inputs[1]
        word_emb = inputs[2]
        pos_emb = inputs[3]
        segment_emb = inputs[4]
        gamma = inputs[5]
        beta = inputs[6]
        mask = inputs[7]
        pos_ids = inputs[8]

        epsilon = attr.get("epsilon", 1e-12)

        (batch_size, seq_len) = [dim.value for dim in input_ids.struct_info.shape]

        if segment_ids:
            assert segment_emb

        if pos_ids is None:
            pos_ids = relax.const([list(range(seq_len))] * batch_size, dtype="int64")
        # TODO(jwfromm) Replace with relax ops once take has better support.
        word_vec = bb.emit_te(topi.take, word_emb, input_ids, 0)
        if segment_ids:
            segment_vec = bb.emit_te(topi.take, segment_emb, segment_ids, 0)
        pos_vec = bb.emit_te(topi.take, pos_emb, pos_ids, 0)

        vec_sum = relax.op.add(word_vec, pos_vec)
        if segment_ids:
            vec_sum = relax.op.add(vec_sum, segment_vec)

        ln = relax.op.nn.layer_norm(vec_sum, gamma, beta, axes=-1, epsilon=epsilon)

        mask_index = relax.const(_np.zeros((batch_size,), dtype="int64"))
        if mask:
            # Caculate number of words per sentence.
            mask_index = relax.op.sum(mask, axis=1)

        return relax.Tuple([ln, mask_index])


class Greater(OnnxOpConverter):
    """Converts an onnx Greater node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = _np.greater(inputs[0].data.numpy(), inputs[1].data.numpy())
            return relax.const(output, output.dtype)
        return relax.op.greater(inputs[0], inputs[1])


class Reciprocal(OnnxOpConverter):
    """Converts an onnx Reciprocal node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        input_dtype = inputs[0].struct_info.dtype
        return relax.op.divide(relax.const(1, dtype=input_dtype), inputs[0])


class OneHot(OnnxOpConverter):
    """Converts an onnx OneHot node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        indices = inputs[0]
        depth = get_constant(inputs[1], params)
        values = get_constant(inputs[2], params)
        axis = attr.get("axis", -1)
        dtype = values.struct_info.dtype
        assert isinstance(depth, relax.Constant), "Only constant depth currently supported."
        depth = depth.data.numpy().tolist()
        assert isinstance(values, relax.Constant), "Only constant values currently supported."
        values = values.data.numpy().tolist()
        off_value, on_value = values
        return bb.emit_te(topi.one_hot, indices, on_value, off_value, depth, axis, dtype)


class Elu(OnnxOpConverter):
    """Converts an onnx Elu node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        alpha = float(attr.get("alpha", 1.0))
        return relax.expr.const(-alpha) * relax.op.nn.relu(
            relax.expr.const(1.0) - relax.op.exp(inputs[0])
        ) + relax.op.nn.relu(inputs[0])


def _get_convert_map():
    return {
        "MatMul": MatMul,
        "Concat": Concat,
        "Add": Add,
        "Mul": Mul,
        "Cast": Cast,
        "Gather": Gather,
        "Gemm": Gemm,
        "Reshape": Reshape,
        "Div": Div,
        "Sigmoid": Sigmoid,
        "Softmax": Softmax,
        "Transpose": Transpose,
        "Unsqueeze": Unsqueeze,
        "Gelu": Gelu,
        "BiasGelu": BiasGelu,
        "Where": Where,
        "Clip": Clip,
        "Equal": Equal,
        "Shape": Shape,
        "Tanh": Tanh,
        "Sqrt": Sqrt,
        "Trilu": Trilu,
        "Relu": Relu,
        "Conv": Conv,
        "Pow": Pow,
        "Erf": Erf,
        "CumSum": CumSum,
        "Squeeze": Squeeze,
        "Constant": Constant,
        "Sub": Sub,
        "Sin": Sin,
        "Cos": Cos,
        "Neg": Neg,
        "Abs": Abs,
        "Min": Min,
        "Max": Max,
        "Log": Log,
        "Exp": Exp,
        "Less": Less,
        "LessOrEqual": LessOrEqual,
        "LayerNormalization": LayerNormalization,
        "SkipLayerNormalization": SkipLayerNormalization,
        "EmbedLayerNormalization": EmbedLayerNormalization,
        "InstanceNormalization": InstanceNormalization,
        # defs/reduction
        "ReduceMax": ReduceMax,
        "ReduceMin": ReduceMin,
        "ReduceSum": ReduceSum,
        "ReduceMean": ReduceMean,
        "ReduceProd": ReduceProd,
        "ReduceLogSumExp": ReduceLogSumExp,
        "ReduceLogSum": ReduceLogSum,
        "ReduceSumSquare": ReduceSumSquare,
        "ReduceL1": ReduceL1,
        "ReduceL2": ReduceL2,
        "ArgMax": ArgMax,
        "ArgMin": ArgMin,
        "Expand": Expand,
        "ConstantOfShape": ConstantOfShape,
        "Slice": Slice,
        "Attention": Attention,
        "Pad": Pad,
        "Split": Split,
        "Tile": Tile,
        "BatchNormalization": BatchNormalization,
        "GlobalAveragePool": GlobalAveragePool,
        "Flatten": Flatten,
        "MaxPool": MaxPool,
        "Identity": Identity,
        "Resize": Resize,
        "Einsum": Einsum,
        "Range": Range,
        "Greater": Greater,
        "Reciprocal": Reciprocal,
        "OneHot": OneHot,
        "Elu": Elu,
    }


class ONNXGraphImporter:
    """A helper class for handling Relax expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

    Parameters
    ----------
    shape_dict : dict of str to tuple, optional
        The input shape to the graph
    dtype_dict : str or dict of str to str
        The input types to the graph
    keep_params_in_input : bool
        If True, parameters will be treated as input variables. If false,
        parameters are treated as constant and folded directly into the graph.
    sanitize : bool
        Whether to sanitize the input names to be valid Relax identifiers.
    """

    current = None

    def __init__(
        self,
        shape_dict: Dict[str, List],
        dtype_dict: Union[str, Dict[str, str]],
        keep_params_in_input: bool = False,
        sanitize: bool = True,
    ):
        self._nodes: Dict[str, relax.Expr] = {}
        self._inputs: Dict[str, relax.Var] = {}
        self._num_input: int = 0
        self._shape = shape_dict.copy() if shape_dict else {}
        self._input_names: List[str] = []
        self._dtype = dtype_dict
        self.opset: int = None
        self._name_supply = NameSupply()
        self._keep_params_in_input = keep_params_in_input
        self._sanitize: bool = sanitize
        self.bb: relax.BlockBuilder = relax.BlockBuilder()  # pylint: disable=invalid-name
        self._params = {}

    def from_onnx(self, graph: onnx.onnx_ml_pb2.ModelProto, opset: int) -> IRModule:
        """Construct Relax expressions from the ONNX graph.
        Onnx graph is a python protobuf object.

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph
        opset : opset version
        Returns
        -------
        mod : tvm.IRModule
            The returned relax module
        """
        with self.bb.function("main"):
            with self.bb.dataflow() as df:  # pylint: disable=invalid-name, unused-variable
                self.opset = opset
                self._parse_graph_initializers(graph)
                self._parse_graph_input(graph)
                self._check_for_unsupported_ops(graph)
                self._construct_nodes(graph)

                # now return the outputs
                outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
                outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                output_var = self.bb.emit_output(outputs)

            # Create function attributes for this module
            func_attrs = {"num_input": self._num_input}
            # Create a function from our output expression and all input variables.
            input_list = [value for value in self._inputs.values() if isinstance(value, relax.Var)]
            # Attach params if they are available.
            if self._keep_params_in_input and self._params:
                param_var_list, param_value_list = map(list, zip(*self._params.values()))
                input_list = input_list + param_var_list
                func_attrs["params"] = param_value_list

            self.bb.emit_func_output(output_var, params=input_list)

        relax_mod = self.bb.get()
        # Attach attributes.
        relax_mod["main"] = relax_mod["main"].with_attrs(func_attrs)
        return relax_mod

    def _parse_graph_initializers(self, graph: onnx.onnx_ml_pb2.GraphProto):
        """Parse network inputs to relax, aka parameters."""
        for init_tensor in graph.initializer:
            # There are two cases for handling parameters, they are either
            # treated as variables or constants.
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            array = self._parse_array(init_tensor)
            # Create variables for constants.
            if self._keep_params_in_input:
                # Pytorch sometimes inserts silly weight prefix. Remove it.
                var_name = init_tensor.name.strip("onnx::")
                init_var = self._new_var(var_name, shape=array.shape, dtype=array.dtype)
                self._nodes[init_tensor.name] = init_var
                # We need to keep track of both the real value and variable for this variable.
                self._params[init_tensor.name] = (init_var, array)
            # Otherwise we can use the weight as a constant.
            else:
                self._nodes[init_tensor.name] = relax.const(array)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name to make it a valid identifier.
        If the name is None, returns a string input_0, input_1, etc.
        If the input is an empty string, returns empty_0, empty_1, etc.
        If the input is a string that does not start with a letter or underscore,
        returns input_<name>. Otherwise, returns an unique input name.

        Parameters
        ----------
        name : str
            The name to sanitize
        Returns
        -------
        new_name : str
        """

        if name == "":
            return self._name_supply.fresh_name("empty_")

        new_name = name.replace(".", "_")
        if not new_name[0].isalpha() and new_name[0] != "_":
            new_name = str(self._name_supply.fresh_name("input_" + new_name))
        else:
            new_name = str(self._name_supply.fresh_name(new_name))

        if new_name != name:
            warnings.warn(("Renaming name %s to %s" % (name, new_name)))
        return new_name

    def _new_var(self, var_name: str, shape: List, dtype: str = "float32"):
        """Creates a new Relax variable."""
        return relax.Var(
            name_hint=var_name, struct_info=relax.TensorStructInfo(shape=shape, dtype=dtype)
        )

    def _parse_graph_input(self, graph: onnx.onnx_ml_pb2.GraphProto):
        """Parse model inputs to Relax parameters."""
        value_dict = {}
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name, value_dict = get_info(i, value_dict)
            if i_name not in self._nodes:
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
                var_name = self._sanitize_name(i_name) if self._sanitize else i_name
                self._nodes[i_name] = self._new_var(var_name, shape=i_shape, dtype=dtype)
            self._inputs[i_name] = self._nodes[i_name]

    def _check_for_unsupported_ops(self, graph: onnx.onnx_ml_pb2.GraphProto):
        convert_map = _get_convert_map()
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                # and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = "The following operators are not supported for frontend ONNX: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

    def _construct_nodes(self, graph: onnx.onnx_ml_pb2.GraphProto):
        """Nodes are stored as directed acyclic graph."""
        for node in graph.node:
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            # Create and populate input list.
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    inputs.append(self._nodes[i])
                else:
                    inputs.append(None)
            i_name = self._parse_value_proto(node)
            outputs = node.output
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(outputs)

            # Perform special handling for shape expressions. If an input is a
            # shape expr, make sure the current op can handle it, otherwise
            # convert it to a tensor.
            shape_compatible_ops = [
                "Reshape",
                "ConstantOfShape",
                "Gather",
                "Slice",
                "Shape",
                "Expand",
                "Concat",
                "Equal",
                "Where",
            ]
            for i, inp in enumerate(inputs):
                if (
                    inp is not None
                    and isinstance(inp, relax.Expr)
                    and isinstance(inp.struct_info, relax.ShapeStructInfo)
                    and op_name not in shape_compatible_ops
                ):
                    raise ValueError(f"Node {node.name} cannot handle ShapeExpr inputs.")
            op = self._convert_operator(op_name, inputs, attr, self.opset)
            # Create struct information for the new operator.
            op = self.bb.normalize(op)

            if not isinstance(op, relax.Tuple):
                if isinstance(op.checked_type, tvm.ir.type.TupleType):
                    # This is a var bound to a tuple. We need to unpack it and create
                    # a new tuple.
                    tuple_items = []
                    for i in range(len(op.checked_type.fields)):
                        tuple_items.append(self.bb.emit(relax.TupleGetItem(op, i)))
                    op = relax.Tuple(tuple_items)
                    outputs_num = len(tuple_items)
                else:
                    outputs_num = 1
            else:
                outputs_num = len(op)
            assert (
                len(outputs) <= outputs_num
            ), "Missing outputs during conversion. Expected {} but Got {} in {}.".format(
                len(outputs), outputs_num, op_name
            )

            if outputs_num == 1:
                self._nodes[outputs[0]] = op
            else:
                for k, i in zip(list(outputs), range(len(outputs))):
                    self._nodes[k] = op[i]

    def _parse_value_proto(self, value_proto: onnx.onnx_ml_pb2.GraphProto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

    def _parse_array(self, tensor_proto: onnx.onnx_ml_pb2.TensorProto) -> tvm.nd.array:
        np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
        return tvm.nd.array(np_array)

    def _parse_attr(self, attr_proto: onnx.onnx_ml_pb2.AttributeProto) -> Dict[str, Any]:
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
                    raise NotImplementedError("Field {} is not supported in relax.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _convert_operator(
        self,
        op_name: str,
        inputs: List[relax.Function],
        attrs: Dict,
        opset: int,
    ) -> relax.Function:
        """Convert ONNX operator into a Relax operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.

        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relax.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version
        Returns
        -------
        sym : tvm.relax.function.Function
            Converted relax function
        """
        convert_map = _get_convert_map()
        if op_name in convert_map:
            convert_class = convert_map[op_name]
            op_function = convert_class.get_converter(opset)
            sym = op_function(self.bb, inputs, attrs, [self._nodes, self._params])
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym


def from_onnx(
    model: onnx.onnx_ml_pb2.GraphProto,
    shape_dict: Optional[Dict[str, List]] = None,
    dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
    opset: int = None,
    keep_params_in_input: bool = False,
    sanitize_input_names: bool = True,
) -> Tuple[IRModule, Dict]:
    """Convert a ONNX model into an equivalent Relax Function.
    ONNX graphs are represented as Python Protobuf objects.

    The current implementation assumes that the input model is after ONNX v1.1.0.

    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0
    shape_dict : dict of str to tuple, optional
        The input shape to the graph
    dtype_dict : str or dict of str to str, optional
        The input types to the graph
    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.
    keep_params_in_input : bool
        If True, parameters will be treated as input variables. If false,
        parameters are treated as constant and folded directly into the graph.
    sanitize_input_names : bool, optional
        Whether to sanitize the input names to ensure they are valid Relax identifiers.

    Returns
    -------
    mod : tvm.IRModule
        The relax module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relax
    """
    # Error if the model version is below 1.1.0
    if model.ir_version < 3:
        raise ValueError(
            "Model IR version {} not supported. Must be at least after 1.1.0.".format(
                model.ir_version
            )
        )

    try:
        import onnx  # pylint: disable=import-outside-toplevel, redefined-outer-name

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except Exception as exception:  # pylint: disable=c-extension-no-member, broad-except
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(exception))
    except ImportError as error:
        raise ImportError("Unable to import onnx which is required {}".format(error))

    g = ONNXGraphImporter(
        shape_dict,
        dtype_dict,
        keep_params_in_input=keep_params_in_input,
        sanitize=sanitize_input_names,
    )
    graph = model.graph

    try:
        opset_in_model = 1
        if model.opset_import:
            # TODO: for now we only really support ai.onnx op set
            # TODO: handle other namespaces well see https://github.com/apache/tvm/issues/10950
            for opset_identifier in model.opset_import:
                # As per https://github.com/onnx/onnx/blob/main/docs/IR.md
                # All operator sets except the default one must specify the operator version
                if str(opset_identifier.domain) in ["ai.onnx", ""]:
                    opset_in_model = opset_identifier.version
                    break
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
    return g.from_onnx(graph, opset)
