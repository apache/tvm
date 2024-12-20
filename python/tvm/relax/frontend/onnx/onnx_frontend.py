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
import math
import operator
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as _np
import onnx.onnx_ml_pb2

import tvm
from tvm import TVMError, relax, tir, topi
from tvm.ir import IRModule
from tvm.ir.supply import NameSupply
from tvm.tir.generic import cast

from ..common import autopad


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
        _, value = params[var.name_hint]
        const_value = relax.const(value)
        graph_nodes[var.name_hint] = const_value
        return const_value
    # Otherwise return variable.
    else:
        return var


def get_value(token, value_dict: Dict[str, tvm.tir.SizeVar]) -> Union[int, tvm.tir.SizeVar]:
    """Converts to token to an integer value if it a constant, otherwise it generates a SizeVar

    Parameters
    ----------
    token: str
        current token to decode.

    value_dict: Dict
        The Dictionary mapping from the name of ValueInfoProto to SizeVar.

    Returns
    -------
    Union[int, tvm.tir.SizeVar]
        The decoded token
    """

    try:
        return int(token)
    except ValueError:
        if token not in value_dict or token == "?":
            value_dict[token] = tvm.tir.SizeVar(token, "int64")
        value = value_dict[token]
        return value


def parse_shape_name(
    name: str, value_dict: Dict[str, tvm.tir.SizeVar]
) -> Union[tir.PrimExpr, tvm.tir.SizeVar]:
    """Converts expressions in the shape dimension name to prim expressions.

    Parameters
    ----------
    name: str
        name of shape dimension.

    value_dict: Dict
        The Dictionary mapping from the name of ValueInfoProto to SizeVar.

    Returns
    -------
    Union[tir.PrimExpr, tvm.tir.SizeVar]
        The expression of the shape dimension.
    """

    tokens = re.split(r"(\+|\-|\*|\/\/|\/)", name.replace(" ", ""))

    operators = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.floordiv,  # is floordiv since the operands are always int
        "//": operator.floordiv,
    }

    value_stack = []
    operator_stack = []

    for token in tokens:
        if token in operators:
            operator_stack.append(token)
        else:
            value = get_value(token, value_dict)
            if value_stack and operator_stack:
                prev_value = value_stack.pop()
                op = operator_stack.pop()
                result = operators[op](prev_value, value)
                value_stack.append(result)
            else:
                value_stack.append(value)

    if value_stack:
        return value_stack[0]
    else:
        raise Exception("Shape dimension could not be inferred")


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
            value = parse_shape_name(name, value_dict)
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
        from onnx.numpy_helper import to_array  # pylint: disable=import-outside-toplevel
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


def _to_numpy(x):
    if isinstance(x, relax.PrimValue):
        x = x.value
        if isinstance(x, (tir.IntImm, tir.FloatImm)):
            x = x.value
        return _np.array(x)
    else:
        return x.data.numpy()


class BinaryBase(OnnxOpConverter):
    """Converts an onnx BinaryBase node into an equivalent Relax expression."""

    numpy_op: Callable = None
    relax_op: Callable = None

    @classmethod
    def base_impl(cls, bb, inputs, attr, params):
        """Base implementation for binary operations."""
        if cls.numpy_op is None or cls.relax_op is None:
            raise ValueError("Numpy and Relax operators must be defined for BinaryBase.")
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            output = cls.numpy_op(  # pylint: disable=not-callable
                inputs[0].data.numpy(), inputs[1].data.numpy()
            )
            return relax.const(output, inputs[0].struct_info.dtype)
        if any([isinstance(inp, relax.PrimValue) for inp in inputs]):
            x = _to_numpy(inputs[0])
            y = _to_numpy(inputs[1])
            return relax.PrimValue(cls.numpy_op(x, y))  # pylint: disable=not-callable

        return cls.relax_op(inputs[0], inputs[1])  # pylint: disable=not-callable


class Add(BinaryBase):
    """Converts an onnx Add node into an equivalent Relax expression."""

    numpy_op = _np.add
    relax_op = relax.op.add

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Sub(BinaryBase):
    """Converts an onnx Sub node into an equivalent Relax expression."""

    numpy_op = _np.subtract
    relax_op = relax.op.subtract

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Mul(BinaryBase):
    """Converts an onnx Mul node into an equivalent Relax expression."""

    numpy_op = _np.multiply
    relax_op = relax.op.multiply

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Div(BinaryBase):
    """Converts an onnx Div node into an equivalent Relax expression."""

    numpy_op = _np.divide
    relax_op = relax.op.divide

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Pow(BinaryBase):
    """Converts an onnx Pow node into an equivalent Relax expression."""

    numpy_op = _np.power
    relax_op = relax.op.power

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Mod(BinaryBase):
    """Converts an onnx Mod node into an equivalent Relax expression."""

    numpy_op = _np.mod
    relax_op = relax.op.mod

    @classmethod
    def _impl_v10(cls, bb, inputs, attr, params):
        if attr.get("fmod", 0) == 0:
            cls.numpy_op = _np.fmod
            cls.relax_op = relax.op.floor_mod
        else:
            cls.numpy_op = _np.mod
            cls.relax_op = relax.op.mod
        return cls.base_impl(bb, inputs, attr, params)


class And(BinaryBase):
    """Converts an onnx And node into an equivalent Relax expression."""

    numpy_op = _np.logical_and
    relax_op = relax.op.logical_and

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Or(BinaryBase):
    """Converts an onnx Or node into an equivalent Relax expression."""

    numpy_op = _np.logical_or
    relax_op = relax.op.logical_or

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Xor(BinaryBase):
    """Converts an onnx Xor node into an equivalent Relax expression."""

    numpy_op = _np.logical_xor
    relax_op = relax.op.logical_xor

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Less(BinaryBase):
    """Converts an onnx Less node into an equivalent Relax expression."""

    numpy_op = _np.less
    relax_op = relax.op.less

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class LessOrEqual(BinaryBase):
    """Converts an onnx LessEqual node into an equivalent Relax expression."""

    numpy_op = _np.less_equal
    relax_op = relax.op.less_equal

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class Greater(BinaryBase):
    """Converts an onnx Greater node into an equivalent Relax expression."""

    numpy_op = _np.greater
    relax_op = relax.op.greater

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class GreaterOrEqual(BinaryBase):
    """Converts an onnx GreaterEqual node into an equivalent Relax expression."""

    numpy_op = _np.greater_equal
    relax_op = relax.op.greater_equal

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


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


class BitwiseBase(BinaryBase):
    """Converts an onnx BitwiseBase node into an equivalent Relax expression."""

    @classmethod
    def base_impl(cls, bb, inputs, attr, params):
        """Base implementation for bitwise operations."""
        valid_types = ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
        for num, inp in enumerate(inputs):
            if inp.struct_info.dtype not in valid_types:
                raise ValueError(
                    f"Bitwise operations expect all inputs to have integer types, "
                    f"got {inp.struct_info.dtype} for input {num}"
                )
        return super().base_impl(bb, inputs, attr, params)


class BitwiseAnd(BitwiseBase):
    """Converts an onnx BitwiseAnd node into an equivalent Relax expression."""

    numpy_op = _np.bitwise_and
    relax_op = relax.op.bitwise_and

    @classmethod
    def _impl_v18(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class BitwiseOr(BitwiseBase):
    """Converts an onnx BitwiseOr node into an equivalent Relax expression."""

    numpy_op = _np.bitwise_or
    relax_op = relax.op.bitwise_or

    @classmethod
    def _impl_v18(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class BitwiseXor(BitwiseBase):
    """Converts an onnx BitwiseXor node into an equivalent Relax expression."""

    numpy_op = _np.bitwise_xor
    relax_op = relax.op.bitwise_xor

    @classmethod
    def _impl_v18(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class BitwiseNot(BitwiseBase):
    """Converts an onnx BitwiseNot node into an equivalent Relax expression."""

    numpy_op = _np.bitwise_not
    relax_op = relax.op.bitwise_not

    @classmethod
    def _impl_v18(cls, bb, inputs, attr, params):
        return cls.base_impl(bb, inputs, attr, params)


class BitShift(BitwiseBase):
    """Converts an onnx BitShift node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        direction = attr.get("direction", "LEFT").decode("ascii")
        if direction == "LEFT":
            cls.numpy_op = _np.left_shift
            cls.relax_op = relax.op.left_shift
        elif direction == "RIGHT":
            cls.numpy_op = _np.right_shift
            cls.relax_op = relax.op.right_shift
        else:
            raise ValueError("Unsupported Shift Direction: " + direction)

        return cls.base_impl(bb, inputs, attr, params)


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


class LogSoftmax(OnnxOpConverter):
    """Converts an onnx LogSoftmax node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", -1)
        return relax.op.nn.log_softmax(inputs[0], axis=axis)


class Hardmax(OnnxOpConverter):
    """Converts an onnx Hardmax node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", -1)
        indices = inputs[0]
        dtype = indices.struct_info.dtype
        axis_len = int(inputs[0].struct_info.shape[axis])
        argmax = relax.op.argmax(indices, axis=axis)
        on_value = relax.PrimValue(tvm.tir.const(1.0, dtype))
        off_value = relax.PrimValue(tvm.tir.const(0.0, dtype))

        one_hot = relax.op.one_hot(argmax, on_value, off_value, axis_len, axis)
        return one_hot


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
    def _impl_v1(cls, bb, inputs, attr, params):
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


class Cast(OnnxOpConverter):
    """Convert an onnx Cast node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        to_type = get_type(attr["to"])
        if isinstance(inputs[0], relax.ShapeExpr):
            shape = inputs[0]
            if all([isinstance(x, tir.IntImm) for x in shape]):
                shape = [int(x) for x in shape]
                return relax.const(shape, to_type)
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

        return relax.op.take(data, indices, axis)


class GatherElements(OnnxOpConverter):
    """Convert an onnx GatherElements node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 0)
        return relax.op.gather_elements(inputs[0], inputs[1], axis)


class GatherND(OnnxOpConverter):
    """Convert an onnx GatherND node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        batch_dims = attr.get("batch_dims", 0)
        return relax.op.gather_nd(inputs[0], inputs[1], batch_dims)


class Scatter(OnnxOpConverter):
    """Convert an onnx Scatter node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 0)
        return relax.op.scatter_elements(inputs[0], inputs[1], inputs[2], axis=axis)

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        raise ValueError("Scatter is deprecated in ONNX 11")


class ScatterElements(OnnxOpConverter):
    """Convert an onnx ScatterElements node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 0)
        return relax.op.scatter_elements(inputs[0], inputs[1], inputs[2], axis=axis)


class ScatterND(OnnxOpConverter):
    """Convert an onnx ScatterND node into an equivalent Relax expression."""

    @staticmethod
    def _reduction_check(attr, valid_reductions: List[str]):
        reduction = attr.get("reduction", None)
        reduction = reduction or b"update"
        reduction = reduction.decode("utf-8")
        reduction = "update" if reduction == "none" else reduction
        assert (
            reduction in valid_reductions
        ), f"Only {valid_reductions} reductions are supported, but {reduction} is gotten"

        return reduction

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        return relax.op.scatter_nd(inputs[0], inputs[1], inputs[2])

    @classmethod
    def _impl_v16(cls, bb, inputs, attr, params):
        reduction = cls._reduction_check(attr, ["update", "add", "mul"])
        return relax.op.scatter_nd(inputs[0], inputs[1], inputs[2], reduction)

    @classmethod
    def _impl_v18(cls, bb, inputs, attr, params):
        reduction = cls._reduction_check(attr, ["update", "add", "mul", "min", "max"])
        return relax.op.scatter_nd(inputs[0], inputs[1], inputs[2], reduction)


class Compress(OnnxOpConverter):
    """Convert an onnx Compress node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        tensor, condition = inputs
        axis = attr.get("axis", None)

        # Change one hot tensor to indices e.g. [0, 1, 1, 0, 1] -> [1, 2, 4]
        if condition.struct_info.dtype != "bool":
            raise ValueError("Condition tensor is expected to be a boolean tensor")
        if condition.struct_info.ndim != 1:
            raise ValueError("Condition tensor is expected to be a 1D boolean tensor")
        indices = relax.op.nonzero(condition)
        num_nonzero = tir.Var("num_nonzero", "int64")
        indices = bb.match_cast(indices, relax.TensorStructInfo([1, num_nonzero], "int64"))
        indices = relax.op.reshape(indices, [-1])

        if axis is not None:
            return relax.op.take(tensor, indices, axis=axis)

        # if axis is None, flatten input tensor before selection
        tensor = relax.op.reshape(tensor, (-1,))
        return relax.op.take(tensor, indices, axis=0)


class Size(OnnxOpConverter):
    """Convert an onnx Size node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        # TODO(tvm-team): add native support for size op
        return relax.op.prod(relax.op.shape_to_tensor(relax.op.shape_of(inputs[0])))


class EyeLike(OnnxOpConverter):
    """Convert an onnx EyeLike node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        k = attr.get("k", 0)
        input_dtype = inputs[0].struct_info.dtype
        if "dtype" in attr and get_type(attr["dtype"]) != input_dtype:
            raise ValueError(
                f"dtype mismatch between input ({input_dtype}) and attribute ({attr['dtype']})"
            )
        return relax.op.eye_like(inputs[0], k, input_dtype)


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
    def _impl_v1(cls, bb, inputs, attr, params):
        min = float(attr.get("min", -_np.inf))
        max = float(attr.get("max", _np.inf))
        results = inputs[0]
        results = bb.emit_te(topi.maximum, results, min)
        results = bb.emit_te(topi.minimum, results, max)
        return results

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        results = inputs[0]
        if inputs[1] is not None:
            results = bb.emit_te(topi.maximum, results, inputs[1])
        if inputs[2] is not None:
            results = bb.emit_te(topi.minimum, results, inputs[2])
        return results


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


class Trilu(OnnxOpConverter):
    """Given a 2-D matrix or batches of 2-D matrices, returns the upper or
    lower triangular part of the tensor(s)
    """

    @classmethod
    def _impl_v14(cls, bb, inputs, attr, params):
        upper = attr.get("upper", True)
        x = inputs[0]
        k = inputs[1] if len(inputs) > 1 else 0

        if len(inputs) > 1:
            k = get_constant(inputs[1], params)
            if isinstance(k, relax.Constant):
                k = int(k.data.numpy().item())
            else:
                raise ValueError("Currently only support constant k for Trilu op.")
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


class Elu(OnnxOpConverter):
    """Converts an onnx Elu node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        alpha = float(attr.get("alpha", 1.0))
        return relax.expr.const(-alpha) * relax.op.nn.relu(
            relax.expr.const(1.0) - relax.op.exp(inputs[0])
        ) + relax.op.nn.relu(inputs[0])


class Selu(OnnxOpConverter):
    """Converts an onnx Selu node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        alpha = attr.get("alpha", 1.67326319217681884765625)
        gamma = attr.get("gamma", 1.05070102214813232421875)
        return relax.const(gamma) * (
            relax.const(-alpha) * relax.op.nn.relu(relax.const(1.0) - relax.op.exp(inputs[0]))
            + relax.op.nn.relu(inputs[0])
        )


class Mish(OnnxOpConverter):
    """Converts an onnx Mish node into an equivalent Relax expression.

    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    """

    @classmethod
    def _impl_v18(cls, bb, inputs, attr, params):
        dtype = inputs[0].checked_type.dtype
        return inputs[0] * relax.op.tanh(
            relax.op.log(relax.const(1.0, dtype) + relax.op.exp(inputs[0]))
        )


class PRelu(OnnxOpConverter):
    """Converts an onnx PRelu node into an equivalent Relax expression.

    f(x) = slope * x for x < 0, x for x >= 0
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        x = inputs[0]
        slope = inputs[1]
        # TODO(tvm-team): Should add a new op for this.
        return x * slope + relax.op.nn.relu(x) * (relax.const(1.0) - slope)


class ThresholdedRelu(OnnxOpConverter):
    """Converts an onnx ThresholdedRelu node into an equivalent Relax expression.

    f(x) = x for x > alpha, 0 otherwise
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        x = inputs[0]
        alpha = attr.get("alpha", 1.0)
        return relax.op.greater(x, relax.const(alpha)).astype("float32") * x


class LeakyRelu(OnnxOpConverter):
    """Converts an onnx LeakyRelu node into an equivalent Relax expression.

    f(x) = x for x > 0, alpha * x otherwise
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        x = inputs[0]
        alpha = attr.get("alpha", 0.01)
        return relax.op.nn.leakyrelu(x, alpha)


class Gelu(OnnxOpConverter):
    """Operator converter for Gelu from Microsoft onnxruntime contrib opset.

    gelu(x) = 0.5x(1 + erf(x/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.nn.gelu(inputs[0])


class FastGelu(OnnxOpConverter):
    """Operator converter for FastGelu from Microsoft onnxruntime contrib opset.

    fast_gelu(x) = 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
                 = 0.5x(1 + tanh((sqrt(2/pi)x + 0.044715(sqrt(2/pi)x^3)))
                 = 0.5x(1 + tanh(c1 * x + c2 * x^3)))
    , where
        c1 = sqrt(2/pi)
        c2 = 0.044715 * sqrt(2/pi)
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if inputs[1]:
            bias = inputs[1]
            bias_shape = bias.struct_info.shape
            assert len(bias_shape) == 1, "bias term must be a 1D tensor"
            x += bias

        # Declare consts
        const_dtype = x.struct_info.dtype
        half = relax.const(0.5, dtype=const_dtype)
        one = relax.const(1.0, dtype=const_dtype)
        const1 = relax.const(math.sqrt(2 / math.pi), dtype=const_dtype)
        const2 = relax.const(0.044715 * math.sqrt(2 / math.pi), dtype=const_dtype)

        # Compute FastGelu
        term1 = relax.op.multiply(half, x)
        term2 = relax.op.multiply(const1, x)
        term3 = relax.op.multiply(const2, relax.op.power(x, relax.const(3, const_dtype)))
        tanh = relax.op.tanh(relax.op.add(term2, term3))
        return relax.op.multiply(term1, relax.op.add(one, tanh))


class BiasGelu(OnnxOpConverter):
    """Operator converter for BiasGelu from Microsoft onnxruntime contrib opset.

    bias_gelu(x, b) = 0.5(x + b)(1 + erf((x + b)/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        inp = relax.op.add(inputs[0], inputs[1])
        return relax.op.nn.gelu(inp)


class Shrink(OnnxOpConverter):
    """Converts an onnx Shrink node into an equivalent Relax expression.

    f(x) = x + bias if x > lambd, x - bias if x < -lambd, 0 otherwise
    """

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        x = inputs[0]
        dtype = x.struct_info.dtype
        lambd = relax.const(attr.get("lambd", 0.5), dtype)
        bias = relax.const(attr.get("bias", 0.0), dtype)
        zeros = relax.op.zeros_like(x)
        return relax.op.where(x > lambd, x - bias, zeros) + relax.op.where(
            x < -lambd, x + bias, zeros
        )


class Conv(OnnxOpConverter):
    """Convert an onnx Conv node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        if hasattr(inputs[0].struct_info, "ndim"):
            ndim = inputs[0].struct_info.ndim
        else:
            ndim = len(inputs[0].struct_info.shape)

        if "kernel_shape" not in attr:
            attr["kernel_shape"] = inputs[1].struct_info.shape.values[2:]

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

        if "auto_pad" in attr:
            attr["auto_pad"] = attr["auto_pad"].decode("utf-8")
            if attr["auto_pad"] in ("SAME_UPPER", "SAME_LOWER"):
                data = autopad(
                    bb,
                    inputs[0],
                    attr.get("strides", [1] * (ndim - 2)),
                    attr["kernel_shape"],
                    attr.get("dilations", [1] * (ndim - 2)),
                    mode=attr["auto_pad"],
                    deconv=False,
                )
            elif attr["auto_pad"] == "VALID":
                attr["pads"] = [0 for _ in range(ndim - 2)]
            elif attr["auto_pad"] == "NOTSET":
                pass
            else:
                msg = (
                    f'Value {attr["auto_pad"]} in attribute "auto_pad" of operator Conv '
                    f"is invalid."
                )
                raise tvm.error.OpAttributeInvalid(msg)
            attr.pop("auto_pad")

        conv_out = bb.normalize(
            op(
                data=data,
                weight=inputs[1],
                strides=attr.get("strides", 1),
                padding=attr.get("pads", 0),
                dilation=attr.get("dilations", 1),
                groups=attr.get("group", 1),
                data_layout=data_layout,
                kernel_layout=kernel_layout,
            )
        )
        if inputs[2] is not None:
            bias = relax.op.reshape(inputs[2], [1, -1] + [1] * (ndim - 2))
            conv_out = relax.op.add(conv_out, bias)

        return conv_out


class ConvTranspose(OnnxOpConverter):
    """Converts an onnx ConvTranspose node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if hasattr(inputs[0].struct_info, "ndim"):
            ndim = inputs[0].struct_info.ndim
        else:
            ndim = len(inputs[0].struct_info.shape)

        if ndim == 3:
            op = relax.op.nn.conv1d_transpose
            data_layout = "NCW"
            kernel_layout = "IOW"
        elif ndim == 4:
            op = relax.op.nn.conv2d_transpose
            data_layout = "NCHW"
            kernel_layout = "IOHW"
        elif ndim == 5:
            raise NotImplementedError("Relax ConvTranspose3d not supported yet")
        else:
            raise NotImplementedError("Ndim > 5 not supported for convolution.")

        conv_out = op(
            data=inputs[0],
            weight=inputs[1],
            strides=attr.get("strides", 1),
            padding=attr.get("pads", 0),
            dilation=attr.get("dilations", 1),
            groups=attr.get("group", 1),
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )

        if inputs[2] is not None:
            bias = relax.op.reshape(inputs[2], [1, -1] + [1] * (ndim - 2))
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
        data = inputs[0]
        axis = get_constant(inputs[1], params)
        if isinstance(axis, relax.Constant):
            axis = tuple([int(x) for x in axis.data.numpy()])

        # If data is constant, perform computation directly.
        if isinstance(data, relax.Constant):
            if isinstance(axis, (tuple, type(None))):
                out_data = _np.squeeze(data.data.numpy(), axis)
            else:
                raise NotImplementedError("Squeeze with symbolic axes not supported")

            return relax.const(out_data, data.struct_info.dtype)

        if isinstance(data, relax.ShapeExpr):
            if axis == (0,):
                return relax.PrimValue(data[0])
            else:
                raise NotImplementedError(
                    "Squeeze with symbolic axes and non-zero axes is not supported."
                )

        return relax.op.squeeze(data, axis)


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


class Sin(OnnxOpConverter):
    """Converts an onnx Sin node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.sin(inputs[0])


class Sinh(OnnxOpConverter):
    """Converts an onnx Sinh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        return relax.op.sinh(inputs[0])


class Cos(OnnxOpConverter):
    """Converts an onnx Cos node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.cos(inputs[0])


class Cosh(OnnxOpConverter):
    """Converts an onnx Cosh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        return relax.op.cosh(inputs[0])


class Tan(OnnxOpConverter):
    """Converts an onnx Tan node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.tan(inputs[0])


class Tanh(OnnxOpConverter):
    """Converts an onnx Tanh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.tanh(inputs[0])


class Acos(OnnxOpConverter):
    """Converts an onnx Acos node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.acos(inputs[0])


class Acosh(OnnxOpConverter):
    """Converts an onnx Acosh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        return relax.op.acosh(inputs[0])


class Asin(OnnxOpConverter):
    """Converts an onnx Asin node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.asin(inputs[0])


class Asinh(OnnxOpConverter):
    """Converts an onnx Asinh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        return relax.op.asinh(inputs[0])


class Atan(OnnxOpConverter):
    """Converts an onnx Atan node into an equivalent Relax expression."""

    @classmethod
    def _impl_v7(cls, bb, inputs, attr, params):
        return relax.op.atan(inputs[0])


class Atanh(OnnxOpConverter):
    """Converts an onnx Atanh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        return relax.op.atanh(inputs[0])


class Neg(OnnxOpConverter):
    """Converts an onnx Neg node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if isinstance(inputs[0], relax.Constant):
            data_np = inputs[0].data.numpy()
            return relax.const(_np.negative(data_np), inputs[0].struct_info.dtype)
        if isinstance(inputs[0], relax.PrimValue):
            return relax.PrimValue(-inputs[0].value)
        return relax.op.negative(inputs[0])


class Abs(OnnxOpConverter):
    """Converts an onnx Abs node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        if isinstance(inputs[0], relax.Constant):
            output = _np.abs(inputs[0].data.numpy())
            return relax.const(output, output.dtype)
        return relax.op.abs(inputs[0])


class Reciprocal(OnnxOpConverter):
    """Converts an onnx Reciprocal node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        input_dtype = inputs[0].struct_info.dtype
        return relax.op.divide(relax.const(1, dtype=input_dtype), inputs[0])


class Floor(OnnxOpConverter):
    """Converts an onnx Floor node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.floor(inputs[0])


class Ceil(OnnxOpConverter):
    """Converts an onnx Ceil node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.ceil(inputs[0])


class Round(OnnxOpConverter):
    """Converts an onnx Round node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.round(inputs[0])


class IsInf(OnnxOpConverter):
    """Converts an onnx IsInf node into an equivalent Relax expression."""

    @classmethod
    def _impl_v10(cls, bb, inputs, attr, params):
        return relax.op.isinf(inputs[0])


class IsNaN(OnnxOpConverter):
    """Converts an onnx IsNaN node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        return relax.op.isnan(inputs[0])


class Sqrt(OnnxOpConverter):
    """Converts an onnx Sqrt node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.sqrt(inputs[0])


class MultiInputBase(OnnxOpConverter):
    """Converts an onnx MultiInputBase node into an equivalent Relax expression."""

    numpy_op: Callable = None
    relax_op: Callable = None

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        if cls.numpy_op is None or cls.relax_op is None:
            raise NotImplementedError("numpy_op and relax_op must be defined for MultiInputBase")
        if all([isinstance(inp, relax.Constant) for inp in inputs]):
            np_inputs = [inp.data.numpy() for inp in inputs]
            output = cls.numpy_op(*np_inputs)  # pylint: disable=not-callable
            return relax.const(output, output.dtype)

        # Expand inputs, stack them, then perform minimum over the new axis.
        inputs = [bb.normalize(relax.op.expand_dims(i, axis=0)) for i in inputs]
        stacked_tensor = relax.op.concat(inputs, axis=0)
        return cls.relax_op(stacked_tensor, axis=0)  # pylint: disable=not-callable


class Min(MultiInputBase):
    """Converts an onnx Min node into an equivalent Relax expression."""

    numpy_op = _np.min
    relax_op = relax.op.min


class Max(MultiInputBase):
    """Converts an onnx Max node into an equivalent Relax expression."""

    numpy_op = _np.max
    relax_op = relax.op.max


class Mean(MultiInputBase):
    """Converts an onnx Mean node into an equivalent Relax expression."""

    numpy_op = _np.mean
    relax_op = relax.op.mean


class Sum(MultiInputBase):
    """Converts an onnx Sum node into an equivalent Relax expression."""

    numpy_op = _np.sum
    relax_op = relax.op.sum


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


class Softplus(OnnxOpConverter):
    """Converts an onnx Softplus node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        dtype = inputs[0].struct_info.dtype
        return relax.op.log(relax.op.exp(inputs[0]) + relax.const(1, dtype=dtype))


class Softsign(OnnxOpConverter):
    """Converts an onnx Softsign node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        dtype = inputs[0].struct_info.dtype
        return inputs[0] / (relax.op.abs(inputs[0]) + relax.const(1, dtype=dtype))


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
                splits = splits.data.numpy()
                indices = []
                index = 0
                for i in splits[:-1]:
                    index += i
                    indices.append(index.item())
            else:
                raise ValueError("Dynamic Split not yet supported")
        # When splits isnt specified divide evenly over axis.
        else:
            indices = attr["tvm_custom"]["num_outputs"]
        return bb.emit_te(topi.split, inputs[0], indices, axis=attr.get("axis", 0))


def get_prim_value_list(values):
    new_values = []
    for v in list(values):
        if isinstance(v, relax.expr.PrimExpr):
            new_values.append(relax.PrimValue(v))
        else:
            new_values.append(v)
    return new_values


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
            shape_data = list(data)
            # Starts, ends, and steps must be 1-d for shape operation.
            assert all(len(i) == 1 for i in [starts, ends, steps])
            sliced_values = shape_data[starts[0] : ends[0] : steps[0]]

            if all([isinstance(val, (tir.IntImm, int)) for val in sliced_values]):
                return relax.const([x.value for x in sliced_values], "int64")
            else:
                return relax.ShapeExpr(sliced_values)

        # If all `starts`, `ends`, and `steps` are constant, use strict mode
        # Otherwise, we assume the slice is inbound.
        assume_inbound = not all(
            [isinstance(param, (tir.IntImm, int)) for param in [*starts, *ends, *steps]]
        )

        # Converting PrimExpr to PrimValue since relax.op.strided_slice does not accept PrimExpr
        starts = get_prim_value_list(starts)
        ends = get_prim_value_list(ends)
        steps = get_prim_value_list(steps)

        return relax.op.strided_slice(
            data, axes, starts, ends, steps, assume_inbound=assume_inbound
        )


class Pad(OnnxOpConverter):
    """Converts an onnx Pad node into an equivalent Relax expression."""

    @classmethod
    def _impl_v2(cls, bb, inputs, attr, params):
        pads = attr.get("pads")
        pads = relax.const(_np.array(pads), inputs[0].struct_info.shape[0].dtype)
        constant_value = attr.get("value")
        if constant_value is None:
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
            data_shape = list(data.struct_info.shape)
            target_shape = list(shape.values)
            data_shape = [1] * (len(target_shape) - len(data_shape)) + data_shape
            assert len(data_shape) == len(target_shape)
            # Fix small target shapes or target shapes assigned to -1
            for i, s in enumerate(target_shape):
                if isinstance(s, tvm.tir.IntImm) and (
                    (isinstance(data_shape[i], tvm.tir.IntImm) and s < data_shape[i])
                    or s.value == -1
                ):
                    target_shape[i] = data_shape[i]
            if target_shape == data_shape:
                return data
            return relax.op.broadcast_to(data, relax.ShapeExpr(target_shape))

        # If possible, directly expand to constant shape.
        if isinstance(shape, relax.Constant):
            new_shape = shape.data.numpy().tolist()
            # For some reason, onnx allows target shapes to be smaller than input shapes.
            # We need to go correct it.
            data_shape = [dim.value for dim in data.struct_info.shape]
            # Dimensions are right alignment.
            data_shape = [1] * (len(new_shape) - len(data_shape)) + data_shape
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
        return relax.op.arange(start, limit, step, out_dtype)


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


class MeanVarianceNormalization(OnnxOpConverter):
    """Converts an onnx MeanVarianceNormalization node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis = attr.get("axes", (0, 2, 3))
        data_mean = relax.op.mean(data, axis=axis, keepdims=True)
        data_mean_squared = relax.op.power(data_mean, relax.const(2, dtype="float32"))
        data_squared = relax.op.power(data, relax.const(2, dtype="float32"))
        data_squared_mean = relax.op.mean(data_squared, axis=axis, keepdims=True)
        return (data - data_mean) / relax.op.sqrt(data_squared_mean - data_mean_squared)


class Pool(OnnxOpConverter):
    """A helper class for pool op converters."""

    name = ""

    @classmethod
    def get_pad_pair(cls, input1d, kernel1d, stride1d, mode):
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

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        # Unpack inputs and attributes.
        data = inputs[0]
        input_shape = data.struct_info.shape
        ndim = len(input_shape)

        auto_pad = attr.get("auto_pad", b"NOTSET").decode("utf-8")
        ceil_mode = attr.get("ceil_mode", 0)
        dilations = attr.get("dilations", [1] * (ndim - 2))
        kernel_shape = attr.get("kernel_shape")
        pads = attr.get("pads", 0)
        strides = attr.get("strides", [1] * (ndim - 2))
        count_include_pad = attr.get("count_include_pad", False)

        assert len(kernel_shape) in [1, 2, 3], "Currently only 1D/2D/3D/ pooling is supported."

        assert auto_pad in [
            "NOTSET",
            "SAME_UPPER",
            "SAME_LOWER",
            "VALID",
        ], f"Value {auto_pad} in attribute auto_pad is invalid."

        if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
            pads = []
            if cls.name == "avg_pool":
                for axis in range(len(input_shape) - 2):
                    axis_shape = input_shape[2 + axis]
                    stride = strides[axis]
                    kernel = kernel_shape[axis]
                    pad = cls.get_pad_pair(axis_shape, kernel, stride, auto_pad)
                    pads.append(pad)
            else:
                input_spatial_shape = cls._get_input_spatial_shape(data)
                output_spatial_shape = [0 for _ in input_spatial_shape]

                for i, _ in enumerate(input_spatial_shape):
                    if auto_pad == "SAME_UPPER":
                        output_spatial_shape[i] = int(_np.ceil(input_spatial_shape[i] / strides[i]))
                    else:
                        output_spatial_shape[i] = int(
                            _np.floor(input_spatial_shape[i] / strides[i])
                        )
                    pad_i = (
                        (output_spatial_shape[i] - 1) * strides[i]
                        + ((kernel_shape[i] - 1) * dilations[i] + 1)
                        - input_spatial_shape[i]
                    )

                    if auto_pad == "SAME_UPPER":
                        pads.append([pad_i // 2, pad_i - pad_i // 2])
                    else:
                        pads.append([pad_i - pad_i // 2, pad_i // 2])

            pads = tuple([val for pair in zip(*pads) for val in pair])

        op = getattr(relax.op.nn, cls.name + str(len(kernel_shape)) + "d")
        return op(data, kernel_shape, strides, pads, dilations, ceil_mode, count_include_pad)

    @classmethod
    def _get_input_spatial_shape(cls, tensor):
        # shape is (N x C x D1 x D2 ... Dn)
        return _np.array([int(d) for d in tensor.struct_info.shape], dtype="int64")[2:]


class MaxPool(Pool):
    """Converts an onnx MaxPool node into an equivalent Relax expression."""

    name = "max_pool"


class AveragePool(Pool):
    """Converts an onnx MaxPool node into an equivalent Relax expression."""

    name = "avg_pool"


class LpPool(OnnxOpConverter):
    """Converts an onnx LpPool node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        dtype = inputs[0].struct_info.dtype
        p = attr.get("p", 2.0)
        reci_p = relax.const(1.0 / p, dtype=dtype)
        # emit for get struct_info
        data = bb.emit(relax.op.power(inputs[0], relax.const(p, dtype=dtype)))
        attr.update({"count_include_pad": True})
        avg_pool = AveragePool._impl_v1(bb, [data], attr, params)
        kernels = attr["kernel_shape"]
        out = avg_pool * relax.const(_np.prod(kernels).astype(dtype))
        return relax.op.power(out, reci_p)


class GlobalAveragePool(OnnxOpConverter):
    """Converts an onnx GlobalAveragePool node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        rank = len(inputs[0].struct_info.shape)
        axes = list(range(2, rank))
        return relax.op.mean(inputs[0], axis=axes, keepdims=True)


class GlobalMaxPool(OnnxOpConverter):
    """Converts an onnx GlobalMaxPool node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        rank = len(inputs[0].struct_info.shape)
        axes = list(range(2, rank))
        return relax.op.max(inputs[0], axis=axes, keepdims=True)


class GlobalLpPool(OnnxOpConverter):
    """Converts an onnx GlobalLpPool node into an equivalent Relax expression."""

    @classmethod
    def _impl_v2(cls, bb, inputs, attr, params):
        p = attr.get("p", 2.0)
        dtype = inputs[0].struct_info.dtype
        rank = len(inputs[0].struct_info.shape)
        axes = list(range(2, rank))
        x_abs = relax.op.abs(inputs[0])
        x_p = relax.op.power(x_abs, relax.const(p, dtype=dtype))
        x_sum = relax.op.sum(x_p, axes, keepdims=True)
        return relax.op.power(x_sum, relax.const(1.0 / p, dtype=dtype))


class MaxUnpool(OnnxOpConverter):
    """Converts an onnx MaxUnpool node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        data = inputs[0]
        indices = inputs[1]
        output_shape = inputs[2]
        kernel_shape = attr.get("kernel_shape")
        pads = attr.get("pads", [0] * len(kernel_shape) * 2)
        strides = attr.get("strides", [1] * len(kernel_shape))

        multiplier = _np.concatenate([[1, 1], list(strides)])
        shape = [v.value for v in data.struct_info.shape]
        total_output_shape = multiplier * shape
        # Add extra dimensions from kernel size and stride mismatch
        total_output_shape += _np.concatenate([[0, 0], list(kernel_shape)], axis=0)
        total_output_shape -= _np.concatenate([[0, 0], list(strides)], axis=0)

        if output_shape is not None:
            total_output_shape = output_shape

        elif pads is not None:
            # Get pads in the proper format for relay.
            pads = _np.concatenate([[0, 0, 0, 0], list(pads)], axis=0)
            pads = _np.reshape(pads, [-1, 2])
            # Compute the total padding per axis.
            total_pad = _np.sum(pads, axis=-1)
            # Reversing maxpool means that padding actually makes our output smaller.
            total_output_shape = total_output_shape - total_pad

        # Create a tensor of zeros then scatter our data through it.
        relax_shape = relax.ShapeExpr(total_output_shape.tolist())
        zeros_tensor = bb.emit(relax.op.zeros(relax_shape, data.struct_info.dtype))
        # We need to flatten all our tensors before scattering.
        flat_tensor = relax.op.scatter_elements(
            relax.op.reshape(zeros_tensor, [-1]),
            relax.op.reshape(indices, [-1]),
            relax.op.reshape(data, [-1]),
            axis=0,
        )
        # Reshape our flattened data back to normal.
        output = relax.op.reshape(flat_tensor, relax_shape)
        return output


class Flatten(OnnxOpConverter):
    """Converts an onnx Flatten node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 1)
        data_shape = list(inputs[0].struct_info.shape)

        if axis == 0:
            new_shape = (1, -1)
        else:
            shape_flags = [isinstance(x, tvm.script.tir.IntImm) for x in data_shape[0:axis]]

            if all(shape_flags):
                data_shape = [x.value for x in data_shape[0:axis]]
                new_shape = (_np.prod(data_shape).astype("int64"), -1)
            else:
                batch_size = 1

                for el in data_shape[0:axis]:
                    batch_size = batch_size * el

                new_shape = (batch_size, -1)

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


class TopK(OnnxOpConverter):
    """Converts an onnx TopK node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        k = inputs[1]
        if not isinstance(k, relax.Constant):
            raise ValueError("TopK k must be a constant")
        k = int(k.data.numpy())
        axis = attr.get("axis", -1)
        largest = attr.get("largest", 1)
        sorted = attr.get("sorted", 1)
        if sorted != 1:
            raise ValueError("TopK sorted must be 1 for Relax frontend")

        return relax.op.topk(data, k, axis, ret_type="both", largest=largest)

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        data = inputs[0]
        k = attr.get("k", 1)
        axis = attr.get("axis", -1)
        return relax.op.topk(data, k, axis, ret_type="both")


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


class OneHot(OnnxOpConverter):
    """Converts an onnx OneHot node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        indices = inputs[0]
        depth = get_constant(inputs[1], params)
        values = get_constant(inputs[2], params)
        axis = attr.get("axis", -1)
        assert isinstance(depth, relax.Constant), "Only constant depth currently supported."
        depth = depth.data.numpy().tolist()
        assert isinstance(values, relax.Constant), "Only constant values currently supported."
        values = values.data.numpy().tolist()
        off_value, on_value = values
        off_value, on_value = relax.PrimValue(off_value), relax.PrimValue(on_value)
        return relax.op.one_hot(indices, on_value, off_value, depth, axis)


class Unique(OnnxOpConverter):
    """Converts an onnx Unique node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        data = inputs[0]
        axis = attr.get("axis", None)
        sorted = bool(attr.get("sorted", 1))
        # TODO(tvm-team): Add support for return_index, return_inverse, return_counts
        unique = relax.op.unique(data, sorted=sorted, axis=axis)
        unique_numbers = tir.Var("unique_numbers", "int64")
        input_shape = data.struct_info.shape
        dtype = data.struct_info.dtype

        if axis is None:
            # flatten the input tensor
            return bb.match_cast(unique, relax.TensorStructInfo((unique_numbers,), dtype))

        axis = axis if axis >= 0 else len(input_shape) + axis
        if axis < 0 or axis >= len(input_shape):
            raise ValueError(f"Axis {axis} is out of bounds")
        output_shape = [
            input_shape[i] if i != axis else unique_numbers for i in range(len(input_shape))
        ]
        return bb.match_cast(unique, relax.TensorStructInfo(output_shape, dtype))


class NonZero(OnnxOpConverter):
    """Converts an onnx NonZero node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        ndim = inputs[0].struct_info.ndim
        ndim = 1 if ndim == 0 else ndim
        nonzero_numbers = tir.Var("nonzero_numbers", "int64")
        return bb.match_cast(
            relax.op.nonzero(inputs[0]), relax.TensorStructInfo((ndim, nonzero_numbers), "int64")
        )


class HardSigmoid(OnnxOpConverter):
    """Converts an onnx HardSigmoid node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        x = inputs[0]
        dtype = x.struct_info.dtype
        alpha = float(attr.get("alpha", 0.2))
        alpha = relax.const(alpha, dtype=dtype)
        beta = float(attr.get("beta", 0.5))
        beta = relax.const(beta, dtype=dtype)
        return relax.op.clip(relax.op.add(relax.op.multiply(alpha, x), beta), 0, 1)


class HardSwish(OnnxOpConverter):
    """Converts an onnx HardSwish node into an equivalent Relax expression."""

    @classmethod
    def _impl_v14(cls, bb, inputs, attr, params):
        x = inputs[0]
        dtype = x.struct_info.dtype
        return relax.op.multiply(
            x,
            relax.op.divide(
                relax.op.clip(relax.op.add(x, relax.const(3, dtype)), 0, 6),
                relax.expr.const(6, dtype),
            ),
        )


class Sign(OnnxOpConverter):
    """Converts an onnx Sign node into an equivalent Relax expression."""

    @classmethod
    def _impl_v9(cls, bb, inputs, attr, params):
        return relax.op.sign(inputs[0])


class Not(OnnxOpConverter):
    """Converts an onnx Not node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        return relax.op.logical_not(inputs[0])


class DepthToSpace(OnnxOpConverter):
    """Converts an onnx DepthToSpace node into an equivalent Relax expression."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        block_size = int(attr["blocksize"])
        mode = attr.get("mode", b"DCR").decode("utf-8")
        b, c, h, w = inputs[0].struct_info.shape
        if mode == "DCR":
            x = relax.op.reshape(
                inputs[0], (b, block_size, block_size, c // (block_size**2), h, w)
            )
            x = relax.op.permute_dims(x, [0, 3, 4, 1, 5, 2])
            return relax.op.reshape(x, (b, c // (block_size**2), h * block_size, w * block_size))
        elif mode == "CRD":
            x = relax.op.reshape(
                inputs[0], (b, c // (block_size**2), block_size, block_size, h, w)
            )
            x = relax.op.permute_dims(x, [0, 1, 4, 2, 5, 3])
            return relax.op.reshape(x, (b, c // (block_size**2), h * block_size, w * block_size))
        else:
            raise ValueError(f"Unsupported mode: {mode}, expected DCR or CRD")


class SpaceToDepth(OnnxOpConverter):
    """Converts an onnx SpaceToDepth node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr, params):
        block_size = int(attr["blocksize"])
        b, c, h, w = inputs[0].struct_info.shape
        x = relax.op.reshape(
            inputs[0], (b, c, h // block_size, block_size, w // block_size, block_size)
        )
        x = relax.op.permute_dims(x, [0, 3, 5, 1, 2, 4])
        return relax.op.reshape(
            x, (b, c * block_size * block_size, h // block_size, w // block_size)
        )


class SequenceConstruct(OnnxOpConverter):
    """Operator converter for sequence construction op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        # Construct a tuple from input tensors.
        return relax.Tuple(inputs)


class SequenceEmpty(OnnxOpConverter):
    """Operator converter for sequence empty op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        # Construct an empty tuple.
        return relax.Tuple([])


class SequenceErase(OnnxOpConverter):
    """Operator converter for sequence erase op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        # Erase tensor from sequence on specified position
        input_sequence = inputs[0]

        if len(inputs) == 2:
            position = inputs[1]
            # Non constant position is not supported.
            if isinstance(position, relax.Constant):
                position = int(position.data.numpy())
            else:
                raise NotImplementedError("Position must be a constant.")
        else:
            position = -1

        seq_len = len(input_sequence)
        if not -seq_len <= position < seq_len:
            raise ValueError(
                f"Position is out of bounds, expected [-{seq_len}, {seq_len}), got {position}"
            )

        if position < 0:
            position = seq_len + position
        # Convert sequence to a list, insert tensors before erased, and repackage as Tuple.
        tensor_list = [input_sequence[i] for i in range(seq_len) if i != position]
        # Create new tuple and return.
        return relax.Tuple(tensor_list)


class SequenceInsert(OnnxOpConverter):
    """Operator converter for sequence insert op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        # Insert a new tensor into a tuple of tensors.
        input_sequence = inputs[0]
        tensor_to_insert = inputs[1]

        if len(inputs) == 3:
            position = inputs[2]
            # Non constant position is not supported.
            if isinstance(position, relax.Constant):
                position = position.data.numpy()
            else:
                raise NotImplementedError("Position must be a constant.")
        else:
            position = -1

        if position < 0:
            position = len(input_sequence) + position + 1
        # Convert sequence to a list, insert new tensor, and repackage as Tuple.
        tensor_list = [input_sequence[i] for i in range(len(input_sequence))]
        # Insert new tensor.
        tensor_list.insert(position, tensor_to_insert)
        # Create new tuple and return.
        return relax.Tuple(tensor_list)


class SequenceLength(OnnxOpConverter):
    """Operator converter for sequence length op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        # Get length of input sequence
        return relax.const(len(inputs[0]), dtype="int64")


class ConcatFromSequence(OnnxOpConverter):
    """Operator converter for sequence concatenation op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 0)
        new_axis = attr.get("new_axis", 0)

        if new_axis == 1:
            raise NotImplementedError("Insert new axis is not supported yet.")

        return relax.op.concat(inputs[0], axis=axis)


class SplitToSequence(OnnxOpConverter):
    """Operator converter for split to sequence op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        axis = attr.get("axis", 0)
        keepdims = attr.get("keepdims", 1)

        input_tensor = inputs[0]
        input_shape = input_tensor.struct_info.shape

        # If split is not provided, we split all values along axis.
        if len(inputs) == 1:
            split = _np.array(1)
            if not keepdims:
                raise NotImplementedError("Only keepdims=1 is supported for now")
        else:
            split = inputs[1]
            if not isinstance(split, relax.Constant):
                raise ValueError("Only constant split supported for SplitToSequence")
            split = split.data.numpy()

        if len(split.shape) == 1 and split.shape[0] > 1:
            split = _np.cumsum(split)
            split = list(split[:-1])
        else:
            chunk_size, dim_size = int(split), input_shape[axis]
            if dim_size % chunk_size != 0:
                raise ValueError(
                    f"Dimension of size {dim_size} along axis {axis} must be "
                    f"evenly divisible by chunk size {chunk_size}"
                )
            split = dim_size // chunk_size

        output = relax.op.split(input_tensor, split, axis=axis)
        return output


class SequenceAt(OnnxOpConverter):
    """Operator converter for sequence at op."""

    @classmethod
    def _impl_v11(cls, bb, inputs, attr, params):
        input_sequence = inputs[0]
        position = inputs[1]
        assert isinstance(
            position, relax.Constant
        ), "Only constant position supported for SequenceAt"
        position = int(position.data.numpy())
        return input_sequence[position]


def _get_convert_map():
    return {
        # defs/experimental
        # "Optional": Optional_,
        # "OptionalHasElement": OptionalHasElement,
        # "OptionalGetElement": OptionalGetElement,
        # Binary operators
        "Add": Add,
        "Sub": Sub,
        "Mul": Mul,
        "Div": Div,
        "Mod": Mod,
        "Less": Less,
        "LessOrEqual": LessOrEqual,
        "Greater": Greater,
        "GreaterOrEqual": GreaterOrEqual,
        "Equal": Equal,
        "BitwiseAnd": BitwiseAnd,
        "BitwiseOr": BitwiseOr,
        "BitwiseXor": BitwiseXor,
        "BitwiseNot": BitwiseNot,
        "BitShift": BitShift,
        "And": And,
        "Or": Or,
        "Xor": Xor,
        "Not": Not,
        # Unary operators
        "Log": Log,
        "Exp": Exp,
        "Acos": Acos,
        "Acosh": Acosh,
        "Asin": Asin,
        "Asinh": Asinh,
        "Atan": Atan,
        "Atanh": Atanh,
        "Cos": Cos,
        "Cosh": Cosh,
        "Sin": Sin,
        "Sinh": Sinh,
        "Tan": Tan,
        "Tanh": Tanh,
        "Neg": Neg,
        "Abs": Abs,
        "Reciprocal": Reciprocal,
        "Floor": Floor,
        "Ceil": Ceil,
        "Round": Round,
        "IsInf": IsInf,
        "IsNaN": IsNaN,
        "Sqrt": Sqrt,
        "Relu": Relu,
        "Selu": Selu,
        "Mish": Mish,
        "Trilu": Trilu,
        "PRelu": PRelu,
        "LeakyRelu": LeakyRelu,
        "ThresholdedRelu": ThresholdedRelu,
        "Elu": Elu,
        "Gelu": Gelu,
        "FastGelu": FastGelu,
        "BiasGelu": BiasGelu,
        "HardSigmoid": HardSigmoid,
        "HardSwish": HardSwish,
        "Sign": Sign,
        "Softplus": Softplus,
        "Softsign": Softsign,
        "Shrink": Shrink,
        "Erf": Erf,
        "Sum": Sum,
        "Min": Min,
        "Max": Max,
        "Mean": Mean,
        "Cast": Cast,
        "Gemm": Gemm,
        "MatMul": MatMul,
        # "MatMulInteger": MatMulInteger,
        # "MatMulInteger16": MatMulInteger16,
        "Reshape": Reshape,
        "Sigmoid": Sigmoid,
        "Softmax": Softmax,
        "LogSoftmax": LogSoftmax,
        "Hardmax": Hardmax,
        "Transpose": Transpose,
        "Unsqueeze": Unsqueeze,
        "Where": Where,
        "Concat": Concat,
        "Clip": Clip,
        "Shape": Shape,
        "Pow": Pow,
        "CumSum": CumSum,
        "Squeeze": Squeeze,
        "Constant": Constant,
        "Gather": Gather,
        "GatherElements": GatherElements,
        "GatherND": GatherND,
        "Scatter": Scatter,
        "ScatterElements": ScatterElements,
        "ScatterND": ScatterND,
        "Compress": Compress,
        "Size": Size,
        "EyeLike": EyeLike,
        # Normalization
        "BatchNormalization": BatchNormalization,
        "LayerNormalization": LayerNormalization,
        "SkipLayerNormalization": SkipLayerNormalization,
        "EmbedLayerNormalization": EmbedLayerNormalization,
        "InstanceNormalization": InstanceNormalization,
        "MeanVarianceNormalization": MeanVarianceNormalization,
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
        "TopK": TopK,
        "Expand": Expand,
        "ConstantOfShape": ConstantOfShape,
        "Slice": Slice,
        "Attention": Attention,
        "Pad": Pad,
        "Split": Split,
        "Tile": Tile,
        "AveragePool": AveragePool,
        "MaxPool": MaxPool,
        "LpPool": LpPool,
        "GlobalAveragePool": GlobalAveragePool,
        "GlobalMaxPool": GlobalMaxPool,
        "GlobalLpPool": GlobalLpPool,
        "MaxUnpool": MaxUnpool,
        "Conv": Conv,
        "ConvTranspose": ConvTranspose,
        "Flatten": Flatten,
        "Identity": Identity,
        "Resize": Resize,
        "Einsum": Einsum,
        "Range": Range,
        "OneHot": OneHot,
        "Unique": Unique,
        "NonZero": NonZero,
        # "If": If,
        # "LRN": LRN,
        # "MaxRoiPool": MaxRoiPool,
        # "RoiAlign": RoiAlign,
        # "NonMaxSuppression": NonMaxSuppression,
        # "GridSample": GridSample,
        # "Upsample": Upsample,
        # others
        "DepthToSpace": DepthToSpace,
        "SpaceToDepth": SpaceToDepth,
        # Sequence operators
        "SequenceConstruct": SequenceConstruct,
        "SequenceEmpty": SequenceEmpty,
        "SequenceErase": SequenceErase,
        "SequenceInsert": SequenceInsert,
        "SequenceLength": SequenceLength,
        "ConcatFromSequence": ConcatFromSequence,
        "SplitToSequence": SplitToSequence,
        "SequenceAt": SequenceAt,
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
                self._params[var_name] = (init_var, array)
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
                "Cast",
                "Squeeze",
            ]
            return_tuple_ops = [
                "SequenceConstruct",
                "SequenceEmpty",
                "SequenceErase",
                "SequenceInsert",
                "ConcatFromSequence",
                "SplitToSequence",
            ]
            for i, inp in enumerate(inputs):
                if (
                    inp is not None
                    and isinstance(inp, relax.Expr)
                    and isinstance(inp.struct_info, relax.ShapeStructInfo)
                    and op_name not in shape_compatible_ops
                ):
                    raise ValueError(f"Node {node.name} cannot handle ShapeExpr inputs.")
            try:
                op = self._convert_operator(op_name, inputs, attr, self.opset)
                # Create struct information for the new operator.
                op = self.bb.normalize(op)
            except TVMError as err:
                print(f"Error converting operator {op_name}, with inputs: {inputs}")
                raise err

            if op_name in return_tuple_ops:
                outputs_num = 1
            elif not isinstance(op, relax.Tuple):
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
        inputs: List[relax.Expr],
        attrs: Dict,
        opset: int,
    ) -> relax.Expr:
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
) -> IRModule:
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
