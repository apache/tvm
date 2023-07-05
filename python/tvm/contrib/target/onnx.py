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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines, redefined-builtin
"""Relay to ONNX codegen """

import os
import struct
import copy
import numpy
import onnx
import onnx.utils
from onnx import numpy_helper, OperatorSetIdProto, defs
from onnx import TensorProto
import tvm
from tvm import relay
import tvm._ffi
from tvm.relay.expr_functor import ExprVisitor
from tvm.relay.ty import TupleType, TensorType

ONNX_OPSET_VERSONS_SUPPORTED = [11]


def run_onnx_optimizer(onnx_model):
    """Run ONNX's optimization routines.

    ONNX Optimizer was moved to an external library in
    version 1.9.  Attempt to use the optimizer in onnx if
    it is available, fall back to the standalone
    onnxoptimizer otherwise, and return the model
    unoptimized if neither are available.

    """
    try:
        onnx_polish_model = onnx.utils.polish_model
    except AttributeError:
        pass
    else:
        return onnx_polish_model(onnx_model)

    try:
        # pylint: disable=import-outside-toplevel
        import onnxoptimizer
    except ImportError:
        pass
    else:
        return onnxoptimizer.optimize(onnx_model)

    return onnx_model


def tvm_array_to_list(arr):
    return tuple(x.value for x in arr)


def get_onnx_version():
    return onnx.__version__


def get_node_shape(node):
    return tuple("Any" if isinstance(i, tvm.tir.Any) else int(i) for i in node.shape)


def infer_type(node):
    """A method to infer the type of a relay expression."""
    mod = tvm.IRModule.from_expr(node)
    mod = relay.transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(node, relay.Function) else entry.body


def call_node_infer_type(node):
    """infer the output types of call node"""
    infer_out = infer_type(node)
    out_type = infer_out._checked_type_
    if isinstance(out_type, TensorType):
        types = [out_type]
    elif isinstance(out_type, TupleType):
        types = list(out_type.fields)
    else:
        raise RuntimeError(f"Unsupported output type {type(out_type)} in operator {node.op.name}")

    return types


def add_input(data, name, prefix, model_container):
    input_name = f"{prefix}_{name}"
    dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[data.dtype]
    tensor_value_info = onnx.helper.make_tensor_value_info(input_name, dtype, shape=data.shape)
    model_container.add_inputs([tensor_value_info])
    data_tensor = numpy_helper.from_array(data, input_name)
    model_container.add_initializers([data_tensor])
    return input_name


class OpConverter(object):
    """Operator converter Base Class."""

    @classmethod
    def convert_attributes(cls, attrs):
        """convert Relay attributes to ONNX attributes.
        The derived classes should implement this method
        if attributes are required by the operator
        otherwise by default no attributes are passed
        """
        return {}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)
        onnx_node = onnx.helper.make_node(
            cls.__name__, node_entry["input_names"], node_entry["output_names"], **attrs
        )
        model_container.add_nodes([onnx_node])


def rename(op_name):
    """This method creates dynamic operator of name op_name with empty attributes"""
    return type(op_name, (OpConverter,), {})


class Reshape(object):
    """Operator converter for Reshape."""

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        """Converts Relay operator Reshape to ONNX operator.
        Relay operator accepts shape as attribute but ONNX operator
        accepts it as a input.
        """
        name = node_entry["name"]
        shape = numpy.asarray(
            [a.value for a in node_entry["relay_node"].attrs.newshape], dtype=numpy.int64
        )

        input_names = [
            node_entry["input_names"][0],
            add_input(shape, name, "shape", model_container),
        ]

        node = onnx.helper.make_node(cls.__name__, input_names, node_entry["output_names"])
        model_container.add_nodes([node])


class Conv(OpConverter):
    """Operator converter for Conv."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "group": attrs.get_int("groups"),
            "pads": attrs.get_int_tuple("padding"),
            "strides": attrs.get_int_tuple("strides"),
            "dilations": attrs.get_int_tuple("dilation"),
            "kernel_shape": attrs.get_int_tuple("kernel_size"),
        }


class ConvTranspose(OpConverter):
    """Operator converter for ConvTranspose."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "group": attrs.get_int("groups"),
            "pads": attrs.get_int_tuple("padding"),
            "strides": attrs.get_int_tuple("strides"),
            "dilations": attrs.get_int_tuple("dilation"),
            "kernel_shape": attrs.get_int_tuple("kernel_size"),
            "output_padding": attrs.get_int_tuple("output_padding"),
        }


class MaxPool(OpConverter):
    """Operator converter for MaxPool."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "pads": attrs.get_int_tuple("padding"),
            "strides": attrs.get_int_tuple("strides"),
            "kernel_shape": attrs.get_int_tuple("pool_size"),
            "ceil_mode": 1 if attrs.ceil_mode else 0,
        }


class Transpose(OpConverter):
    """Operator converter for Transpose."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"perm": attrs.get_int_tuple("axes")} if attrs["axes"] else {}


class MatMul(OpConverter):
    """Operator converter for MatMul."""

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        inter_output_name = f"inter{node_entry['name']}"
        transpose_node = onnx.helper.make_node(
            Transpose.__name__, [node_entry["input_names"][1]], [inter_output_name], perm=(1, 0)
        )
        model_container.add_nodes([transpose_node])

        inputs = [node_entry["input_names"][0], inter_output_name]
        matmul_node = onnx.helper.make_node(cls.__name__, inputs, node_entry["output_names"])
        model_container.add_nodes([matmul_node])


class Flatten(OpConverter):
    """Operator converter for Flatten."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"axis": 1}


class BatchNormalization(OpConverter):
    """Operator converter for BatchNormalization."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"epsilon": float(attrs.get_str("epsilon")), "axis": float(attrs.get_int("axis"))}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        """Converts Relay operator batch_norm to ONNX operator.
        Relay operator has property axis to handle data in NHWC format.
        """
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)
        transpose_out_name = node_entry["input_names"][0]
        inter_output_names = [node_entry["output_names"][0]]
        # axis==3 means channel is specified along the 3rd axis
        if attrs["axis"] == 3:
            transpose_out_name = f"transpose_{node_entry['name']}"
            node_transposed = onnx.helper.make_node(
                Transpose.__name__,
                [node_entry["input_names"][0]],
                [transpose_out_name],
                perm=[0, 3, 1, 2],
            )
            model_container.add_nodes([node_transposed])
            inter_output_names = [f"batch_norm_{node_entry['name']}"]

        input_names = [transpose_out_name] + node_entry["input_names"][1:]
        batch_norm_node = onnx.helper.make_node(
            cls.__name__, input_names, inter_output_names, epsilon=attrs["epsilon"]
        )
        model_container.add_nodes([batch_norm_node])

        if attrs["axis"] == 3:
            node_transposed = onnx.helper.make_node(
                Transpose.__name__,
                inter_output_names,
                [node_entry["output_names"][0]],
                perm=[0, 2, 3, 1],
            )
            model_container.add_nodes([node_transposed])


class Dropout(OpConverter):
    """Operator converter for Dropout."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"ratio": float(attrs.get_str("rate"))}


class AveragePool(MaxPool):
    """Operator converter for AveragePool."""


class Concat(OpConverter):
    """Operator converter for Concat."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"axis": attrs.get_int("axis")}


class BiasAdd(OpConverter):
    """Operator converter for BiasAdd."""

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node_entry can not be a Tuple"
        input_node = input_node[0]
        data_ndim = len(input_node["types"][0].shape)
        axis = node_entry["relay_node"].attrs.get_int("axis")
        if axis < 0:
            axis = axis + data_ndim
        new_axes = data_ndim - axis - 1
        if new_axes:
            inter_output_name = f"inter{node_entry['name']}"
            unsqueeze_node = onnx.helper.make_node(
                "Unsqueeze",
                [node_entry["input_names"][1]],
                [inter_output_name],
                axes=tuple(range(1, new_axes + 1)),
            )
            model_container.add_nodes([unsqueeze_node])
        else:
            inter_output_name = node_entry["input_names"][1]

        inputs = [node_entry["input_names"][0], inter_output_name]
        matmul_node = onnx.helper.make_node("Add", inputs, node_entry["output_names"])
        model_container.add_nodes([matmul_node])


class ReduceMean(OpConverter):
    """Operator converter for ReduceMean."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "axes": attrs.axis,
            "keepdims": 0 if bool(attrs.get_int("keepdims", 0)) is False else 1,
        }

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node can not be a Tuple"
        input_node = input_node[0]
        shape = input_node["types"][0].shape
        axis = node_entry["relay_node"].attrs.axis
        axis = list(range(shape.size())) if not axis else tvm_array_to_list(axis)
        exclude = 0 if not bool(node_entry["relay_node"].attrs.exclude) else 1
        keepdims = 0 if not bool(node_entry["relay_node"].attrs.keepdims) else 1
        if exclude:
            all_axis = list(range(len(shape)))
            axis = set(all_axis) - set(axis)

        node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            axes=axis,
            keepdims=keepdims,
        )
        model_container.add_nodes([node])


class Pad(OpConverter):
    """Operator converter for Pad."""

    @classmethod
    def convert_attributes(cls, attrs):
        before = []
        after = []
        for axis_pads in attrs.pad_width:
            before.append(axis_pads[0])
            after.append(axis_pads[1])
        pads = before + after
        pads = numpy.asarray(pads, dtype=pads[0].dtype)
        return {"pads": pads, "mode": attrs.get_str("pad_mode")}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        """Converts Relay operator Pad to ONNX operator.
        Relay operator accepts pads as attribute but ONNX operator
        accepts it as a input.
        """
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)

        name = node_entry["name"]
        pad_data = numpy.asarray(attrs["pads"], dtype=attrs["pads"][0].dtype).astype(numpy.int64)

        input_names = [
            node_entry["input_names"][0],
            add_input(pad_data, name, "pads", model_container),
            node_entry["input_names"][1],
        ]

        node = onnx.helper.make_node(
            cls.__name__, input_names, node_entry["output_names"], mode=attrs["mode"]
        )
        model_container.add_nodes([node])


class Softmax(OpConverter):
    """Operator converter for SoftMax."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"axis": attrs.axis}


class Squeeze(OpConverter):
    """Operator converter for Squeeze."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"axes": attrs.axis}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node can not be a Tuple"
        input_node = input_node[0]
        shape = input_node["types"][0].shape
        axis = node_entry["relay_node"].attrs.get_int("axis")
        if not axis:
            axis = []
            for axis_idx, val in enumerate(shape):
                if val.value == 1:
                    axis.append(axis_idx)
        else:
            axis = node_entry["relay_node"].attrs.get_int_tuple("axis")

        node = onnx.helper.make_node(
            cls.__name__, node_entry["input_names"], node_entry["output_names"], axes=axis
        )
        model_container.add_nodes([node])


class Slice(OpConverter):
    """Operator converter for Slice."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {
            "starts": attrs.get_int_tuple("begin"),
            "ends": attrs.get_int_tuple("end"),
            "steps": attrs.get_int_tuple("strides"),
            "slice_mode": attrs.get_str("slice_mode"),
        }

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)

        name = node_entry["name"]
        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node can not be a Tuple"
        input_node = input_node[0]
        shape = input_node["types"][0].shape

        starts = list(attrs["starts"])
        ends = list(attrs["ends"])
        steps = list(attrs["steps"])
        starts += [0] * (len(shape) - len(starts))
        ends += [shape[i] + 1 for i in range(len(ends), len(shape))]
        axes = list(range(len(shape)))

        if attrs["slice_mode"] == "size":
            ends = [
                starts[i] + (shape[i] + 1 if ends[i] < 0 else ends[i]) for i in range(len(shape))
            ]
            steps = [1] * len(shape)
        else:
            steps += [1] * (len(shape) - len(steps))

        starts = numpy.asarray(starts).astype(numpy.int64)
        ends = numpy.asarray(ends).astype(numpy.int64)
        axes = numpy.asarray(axes).astype(numpy.int64)
        steps = numpy.asarray(steps).astype(numpy.int64)

        input_names = []
        input_names.append(add_input(starts, name, "starts", model_container))
        input_names.append(add_input(ends, name, "ends", model_container))
        input_names.append(add_input(axes, name, "axes", model_container))
        input_names.append(add_input(steps, name, "steps", model_container))

        input_names = [node_entry["input_names"][0]] + input_names

        slice_node = onnx.helper.make_node(cls.__name__, input_names, node_entry["output_names"])
        model_container.add_nodes([slice_node])


class Split(OpConverter):
    """Operator converter for Split."""

    @classmethod
    def convert_attributes(cls, attrs):
        indices_or_sections = attrs["indices_or_sections"]

        if isinstance(indices_or_sections, (list, tvm.ir.container.Array)):
            indices_or_sections = attrs.get_int_tuple("indices_or_sections")
        if isinstance(indices_or_sections, tvm.ir.PrimExpr):
            indices_or_sections = indices_or_sections.value

        return {"indices_or_section": indices_or_sections, "axis": attrs.get_int("axis")}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)

        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node can not be a Tuple"
        input_node = input_node[0]
        shape = get_node_shape(input_node["types"][0])

        indices_or_sect = attrs["indices_or_section"]
        axis = attrs["axis"]
        axis_length = shape[axis]

        if isinstance(indices_or_sect, int):
            split = [axis_length // indices_or_sect] * indices_or_sect
        else:
            split = []
            for i in range(len(indices_or_sect) + 1):
                if i == 0:
                    split.append(indices_or_sect[0])
                elif i == len(indices_or_sect):
                    split.append(axis_length - indices_or_sect[-1])
                else:
                    split.append(indices_or_sect[i] - indices_or_sect[i - 1])

        slice_node = onnx.helper.make_node(
            cls.__name__,
            node_entry["input_names"],
            node_entry["output_names"],
            split=split,
            axis=axis,
        )
        model_container.add_nodes([slice_node])


class LayoutTransform(OpConverter):
    """Operator converter for Layouttransform"""

    @classmethod
    def convert_attributes(cls, attrs):
        src_layout = attrs.get_str("src_layout")
        dst_layout = attrs.get_str("dst_layout")

        perm = [src_layout.index(c) for c in dst_layout]
        return {"perm": tuple(perm)}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)
        onnx_node = onnx.helper.make_node(
            "Transpose", node_entry["input_names"], node_entry["output_names"], **attrs
        )
        model_container.add_nodes([onnx_node])


class Clip(OpConverter):
    """Operator converter for Clip."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"min": attrs.a_min, "max": attrs.a_max}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)

        name = node_entry["name"]

        min_val = numpy.asarray(attrs["min"]).astype(numpy.float32)
        max_val = numpy.asarray(attrs["max"]).astype(numpy.float32)

        input_names = []
        input_names.append(add_input(min_val, name, "min", model_container))
        input_names.append(add_input(max_val, name, "max", model_container))

        input_names = [node_entry["input_names"][0]] + input_names

        node = onnx.helper.make_node(cls.__name__, input_names, node_entry["output_names"])
        model_container.add_nodes([node])


class Expand(OpConverter):
    """Operator converter for Expand_dims."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"axis": attrs.axis, "num_newaxis": attrs.num_newaxis}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)

        name = node_entry["name"]

        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node_entry can not be a Tuple"
        input_node = input_node[0]
        data_shape = input_node["types"][0].shape
        new_shape = list(data_shape)

        for _ in range(attrs["num_newaxis"]):
            new_shape.insert(attrs["axis"], 1)

        new_shape = numpy.asarray(new_shape).astype(numpy.int64)
        input_names = []
        input_names.append(add_input(new_shape, name, "shape", model_container))

        input_names = [node_entry["input_names"][0]] + input_names

        node = onnx.helper.make_node(cls.__name__, input_names, node_entry["output_names"])
        model_container.add_nodes([node])


class ConstantOfShapeZeros(OpConverter):
    """Operator converter for ConstantOfShape."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"value": 0}

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)
        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node can not be a Tuple"
        input_node = input_node[0]
        dtype = input_node["types"][0].dtype

        name = node_entry["name"]
        shape = [val.value for val in input_node["types"][0].shape]
        shape = numpy.asarray(shape).astype(numpy.int64)

        input_names = []
        input_names.append(add_input(shape, name, "shape", model_container))

        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(dtype)]
        tensor_value = onnx.helper.make_tensor("value", dtype, [1], [attrs["value"]])

        node = onnx.helper.make_node(
            "ConstantOfShape", input_names, node_entry["output_names"], value=tensor_value
        )
        model_container.add_nodes([node])


class ConstantOfShapeOnes(ConstantOfShapeZeros):
    """Operator converter for ConstantOfShape."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"value": 1}


class LRN(OpConverter):
    """Operator converter for LRN."""

    @classmethod
    def convert_attributes(cls, attrs):
        """axis attr is not supported as an argument in onnx.
        Onnx only supports axis=1 (channels)."""
        if attrs.get_int("axis") != 1:
            raise RuntimeError(
                f"Unsupported axis {attrs.get_int('axis')} in operator relay lrn operator. "
                f"Only axis = 1 is supported by Onnx."
            )

        return {"alpha": attrs.alpha, "beta": attrs.beta, "bias": attrs.bias, "size": attrs.size}


class Cast(OpConverter):
    """Operator converter for Cast."""

    @classmethod
    def convert_attributes(cls, attrs):
        return {"to": getattr(TensorProto, attrs.dtype.upper())}


class Resize(OpConverter):
    """Operator converter for Resize."""

    @classmethod
    def convert_attributes(cls, attrs):
        method = attrs.get_str("method")
        if method == "nearest_neighbor":
            mode = b"nearest"
        elif "linear" in method:  # linear / bilinear
            mode = b"linear"
        elif "cubic" in method:  # cubic / bicubic
            mode = b"cubic"
        else:
            raise RuntimeError(f"Unsupported method {method} in operator Resize")

        coord_trans = attrs.get_str("coordinate_transformation_mode")
        if coord_trans == "half_pixel":
            coord_trans = b"half_pixel"
        elif coord_trans == "align_corners":
            coord_trans = b"align_corners"
        elif coord_trans == "asymmetric":
            coord_trans = b"asymmetric"
        else:
            raise RuntimeError(
                f"Unsupported coordinate transform mode {coord_trans} in operator Resize"
            )

        rounding_method = attrs.get_str("rounding_method")
        if rounding_method == "round":
            rounding_method = b"round_prefer_ceil"
        elif rounding_method == "floor":
            rounding_method = b"floor"
        elif rounding_method == "ceil":
            rounding_method = b"ceil"
        else:
            raise RuntimeError(f"Unsupported rounding method {rounding_method} in operator Resize")

        size = attrs.get_int_tuple("size")

        return {
            "mode": mode,
            "coord_trans": coord_trans,
            "size": size,
            "nearest_mode": rounding_method,
        }

    @classmethod
    def convert(cls, node_entry, model_container, node_dict):
        attrs = cls.convert_attributes(node_entry["relay_node"].attrs)

        name = node_entry["name"]
        input_node = node_dict[node_entry["inputs"][0]]
        assert len(input_node) == 1, "input node can not be a Tuple"
        input_node = input_node[0]
        input_shape = input_node["types"][0].shape

        # (TBD) needed in opset 11
        roi = [0] * len(input_shape) + [1] * len(input_shape)
        roi_array = numpy.asarray(roi).astype(numpy.float64)
        roi_node = add_input(roi_array, name, "roi", model_container)

        out_size = attrs["size"]

        # (onnx) rank of scale / size must match rank of X
        # relay size node contains only spatial dimensions
        # pad with 1s to match rank
        match_rank_pad = len(input_shape) - len(out_size)
        out_size_full_rank = input_shape[:match_rank_pad] + list(out_size)
        out_size_array = numpy.asarray(out_size_full_rank).astype(numpy.int64)

        input_size_array = numpy.asarray(list(input_shape)).astype(numpy.int64)

        scale_array = numpy.divide(out_size_array, input_size_array).astype(numpy.float32)
        scale_node = add_input(scale_array, name, "scales", model_container)

        input_names = [node_entry["input_names"][0], roi_node, scale_node]

        resize_node = onnx.helper.make_node(
            cls.__name__,
            input_names,
            node_entry["output_names"],
            mode=attrs["mode"],
            coordinate_transformation_mode=attrs["coord_trans"],
            nearest_mode=attrs["nearest_mode"],
        )
        model_container.add_nodes([resize_node])


relay_to_onnx_op_mapping = {
    "reshape": Reshape,
    "nn.conv2d": Conv,
    "nn.conv2d_transpose": ConvTranspose,
    "add": rename("Add"),
    "nn.relu": rename("Relu"),
    "transpose": Transpose,
    "nn.dense": MatMul,
    "nn.max_pool2d": MaxPool,
    "nn.batch_flatten": Flatten,
    "multiply": rename("Mul"),
    "nn.bias_add": BiasAdd,
    "nn.batch_norm": BatchNormalization,
    "nn.global_avg_pool2d": rename("GlobalAveragePool"),
    "concatenate": Concat,
    "nn.dropout": Dropout,
    "nn.avg_pool2d": AveragePool,
    "divide": rename("Div"),
    "mean": ReduceMean,
    "nn.pad": Pad,
    "nn.softmax": Softmax,
    "squeeze": Squeeze,
    "strided_slice": Slice,
    "greater": rename("Greater"),
    "less": rename("Less"),
    "equal": rename("Equal"),
    "zeros_like": ConstantOfShapeZeros,
    "ones_like": ConstantOfShapeOnes,
    "subtract": rename("Sub"),
    "split": Split,
    "exp": rename("Exp"),
    "layout_transform": LayoutTransform,
    "clip": Clip,
    "expand_dims": Expand,
    "nn.lrn": LRN,
    "sigmoid": rename("Sigmoid"),
    "copy": rename("Identity"),
    "round": rename("Round"),
    "cast": Cast,
    "image.resize2d": Resize,
}


class ModelContainer(object):
    """A container class to hold  different attributes of ONNX model graph"""

    def __init__(self, name, opset_version):
        self._name = name
        self._opset_version = opset_version
        self._inputs = []
        self._outputs = []
        self._nodes = []
        self._initializers = []

    def add_inputs(self, inputs):
        self._inputs.extend(inputs)

    def add_outputs(self, outputs):
        self._outputs.extend(outputs)

    def add_nodes(self, nodes):
        self._nodes.extend(nodes)

    def add_initializers(self, initializers):
        self._initializers.extend(initializers)

    def _get_opsets(self):
        opsets = []
        imp = OperatorSetIdProto()
        imp.version = self._opset_version
        opsets.append(imp)
        return opsets

    def make_model(self):
        """Creates the onnx model from the graph"""
        onnx_graph = onnx.helper.make_graph(
            self._nodes, self._name, self._inputs, self._outputs, self._initializers
        )
        kwargs = {}
        kwargs["opset_imports"] = self._get_opsets()
        kwargs["producer_name"] = "TVM Relay"
        kwargs["producer_version"] = tvm.__version__

        return onnx.helper.make_model(onnx_graph, **kwargs)


class RelayToONNXConverter(ExprVisitor):
    """A helper class to traverse the Relay graph and convert Relay nodes to ONNX model

    Parameters
    ----------
    name : str
       name of the model

    params : dict
        dict of the parameter names and NDarray values

    opset_version : int
        target onnx opset version

    """

    def __init__(self, name, params, opset_version):
        super().__init__()
        self._name = name
        self._mc = ModelContainer(name, opset_version)
        self._params = params
        self._node_dict = {}
        self._node_count = 0
        self.last_node = None

    @classmethod
    def _get_node_entry(cls, relay_node, name):
        return {
            "relay_node": relay_node,
            "inputs": [relay_node],  # inputs in the form of relay nodes
            "types": [],  # output types in case of call nodes else self type
            "name": name,  # name of the node
            "input_names": [name],  # input names in case of call nodes else self name
            "output_names": [name],  # output names in case of call nodes else self name
            "op": None,  # op name in case of call node else None
        }

    def convert_to_onnx(self, func):
        """Traverse Relay graph and generate a ONNX model"""

        self.visit(func)
        self._add_output(self._node_dict[self.last_node])
        model = self._mc.make_model()
        return run_onnx_optimizer(model)

    def visit(self, expr):
        self._node_count += 1
        super().visit(expr)

    def visit_constant(self, const):
        node_index = self._node_count
        name = self._name + "_const_" + str(node_index)
        node_entry = self._get_node_entry(const, name)
        node_entry["types"] = [const.checked_type]

        self._add_constant_input(node_entry, node_index)
        self._node_dict[const] = [node_entry]

    def visit_var(self, var):
        node_index = self._node_count
        node_entry = self._get_node_entry(var, var.name_hint)
        node_entry["types"] = [var.type_annotation]

        self._add_input(node_entry, node_index)
        self._node_dict[var] = [node_entry]

    def visit_tuple(self, tup):
        self._node_dict[tup] = []
        for f in tup.fields:
            self.visit(f)
            self._node_dict[tup].extend(self._node_dict[f])

        self.last_node = tup

    def visit_tuple_getitem(self, t):
        self.visit(t.tuple_value)
        tup_node = self._node_dict[t.tuple_value]
        if len(tup_node) > 1:
            self._node_dict[t] = tup_node[t.index]
        else:
            node_entry = copy.deepcopy(tup_node[0])
            output_names = [node_entry["output_names"][t.index]]
            node_entry["output_names"] = output_names
            self._node_dict[t] = [node_entry]
        self.last_node = t

    def visit_call(self, call):
        node_index = self._node_count
        op = call.op
        name = f"{op}_{node_index}"
        node_entry = self._get_node_entry(call, name)

        node_entry["op"] = op
        node_entry["input_names"] = []
        node_entry["inputs"] = []
        node_entry["output_names"] = None
        for input_arg in call.args:
            self.visit(input_arg)
            input_names = []
            for arg_node_entry in self._node_dict[input_arg]:
                input_names.extend(arg_node_entry["output_names"])
            node_entry["input_names"].extend(input_names)
            node_entry["inputs"].extend([input_arg])

        node_entry["types"] = call_node_infer_type(call)
        node_entry["output_names"] = []
        for i in range(len(node_entry["types"])):
            node_entry["output_names"].append(name + str(i))
        self.last_node = call
        self._add_node(node_entry, node_index)
        self._node_dict[call] = [node_entry]

    def _add_node(self, node_entry, idx):
        """Convert Relay operator node to ONNX operator and add it to container nodes list"""
        if node_entry["op"].name not in relay_to_onnx_op_mapping:
            raise NotImplementedError(
                f"Currently the operator '{node_entry['op'].name}' is " "not supported."
            )
        converter = relay_to_onnx_op_mapping[node_entry["op"].name]()

        return converter.convert(node_entry, self._mc, self._node_dict)

    def _add_params(self, node_entry, idx):
        """Add param value to initializer and name to inputs"""
        param_name = node_entry["name"]
        assert param_name in self._params, (
            f"The parameter {param_name} is not present" "in params dict provided."
        )
        value = self._params[param_name]
        numpy_array = value.numpy()
        tensor = numpy_helper.from_array(numpy_array, param_name)
        self._mc.add_initializers([tensor])
        dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy_array.dtype]
        input = onnx.helper.make_tensor_value_info(param_name, dtype, shape=numpy_array.shape)
        self._mc.add_inputs([input])

    def _add_constant_input(self, node_entry, idx):
        """Create named input for constant and add it to container inputs.
        If input is a parameter then add to param
        """
        node = node_entry["relay_node"]
        param_name = node_entry["name"]
        self._params[param_name] = node.data
        self._add_params(node_entry, idx)

    def _add_input(self, node_entry, idx):
        """Add input node to container inputs. If input is a parameter then add to param"""
        if node_entry["name"] in self._params:
            self._add_params(node_entry, idx)
        else:
            node_type = node_entry["types"][0]
            dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(node_type.dtype)]
            input = onnx.helper.make_tensor_value_info(
                node_entry["name"], dtype, shape=get_node_shape(node_type)
            )
            self._mc.add_inputs([input])

    def _add_output(self, node_entries):
        """Add output node to container outputs."""

        for node_entry in node_entries:
            for node_type, output_name in zip(node_entry["types"], node_entry["output_names"]):
                dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[numpy.dtype(node_type.dtype)]
                output = onnx.helper.make_tensor_value_info(
                    output_name, dtype, shape=get_node_shape(node_type)
                )
                self._mc.add_outputs([output])


def to_onnx(relay_ir, params, name, opset_version=11, path=None):
    """Convert a Relay Function Module into an equivalent ONNX and serialize it to the path

    Parameters
    ----------
    relay_ir : tvm.ir.IRModule or tvm.relay.Function
        The relay module object

    params : dict
        dict of the parameter names and NDarray values

    name : str
        name of the output ONNX graph

    opset_version : int
        target onnx opset version

    path : str
        The path where ONNX model will be saved

    Returns
    -------
    onnx_model : onnx.ModelProto
        converted ONNX model as a ModelProto.

    """

    if opset_version not in ONNX_OPSET_VERSONS_SUPPORTED:
        raise NotImplementedError("Currently only opset version 11 is supported.")

    if opset_version > defs.onnx_opset_version():
        raise Exception(
            f"The ONNX package installed of version {get_onnx_version()} does not support the "
            f"opset version {opset_version}. Upgrade the ONNX package to latest version."
        )

    func = relay_ir["main"] if isinstance(relay_ir, tvm.ir.IRModule) else relay_ir
    converter = RelayToONNXConverter(name, params, opset_version)
    onnx_model = converter.convert_to_onnx(func)

    if path:
        onnx.save(onnx_model, path)
    return onnx_model


@tvm._ffi.register_func("relay.ext.onnx")
def onnx_compiler(func):
    """Create a runtime module for ONNX from Relay Function

    :param func: Relay function
    :return: runtime module for ONNX
    """

    assert isinstance(func, tvm.relay.function.Function)
    name = str(func.attrs.global_symbol)
    model = to_onnx(func, {}, name)
    const_vars = [const.name for const in model.graph.initializer]
    name_bytes = bytes(name, "utf-8")
    name_size = struct.pack("I", len(name_bytes))
    model_serialized = model.SerializeToString()
    model_size = struct.pack("I", model.ByteSize())
    data = b"" + name_size + name_bytes + model_size + model_serialized

    runtime_func = "runtime.ONNXModuleCreate"
    fcreate = tvm._ffi.get_global_func(runtime_func)
    return fcreate(data.hex(), name, const_vars)


@tvm._ffi.register_func("relay.ext.onnx.save_to_file")
def save_to_file(hex_str, path=None, fmt="onnx"):
    """Store the ONNX subgraphs in the path folder

    :param hex_str: Subgrah names and corresponding serialized onnx hex string
    :param path: path to which ONNX files to be stored
                It is assumed that path exists
    :param fmt: extension of the files to be stored
    """
    onnx_ir = bytes.fromhex(hex_str)

    offset = 0
    while offset < len(onnx_ir):
        stop = offset + 4
        (name_size,) = struct.unpack("I", onnx_ir[offset:stop])
        name = onnx_ir[stop : stop + name_size].decode("utf-8")
        stop = stop + name_size
        (model_size,) = struct.unpack("I", onnx_ir[stop : stop + 4])
        stop = stop + 4
        model_serialized = onnx_ir[stop : stop + model_size]
        offset = stop + model_size

        model_onnx = onnx.load_model_from_string(model_serialized)
        onnx.save(model_onnx, f"{path}{os.path.sep}{name}.{fmt}")
