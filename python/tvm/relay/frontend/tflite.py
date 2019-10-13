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
"""Tensorflow lite frontend."""
from __future__ import absolute_import as _abs
import math
import numpy as np
import tvm
from .. import analysis
from .. import expr as _expr
from .. import module as _module
from .. import op as _op
from ... import nd as _nd
from .common import ExprTable
from .common import infer_shape as _infer_shape

__all__ = ['from_tflite']

class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""
    def __init__(self, tensor_idx, tensor, buffer):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer

class OperatorConverter(object):
    """Operator Converted for converting TFLite ops to Relay ops"""
    def __init__(self, model, subgraph, exp_tab):

        try:
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.model = model
        self.subgraph = subgraph
        self.exp_tab = exp_tab
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())

        # Add more operators
        self.convert_map = {
            'CONV_2D': self.convert_conv2d,
            'DEPTHWISE_CONV_2D': self.convert_depthwise_conv2d,
            'AVERAGE_POOL_2D': self.convert_average_pool2d,
            'RESHAPE': self.convert_reshape,
            'RESIZE_BILINEAR': self.convert_resize_bilinear,
            'RESIZE_NEAREST_NEIGHBOR': self.convert_resize_nearest_neighbor,
            'SOFTMAX': self.convert_softmax,
            'SQUEEZE': self.convert_squeeze,
            'MAX_POOL_2D': self.convert_max_pool2d,
            'CONCATENATION': self.convert_concatenation,
            'ADD': self.convert_add,
            'SUB': self.convert_sub,
            'MUL': self.convert_mul,
            'DIV': self.convert_div,
            'POW': self.convert_pow,
            'MAXIMUM': self.convert_maximum,
            'MINIMUM': self.convert_minimum,
            'GREATER': self.convert_greater,
            'ZEROS_LIKE': self.convert_zeros_like,
            'REDUCE_MIN': self._convert_reduce_min,
            'REDUCE_MAX': self._convert_reduce_max,
            'MEAN': self._convert_reduce_mean,
            'REDUCE_PROD': self._convert_reduce_prod,
            'FULLY_CONNECTED': self.convert_fully_connected,
            'PAD': self.convert_pad,
            'PACK': self.convert_pack,
            'LOGISTIC': self.convert_logistic,
            'TANH':self.convert_tanh,
            'RELU':self.convert_relu,
            'SPLIT': self.convert_split,
            'TRANSPOSE': self.convert_transpose,
            'CAST': self.convert_cast,
            'TILE': self.convert_tile,
            'BATCH_TO_SPACE_ND': self.convert_batch_to_space_nd,
            'SPACE_TO_BATCH_ND': self.convert_space_to_batch_nd
        }

    def check_unsupported_ops(self):
        """Check unsupported TFLite ops in our converter."""
        unsupported_ops_set = set()

        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            if op_code_str not in self.convert_map:
                unsupported_ops_set.add(op_code_str)

        if unsupported_ops_set:
            msg = 'The following operators are not supported in frontend ' \
                  'TFLite: {}'
            ops = str(list(unsupported_ops_set)).strip('[,]')
            raise tvm.error.OpNotImplemented(msg.format(ops))

    def convert_op_to_relay(self):
        """Convert TFLite ops to relay ops"""
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            output_tensors = self.get_output_tensors(op)

            ret = self.convert_map[op_code_str](op)

            if len(output_tensors) == 1:
                tensor_idx = output_tensors[0].tensor_idx
                self.exp_tab.set_expr(get_tensor_name(self.subgraph, tensor_idx), ret)
            else:
                for idx, output_tensor in enumerate(output_tensors):
                    self.exp_tab.set_expr(get_tensor_name(self.subgraph, output_tensor.tensor_idx),
                                          ret[idx])

    def get_op_code_str(self, op):
        """Get TFLite ops string representation"""
        try:
            from tflite.BuiltinOperator import BuiltinOperator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        op_code_list_idx = op.OpcodeIndex()
        op_code_id = self.model.OperatorCodes(op_code_list_idx).BuiltinCode()
        op_code_str = self.builtin_op_code[op_code_id]
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            raise NotImplementedError("Custom operators are currently not supported")
        return op_code_str

    def get_input_tensors(self, op):
        operator_inputs = op.InputsAsNumpy()
        return self.get_tensors(operator_inputs)

    def get_output_tensors(self, op):
        operator_outputs = op.OutputsAsNumpy()
        return self.get_tensors(operator_outputs)

    def get_tensors(self, tensors_idx_list):
        """Get tensor wrapper list from given TFLite tensor index list"""
        return_list = list()
        for tensor_idx in tensors_idx_list:
            if tensor_idx < 0:
                return_list.append(TensorWrapper(tensor_idx, 0, 0))
                continue

            tensor = self.subgraph.Tensors(tensor_idx)
            buffer_idx = tensor.Buffer()
            buffer = self.model.Buffers(buffer_idx)
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer))
        return return_list

    def get_tensor_value(self, tensor_wrapper):
        """Get tensor buffer value from given tensor wrapper"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_wrapper.tensor.Type() == TensorType.UINT8:
            return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.uint8).reshape(
                tensor_wrapper.tensor.ShapeAsNumpy())
        if tensor_wrapper.tensor.Type() == TensorType.FLOAT32:
            return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.float32).reshape(
                tensor_wrapper.tensor.ShapeAsNumpy())
        if tensor_wrapper.tensor.Type() == TensorType.INT32:
            return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.int32).reshape(
                tensor_wrapper.tensor.ShapeAsNumpy())
        if tensor_wrapper.tensor.Type() == TensorType.INT64:
            return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.int64).reshape(
                tensor_wrapper.tensor.ShapeAsNumpy())
        raise NotImplementedError("Tensor type {} is currently not supported"
                                  .format(str(tensor_wrapper.tensor.Type())))

    def get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        raise NotImplementedError("Tensor type {} is currently not supported"
                                  .format(str(tensor_type)))

    def convert_conv2d(self, op):
        """Convert TFLite conv2d"""
        return self.convert_conv(op, "conv2d")

    def convert_depthwise_conv2d(self, op):
        """Convert TFLite depthwise conv2d"""
        return self.convert_conv(op, "depthwise")

    def convert_average_pool2d(self, op):
        """Convert TFLite average pool2d"""
        return self.convert_pool2d(op, "average")

    def convert_max_pool2d(self, op):
        """Convert TFLite max pool2d"""
        return self.convert_pool2d(op, "max")

    def convert_reshape(self, op):
        """Convert TFLite reshape"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Operator import Operator
            from tflite.ReshapeOptions import ReshapeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
        op_options = op.BuiltinOptions()
        reshape_options = ReshapeOptions()
        reshape_options.Init(op_options.Bytes, op_options.Pos)
        target_shape = reshape_options.NewShapeAsNumpy()

        in_expr = self.get_expr(input_tensor_idx)
        out = _op.reshape(in_expr, newshape=tuple(target_shape))

        return out

    def _convert_resize(self, method, op):
        """Generic method to Convert TFLite RESIZE operators"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Operator import Operator
            from tflite.ResizeBilinearOptions import ResizeBilinearOptions
            # ResizeNearestNeighborOptions was added in tflite v1.13
            tflite_ver = 1120
            if 'ResizeNearestNeighborOptions' in dir(BuiltinOptions):
                from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
                tflite_ver = 1130
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # images, 4-D Tensor with shape NHWC.
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # size - 1-D int32 Tensor of 2 elements: new_height, new_width
        target_size = tuple(self.get_tensor_value(input_tensors[1]))

        # Options - align_corners (bool)
        resize_options = None
        align_corners = False
        if method == "bilinear":
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeBilinearOptions
            resize_options = ResizeBilinearOptions()
        elif tflite_ver >= 1130:
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeNearestNeighborOptions
            resize_options = ResizeNearestNeighborOptions()

        if resize_options is not None:
            op_options = op.BuiltinOptions()
            resize_options.Init(op_options.Bytes, op_options.Pos)
            align_corners = resize_options.AlignCorners()

        # Use layout NHWC
        out = _op.image.resize(in_expr, target_size, "NHWC", method, align_corners)
        return out

    def convert_resize_bilinear(self, op):
        """Convert TFLite RESIZE_BILINEAR"""
        return self._convert_resize("bilinear", op)

    def convert_resize_nearest_neighbor(self, op):
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR"""
        return self._convert_resize("nearest_neighbor", op)

    def convert_logistic(self, op):
        """Convert TFLite LOGISTIC"""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        out = _op.sigmoid(in_expr)
        return out

    def convert_softmax(self, op):
        """Convert TFLite softmax"""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        params = {'axis': 1}  # 1 is channel
        in_expr = self.get_expr(input_tensor_idx)
        out = _op.nn.softmax(in_expr, **params)

        return out

    def convert_tanh(self, op):
        """Convert TFLite TANH"""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = _op.tanh(in_expr)

        return out

    def convert_relu(self, op):
        """Convert TFLite ReLU"""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = _op.nn.relu(in_expr)

        return out

    def convert_concatenation(self, op):
        """Convert TFLite concatenation"""
        try:
            from tflite.Operator import Operator
            from tflite.ConcatenationOptions import ConcatenationOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        in_exprs = [self.get_expr(input_tensor.tensor_idx) for input_tensor in input_tensors]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.ConcatenationOptions
        op_options = op.BuiltinOptions()
        concatenation_options = ConcatenationOptions()
        concatenation_options.Init(op_options.Bytes, op_options.Pos)
        concatenation_axis = concatenation_options.Axis()
        fused_activation_fn = concatenation_options.FusedActivationFunction()

        # with axis in N H W C
        out = _op.concatenate(in_exprs, axis=concatenation_axis)

        # if we have activation fn
        if fused_activation_fn != ActivationFunctionType.NONE:
            out = self.convert_fused_activation_function(out, fused_activation_fn)
        return out

    def _convert_elemwise(self, relay_op, op):
        """Generic method to Convert TFLite elemwise"""
        try:
            from tflite.Operator import Operator
            from tflite.AddOptions import AddOptions
            from tflite.SubOptions import SubOptions
            from tflite.MulOptions import MulOptions
            from tflite.DivOptions import DivOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        lhs_expr = self.get_expr(lhs_tensor.tensor_idx)

        rhs_tensor = input_tensors[1]
        if self.has_expr(rhs_tensor.tensor_idx):
            # In most cases, we can assume that TOCO fuses elemwise operators
            # with constants - it means both will be tensors.
            rhs_expr = self.get_expr(rhs_tensor.tensor_idx)
        else:
            # However, in some corner cases, the elemwise operator is not fused,
            # we can receive as constant.
            rhs_type_str = self.get_tensor_type_str(rhs_tensor.tensor.Type())
            rhs_expr = self.exp_tab.new_const(self.get_tensor_value(rhs_tensor),
                                              dtype=rhs_type_str)
        out = relay_op(lhs_expr, rhs_expr)

        # Options (fused_activation_function)
        options = None
        if op.BuiltinOptionsType() == BuiltinOptions.AddOptions:
            options = AddOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.SubOptions:
            options = SubOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.MulOptions:
            options = MulOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.DivOptions:
            options = DivOptions()

        if options is not None:
            op_options = op.BuiltinOptions()
            options.Init(op_options.Bytes, op_options.Pos)
            fused_activation_fn = options.FusedActivationFunction()
            # if we have activation fn
            if fused_activation_fn != ActivationFunctionType.NONE:
                out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_add(self, op):
        """Convert TFLite ADD"""
        return self._convert_elemwise(_op.add, op)

    def convert_sub(self, op):
        """Convert TFLite SUB"""
        return self._convert_elemwise(_op.subtract, op)

    def convert_mul(self, op):
        """Convert TFLite MUL"""
        return self._convert_elemwise(_op.multiply, op)

    def convert_div(self, op):
        """Convert TFLite DIV"""
        return self._convert_elemwise(_op.divide, op)

    def convert_pow(self, op):
        return self._convert_elemwise(_op.power, op)

    def convert_maximum(self, op):
        return self._convert_elemwise(_op.maximum, op)

    def convert_minimum(self, op):
        return self._convert_elemwise(_op.minimum, op)

    def convert_greater(self, op):
        return self._convert_elemwise(_op.greater, op)

    def convert_zeros_like(self, op):
        """Convert TFLite ZEROS LIKE"""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = _op.zeros_like(in_expr)

        return out

    def _convert_reduce(self, relay_op, op):
        """Generic method to Convert TFLite MEAN operators"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Operator import Operator
            from tflite.ReducerOptions import ReducerOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # input_tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # axis
        axis = tuple(self.get_tensor_value(input_tensors[1]))

        # Options - keep_dims (bool)
        assert op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions
        reduce_options = ReducerOptions()
        op_options = op.BuiltinOptions()
        reduce_options.Init(op_options.Bytes, op_options.Pos)
        keep_dims = reduce_options.KeepDims()

        out = relay_op(in_expr, axis, keep_dims)
        return out

    def _convert_reduce_min(self, op):
        return self._convert_reduce(_op.reduce.min, op)

    def _convert_reduce_max(self, op):
        return self._convert_reduce(_op.reduce.max, op)

    def _convert_reduce_mean(self, op):
        return self._convert_reduce(_op.reduce.mean, op)

    def _convert_reduce_prod(self, op):
        return self._convert_reduce(_op.reduce.prod, op)

    def convert_fully_connected(self, op):
        """Convert TFLite fully connected"""
        try:
            from tflite.Operator import Operator
            from tflite.FullyConnectedOptions import FullyConnectedOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.TensorType import TensorType
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        input_tensor_shape = input_tensor.tensor.ShapeAsNumpy()
        weight_tensor_shape = weight_tensor.tensor.ShapeAsNumpy()

        # reshape input tensor from N H W C to N H*W*C
        input_size_per_batch = 1
        for s in range(1, len(input_tensor_shape)):
            input_size_per_batch *= input_tensor_shape[s]
        assert input_size_per_batch == weight_tensor_shape[1], \
            "input size and weight size are mismatched"
        target_shape = tuple((input_tensor_shape[0], input_size_per_batch))
        in_expr = self.get_expr(input_tensor_idx)
        in_expr = _op.reshape(in_expr, target_shape)

        assert op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions
        op_options = op.BuiltinOptions()
        fully_connected_options = FullyConnectedOptions()
        fully_connected_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = fully_connected_options.FusedActivationFunction()

        # weight tensor type should be UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        weight_value = self.get_tensor_value(weight_tensor)
        weight_expr = self.exp_tab.new_const(weight_value, dtype=weight_tensor_type_str)

        out = _op.nn.dense(in_expr, weight_expr)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (quantization) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            bias_expr = self.exp_tab.new_const(self.get_tensor_value(bias_tensor),
                                               dtype=bias_tensor_type_str)
            out = _op.nn.bias_add(out, bias_expr)

        # If we have fused activations
        if fused_activation_fn != ActivationFunctionType.NONE:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_squeeze(self, op):
        """Convert TFLite squeeze"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Operator import Operator
            from tflite.SqueezeOptions import SqueezeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        output_tensors = self.get_output_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        assert len(output_tensors) == 1, "output tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.SqueezeOptions
        op_options = op.BuiltinOptions()
        squeeze_options = SqueezeOptions()
        squeeze_options.Init(op_options.Bytes, op_options.Pos)
        squeeze_axis = squeeze_options.SqueezeDimsAsNumpy()

        in_expr = self.get_expr(input_tensor_idx)
        out = _op.squeeze(in_expr, axis=tuple(squeeze_axis))

        return out

    def convert_fused_activation_function(self, in_expr, fused_activation_fn):
        """Convert TFLite fused activation function"""
        try:
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")
        assert fused_activation_fn != ActivationFunctionType.NONE
        if fused_activation_fn == ActivationFunctionType.RELU6:
            return _op.clip(in_expr, a_min=0, a_max=6)
        if fused_activation_fn == ActivationFunctionType.RELU:
            return _op.nn.relu(in_expr)
        if fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            return _op.clip(in_expr, a_min=-1, a_max=1)
        if fused_activation_fn == ActivationFunctionType.TANH:
            return _op.tanh(in_expr)
        fused_activation_fn_str = self.activation_fn_type[fused_activation_fn]
        raise tvm.error.OpNotImplemented(
            'Operator {} is not supported for frontend TFLite.'.format(fused_activation_fn_str))

    def convert_conv(self, op, conv_type):
        """convolution implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
            from tflite.TensorType import TensorType
            from tflite.Operator import Operator
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        is_depthwise_conv = False
        if conv_type == 'conv2d':
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == 'depthwise':
            is_depthwise_conv = True
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported for frontend TFLite.'.format(conv_type))

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        _, input_h, input_w, input_c = input_tensor.tensor.ShapeAsNumpy()

        if is_depthwise_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            _, kernel_h, kernel_w, in_channels = weight_tensor.tensor.ShapeAsNumpy()
            assert in_channels == input_c * depth_multiplier
        else:
            output_channels, kernel_h, kernel_w, _ = weight_tensor.tensor.ShapeAsNumpy()

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {'kernel_size': [kernel_h, kernel_w],
                  'strides': [stride_h, stride_w],
                  'dilation': [dilation_h, dilation_w],
                  'padding': [0, 0],
                  'data_layout': 'NHWC'}

        if is_depthwise_conv:
            params['channels'] = int(in_channels)
            params['groups'] = int(in_channels)
            params['kernel_layout'] = 'HWOI'
        else:
            params['channels'] = int(output_channels)
            params['kernel_layout'] = 'HWIO'

        # weight tensor type should be UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        in_expr = self.get_expr(input_tensor_idx)
        weight_value = self.get_tensor_value(weight_tensor)

        # TFLite kernel layout:
        # convolution:
        # OC KH KW IC, we require KH KW IC OC (HWIO)
        # depthwise convolution:
        # 1 KH KW C(input_c * depth_multiplier), we require
        # KH KW IC M (depth_multiplier) (HWOI)
        if is_depthwise_conv:
            weight_value = weight_value.reshape(kernel_h, kernel_w, input_c, depth_multiplier)
        else:
            weight_value = weight_value.transpose((1, 2, 3, 0))

        weight_expr = self.exp_tab.new_const(weight_value, dtype=weight_tensor_type_str)

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, dilated_kernel_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, dilated_kernel_w, stride_w)
            do_pad = not (pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0)
            if do_pad:
                in_expr = _op.nn.pad(data=in_expr, pad_width=((0, 0),
                                                              (pad_top, pad_bottom),
                                                              (pad_left, pad_right),
                                                              (0, 0)))
        else:
            raise tvm.error.OpAttributeUnImplemented(
                'Padding format {} is not supported for operator Conv.'.format(padding))

        out = _op.nn.conv2d(data=in_expr, weight=weight_expr, **params)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (quantization) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            bias_expr = self.exp_tab.new_const(self.get_tensor_value(bias_tensor),
                                               dtype=bias_tensor_type_str)
            channel_axis = 3
            out = _op.nn.bias_add(out, bias_expr, axis=channel_axis)

        # If we have fused activations
        if fused_activation_fn != ActivationFunctionType.NONE:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_split(self, op):
        """split implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Operator import Operator
            from tflite.SplitOptions import SplitOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)

        assert len(input_tensors) == 2, "input tensors length should be == 2"

        axis_tensor = input_tensors[0]
        split_axis = self.get_tensor_value(axis_tensor)
        input_tensor = input_tensors[1]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.SplitOptions
        op_options = op.BuiltinOptions()
        split_options = SplitOptions()
        split_options.Init(op_options.Bytes, op_options.Pos)
        num_splits = split_options.NumSplits()

        in_expr = self.get_expr(input_tensor_idx)
        out = _op.split(in_expr, num_splits, axis=int(split_axis))
        # Relay does not like a TupleWrapper of 1 element, further this
        # only shows up with tf1.13 if we use a split with num_splits==1.
        # In tf 1.14 this doesn't appear as it is automatically a reshape
        # operation.
        if isinstance(out, _expr.TupleWrapper):
            if out.size == 1:
                out = out[0]

        return out

    def convert_transpose(self, op):
        """transpose implementation."""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        in_expr = self.get_expr(input_tensor_idx)

        # axis
        in_axis = tuple(self.get_tensor_value(input_tensors[1]))

        if not in_axis:
            out = _op.transpose(in_expr)
        else:
            out = _op.transpose(in_expr, in_axis)

        return out

    def convert_cast(self, op):
        """Convert TFLite CAST"""
        try:
            from tflite.Operator import Operator
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.CastOptions import CastOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.CastOptions
        op_options = op.BuiltinOptions()
        cast_options = CastOptions()
        cast_options.Init(op_options.Bytes, op_options.Pos)
        cast_dtype = cast_options.OutDataType()

        out = _op.cast(in_expr, self.get_tensor_type_str(cast_dtype))

        return out

    def convert_tile(self, op):
        """tile implementation."""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        in_expr = self.get_expr(input_tensor_idx)

        # reps (tuple of int) – The number of times repeating the tensor data.
        reps = tuple(self.get_tensor_value(input_tensors[1]))

        out = _op.tile(in_expr, reps)

        return out

    def convert_pool2d(self, op, pool_type):
        """pool2d implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
            from tflite.Operator import Operator
            from tflite.Pool2DOptions import Pool2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.Pool2DOptions
        op_options = op.BuiltinOptions()
        pool2d_options = Pool2DOptions()
        pool2d_options.Init(op_options.Bytes, op_options.Pos)
        stride_h = pool2d_options.StrideH()
        stride_w = pool2d_options.StrideW()
        padding = pool2d_options.Padding()
        filter_h = pool2d_options.FilterHeight()
        filter_w = pool2d_options.FilterWidth()
        fused_activation_fn = pool2d_options.FusedActivationFunction()

        params = {'pool_size': (filter_h, filter_w),
                  'strides': (stride_h, stride_w),
                  'padding': [0, 0],
                  'layout': 'NHWC'}

        in_expr = self.get_expr(input_tensor_idx)

        _, input_h, input_w, _ = input_tensor.tensor.ShapeAsNumpy()
        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, filter_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, filter_w, stride_w)
            params['padding'] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            raise tvm.error.OpAttributeUnImplemented(
                'Padding format {} for operator Pool2D is not supported.'.format(padding))

        if pool_type == "average":
            out = _op.nn.avg_pool2d(in_expr, **params)
        elif pool_type == "max":
            out = _op.nn.max_pool2d(in_expr, **params)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported for frontend TFLite.'.format(pool_type + ' pool'))

        # If we have fused activations
        if fused_activation_fn != ActivationFunctionType.NONE:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_pad(self, op):
        """Convert TFLite PAD"""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # TFLite only support CONSTANT mode and does not support constant_values parameter.
        # tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])
        # convert list of lists to tuple of tuples
        paddings = tuple(tuple(l) for l in pad_list)

        # Use default pad_value 0 because TFLite does not support constant_values parameter
        out = _op.nn.pad(in_expr, paddings)
        return out

    def convert_pack(self, op):
        """Convert TFLite pack"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Operator import Operator
            from tflite.PackOptions import PackOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        in_exprs = [self.get_expr(input_tensor.tensor_idx) for input_tensor in input_tensors]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.PackOptions
        op_options = op.BuiltinOptions()
        pack_options = PackOptions()
        pack_options.Init(op_options.Bytes, op_options.Pos)
        pack_axis = pack_options.Axis()

        in_exprs_reshaped = [_op.expand_dims(i, axis=pack_axis, num_newaxis=1) for i in in_exprs]
        out = _op.concatenate(in_exprs_reshaped, pack_axis)
        return out

    def convert_batch_to_space_nd(self, op):
        """batch_to_space_nd implementation."""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        input_shape = list(input_tensor.tensor.ShapeAsNumpy())
        batch = input_shape[0]

        block_shape = list(self.get_tensor_value(input_tensors[1]))
        M = len(block_shape)

        crops = list(self.get_tensor_value(input_tensors[2]))

        # From https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d:
        # Reshape input to reshaped of shape
        shape1 = block_shape + [batch // np.prod(block_shape)] + input_shape[1:]
        reshaped = _op.reshape(in_expr, newshape=shape1)

        # Permute dimensions of reshaped to produce permuted of shape
        axes = [M] + [axis for i in range(M) for axis in [M + i + 1, i]] + \
            list(range(2 * M + 1, len(shape1)))
        permuted = _op.transpose(reshaped, axes=axes)

        # Reshape permuted to produce reshaped_permuted of shape
        shape2 = [0] + [-3] * M + [-2]
        reshaped_permuted = _op.reshape(permuted, newshape=shape2)

        # Crop the start and end of dimensions [1, ..., M] of reshaped_permuted according to crops
        # to produce the output of shape:
        reshaped_permuted_shape = _infer_shape(reshaped_permuted)
        cropped = reshaped_permuted
        for axis in range(1, M + 1):
            crop = crops[axis - 1]
            if (crop != [0, 0]).all():
                indices = _op.arange(
                    _expr.const(crop[0]),
                    _expr.const(reshaped_permuted_shape[axis] - crop[1]),
                    dtype='int32'
                )
                cropped = _op.take(cropped, indices=indices, axis=axis)

        return cropped

    def convert_space_to_batch_nd(self, op):
        """space_to_batch_nd implementation."""
        try:
            from tflite.Operator import Operator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        assert isinstance(op, Operator)
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        input_shape = list(input_tensor.tensor.ShapeAsNumpy())
        batch = input_shape[0]
        N = len(input_shape)

        block_shape = list(self.get_tensor_value(input_tensors[1]))
        M = len(block_shape)

        paddings = list(self.get_tensor_value(input_tensors[2]))

        # From https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd:
        # Zero-pad the start and end of dimensions [1, ..., M] of the input according to paddings
        # to produce padded of shape padded_shape.
        remaining_shape_length = N - M - 1
        padded_list = [(0, 0)] + paddings + [(0, 0)] * remaining_shape_length

        padded_shape = []
        for element in padded_list:
            if isinstance(element, np.ndarray):
                element = element.tolist()

            padded_shape.append(element)

        padded_shape = tuple(padded_shape)
        padded = _op.nn.pad(in_expr, pad_width=tuple(padded_shape))

        # Reshape padded to reshaped_padded of shape:
        shape1 = [batch] + [item for i in range(M) for item in [-4, -1, block_shape[i]]] + [-2]
        reshaped_padded = _op.reshape(padded, newshape=shape1)

        # Permute dimensions of reshaped_padded to produce permuted_reshaped_padded of shape:
        axes = [2 * i + 2 for i in range(M)] + [0] + [2 * i + 1 for i in range(M)] + \
            list(range(1 + 2 * M, 1 + 2 * M + remaining_shape_length))
        permuted_reshaped_padded = _op.transpose(reshaped_padded, axes=axes)
        permuted_reshaped_padded_shape = _infer_shape(permuted_reshaped_padded)

        # Reshape permuted_reshaped_padded to flatten block_shape into the batch dimension,
        # producing an output tensor of shape:
        shape2 = [batch * np.prod(block_shape)] + list(permuted_reshaped_padded_shape)[M + 1:]
        reshaped_permuted_reshaped_padded = _op.reshape(permuted_reshaped_padded, newshape=shape2)

        return reshaped_permuted_reshaped_padded

    def get_expr(self, input_tensor_idx):
        return self.exp_tab.get_expr(get_tensor_name(self.subgraph, input_tensor_idx))

    def has_expr(self, input_tensor_idx):
        return self.exp_tab.has_expr(get_tensor_name(self.subgraph, input_tensor_idx))

def build_str_map(obj):
    """Build string map of TFLite enum int value

    Parameters
    ----------
    obj:
        TFLite class which contains enum int value, such as BuiltInOptions

    Returns
    -------
        String representation map of TFLite class enum int value
    """
    ret = {}
    for field_name in dir(obj):
        if not field_name.startswith('_'):
            field_value = getattr(obj, field_name)
            if isinstance(field_value, int):
                ret[field_value] = field_name
    return ret

# SAME padding: https://www.tensorflow.org/api_guides/python/nn
def get_pad_value(data, kernel, stride):
    """Get the pad tuple of value for SAME padding

    Parameters
    ----------
    data:
        1D input data

    kernel:
        1D input kernel

    stride:
        1D input stride

    Returns
    -------
        pad tuple of value
    """

    out = int(math.ceil(float(data) / float(stride)))
    pad = max(0, (out - 1) * stride + kernel - data)
    pad_before = pad // 2
    pad_after = pad - pad_before
    return pad_before, pad_after


def get_tensor_name(subgraph, tensor_idx):
    """Get the tensor name.

    Parameters
    ----------
    subgraph:
        tflite.Subgraph.Subgraph

    tensor:
        tensor index in subgraph

    Returns
    -------
        tensor name in UTF-8 encoding
    """
    return subgraph.Tensors(tensor_idx).Name().decode("utf-8")


def from_tflite(model, shape_dict, dtype_dict):
    """Convert from tflite model into compatible relay Function.

    Parameters
    ----------
    model:
        tflite.Model.Model

    shape_dict : dict of str to int list/tuple
        Input shapes of the model.

    dtype_dict : dict of str to str
        Input types of the model.

    Returns
    -------
    mod : tvm.relay.Module
        The relay module for compilation.

    params : dict of str to tvm.NDArray
        The parameter dict to be used by relay
    """
    try:
        import tflite.Model
        import tflite.SubGraph
        import tflite.BuiltinOperator
    except ImportError:
        raise ImportError("The tflite package must be installed")
    assert isinstance(model, tflite.Model.Model)

    # keep the same as tflite
    assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"
    subgraph = model.Subgraphs(0)

    # model inputs / outputs
    model_inputs = subgraph.InputsAsNumpy()
    model_outputs = subgraph.OutputsAsNumpy()

    exp_tab = ExprTable()
    for model_input in model_inputs:
        model_input_name = get_tensor_name(subgraph, model_input)
        shape = shape_dict[model_input_name] if model_input_name in shape_dict else None
        dtype = dtype_dict[model_input_name] if model_input_name in dtype_dict else "float32"
        exp_tab.set_expr(model_input_name, _expr.var(model_input_name, shape=shape, dtype=dtype))

    # op code in model
    op_converter = OperatorConverter(model, subgraph, exp_tab)
    op_converter.check_unsupported_ops()
    op_converter.convert_op_to_relay()

    # params and outputs
    params = {k:_nd.array(np.array(v)) for k, v in exp_tab.params.items()}
    outputs = [exp_tab.get_expr(get_tensor_name(subgraph, i)) for i in model_outputs]
    outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    func = _expr.Function(analysis.free_vars(outputs), outputs)
    return _module.Module.from_expr(func), params
