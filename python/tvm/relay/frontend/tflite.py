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
# pylint: disable=invalid-name, unused-argument, too-many-lines, import-outside-toplevel
"""Tensorflow lite frontend."""
import math
import itertools
import numpy as np
import tvm
from tvm.ir import IRModule

from tvm import relay
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import qnn as _qnn
from ... import nd as _nd
from .common import ExprTable
from .common import infer_shape as _infer_shape

__all__ = ['from_tflite']

class TensorWrapper(object):
    """Tensor wrapper for TFLite Tensor"""
    def __init__(self, tensor_idx, tensor, buffer, qnn_params=None):
        self.tensor_idx = tensor_idx
        self.tensor = tensor
        self.buffer = buffer
        self.qnn_params = qnn_params

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
            'ABS': self.convert_abs,
            'ADD': self.convert_add,
            'AVERAGE_POOL_2D': self.convert_average_pool2d,
            'BATCH_TO_SPACE_ND': self.convert_batch_to_space_nd,
            'CAST': self.convert_cast,
            'CEIL': self.convert_ceil,
            'CONCATENATION': self.convert_concatenation,
            'CONV_2D': self.convert_conv2d,
            'COS': self.convert_cos,
            'DEPTH_TO_SPACE': self.convert_depth_to_space,
            'DEPTHWISE_CONV_2D': self.convert_depthwise_conv2d,
            'DETECTION_POSTPROCESS': self.convert_detection_postprocess,
            'DIV': self.convert_div,
            'ELU': self.convert_elu,
            'EQUAL': self.convert_equal,
            'EXP': self.convert_exp,
            'FILL': self.convert_fill,
            'FLOOR_DIV': self.convert_floor_div,
            'FLOOR_MOD': self.convert_floor_mod,
            'FLOOR': self.convert_floor,
            'FULLY_CONNECTED': self.convert_fully_connected,
            'GATHER': self.convert_gather,
            'GREATER_EQUAL': self.convert_greater_equal,
            'GREATER': self.convert_greater,
            'HARD_SWISH': self.convert_hard_swish,
            'L2_NORMALIZATION': self.convert_l2_normalization,
            'L2_POOL_2D': self.convert_l2_pool2d,
            'LESS_EQUAL': self.convert_less_equal,
            'LESS': self.convert_less,
            'LOCAL_RESPONSE_NORMALIZATION': self.convert_lrn,
            'LOG': self.convert_log,
            'LOGICAL_AND': self.convert_logical_and,
            'LOGICAL_NOT': self.convert_logical_not,
            'LOGICAL_OR': self.convert_logical_or,
            'LOGISTIC': self.convert_logistic,
            'MAX_POOL_2D': self.convert_max_pool2d,
            'MAXIMUM': self.convert_maximum,
            'MEAN': self.convert_reduce_mean,
            'MINIMUM': self.convert_minimum,
            'MIRROR_PAD': self.convert_mirror_pad,
            'MUL': self.convert_mul,
            'NEG': self.convert_neg,
            'NOT_EQUAL': self.convert_not_equal,
            'PACK': self.convert_pack,
            'PAD': self.convert_pad,
            'POW': self.convert_pow,
            'PRELU': self.convert_prelu,
            'REDUCE_ANY': self.convert_reduce_any,
            'REDUCE_MAX': self.convert_reduce_max,
            'REDUCE_MIN': self.convert_reduce_min,
            'REDUCE_PROD': self.convert_reduce_prod,
            'RELU':self.convert_relu,
            'RESHAPE': self.convert_reshape,
            'RESIZE_BILINEAR': self.convert_resize_bilinear,
            'RESIZE_NEAREST_NEIGHBOR': self.convert_resize_nearest_neighbor,
            'ROUND': self.convert_round,
            'RSQRT': self.convert_rsqrt,
            'SIN': self.convert_sin,
            'SLICE': self.convert_slice,
            'SOFTMAX': self.convert_softmax,
            'SPACE_TO_BATCH_ND': self.convert_space_to_batch_nd,
            'SPACE_TO_DEPTH': self.convert_space_to_depth,
            'SPLIT': self.convert_split,
            'SPLIT_V': self.convert_split_v,
            'SQRT': self.convert_sqrt,
            'SQUARE': self.convert_square,
            'SQUARED_DIFFERENCE': self.convert_squared_difference,
            'SQUEEZE': self.convert_squeeze,
            'STRIDED_SLICE': self.convert_strided_slice,
            'SUB': self.convert_sub,
            'SUM': self.convert_reduce_sum,
            'TAN': self.convert_tan,
            'TANH':self.convert_tanh,
            'TILE': self.convert_tile,
            'TOPK_V2': self.convert_topk_v2,
            'TRANSPOSE_CONV': self.convert_transpose_conv,
            'TRANSPOSE': self.convert_transpose,
            'UNPACK': self.convert_unpack,
            'ZEROS_LIKE': self.convert_zeros_like,
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
            try:
                from tflite.Operator import Operator
            except ImportError:
                raise ImportError("The tflite package must be installed")

            assert isinstance(op, Operator)
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
        try:
            op_code_str = self.builtin_op_code[op_code_id]
        except KeyError:
            raise NotImplementedError('TFLite operator with code ' + str(op_code_id) + \
                                      ' is not supported by this version of the fbs schema.')
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            custom_op_code_str = self.model.OperatorCodes(op_code_list_idx).CustomCode()
            if custom_op_code_str == b'TFLite_Detection_PostProcess':
                return "DETECTION_POSTPROCESS"

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

            # Check if the tensors are quantized. Parse if yes.
            qnn_params = None
            tflite_qnn_params = tensor.Quantization()
            if tflite_qnn_params is not None:
                scale = float(tflite_qnn_params.ScaleAsNumpy())
                zero_point = int(tflite_qnn_params.ZeroPointAsNumpy())
                # Check that the scale and zero points are valid.
                if scale != 0 or zero_point != 0:
                    qnn_params = dict()
                    qnn_params['scale'] = relay.const(scale, 'float32')
                    qnn_params['zero_point'] = relay.const(zero_point, 'int32')
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
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
        if tensor_wrapper.tensor.Type() == TensorType.BOOL:
            return np.frombuffer(tensor_wrapper.buffer.DataAsNumpy(), dtype=np.bool_).reshape(
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
        if tensor_type == TensorType.BOOL:
            return "bool"
        raise NotImplementedError("Tensor type {} is currently not supported"
                                  .format(str(tensor_type)))

    def has_same_qnn_params(self, lhs_tensor, rhs_tensor):
        lhs_scale = lhs_tensor.qnn_params['scale']
        rhs_scale = rhs_tensor.qnn_params['scale']
        lhs_zero_point = lhs_tensor.qnn_params['zero_point']
        rhs_zero_point = rhs_tensor.qnn_params['zero_point']
        lhs_scale_value = get_scalar_from_constant(lhs_scale)
        rhs_scale_value = get_scalar_from_constant(rhs_scale)
        lhs_zero_point_value = get_scalar_from_constant(lhs_zero_point)
        rhs_zero_point_value = get_scalar_from_constant(rhs_zero_point)
        return lhs_scale_value == rhs_scale_value and \
                lhs_zero_point_value == rhs_zero_point_value

    def is_quantized(self, op):
        """Check if an input tensor is quantized."""
        input_tensors = self.get_input_tensors(op)
        first_tensor = input_tensors[0]
        return first_tensor.qnn_params is not None

    def quantize(self, expr, tensor_to_quantize):
        """ Helper function to quantize a tensor with Relay """
        tensor_type = tensor_to_quantize.tensor.Type()
        tensor_type_str = self.get_tensor_type_str(tensor_type)
        quantized = _qnn.op.quantize(data=expr,
                                     output_scale=tensor_to_quantize.qnn_params['scale'],
                                     output_zero_point=tensor_to_quantize.qnn_params['zero_point'],
                                     out_dtype=tensor_type_str)
        return quantized

    def dequantize(self, expr, tensor):
        """ Helper function to dequantize a tensor with Relay """
        dequantized = _qnn.op.dequantize(data=expr,
                                         input_scale=tensor.qnn_params['scale'],
                                         input_zero_point=tensor.qnn_params['zero_point'])
        return dequantized

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

    def convert_l2_pool2d(self, op):
        """Convert TFLite l2 pool2d"""
        return self.convert_pool2d(op, "l2")

    def convert_reshape(self, op):
        """Convert TFLite reshape"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ReshapeOptions import ReshapeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert input_tensors, "input tensors should not be empty"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
        op_options = op.BuiltinOptions()
        reshape_options = ReshapeOptions()
        reshape_options.Init(op_options.Bytes, op_options.Pos)
        target_shape = reshape_options.NewShapeAsNumpy()

        in_expr = self.get_expr(input_tensor_idx)

        # If the tensors are quantized, ensure that input/output qnn params are same.
        if input_tensor.qnn_params:
            output_tensors = self.get_output_tensors(op)
            assert len(output_tensors) == 1, "There should be only 1 output tensor"
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(input_tensor, output_tensor), \
                    "TFLite reshape requires input and output scale and zero points to be equal"
        out = _op.reshape(in_expr, newshape=tuple(target_shape))
        return out

    def _convert_resize(self, method, op):
        """Generic method to Convert TFLite RESIZE operators"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ResizeBilinearOptions import ResizeBilinearOptions
            # ResizeNearestNeighborOptions was added in tflite v1.13
            tflite_ver = 1120
            if 'ResizeNearestNeighborOptions' in dir(BuiltinOptions):
                from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions
                tflite_ver = 1130
        except ImportError:
            raise ImportError("The tflite package must be installed")

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
        coord_trans = "align_corners" if align_corners else "asymmetric"
        out = _op.image.resize(in_expr, target_size, "NHWC", method,
                               coordinate_transformation_mode=coord_trans)
        return out

    def convert_resize_bilinear(self, op):
        """Convert TFLite RESIZE_BILINEAR"""
        return self._convert_resize("bilinear", op)

    def convert_resize_nearest_neighbor(self, op):
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR"""
        return self._convert_resize("nearest_neighbor", op)

    def convert_l2_normalization(self, op):
        """Convert TFLite L2_NORMALIZATION """
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.L2NormOptions import L2NormOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.L2NormOptions
        op_options = op.BuiltinOptions()
        l2_norm_options = L2NormOptions()
        l2_norm_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = l2_norm_options.FusedActivationFunction()

        # TFLite supports normalization only over the last dim
        input_tensor_rank = len(input_tensor.tensor.ShapeAsNumpy())

        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFLite quantized L2_NORMALIZATION operator is not supported yet.')
        # TFL uses only the default epsilon value
        out = _op.nn.l2_normalize(in_expr, eps=1e-12, axis=[input_tensor_rank - 1])

        # if we have fused activation fn
        if fused_activation_fn != ActivationFunctionType.NONE:
            if not output_tensor.qnn_params:
                out = self.convert_fused_activation_function(out, fused_activation_fn)
            else:
                raise tvm.error.OpNotImplemented(
                    'TFLite quantized L2_NORMALIZATION operator\
                    with fused activation function is not supported yet.')

        return out

    def convert_lrn(self, op):
        """Convert TFLite LOCAL_RESPONSE_NORMALIZATION """
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.LocalResponseNormalizationOptions import LocalResponseNormalizationOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized LRN operator is not supported yet.')

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.LocalResponseNormalizationOptions
        op_options = op.BuiltinOptions()
        lrn_options = LocalResponseNormalizationOptions()
        lrn_options.Init(op_options.Bytes, op_options.Pos)
        radius = lrn_options.Radius()
        bias = lrn_options.Bias()
        alpha = lrn_options.Alpha()
        beta = lrn_options.Beta()
        size = (radius * 2) + 1
        alpha = alpha * size
        axis = 3 # NHWC format
        out = _op.nn.lrn(in_expr, size=size, axis=axis, bias=bias, alpha=alpha, beta=beta)

        return out

    def convert_logistic(self, op):
        """Convert TFLite LOGISTIC"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = _op.sigmoid(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_softmax(self, op):
        """Convert TFLite softmax"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        params = {'axis': 1}  # 1 is channel
        in_expr = self.get_expr(input_tensor_idx)

        # TODO - Naive softmax int8 implementation leads to bad accuracy. Currently, we can
        # dequantize to FP32 and perform softmax on FP32. We can investigate an integer only softmax
        # implementation in future.
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        out = _op.nn.softmax(in_expr, **params)

        # Go back to integer dataype if the original operator was quantized.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_tanh(self, op):
        """Convert TFLite TANH"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = _op.tanh(in_expr)

        return out

    def convert_relu(self, op):
        """Convert TFLite ReLU"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = _op.nn.relu(in_expr)

        return out

    def convert_hard_swish(self, op):
        """Convert TFLite Hard swish"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        def _relu6(data):
            return _op.tensor.clip(data, 0.0, 6.0)

        def _hard_swish(data):
            return data * _relu6(data + relay.const(3.0)) / relay.const(6.0)

        # Dequantize if the input is quantized.
        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)

        # Perform hardswish
        out = _hard_swish(in_expr)

        # Go back to integer dataype if the original operator was quantized.
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_concatenation(self, op):
        """Convert TFLite concatenation"""
        try:
            from tflite.ConcatenationOptions import ConcatenationOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        in_exprs = [self.get_expr(input_tensor.tensor_idx) for input_tensor in input_tensors]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        assert op.BuiltinOptionsType() == BuiltinOptions.ConcatenationOptions
        op_options = op.BuiltinOptions()
        concatenation_options = ConcatenationOptions()
        concatenation_options.Init(op_options.Bytes, op_options.Pos)
        concatenation_axis = concatenation_options.Axis()
        fused_activation_fn = concatenation_options.FusedActivationFunction()

        if not input_tensors[0].qnn_params:
            out = _op.concatenate(in_exprs, axis=concatenation_axis)
        else:
            input_scales = [input_tensor.qnn_params['scale'] for input_tensor in input_tensors]
            input_zero_points = \
                    [input_tensor.qnn_params['zero_point'] for input_tensor in input_tensors]
            out = _qnn.op.concatenate(in_exprs,
                                      input_scales=input_scales,
                                      input_zero_points=input_zero_points,
                                      output_scale=output_tensor.qnn_params['scale'],
                                      output_zero_point=output_tensor.qnn_params['zero_point'],
                                      axis=concatenation_axis)

        # if we have activation fn
        if fused_activation_fn != ActivationFunctionType.NONE:
            if not output_tensor.qnn_params:
                out = self.convert_fused_activation_function(out, fused_activation_fn)
            else:
                raise tvm.error.OpNotImplemented(
                    'Operator {} with fused activation is not supported yet.'
                    .format('qnn.op.concatenate'))
        return out

    def _convert_unary_elemwise(self, relay_op, op):
        """Generic method to convert TFLite unary elemwise functions"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = relay_op(in_expr)

        return out

    def convert_abs(self, op):
        """Convert TFLite ABS"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized ABS operator is not supported yet.')
        return self._convert_unary_elemwise(_op.abs, op)

    def convert_ceil(self, op):
        """Convert TFLite CEIL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized CEIL operator is not supported yet.')
        return self._convert_unary_elemwise(_op.ceil, op)

    def convert_floor(self, op):
        """Convert TFLite FLOOR"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized FLOOR operator is not supported yet.')
        return self._convert_unary_elemwise(_op.floor, op)

    def convert_round(self, op):
        """Convert TFLite ROUND"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized ROUND operator is not supported yet.')
        return self._convert_unary_elemwise(_op.round, op)

    def convert_exp(self, op):
        """Convert TFLite EXP"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized EXP operator is not supported yet.')
        return self._convert_unary_elemwise(_op.exp, op)

    def convert_log(self, op):
        """Convert TFLite LOG"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized LOG operator is not supported yet.')
        return self._convert_unary_elemwise(_op.log, op)

    def convert_sin(self, op):
        """Convert TFLite SIN"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized SIN operator is not supported yet.')
        return self._convert_unary_elemwise(_op.sin, op)

    def convert_tan(self, op):
        """Convert TFLite TAN"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized TAN operator is not supported yet.')
        return self._convert_unary_elemwise(_op.tan, op)

    def convert_cos(self, op):
        """Convert TFLite COS"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized COS operator is not supported yet.')
        return self._convert_unary_elemwise(_op.cos, op)

    def convert_sqrt(self, op):
        """Convert TFLite SQRT"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized SQRT operator is not supported yet.')
        return self._convert_unary_elemwise(_op.sqrt, op)

    def convert_rsqrt(self, op):
        """Convert TFLite RSQRT"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized RSQRT operator is not supported yet.')
        return self._convert_unary_elemwise(_op.rsqrt, op)

    def convert_neg(self, op):
        """Convert TFLite NEG"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized NEG operator is not supported yet.')
        return self._convert_unary_elemwise(_op.negative, op)

    def convert_elu(self, op):
        """Convert TFLite ELU"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized ELU operator is not supported yet.')
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        exp_type = self.get_tensor_type_str(input_tensor.tensor.Type())
        out = relay.const(-1.0, exp_type) * \
              _op.nn.relu(relay.const(1., exp_type) - _op.exp(in_expr)) + \
              _op.nn.relu(in_expr)

        return out

    def convert_square(self, op):
        """Convert TFLite SQUARE"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized SQUARE operator is not supported yet.')

        exp_type = self.get_tensor_type_str(output_tensor.tensor.Type())
        out = _op.power(in_expr, relay.const(2, exp_type))

        return out

    def _convert_elemwise(self, relay_op, op):
        """Generic method to Convert TFLite elemwise"""
        try:
            from tflite.AddOptions import AddOptions
            from tflite.SubOptions import SubOptions
            from tflite.MulOptions import MulOptions
            from tflite.DivOptions import DivOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        if self.has_expr(lhs_tensor.tensor_idx):
            # In most cases, we can assume that TOCO fuses elemwise operators
            # with constants - it means both will be tensors.
            lhs_expr = self.get_expr(lhs_tensor.tensor_idx)
        else:
            # However, in some corner cases, the elemwise operator is not fused,
            # we can receive as constant.
            lhs_type_str = self.get_tensor_type_str(lhs_tensor.tensor.Type())
            lhs_expr = self.exp_tab.new_const(self.get_tensor_value(lhs_tensor),
                                              dtype=lhs_type_str)

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

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # If quantized, extracts qnn params and call QNN add operator.
        if lhs_tensor.qnn_params:
            assert rhs_tensor.qnn_params, "Both tensors should be quantized."
            assert output_tensor.qnn_params, "Output tensor should be quantized."
            out = relay_op(lhs=lhs_expr,
                           rhs=rhs_expr,
                           lhs_scale=lhs_tensor.qnn_params['scale'],
                           lhs_zero_point=lhs_tensor.qnn_params['zero_point'],
                           rhs_scale=rhs_tensor.qnn_params['scale'],
                           rhs_zero_point=rhs_tensor.qnn_params['zero_point'],
                           output_scale=output_tensor.qnn_params['scale'],
                           output_zero_point=output_tensor.qnn_params['zero_point'])
        else:
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
                if output_tensor.qnn_params:
                    raise tvm.error.OpNotImplemented(
                        'Elemwise operators with fused activation are not supported yet.')
                out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_add(self, op):
        """Convert TFLite ADD"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            return self._convert_elemwise(_qnn.op.add, op)
        return self._convert_elemwise(_op.add, op)

    def convert_sub(self, op):
        """Convert TFLite SUB"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            return self._convert_elemwise(_qnn.op.subtract, op)
        return self._convert_elemwise(_op.subtract, op)

    def convert_mul(self, op):
        """Convert TFLite MUL"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            return self._convert_elemwise(_qnn.op.mul, op)
        return self._convert_elemwise(_op.multiply, op)

    def convert_div(self, op):
        """Convert TFLite DIV"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized DIV operator is not supported yet.')
        return self._convert_elemwise(_op.divide, op)

    def convert_pow(self, op):
        """Convert TFLite POW"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized POW operator is not supported yet.')
        return self._convert_elemwise(_op.power, op)

    def convert_maximum(self, op):
        """Convert TFLite MAXIMUM"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized MAXIMUM operator is not supported yet.')
        return self._convert_elemwise(_op.maximum, op)

    def convert_minimum(self, op):
        """Convert TFLite MINIMUM"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized MINIMUM operator is not supported yet.')
        return self._convert_elemwise(_op.minimum, op)

    def convert_greater(self, op):
        """Convert TFLite GREATER"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized GREATER operator is not supported yet.')
        return self._convert_elemwise(_op.greater, op)

    def convert_squared_difference(self, op):
        """Convert TFLite SQUARED DIFFERENCE"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized squared difference operator is not supported yet.')
        difference = self._convert_elemwise(_op.subtract, op)
        # _convert_elemwise has guaranteed only have one output tensor
        exp_type = self.get_tensor_type_str(self.get_output_tensors(op)[0].tensor.Type())
        out = _op.power(difference, relay.const(2, exp_type))
        return out

    def convert_greater_equal(self, op):
        """Convert TFLite GREATER_EQUAL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized GREATER_EQUAL operator is not supported yet.')
        return self._convert_elemwise(_op.greater_equal, op)

    def convert_less(self, op):
        """Convert TFLite LESS"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized LESS operator is not supported yet.')
        return self._convert_elemwise(_op.less, op)

    def convert_less_equal(self, op):
        """Convert TFLite LESS_EQUAL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized LESS_EQUAL operator is not supported yet.')
        return self._convert_elemwise(_op.less_equal, op)

    def convert_equal(self, op):
        """Convert TFLite EQUAL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized EQUAL operator is not supported yet.')
        return self._convert_elemwise(_op.equal, op)

    def convert_not_equal(self, op):
        """Convert TFLite NOT_EQUAL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized NOT_EQUAL operator is not supported yet.')
        return self._convert_elemwise(_op.not_equal, op)

    def _convert_logical_binary(self, relay_op, op):
        """Generic method to convert logical binary ops"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        lhs_expr = self.get_expr(lhs_tensor.tensor_idx)
        rhs_tensor = input_tensors[1]
        rhs_expr = self.get_expr(rhs_tensor.tensor_idx)
        out = relay_op(lhs_expr, rhs_expr)

        return out

    def convert_logical_and(self, op):
        """Convert tflite LOGICAL_AND"""
        return self._convert_logical_binary(_op.logical_and, op)

    def convert_logical_or(self, op):
        """Convert tflite LOGICAL_OR"""
        return self._convert_logical_binary(_op.logical_or, op)

    def convert_logical_not(self, op):
        """Convert tflite LOGICAL_NOT"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        data = self.get_expr(input_tensors[0].tensor_idx)
        out = _op.logical_not(data)

        return out

    def convert_gather(self, op):
        """Method to Convert TFLite GATHER operator"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.GatherOptions import GatherOptions
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        data = self.get_expr(input_tensors[0].tensor_idx)

        indices = input_tensors[1]
        indices_type = indices.tensor.Type()
        assert indices_type in (TensorType.INT32, TensorType.INT64)
        indices_type_str = self.get_tensor_type_str(indices_type)
        indices = self.exp_tab.new_const(self.get_tensor_value(indices),
                                         dtype=indices_type_str)

        assert op.BuiltinOptionsType() == BuiltinOptions.GatherOptions
        op_options = op.BuiltinOptions()
        gather_options = GatherOptions()
        gather_options.Init(op_options.Bytes, op_options.Pos)
        axis = gather_options.Axis()

        # Check the indices are with in bounds.
        data_shape = list(input_tensors[0].tensor.ShapeAsNumpy())
        data_dim = len(data_shape)

        axis_n = axis
        if axis_n < 0:
            axis_n += axis_n + data_dim
        assert axis_n >= 0, "Axis out of bounds"
        assert axis_n < data_dim, "Axis out of bounds"

        indices_val = self.get_tensor_value(input_tensors[1])
        indices_shape = list(indices_val.shape)
        indices_len = len(indices_shape)

        out_shape = []
        for i in range(data_dim):
            if axis_n == i:
                for j in range(indices_len):
                    out_shape.append(indices_shape[j])
            else:
                out_shape.append(data_shape[i])

        loopover = [range(s) for s in out_shape]
        for idx in list(itertools.product(*loopover)):
            indices_position = [idx[j] for j in range(axis_n, axis_n+indices_len)]

            real_indices = [idx[j] for j in range(axis_n)]
            real_indices.append(indices_val[tuple(indices_position)])
            real_indices.extend([idx[j] for j in range(axis_n + indices_len, len(idx))])
            for r, d in zip(real_indices, data_shape):
                if r >= d:
                    raise ValueError("TFLite out of bound indices are not supported.")

        # Use mode 'fast' since indices are already checked within bounds.
        out = _op.take(data, indices, axis=axis, mode="fast")
        return out

    def convert_strided_slice(self, op):
        """Method to Convert TFLite STRIDED_SLICE operator.
           NOTE: Eventhough tensorflow supports begin_mask, end_mask, ellipsis_mask, new_axis_mask
           and shrink_axis_mask, tflite doesn't support these and expect these values to be zero.
           But in future, they may open up the mask implementation, so kept the implementation
           same as tensorflow.

           This op extracts a slice of size (end - begin) / stride from the given input tensor.
           Starting at the location specified by begin the slice continues by adding stride to the
           index until all dimensions are not less than end. Note that a stride can be negative,
           which causes a reverse slice.

           For slice input[val0, val1, ..., valn], begin/end/strides will be vectors of length n.

           In each mask field(begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
           the ith bit will correspond to the ith val.

           If the ith bit of begin_mask is set, begin[i] is ignored and the fullest possible range
           in that dimension is used instead.

           If the ith bit of ellipsis_mask is set, as many unspecified dimensions as needed will be
           inserted between other dimensions. Only one non-zero bit is allowed in ellipsis_mask.

           If the ith bit of new_axis_mask is set, then begin, end, and stride are ignored and a
           new length 1 dimension is added at this point in the output tensor.

           If the ith bit of shrink_axis_mask is set, it implies that the ith specification shrinks
           the dimensionality by 1, taking on the value at index begin[i]. end[i] and strides[i]
           are ignored in this case.
           begin and end are zero-indexed. strides entries must be non-zero.

           TVM Relay implementation of doesn't support mask, so the mask values are processed in
           this function and begin/end/strides are updated accordingly. If any mask is present, and
           since tvm doesn't support mask computation directly, the output need a final reshape.
        """
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.StridedSliceOptions import StridedSliceOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 4, "input tensors length should be 4"

        data_expr = self.get_expr(input_tensors[0].tensor_idx)

        begin = list(self.get_tensor_value(input_tensors[1]))
        end = list(self.get_tensor_value(input_tensors[2]))
        stride = list(self.get_tensor_value(input_tensors[3]))

        assert op.BuiltinOptionsType() == BuiltinOptions.StridedSliceOptions
        op_options = op.BuiltinOptions()
        options = StridedSliceOptions()
        options.Init(op_options.Bytes, op_options.Pos)
        begin_mask = options.BeginMask()
        end_mask = options.EndMask()
        ellipsis_mask = options.EllipsisMask()
        new_axis_mask = options.NewAxisMask()
        shrink_axis_mask = options.ShrinkAxisMask()

        data_shape = list(input_tensors[0].tensor.ShapeAsNumpy())
        data_dim = len(data_shape)
        stride_dim = len(stride)
        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            #Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                #Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= (1 << stride_dim)
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    #Identify the end index for applying ellipsis_mask
                    to_index = min(((data_dim - (stride_dim-index)) + 1 \
                                     + new_axes_after_ellipsis), data_dim)
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask &new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = data_shape[final_index] \
                                                     if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = 0 if stride[index] < 0 \
                                                 else data_shape[final_index]
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        #Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = data_shape[final_index] + begin[index] \
                                                 if begin[index] < 0 else begin[index]
                        m_end[final_index] = begin[index] + 1
                        m_stride[final_index] = 1
                        fshape_indices.append(-2)
                    else:
                        fshape_indices.append(final_index)

                    final_index += 1
            return m_begin, m_end, m_stride, fshape_indices

        fshape_indices = None
        if begin_mask or end_mask or ellipsis_mask or new_axis_mask or shrink_axis_mask:
            begin, end, stride, fshape_indices = _transform_mask(stride_dim, ellipsis_mask)

        out = _op.strided_slice(data_expr, begin=begin, end=end, strides=stride)
        out_shape = _infer_shape(out)
        if not fshape_indices:
            fshape_indices = range(len(out_shape))

        #Create final output shape.
        final_output = []
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
            elif gather_index == -2:
                pass
            else:
                final_output.append(out_shape[gather_index])

        if not final_output:
            return out
        return _op.reshape(out, newshape=tuple(final_output))

    def convert_zeros_like(self, op):
        """Convert TFLite ZEROS LIKE"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = _op.zeros_like(in_expr)

        return out

    def convert_fill(self, op):
        """Convert TFLite FILL"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        if self.has_expr(input_tensors[0].tensor_idx):
            raise tvm.error.OpNotImplemented("For dims parameter of Fill operator,"
                                             " only constant values are supported.")

        in_dims = list(self.get_tensor_value(input_tensors[0]))
        in_value_expr = self.get_expr(input_tensors[1].tensor_idx)
        out = _op.full(in_value_expr, in_dims)

        return out

    def _convert_reduce(self, relay_op, op):
        """Generic method to Convert TFLite REDUCE operators"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ReducerOptions import ReducerOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

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

        if input_tensor.qnn_params:
            in_expr = _op.cast(in_expr, "int32")

        out = relay_op(in_expr, axis, keep_dims)

        # Finally if the reduce is quantized. Add a requantize at the end.
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
        if output_tensor.qnn_params:
            out = _qnn.op.requantize(out,
                                     input_scale=input_tensor.qnn_params['scale'],
                                     input_zero_point=input_tensor.qnn_params['zero_point'],
                                     output_scale=output_tensor.qnn_params['scale'],
                                     output_zero_point=output_tensor.qnn_params['zero_point'],
                                     out_dtype=output_tensor_type_str)

        return out

    def convert_reduce_min(self, op):
        return self._convert_reduce(_op.reduce.min, op)

    def convert_reduce_max(self, op):
        return self._convert_reduce(_op.reduce.max, op)

    def convert_reduce_mean(self, op):
        return self._convert_reduce(_op.reduce.mean, op)

    def convert_reduce_prod(self, op):
        return self._convert_reduce(_op.reduce.prod, op)

    def convert_reduce_sum(self, op):
        return self._convert_reduce(_op.reduce.sum, op)

    def convert_reduce_any(self, op):
        return self._convert_reduce(_op.reduce.any, op)

    def convert_fully_connected(self, op):
        """Convert TFLite fully connected"""
        try:
            from tflite.FullyConnectedOptions import FullyConnectedOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.TensorType import TensorType
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

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
        weight_shape = _infer_shape(weight_expr)

        if input_tensor.qnn_params:
            out = _qnn.op.dense(in_expr, weight_expr,
                                input_zero_point=input_tensor.qnn_params['zero_point'],
                                kernel_zero_point=weight_tensor.qnn_params['zero_point'],
                                input_scale=input_tensor.qnn_params['scale'],
                                kernel_scale=weight_tensor.qnn_params['scale'],
                                units=weight_shape[0],
                                out_dtype='int32')
        else:
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
            if not output_tensor.qnn_params:
                out = self.convert_fused_activation_function(out, fused_activation_fn)
            else:
                raise tvm.error.OpNotImplemented(
                    'Operator {} with fused activation is not supported yet.'
                    .format('qnn.op.dense'))

        # Finally if the dense is quantized. Add a requantize at the end.
        if output_tensor.qnn_params:
            data_scale = input_tensor.qnn_params['scale']
            weight_scale = weight_tensor.qnn_params['scale']
            data_scale_val = get_scalar_from_constant(data_scale)
            weight_scale_val = get_scalar_from_constant(weight_scale)
            new_input_scale_val = data_scale_val * weight_scale_val
            new_input_scale = relay.const(new_input_scale_val, 'float32')
            new_input_zero_point = relay.const(0, 'int32')
            out = _qnn.op.requantize(out,
                                     input_scale=new_input_scale,
                                     input_zero_point=new_input_zero_point,
                                     output_scale=output_tensor.qnn_params['scale'],
                                     output_zero_point=output_tensor.qnn_params['zero_point'],
                                     out_dtype=output_tensor_type_str)

        return out

    def convert_squeeze(self, op):
        """Convert TFLite squeeze"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.SqueezeOptions import SqueezeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

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
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 2, "input tensors length should be >= 2"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

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
            params['groups'] = int(input_c)
            # If number of input channels is 1, treat as normal
            # convolution.
            params['kernel_layout'] = 'HWIO' if input_c == 1 else 'HWOI'
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
                params['padding'] = [pad_top, pad_left, pad_bottom, pad_right]

        else:
            raise tvm.error.OpAttributeUnImplemented(
                'Padding format {} is not supported for operator Conv.'.format(padding))

        if input_tensor.qnn_params:
            qnn_conv2d_params = dict(params)
            qnn_conv2d_params['input_zero_point'] = input_tensor.qnn_params['zero_point']
            qnn_conv2d_params['kernel_zero_point'] = weight_tensor.qnn_params['zero_point']
            qnn_conv2d_params['out_dtype'] = 'int32'
            qnn_conv2d_params['input_scale'] = input_tensor.qnn_params['scale']
            qnn_conv2d_params['kernel_scale'] = weight_tensor.qnn_params['scale']
            out = _qnn.op.conv2d(in_expr, weight_expr, **qnn_conv2d_params)
        else:
            out = _op.nn.conv2d(in_expr, weight_expr, **params)

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
            if not output_tensor.qnn_params:
                out = self.convert_fused_activation_function(out, fused_activation_fn)
            else:
                raise tvm.error.OpNotImplemented(
                    'Operator {} with fused activation is not supported yet.'
                    .format('qnn.op.conv2d'))

        # Finally if the conv is quantized. Add a requantize at the end.
        if output_tensor.qnn_params:
            data_scale = input_tensor.qnn_params['scale']
            weight_scale = weight_tensor.qnn_params['scale']
            data_scale_val = get_scalar_from_constant(data_scale)
            weight_scale_val = get_scalar_from_constant(weight_scale)
            new_input_scale_val = data_scale_val * weight_scale_val
            new_input_scale = relay.const(new_input_scale_val, 'float32')
            new_input_zero_point = relay.const(0, 'int32')
            out = _qnn.op.requantize(out,
                                     input_scale=new_input_scale,
                                     input_zero_point=new_input_zero_point,
                                     output_scale=output_tensor.qnn_params['scale'],
                                     output_zero_point=output_tensor.qnn_params['zero_point'],
                                     out_dtype=output_tensor_type_str)

        return out

    def convert_split(self, op):
        """split implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.SplitOptions import SplitOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

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

    def convert_split_v(self, op):
        """SPLIT_V implementation."""
        input_tensors = self.get_input_tensors(op)

        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        if self.has_expr(input_tensors[1].tensor_idx):
            raise tvm.error.OpNotImplemented("For size_splits parameter of SPLIT_V operator, "
                                             "only constant values are supported.")
        size_splits = list(self.get_tensor_value(input_tensors[1]))
        size_splits = tuple(np.cumsum(size_splits)[:-1])

        axis_tensor = input_tensors[2]
        split_axis = self.get_tensor_value(axis_tensor)

        out = _op.split(in_expr, size_splits, axis=int(split_axis))
        # Relay does not like a TupleWrapper of 1 element, further this
        # only shows up with tf1.13 if we use a split with num_splits==1.
        # In tf 1.14 this doesn't appear as it is automatically a reshape
        # operation.
        if isinstance(out, _expr.TupleWrapper) and out.size == 1:
            out = out[0]

        return out

    def convert_slice(self, op):
        """Convert TFLite SLICE"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be == 3"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        begin = list(self.get_tensor_value(input_tensors[1]))
        size = list(self.get_tensor_value(input_tensors[2]))
        # strided_slice(Relay) needs the slice's end indices, not the size
        end = size
        input_tensor_shape = input_tensor.tensor.ShapeAsNumpy()
        input_tensor_rank = len(input_tensor_shape)
        for i in range(input_tensor_rank):
            if size[i] == -1:
                end[i] = input_tensor_shape[i]
            else:
                end[i] += begin[i]

        out = _op.strided_slice(in_expr, begin, end)

        return out

    def convert_transpose(self, op):
        """transpose implementation."""
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
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.CastOptions import CastOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

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
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        in_expr = self.get_expr(input_tensor_idx)

        # reps (tuple of int) – The number of times repeating the tensor data.
        reps = tuple(self.get_tensor_value(input_tensors[1]))

        out = _op.tile(in_expr, reps)

        return out

    def convert_topk_v2(self, op):
        """ Convert TFLite TOPK_v2 """
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)
        k = self.get_tensor_value(input_tensors[1])
        out = _op.topk(in_expr, int(k))

        return out

    def convert_pool2d(self, op, pool_type):
        """pool2d implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ActivationFunctionType import ActivationFunctionType
            from tflite.Pool2DOptions import Pool2DOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

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
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(input_tensor, output_tensor), \
                        'TFLite avg_pool2dreshape requires input and output scale' \
                        'and zero points to be equal'
                out = _op.cast(in_expr, dtype="int32")
                out = _op.nn.avg_pool2d(out, **params)
                out = _op.cast(out, dtype=output_tensor_type_str)
            else:
                out = _op.nn.avg_pool2d(in_expr, **params)
        elif pool_type == "max":
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(input_tensor, output_tensor), \
                        "qnn.op.max_pool2d requires input and output qnn params to be same"
            out = _op.nn.max_pool2d(in_expr, **params)
        elif pool_type == "l2":
            # L2_POOL_2D is equivalent to square_root(avg_pool(square(in_data)))
            # TFLite does not have support for quantised L2_POOL_2D op.
            assert not input_tensor.qnn_params, \
                "As TFLite does not have support for quantized L2_POOL_2D, \
                Quantized input is not expected."
            exp_type = self.get_tensor_type_str(output_tensor.tensor.Type())
            square_exp = _op.power(in_expr, relay.const(2, exp_type))
            avg_pool_exp = _op.nn.avg_pool2d(square_exp, **params)
            out = _op.sqrt(avg_pool_exp)
        else:
            raise tvm.error.OpNotImplemented(
                'Operator {} is not supported for frontend TFLite.'.format(pool_type + ' pool'))

        # If we have fused activations
        if fused_activation_fn != ActivationFunctionType.NONE:
            if input_tensor.qnn_params:
                raise tvm.error.OpNotImplemented(
                    'Operator {} with fused activation is not supported yet.'
                    .format('qnn.op.pool2d'))
            out = self.convert_fused_activation_function(out, fused_activation_fn)
        return out

    def convert_pad(self, op):
        """Convert TFLite PAD"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # TFLite PAD only support CONSTANT mode and does not support constant_values parameter.
        # tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])
        # convert list of lists to tuple of tuples
        paddings = tuple(tuple(l) for l in pad_list)

        # Set the pad value
        pad_value = 0
        if input_tensor.qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(input_tensor, output_tensor), \
                    "TFLite reshape requires input and output scale and zero points to be equal"

            # The pad value for quantized pad is the input zero point.
            pad_value = float(input_tensor.qnn_params['zero_point'].data.asnumpy())

        out = _op.nn.pad(in_expr, pad_width=paddings, pad_value=pad_value)
        return out

    def convert_floor_div(self, op):
        """Convert TFLite FLOOR_DIV"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized FLOOR DIV operator is not supported yet.')
        return self._convert_elemwise(_op.floor_divide, op)

    def convert_floor_mod(self, op):
        """Convert TFLite FLOOR_MOD"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized FLOOR MOD operator is not supported yet.')
        return self._convert_elemwise(_op.floor_mod, op)

    def convert_mirror_pad(self, op):
        """Convert TFLite MIRROR_PAD"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.MirrorPadOptions import MirrorPadOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        # the quantized form MirrorPad is not yet implemented in TFLite.
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                'TFlite quantized MIRROR_PAD operator is not supported yet.')

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])
        # convert list of lists to tuple of tuples
        paddings = tuple(tuple(l) for l in pad_list)

        assert op.BuiltinOptionsType() == BuiltinOptions.MirrorPadOptions
        op_options = op.BuiltinOptions()
        mirror_pad_options = MirrorPadOptions()
        mirror_pad_options.Init(op_options.Bytes, op_options.Pos)
        mode_byte = mirror_pad_options.Mode()

        mode = "REFLECT" if mode_byte == 0 else "SYMMETRIC"
        out = _op.nn.mirror_pad(in_expr, paddings, mode)

        return out

    def convert_pack(self, op):
        """Convert TFLite pack"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.PackOptions import PackOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        in_exprs = [self.get_expr(input_tensor.tensor_idx) for input_tensor in input_tensors]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.PackOptions
        op_options = op.BuiltinOptions()
        pack_options = PackOptions()
        pack_options.Init(op_options.Bytes, op_options.Pos)
        pack_axis = pack_options.Axis()

        in_exprs_reshaped = [_op.expand_dims(i, axis=pack_axis, num_newaxis=1) for i in in_exprs]
        out = _op.concatenate(in_exprs_reshaped, pack_axis)
        return out

    def convert_unpack(self, op):
        """Convert TFLite unpack"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.UnpackOptions import UnpackOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        assert op.BuiltinOptionsType() == BuiltinOptions.UnpackOptions
        op_options = op.BuiltinOptions()
        unpack_options = UnpackOptions()
        unpack_options.Init(op_options.Bytes, op_options.Pos)
        num_unpacks = unpack_options.Num()
        unpack_axis = unpack_options.Axis()

        # Relay doesn't support 'unpack' operator so we use 'split' & 'squeeze' instead.
        # We have to do 'squeeze' along the split axis but Relay expects
        # squeeze_axis to be either None or List.
        squeeze_axis = None if unpack_axis == 0 else [unpack_axis]

        # Relay doesn't like TupleWrapper of 1 element so we isolate the case of unpacking
        # a tensor by an axis with len(axis) == 1. For reference see convert_split().
        # Such unpacking will result in the same tensor so we omit 'split' and only squeeze
        # along the axis of dim == 1.
        if num_unpacks == 1:
            squeezed = _op.squeeze(in_expr, axis=squeeze_axis)
            if isinstance(squeezed, _expr.TupleWrapper):
                squeezed = squeezed[0]
        else:
            splitted = _op.split(in_expr,
                                 indices_or_sections=num_unpacks,
                                 axis=unpack_axis)
            squeezed = _expr.TupleWrapper(
                _expr.Tuple([_op.squeeze(split_item, axis=squeeze_axis) \
                             for split_item in splitted]), len(splitted))

        return squeezed

    def convert_batch_to_space_nd(self, op):
        """batch_to_space_nd implementation."""

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

    def convert_depth_to_space(self, op):
        """Convert TFLite DEPTH_TO_SPACE"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.DepthToSpaceOptions import DepthToSpaceOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.DepthToSpaceOptions
        op_options = op.BuiltinOptions()
        depth_to_space_options = DepthToSpaceOptions()
        depth_to_space_options.Init(op_options.Bytes, op_options.Pos)
        block_size = depth_to_space_options.BlockSize()
        out = _op.nn.depth_to_space(in_expr, block_size, layout='NHWC')

        return out

    def convert_space_to_depth(self, op):
        """Convert TFLite SPACE_TO_DEPTH"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.SpaceToDepthOptions import SpaceToDepthOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.SpaceToDepthOptions
        op_options = op.BuiltinOptions()
        space_to_depth_options = SpaceToDepthOptions()
        space_to_depth_options.Init(op_options.Bytes, op_options.Pos)
        block_size = space_to_depth_options.BlockSize()
        out = _op.nn.space_to_depth(in_expr, block_size, layout='NHWC')

        return out

    def convert_prelu(self, op):
        """Convert TFLite PReLU"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        input_tensor = input_tensors[0]
        alpha_tensor = input_tensors[1]
        alpha_tensor_type = alpha_tensor.tensor.Type()
        alpha_tensor_type_str = self.get_tensor_type_str(alpha_tensor_type)
        alpha_expr = self.exp_tab.new_const(self.get_tensor_value(alpha_tensor).flatten(),
                                            dtype=alpha_tensor_type_str)
        in_expr = self.get_expr(input_tensor.tensor_idx)
        out = _op.nn.prelu(in_expr, alpha_expr, axis=3)

        return out

    def convert_transpose_conv(self, op):
        """Convert TFLite TRANSPOSE_CONV"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.TensorType import TensorType
            from tflite.TransposeConvOptions import TransposeConvOptions
            from tflite.Padding import Padding
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        # Input (data) Tensor. NHWC layout
        input_tensor = input_tensors[2]
        _, _, _, input_c = input_tensor.tensor.ShapeAsNumpy()
        # Weights tensor. TFLite uses OHWI layout
        weights_tensor = input_tensors[1]
        out_channels, kernel_h, kernel_w, in_channels = weights_tensor.tensor.ShapeAsNumpy()
        assert input_c == in_channels, \
            "Input channel in the filter should match to channel in the input"
        # output_shape Tensor. NHWC layout
        output_shape_tensor = input_tensors[0]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        assert op.BuiltinOptionsType() == BuiltinOptions.TransposeConvOptions
        op_options = op.BuiltinOptions()
        deconv_options = TransposeConvOptions()
        deconv_options.Init(op_options.Bytes, op_options.Pos)

        padding = deconv_options.Padding()
        stride_h = deconv_options.StrideH()
        stride_w = deconv_options.StrideW()
        assert padding in (Padding.VALID, Padding.SAME), \
            'Padding format {} is not supported for operator TRANSPOSE_CONV'.format(padding)

        # Data
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # Weights
        weights_tensor_type = weights_tensor.tensor.Type()
        # weights tensor type should be UINT8 (quantization) or FLOAT32
        assert weights_tensor_type in (TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weights_tensor_type)
        weight_value_ohwi = self.get_tensor_value(weights_tensor)
        # Relay kernel_layout should be OIHW
        # Relay weights layout should be different from kernel_layout - it should be IOHW
        weight_value_iohw = np.transpose(weight_value_ohwi, (3, 0, 1, 2))
        weight_expr_iohw = self.exp_tab.new_const(weight_value_iohw, dtype=weight_tensor_type_str)

        # Output shape value
        output_shape_value = self.get_tensor_value(output_shape_tensor)
        # Relay expects filter output channel to match to output tensor channel.
        assert out_channels == output_shape_value[3], \
            "Output channel in the filter should match to channel in the output_shape"

        # TF frontend supports 'SAME' padding for kernel 1x1 only. Lets do the same here
        if padding == Padding.SAME:
            assert (kernel_h, kernel_w) == (1, 1), \
                "SAME padding is supported for kernel (1,1) only"

        out = _op.nn.conv2d_transpose(in_expr, weight_expr_iohw,
                                      strides=(stride_h, stride_w),
                                      channels=int(out_channels),
                                      kernel_size=(int(kernel_h), int(kernel_w)),
                                      data_layout="NHWC",
                                      kernel_layout="OIHW",
                                      out_dtype=output_tensor_type_str)

        return out

    def convert_detection_postprocess(self, op):
        """Convert TFLite_Detection_PostProcess"""
        _option_names = [
            "w_scale",
            "max_detections",
            "_output_quantized",
            "detections_per_class",
            "x_scale",
            "nms_score_threshold",
            "num_classes",
            "max_classes_per_detection",
            "use_regular_nms",
            "y_scale",
            "h_scale",
            "_support_output_type_float_in_quantized_op",
            "nms_iou_threshold"
        ]

        custom_options = get_custom_options(op, _option_names)
        if custom_options["use_regular_nms"]:
            raise tvm.error.OpAttributeUnImplemented(
                "use_regular_nms=True is not yet supported for operator {}."
                .format("TFLite_Detection_PostProcess")
            )

        inputs = self.get_input_tensors(op)
        assert len(inputs) == 3, "inputs length should be 3"
        cls_pred = self.get_expr(inputs[1].tensor_idx)
        loc_prob = self.get_expr(inputs[0].tensor_idx)
        batch_size = inputs[1].tensor.Shape(0)
        anchor_values = self.get_tensor_value(inputs[2])
        anchor_boxes = len(anchor_values)
        anchor_type = self.get_tensor_type_str(inputs[2].tensor.Type())
        anchor_expr = self.exp_tab.new_const(anchor_values, dtype=anchor_type)

        if inputs[0].qnn_params:
            loc_prob = _qnn.op.dequantize(data=loc_prob,
                                          input_scale=inputs[0].qnn_params['scale'],
                                          input_zero_point=inputs[0].qnn_params['zero_point'])
        if inputs[1].qnn_params:
            cls_pred = _qnn.op.dequantize(data=cls_pred,
                                          input_scale=inputs[1].qnn_params['scale'],
                                          input_zero_point=inputs[1].qnn_params['zero_point'])
        if inputs[2].qnn_params:
            anchor_expr = _qnn.op.dequantize(data=anchor_expr,
                                             input_scale=inputs[2].qnn_params['scale'],
                                             input_zero_point=inputs[2].qnn_params['zero_point'])

        # reshape the cls_pred and loc_prob tensors so
        # they can be consumed by multibox_transform_loc
        cls_pred = _op.transpose(cls_pred, [0, 2, 1])
        # loc_prob coords are in yxhw format
        # need to convert to xywh
        loc_coords = _op.split(loc_prob, 4, axis=2)
        loc_prob = _op.concatenate(
            [loc_coords[1], loc_coords[0], loc_coords[3], loc_coords[2]], axis=2
        )
        loc_prob = _op.reshape(loc_prob, [batch_size, anchor_boxes*4])

        # anchor coords are in yxhw format
        # need to convert to ltrb
        anchor_coords = _op.split(anchor_expr, 4, axis=1)
        anchor_y = anchor_coords[0]
        anchor_x = anchor_coords[1]
        anchor_h = anchor_coords[2]
        anchor_w = anchor_coords[3]
        plus_half = _expr.const(0.5, dtype='float32')
        minus_half = _expr.const(-0.5, dtype='float32')
        anchor_l = _op.add(anchor_x, _op.multiply(anchor_w, minus_half))
        anchor_r = _op.add(anchor_x, _op.multiply(anchor_w, plus_half))
        anchor_t = _op.add(anchor_y, _op.multiply(anchor_h, minus_half))
        anchor_b = _op.add(anchor_y, _op.multiply(anchor_h, plus_half))
        anchor_expr = _op.concatenate([anchor_l, anchor_t, anchor_r, anchor_b], axis=1)
        anchor_expr = _op.expand_dims(anchor_expr, 0)

        # attributes for multibox_transform_loc
        multibox_transform_loc_attrs = {}
        multibox_transform_loc_attrs["clip"] = False
        multibox_transform_loc_attrs["threshold"] = custom_options["nms_score_threshold"]
        multibox_transform_loc_attrs["variances"] = (
            1 / custom_options["x_scale"],
            1 / custom_options["y_scale"],
            1 / custom_options["w_scale"],
            1 / custom_options["h_scale"],
        )

        # attributes for non_max_suppression
        non_max_suppression_attrs = {}
        non_max_suppression_attrs["return_indices"] = False
        non_max_suppression_attrs["iou_threshold"] = custom_options["nms_iou_threshold"]
        non_max_suppression_attrs["force_suppress"] = False
        non_max_suppression_attrs["top_k"] = anchor_boxes
        non_max_suppression_attrs["max_output_size"] = custom_options["max_detections"]
        non_max_suppression_attrs["invalid_to_bottom"] = False

        ret = _op.vision.multibox_transform_loc(cls_pred, loc_prob,
                                                anchor_expr, **multibox_transform_loc_attrs)
        ret = _op.vision.non_max_suppression(ret[0], ret[1], **non_max_suppression_attrs)
        ret = _op.vision.get_valid_counts(ret, 0)
        valid_count = ret[0]
        # keep only the top 'max_detections' rows
        ret = _op.strided_slice(ret[1],
                                [0, 0, 0],
                                [batch_size, custom_options["max_detections"], anchor_boxes])
        # the output needs some reshaping to match tflite
        ret = _op.split(ret, 6, axis=2)
        cls_ids = _op.reshape(ret[0], [batch_size, -1])
        scores = _op.reshape(ret[1], [batch_size, -1])
        boxes = _op.concatenate([ret[3], ret[2], ret[5], ret[4]], axis=2)
        ret = _expr.TupleWrapper(_expr.Tuple([boxes, cls_ids, scores, valid_count]), size=4)
        return ret

    def get_expr(self, input_tensor_idx):
        return self.exp_tab.get_expr(get_tensor_name(self.subgraph, input_tensor_idx))

    def has_expr(self, input_tensor_idx):
        return self.exp_tab.has_expr(get_tensor_name(self.subgraph, input_tensor_idx))


def get_scalar_from_constant(expr):
    """ Returns scalar value from Relay constant scalar. """
    assert isinstance(expr, _expr.Constant) and not expr.data.shape, \
        "Expr is not a constant scalar."
    value = expr.data.asnumpy()
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(np.float32), \
        "value must be float32/int32"
    return np.asscalar(value)


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


def get_custom_options(op, option_names):
    """Get the options of a custom operator.

    This implements partial flexbuffer deserialization to be able
    to read custom options. It is not intended to be a general
    purpose flexbuffer deserializer and as such only supports a
    limited number of types and assumes the data is a flat map.

    Parameters
    ----------
    op:
        A custom TFlite operator.
    option_names: list
        A complete list of the custom option names.

    Returns
    -------
    options: dict
        A dictionary of the custom options.

    """
    import struct
    from enum import IntEnum

    class _FlexBufferType(IntEnum):
        """Flexbuffer type schema from flexbuffers.h"""
        FBT_NULL = 0
        FBT_INT = 1
        FBT_UINT = 2
        FBT_FLOAT = 3
        # Types above stored inline, types below store an offset.
        FBT_KEY = 4
        FBT_STRING = 5
        FBT_INDIRECT_INT = 6
        FBT_INDIRECT_UINT = 7
        FBT_INDIRECT_FLOAT = 8
        FBT_MAP = 9
        FBT_VECTOR = 10 # Untyped.
        FBT_VECTOR_INT = 11 # Typed any size (stores no type table).
        FBT_VECTOR_UINT = 12
        FBT_VECTOR_FLOAT = 13
        FBT_VECTOR_KEY = 14
        FBT_VECTOR_STRING = 15
        FBT_VECTOR_INT2 = 16 # Typed tuple (no type table, no size field).
        FBT_VECTOR_UINT2 = 17
        FBT_VECTOR_FLOAT2 = 18
        FBT_VECTOR_INT3 = 19 # Typed triple (no type table, no size field).
        FBT_VECTOR_UINT3 = 20
        FBT_VECTOR_FLOAT3 = 21
        FBT_VECTOR_INT4 = 22 # Typed quad (no type table, no size field).
        FBT_VECTOR_UINT4 = 23
        FBT_VECTOR_FLOAT4 = 24
        FBT_BLOB = 25
        FBT_BOOL = 26
        FBT_VECTOR_BOOL = 36 # To Allow the same type of conversion of type to vector type

    buffer = op.CustomOptionsAsNumpy().tobytes()
    value_vector_offset = buffer[-3]
    buffer = buffer[:-3]
    num_bytes = 4 # Assume all values are stored in 32 bit width
    value_vector_size = struct.unpack(
        "<i", buffer[-value_vector_offset - num_bytes:-value_vector_offset]
    )[0]
    type_offset = value_vector_size
    types = buffer[-type_offset:]
    values = []
    for i, t in enumerate(types):
        flex_type = _FlexBufferType(t >> 2)
        value_offset = -value_vector_offset + i*num_bytes
        value_bytes = buffer[value_offset:value_offset+num_bytes]
        if flex_type == _FlexBufferType.FBT_BOOL:
            value = bool(value_bytes[0])
        if flex_type == _FlexBufferType.FBT_INT:
            value = struct.unpack("<i", value_bytes)[0]
        if flex_type == _FlexBufferType.FBT_UINT:
            value = struct.unpack("<I", value_bytes)[0]
        if flex_type == _FlexBufferType.FBT_FLOAT:
            value = struct.unpack("<f", value_bytes)[0]

        values.append(value)

    custom_options = dict(zip(sorted(option_names), values))
    return custom_options


def from_tflite(model, shape_dict, dtype_dict):
    """Convert from tflite model into compatible relay Function.

    Parameters
    ----------
    model:
        tflite.Model or tflite.Model.Model (depending on tflite version)

    shape_dict : dict of str to int list/tuple
        Input shapes of the model.

    dtype_dict : dict of str to str
        Input types of the model.

    Returns
    -------
    mod : tvm.IRModule
        The relay module for compilation.

    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relay
    """
    try:
        import tflite.SubGraph
        import tflite.BuiltinOperator
    except ImportError:
        raise ImportError("The tflite package must be installed")

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite
        assert isinstance(model, tflite.Model)
    except TypeError:
        import tflite.Model
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
    func = _function.Function(analysis.free_vars(outputs), outputs)
    mod = IRModule.from_expr(func)
    return mod, params
