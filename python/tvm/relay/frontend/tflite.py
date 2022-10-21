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
import itertools
import math

import numpy as np
import tvm
from tvm import relay
from tvm.ir import IRModule
from tvm.runtime.name_transforms import sanitize_name

from ... import nd as _nd
from .. import analysis
from .. import expr as _expr
from .. import function as _function
from .. import op as _op
from .. import qnn as _qnn
from .common import ExprTable
from .common import infer_shape as _infer_shape
from .common import lstm_cell, to_int_list, shape_of, try_infer_value
from .tflite_flexbuffer import FlexBufferDecoder

__all__ = ["from_tflite"]


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
            from tflite.ActivationFunctionType import ActivationFunctionType
            from tflite.BuiltinOperator import BuiltinOperator
            from tflite.BuiltinOptions import BuiltinOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        self.model = model
        self.subgraph = subgraph
        self.exp_tab = exp_tab
        self.builtin_op_code = build_str_map(BuiltinOperator())
        self.activation_fn_type = build_str_map(ActivationFunctionType())
        self.builtin_options = build_str_map(BuiltinOptions())
        self.prefetched_nodes = {}
        self.allow_custom_ops = False

        # Add more operators
        self.convert_map = {
            "ABS": self.convert_abs,
            "ADD": self.convert_add,
            "ADD_N": self.convert_add_n,
            "ARG_MAX": self.convert_arg_max,
            "ARG_MIN": self.convert_arg_min,
            "AVERAGE_POOL_2D": self.convert_average_pool2d,
            "BATCH_TO_SPACE_ND": self.convert_batch_to_space_nd,
            "CAST": self.convert_cast,
            "CEIL": self.convert_ceil,
            "CONCATENATION": self.convert_concatenation,
            "CONV_2D": self.convert_conv2d,
            "COS": self.convert_cos,
            "DENSIFY": self.convert_densify,
            "DEPTH_TO_SPACE": self.convert_depth_to_space,
            "DEPTHWISE_CONV_2D": self.convert_depthwise_conv2d,
            "DEQUANTIZE": self.convert_dequantize,
            "DETECTION_POSTPROCESS": self.convert_detection_postprocess,
            "DIV": self.convert_div,
            "ELU": self.convert_elu,
            "EQUAL": self.convert_equal,
            "EXP": self.convert_exp,
            "EXPAND_DIMS": self.convert_expand_dims,
            "FAKE_QUANT": self.convert_fake_quant,
            "FILL": self.convert_fill,
            "FLOOR_DIV": self.convert_floor_div,
            "FLOOR_MOD": self.convert_floor_mod,
            "FLOOR": self.convert_floor,
            "FULLY_CONNECTED": self.convert_fully_connected,
            "GATHER": self.convert_gather,
            "GATHER_ND": self.convert_gather_nd,
            "GREATER_EQUAL": self.convert_greater_equal,
            "GREATER": self.convert_greater,
            "HARD_SWISH": self.convert_hard_swish,
            "L2_NORMALIZATION": self.convert_l2_normalization,
            "L2_POOL_2D": self.convert_l2_pool2d,
            "LEAKY_RELU": self.convert_leaky_relu,
            "LESS_EQUAL": self.convert_less_equal,
            "LESS": self.convert_less,
            "LOCAL_RESPONSE_NORMALIZATION": self.convert_lrn,
            "LOG": self.convert_log,
            "LOG_SOFTMAX": self.convert_log_softmax,
            "LOGICAL_AND": self.convert_logical_and,
            "LOGICAL_NOT": self.convert_logical_not,
            "LOGICAL_OR": self.convert_logical_or,
            "LOGISTIC": self.convert_logistic,
            "MATRIX_DIAG": self.convert_matrix_diag,
            "MATRIX_SET_DIAG": self.convert_matrix_set_diag,
            "MAX_POOL_2D": self.convert_max_pool2d,
            "MAXIMUM": self.convert_maximum,
            "MEAN": self.convert_reduce_mean,
            "MINIMUM": self.convert_minimum,
            "MIRROR_PAD": self.convert_mirror_pad,
            "MUL": self.convert_mul,
            "NEG": self.convert_neg,
            "NOT_EQUAL": self.convert_not_equal,
            "ONE_HOT": self.convert_one_hot,
            "PACK": self.convert_pack,
            "PAD": self.convert_pad,
            "PADV2": self.convert_pad,
            "POW": self.convert_pow,
            "PRELU": self.convert_prelu,
            "RANGE": self.convert_range,
            "QUANTIZE": self.convert_quantize,
            "REDUCE_ANY": self.convert_reduce_any,
            "REDUCE_MAX": self.convert_reduce_max,
            "REDUCE_MIN": self.convert_reduce_min,
            "REDUCE_PROD": self.convert_reduce_prod,
            "RELU": self.convert_relu,
            "RELU6": self.convert_relu6,
            "RELU_N1_TO_1": self.convert_relu_n1_to_1,
            "RESHAPE": self.convert_reshape,
            "RESIZE_BILINEAR": self.convert_resize_bilinear,
            "RESIZE_NEAREST_NEIGHBOR": self.convert_resize_nearest_neighbor,
            "ROUND": self.convert_round,
            "RSQRT": self.convert_rsqrt,
            "REVERSE_SEQUENCE": self.convert_reverse_sequence,
            "REVERSE_V2": self.convert_reverse_v2,
            "SELECT": self.convert_select,
            "SHAPE": self.convert_shape,
            "SIN": self.convert_sin,
            "SLICE": self.convert_slice,
            "SOFTMAX": self.convert_softmax,
            "SPACE_TO_BATCH_ND": self.convert_space_to_batch_nd,
            "SPACE_TO_DEPTH": self.convert_space_to_depth,
            "SPARSE_TO_DENSE": self.convert_sparse_to_dense,
            "SPLIT": self.convert_split,
            "SPLIT_V": self.convert_split_v,
            "SQRT": self.convert_sqrt,
            "SQUARE": self.convert_square,
            "SQUARED_DIFFERENCE": self.convert_squared_difference,
            "SQUEEZE": self.convert_squeeze,
            "STRIDED_SLICE": self.convert_strided_slice,
            "SUB": self.convert_sub,
            "SUM": self.convert_reduce_sum,
            "TAN": self.convert_tan,
            "TANH": self.convert_tanh,
            "TILE": self.convert_tile,
            "TOPK_V2": self.convert_topk_v2,
            "TRANSPOSE_CONV": self.convert_transpose_conv,
            "TRANSPOSE": self.convert_transpose,
            "UNPACK": self.convert_unpack,
            "UNIDIRECTIONAL_SEQUENCE_LSTM": self.convert_unidirectional_sequence_lstm,
            "WHERE": self.convert_select,
            "ZEROS_LIKE": self.convert_zeros_like,
            "NON_MAX_SUPPRESSION_V5": self.convert_nms_v5,
        }

    def check_unsupported_ops(self):
        """Check unsupported TFLite ops in our converter."""
        unsupported_ops_set = set()
        dynamic_range_ops_set = set()
        for op_idx in range(self.subgraph.OperatorsLength()):
            op = self.subgraph.Operators(op_idx)
            op_code_str = self.get_op_code_str(op)
            if op_code_str not in self.convert_map:
                unsupported_ops_set.add(op_code_str)
                continue

            # Trying to exclude "dynamic range quantization" optimized ops as not supported in TVM
            qnn_in_cnt = len(
                [_.qnn_params for _ in self.get_input_tensors(op)[0:1] if _.qnn_params is not None]
            )
            qnn_weight_cnt = len(
                [_.qnn_params for _ in self.get_input_tensors(op)[1:] if _.qnn_params is not None]
            )
            qnn_out_cnt = len(
                [_.qnn_params for _ in self.get_output_tensors(op) if _.qnn_params is not None]
            )

            if qnn_in_cnt == 0 and qnn_out_cnt == 0 and qnn_weight_cnt > 0:
                dynamic_range_ops_set.add(op_code_str)

        raise_msg = ""

        if unsupported_ops_set:
            msg = "The following operators are not supported in frontend " "TFLite: {}\n"
            ops = str(list(unsupported_ops_set)).strip("[,]")
            raise_msg += msg.format(ops)

        if dynamic_range_ops_set:
            msg = (
                "The following operators are likely to have dynamic range quantization: {}. "
                "If you are running an optimized graph, please turn off dynamic range quantization "
                "or use full integer quantization"
            )
            raise_msg += msg.format(str(list(dynamic_range_ops_set)).strip("[,]"))

        if len(raise_msg) > 0:
            raise tvm.error.OpNotImplemented(raise_msg)

    def unbind(self, data, axis=1):
        """
        This is a modified version compared to the one in common.py.
        The onnx version takes a relay.Expr.Call, the tflite
        version a TensorWrapper. Also this version by default splits
        along axis 1 and not axis 0 as the onnx version.

         Parameters
         ----------
         data : tvm.relay.frontend.tflite.TensorWrapper
             Input tensor
         axis : int
             Axis along which tensor is split.
         Returns
         -------
         result : List[relay.Expr]
             The sequence of computed tensors
        """
        shape = to_int_list(self.get_tensor_shape(data))
        if axis >= len(shape):
            msg = "Please check input dim, it shouldn't be greater than or equal to rank."
            raise AttributeError(msg)

        selections = shape[axis]
        shape.pop(axis)
        timestep = 0  # Reshape to make time step as the first dim
        shape.insert(timestep, selections)
        res_split = _op.split(
            _op.reshape(self.get_expr(data.tensor_idx), tuple(shape)), selections, timestep
        )
        ret = []
        for i in range(selections):
            ret.append(_op.squeeze(res_split[i], axis=[timestep]))
        return _expr.TupleWrapper(_expr.Tuple(ret), selections)

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

            # In case the Op can be prefetched, the output can be optimized out
            if ret is None:
                continue

            if len(output_tensors) == 1:
                tensor_idx = output_tensors[0].tensor_idx
                self.exp_tab.set_expr(get_tensor_name(self.subgraph, tensor_idx), ret)
            else:
                for idx, output_tensor in enumerate(output_tensors):
                    self.exp_tab.set_expr(
                        get_tensor_name(self.subgraph, output_tensor.tensor_idx), ret[idx]
                    )

    def get_op_code_str(self, op):
        """Get TFLite ops string representation"""
        try:
            from tflite.BuiltinOperator import BuiltinOperator
        except ImportError:
            raise ImportError("The tflite package must be installed")

        op_code_list_idx = op.OpcodeIndex()

        op_c = self.model.OperatorCodes(op_code_list_idx)
        # In TFlite 2.4.x there was a change where the type of the field that contained
        # the builtin code changed from int8 to int32 in the flat buffer representation.
        # However, to retain support for old flat buffers that were created, they retained
        # the original 8 bit field, but named it "deprecated_builtin_code" in TFLite 2.4.
        # This means that the API function BuiltinCode() which originally returned the value
        # of the 8 bit field would now look for the value in the new int32 field in the
        # schema and DeprecatedBuiltinCode() will look at the old 8 bit field.
        # In TFLite 2.4, if the opcode value is less than 127, it can be in either field
        # (however, if it is only in the "builtin_code" field, the model is not backward
        # compatible), so similarly to TFLite 2.4 reader, we'll pick the higher value of the
        # two fields.
        # Remember however that this value came into existence only after Tensorflow
        # lite 2.4.x and hence encase it in a try -except block.
        # Phew !
        try:
            opc = max(op_c.DeprecatedBuiltinCode(), op_c.BuiltinCode())
        except AttributeError:
            # In versions before 2.4 the int8 field that holds the builtin code is accessed
            # by BuiltinCode() and DeprecatedBuiltinCode() doesn't exist
            opc = op_c.BuiltinCode()

        op_code_id = opc
        try:
            op_code_str = self.builtin_op_code[op_code_id]
        except KeyError:
            raise NotImplementedError(
                "TFLite operator with code "
                + str(op_code_id)
                + " is not supported by this version of the fbs schema."
            )
        if op_code_id == BuiltinOperator.CUSTOM:
            # Custom operator
            custom_op_code_str = self.model.OperatorCodes(op_code_list_idx).CustomCode()

            if self.allow_custom_ops:
                return "CUSTOM"

            if custom_op_code_str == b"TFLite_Detection_PostProcess":
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
                # TFLite supports both per-tensor and per-axis (aka channel) quantization.  For
                # per-tensor quantization, scale and zero points are scalar values.  For per-axis
                # quantization, scale and zero points for the weights are tensors (activations are
                # per-tensor quantized). However, the TFLite quantization spec puts restrictions on
                # zero points for per-axis quantization.  Specifically, the zero point is a tensor
                # but all values are 0. More information can be found here -
                # https://www.tensorflow.org/lite/performance/quantization_spec

                tflite_scale = tflite_qnn_params.ScaleAsNumpy()
                tflite_zero_point = tflite_qnn_params.ZeroPointAsNumpy()
                is_qnn_params_valid = True

                # Handle Per-axis and per-tensor cases
                if isinstance(tflite_scale, np.ndarray):
                    assert isinstance(tflite_zero_point, np.ndarray)

                    # Tensor - Per-axis quantization
                    if tflite_scale.size != 1 and tflite_zero_point.size != 1:
                        scale = tflite_scale
                        # Ensure that all zero points are zeros
                        zero_point = tflite_zero_point
                        if not np.all(zero_point == 0):
                            raise tvm.error.OpAttributeInvalid(
                                "TFLite per-axis quantization restricts all zero points to be"
                                + " 0, but a non-zero value is observed"
                            )
                        zero_point = int(zero_point[0])

                    # Scalar - Per-tensor quantization
                    elif tflite_scale.size == 1 and tflite_zero_point.size == 1:
                        scale = float(tflite_scale[0])
                        zero_point = int(tflite_zero_point[0])

                    else:
                        raise NotImplementedError(
                            "Quantized type {} (scale) and  {} (zero point) not supported".format(
                                type(tflite_scale), type(tflite_zero_point)
                            )
                        )
                elif tflite_scale == 0 and tflite_zero_point == 0:
                    # Handle corner case for ops like quantized reshape whose second operand (shape)
                    # has zero scale and zero zero point. This is not used.
                    is_qnn_params_valid = False
                else:
                    raise NotImplementedError(
                        "Quantized type {} not supported".format(type(tflite_scale))
                    )

                # Check that the scale and zero points are valid.
                if is_qnn_params_valid:
                    qnn_params = dict()
                    qnn_params["scale"] = relay.const(scale, "float32")
                    qnn_params["zero_point"] = relay.const(zero_point, "int32")
            return_list.append(TensorWrapper(tensor_idx, tensor, buffer, qnn_params))
        return return_list

    def get_tensor_type_as_numpy(self, tensor_wrapper):
        """Returns np.dtype out of TensorType"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        try:
            from tflite.TensorType import TensorType

            return {
                TensorType.UINT8: np.uint8,
                TensorType.INT8: np.int8,
                TensorType.INT16: np.int16,
                TensorType.FLOAT16: np.float16,
                TensorType.FLOAT32: np.float32,
                TensorType.INT32: np.int32,
                TensorType.INT64: np.int64,
                TensorType.BOOL: np.bool_,
            }[tensor_wrapper.tensor.Type()]
        except ImportError:
            raise ImportError("The tflite package must be installed")
        except KeyError:
            raise NotImplementedError(
                "Tensor type '{}' currently not supported".format(tensor_wrapper.tensor.Type())
            )

    # pylint: disable=no-else-return
    def get_tensor_value(self, tensor_wrapper, is_sparse=False):
        """Get tensor buffer value from given tensor wrapper"""
        assert isinstance(tensor_wrapper, TensorWrapper)

        dtype = self.get_tensor_type_as_numpy(tensor_wrapper)
        data = tensor_wrapper.buffer.DataAsNumpy()

        if tensor_wrapper.tensor.ShapeLength() != 0:
            shape = to_int_list(self.get_tensor_shape(tensor_wrapper))
        else:
            shape = []

        if is_sparse:
            return np.frombuffer(data, dtype=dtype)
        else:
            return np.frombuffer(data, dtype=dtype).reshape(shape)

    def get_tensor_type_str(self, tensor_type):
        """Get tensor type string representation when given TFLite tensor type"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if tensor_type == TensorType.INT8:
            return "int8"
        if tensor_type == TensorType.INT16:
            return "int16"
        if tensor_type == TensorType.UINT8:
            return "uint8"
        if tensor_type == TensorType.FLOAT16:
            return "float16"
        if tensor_type == TensorType.FLOAT32:
            return "float32"
        if tensor_type == TensorType.INT32:
            return "int32"
        if tensor_type == TensorType.INT64:
            return "int64"
        if tensor_type == TensorType.BOOL:
            return "bool"
        raise NotImplementedError(
            "Tensor type {} is currently not supported".format(str(tensor_type))
        )

    def has_same_qnn_params(self, lhs_tensor, rhs_tensor):
        lhs_scale = lhs_tensor.qnn_params["scale"]
        rhs_scale = rhs_tensor.qnn_params["scale"]
        lhs_zero_point = lhs_tensor.qnn_params["zero_point"]
        rhs_zero_point = rhs_tensor.qnn_params["zero_point"]
        # 0.1 + 0.2 != 0.3
        return np.allclose(
            lhs_scale.data.numpy(), rhs_scale.data.numpy(), rtol=1e-5, atol=1e-5
        ) and np.allclose(
            lhs_zero_point.data.numpy(), rhs_zero_point.data.numpy(), rtol=1e-5, atol=1e-5
        )

    def is_quantized(self, op):
        """Check if an input tensor is quantized."""
        input_tensors = self.get_input_tensors(op)
        first_tensor = input_tensors[0]
        return first_tensor.qnn_params is not None

    def quantize(self, expr, tensor_to_quantize):
        """Helper function to quantize a tensor with Relay"""
        tensor_type = tensor_to_quantize.tensor.Type()
        tensor_type_str = self.get_tensor_type_str(tensor_type)
        quantized = _qnn.op.quantize(
            data=expr,
            output_scale=tensor_to_quantize.qnn_params["scale"],
            output_zero_point=tensor_to_quantize.qnn_params["zero_point"],
            out_dtype=tensor_type_str,
        )
        return quantized

    def dequantize(self, expr, tensor):
        """Helper function to dequantize a tensor with Relay"""
        dequantized = _qnn.op.dequantize(
            data=expr,
            input_scale=tensor.qnn_params["scale"],
            input_zero_point=tensor.qnn_params["zero_point"],
        )
        return dequantized

    def convert_qnn_fused_activation_function(
        self, expr, fused_activation_fn, scale, zero_point, dtype
    ):
        """Convert TFLite fused activation function. The expr is an input quantized tensor with
        scale and zero point"""
        try:
            from tflite.ActivationFunctionType import ActivationFunctionType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        # Quantize a float value to an quantized integer value
        quantize = lambda x: float(int(round(x / scale)) + zero_point)

        # Get min/max of the output dtype. This will be used to ensure that clip a_min/a_max are not
        # beyond the dtype range.
        qmin = float(tvm.tir.op.min_value(dtype).value)
        qmax = float(tvm.tir.op.max_value(dtype).value)

        # The input expr is a quantized tensor with its scale and zero point. We calculate the
        # suitable clip off points based on these scale and zero point.
        if fused_activation_fn == ActivationFunctionType.NONE:
            return expr
        if fused_activation_fn == ActivationFunctionType.RELU6:
            return _op.clip(expr, a_min=max(qmin, quantize(0)), a_max=min(qmax, quantize(6.0)))
        if fused_activation_fn == ActivationFunctionType.RELU_N1_TO_1:
            return _op.clip(expr, a_min=max(qmin, quantize(-1.0)), a_max=min(qmax, quantize(1.0)))
        if fused_activation_fn == ActivationFunctionType.RELU:
            return _op.clip(expr, a_min=max(qmin, quantize(0.0)), a_max=qmax)

        fused_activation_fn_str = self.activation_fn_type[fused_activation_fn]
        raise tvm.error.OpNotImplemented(
            "Quantized activation {} is not supported yet.".format(fused_activation_fn_str)
        )

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
        assert len(input_tensors) in (1, 2), "input tensors should not be empty"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "There should be only 1 output tensor"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx

        if len(input_tensors) == 2:
            shape_tensor = input_tensors[1]
            if self.has_expr(shape_tensor.tensor_idx):
                target_expr = self.get_expr(shape_tensor.tensor_idx)
                target_value, success = try_infer_value(
                    target_expr,
                    parameters={k: _nd.array(np.array(v)) for k, v in self.exp_tab.params.items()},
                )
                if success:
                    # convert to flattened list
                    from itertools import chain

                    try:
                        target_shape = list(chain(*target_value))
                    except TypeError:
                        target_shape = list(chain(target_value))
                else:
                    target_shape = target_expr
            else:
                target_shape = self.get_tensor_value(shape_tensor)
                # convert to flattened list
                from itertools import chain

                try:
                    target_shape = list(chain(*target_shape))
                except TypeError:
                    target_shape = list(chain(target_shape))

        else:
            assert op.BuiltinOptionsType() == BuiltinOptions.ReshapeOptions
            op_options = op.BuiltinOptions()
            reshape_options = ReshapeOptions()
            reshape_options.Init(op_options.Bytes, op_options.Pos)
            target_shape = to_int_list(reshape_options.NewShapeAsNumpy())

        in_expr = self.get_expr(input_tensor_idx)

        # If the tensors are quantized, ensure that input/output qnn params are same.

        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        if input_tensor.qnn_params and input_tensor_type_str == "int8":
            # TFLite 2.x quantization spec requires qnn params to be same and dtype to be int8.
            # For TFLite 1.x, dtype can be uint8 and qnn params can be different
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(
                input_tensor, output_tensor
            ), "TFLite reshape requires input and output scale and zero points to be equal"

        out = _op.reshape(in_expr, newshape=target_shape)
        if input_tensor.qnn_params and input_tensor_type_str == "uint8":
            output_tensor = output_tensors[0]
            if not self.has_same_qnn_params(input_tensor, output_tensor):
                output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
                out = _qnn.op.requantize(
                    out,
                    input_scale=input_tensor.qnn_params["scale"],
                    input_zero_point=input_tensor.qnn_params["zero_point"],
                    output_scale=output_tensor.qnn_params["scale"],
                    output_zero_point=output_tensor.qnn_params["zero_point"],
                    out_dtype=output_tensor_type_str,
                )

        return out

    def _convert_resize(self, method, op):
        """Generic method to Convert TFLite RESIZE operators"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ResizeBilinearOptions import ResizeBilinearOptions

            # ResizeNearestNeighborOptions was added in tflite v1.13
            tflite_ver = 1120
            if "ResizeNearestNeighborOptions" in dir(BuiltinOptions):
                from tflite.ResizeNearestNeighborOptions import ResizeNearestNeighborOptions

                tflite_ver = 1130
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # images, 4-D Tensor with shape NHWC.
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # size - 1-D int32 Tensor of 2 elements: new_height, new_width
        target_size = tuple(self.get_tensor_value(input_tensors[1]))

        # Options - align_corners (bool)
        resize_options = None
        align_corners = False
        bilinear_method = method == "linear"
        if bilinear_method:
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeBilinearOptions
            resize_options = ResizeBilinearOptions()
        elif tflite_ver >= 1130:
            assert op.BuiltinOptionsType() == BuiltinOptions.ResizeNearestNeighborOptions
            resize_options = ResizeNearestNeighborOptions()

        if resize_options is not None:
            op_options = op.BuiltinOptions()
            resize_options.Init(op_options.Bytes, op_options.Pos)
            align_corners = resize_options.AlignCorners()
            half_pixel_centers = resize_options.HalfPixelCenters()

        # Use layout NHWC
        coord_trans = "align_corners" if align_corners else "asymmetric"
        coord_trans = "half_pixel" if half_pixel_centers else coord_trans

        rounding_method = ""
        if method == "nearest_neighbor":
            if not align_corners and half_pixel_centers:
                rounding_method = "round_prefer_ceil"

        if bilinear_method and input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = _op.image.resize2d(
            in_expr, target_size, None, "NHWC", method, coord_trans, rounding_method
        )
        if bilinear_method and output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_resize_bilinear(self, op):
        """Convert TFLite RESIZE_BILINEAR"""
        return self._convert_resize("linear", op)

    def convert_resize_nearest_neighbor(self, op):
        """Convert TFLite RESIZE_NEAREST_NEIGHBOR"""
        return self._convert_resize("nearest_neighbor", op)

    def convert_l2_normalization(self, op):
        """Convert TFLite L2_NORMALIZATION"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.L2NormOptions import L2NormOptions
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
                "TFLite quantized L2_NORMALIZATION operator is not supported yet."
            )

        # TFL uses only the default epsilon value
        out = _op.nn.l2_normalize(in_expr, eps=1e-12, axis=[input_tensor_rank - 1])

        # if we have fused activation fn
        if output_tensor.qnn_params:
            raise tvm.error.OpNotImplemented(
                "TFLite quantized L2_NORMALIZATION operator is not supported yet."
            )
        out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_lrn(self, op):
        """Convert TFLite LOCAL_RESPONSE_NORMALIZATION"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.LocalResponseNormalizationOptions import LocalResponseNormalizationOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented("TFlite quantized LRN operator is not supported yet.")

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
        axis = 3  # NHWC format
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

        params = {"axis": -1}  # -1 is channel
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

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = _op.tanh(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_range(self, op):
        """Convert TFLite Range"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        start, limit, delta = input_tensors[0], input_tensors[1], input_tensors[2]

        expressions = [self.get_tensor_expr(t) for t in [start, limit, delta]]

        # out type inference
        if delta.tensor.Type() == TensorType.FLOAT32:
            out_type = self.get_tensor_type_str(delta.tensor.Type())
        else:
            out_type = self.get_tensor_type_str(start.tensor.Type())

        out = _op.arange(expressions[0], expressions[1], expressions[2], out_type)

        return out

    def convert_shape(self, op):
        """Convert TFLite Shape"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ShapeOptions import ShapeOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        assert op.BuiltinOptionsType() == BuiltinOptions.ShapeOptions
        op_options = op.BuiltinOptions()
        shape_options = ShapeOptions()
        shape_options.Init(op_options.Bytes, op_options.Pos)

        out_type = self.get_tensor_type_str(shape_options.OutType())
        out = shape_of(self.get_tensor_expr(input_tensors[0]), dtype=out_type)

        return out

    def convert_relu(self, op):
        """Convert TFLite ReLU"""
        try:
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

        if input_tensor.qnn_params:
            # Quantize a float value to an quantized integer value
            scale_val = get_scalar_from_constant(input_tensor.qnn_params["scale"])
            zero_point_val = get_scalar_from_constant(input_tensor.qnn_params["zero_point"])

            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = self.convert_qnn_fused_activation_function(
                expr=in_expr,
                fused_activation_fn=ActivationFunctionType.RELU,
                scale=scale_val,
                zero_point=zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = _op.nn.relu(in_expr)

        if output_tensor.qnn_params:
            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = _qnn.op.requantize(
                out,
                input_scale=input_tensor.qnn_params["scale"],
                input_zero_point=input_tensor.qnn_params["zero_point"],
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
            )

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

    def convert_relu6(self, op):
        """Convert TFLite ReLU6"""
        try:
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

        if input_tensor.qnn_params:
            # Quantize a float value to an quantized integer value
            scale_val = get_scalar_from_constant(input_tensor.qnn_params["scale"])
            zero_point_val = get_scalar_from_constant(input_tensor.qnn_params["zero_point"])

            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = self.convert_qnn_fused_activation_function(
                expr=in_expr,
                fused_activation_fn=ActivationFunctionType.RELU6,
                scale=scale_val,
                zero_point=zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = _op.clip(in_expr, a_min=0, a_max=6)

        if output_tensor.qnn_params:
            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = _qnn.op.requantize(
                out,
                input_scale=input_tensor.qnn_params["scale"],
                input_zero_point=input_tensor.qnn_params["zero_point"],
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
            )

        return out

    def convert_leaky_relu(self, op):
        """Convert TFLite LEAKY_RELU"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.LeakyReluOptions import LeakyReluOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        assert op.BuiltinOptionsType() == BuiltinOptions.LeakyReluOptions
        op_options = op.BuiltinOptions()
        leaky_relu_options = LeakyReluOptions()
        leaky_relu_options.Init(op_options.Bytes, op_options.Pos)
        alpha_tensor = leaky_relu_options.Alpha()

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = _op.nn.leaky_relu(in_expr, alpha_tensor)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_relu_n1_to_1(self, op):
        """Convert TFLite RELU_N1_TO_1"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            # Quantize a float value to an quantized integer value
            scale_val = get_scalar_from_constant(input_tensor.qnn_params["scale"])
            zero_point_val = get_scalar_from_constant(input_tensor.qnn_params["zero_point"])
            quantize = lambda x: float(int(round(x / scale_val)) + zero_point_val)

            # Get min/max of the input dtype. This will be used to ensure that
            # clip a_min/a_max are not beyond the dtype range.
            input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
            qmin = float(tvm.tir.op.min_value(input_tensor_type_str).value)
            qmax = float(tvm.tir.op.max_value(input_tensor_type_str).value)

            out = _op.clip(in_expr, a_min=max(qmin, quantize(-1.0)), a_max=min(qmax, quantize(1.0)))
        else:
            out = _op.clip(in_expr, a_min=-1, a_max=1)

        if output_tensor.qnn_params:
            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = _qnn.op.requantize(
                out,
                input_scale=input_tensor.qnn_params["scale"],
                input_zero_point=input_tensor.qnn_params["zero_point"],
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
            )

        return out

    def convert_log_softmax(self, op):
        """Convert TFLite LOG_SOFTMAX"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = _op.nn.log_softmax(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)

        return out

    def convert_concatenation(self, op):
        """Convert TFLite concatenation"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ConcatenationOptions import ConcatenationOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 1, "input tensors should greater than 1"
        in_exprs = [self.get_tensor_expr(_) for _ in input_tensors]

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
            input_scales = [input_tensor.qnn_params["scale"] for input_tensor in input_tensors]
            input_zero_points = [
                input_tensor.qnn_params["zero_point"] for input_tensor in input_tensors
            ]
            out = _qnn.op.concatenate(
                in_exprs,
                input_scales=input_scales,
                input_zero_points=input_zero_points,
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                axis=concatenation_axis,
            )

        # Handle fused activations
        if output_tensor.qnn_params:
            scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=scale_val,
                zero_point=zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def _convert_unary_elemwise(self, relay_op, op):
        """Generic method to convert TFLite unary elemwise functions"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        if input_tensor.qnn_params:
            in_expr = self.dequantize(in_expr, input_tensor)
        out = relay_op(in_expr)
        if output_tensor.qnn_params:
            out = self.quantize(out, output_tensor)
        return out

    def convert_abs(self, op):
        """Convert TFLite ABS"""
        return self._convert_unary_elemwise(_op.abs, op)

    def convert_ceil(self, op):
        """Convert TFLite CEIL"""
        return self._convert_unary_elemwise(_op.ceil, op)

    def convert_floor(self, op):
        """Convert TFLite FLOOR"""
        return self._convert_unary_elemwise(_op.floor, op)

    def convert_round(self, op):
        """Convert TFLite ROUND"""
        return self._convert_unary_elemwise(_op.round, op)

    def convert_exp(self, op):
        """Convert TFLite EXP"""
        return self._convert_unary_elemwise(_op.exp, op)

    def convert_log(self, op):
        """Convert TFLite LOG"""
        return self._convert_unary_elemwise(_op.log, op)

    def convert_sin(self, op):
        """Convert TFLite SIN"""
        return self._convert_unary_elemwise(_op.sin, op)

    def convert_tan(self, op):
        """Convert TFLite TAN"""
        return self._convert_unary_elemwise(_op.tan, op)

    def convert_cos(self, op):
        """Convert TFLite COS"""
        return self._convert_unary_elemwise(_op.cos, op)

    def convert_sqrt(self, op):
        """Convert TFLite SQRT"""
        return self._convert_unary_elemwise(_op.sqrt, op)

    def convert_rsqrt(self, op):
        """Convert TFLite RSQRT"""
        return self._convert_unary_elemwise(_op.rsqrt, op)

    def convert_neg(self, op):
        """Convert TFLite NEG"""
        return self._convert_unary_elemwise(_op.negative, op)

    def convert_elu(self, op):
        """Convert TFLite ELU"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented("TFlite quantized ELU operator is not supported yet.")
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        exp_type = self.get_tensor_type_str(input_tensor.tensor.Type())
        out = relay.const(-1.0, exp_type) * _op.nn.relu(
            relay.const(1.0, exp_type) - _op.exp(in_expr)
        ) + _op.nn.relu(in_expr)

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
                "TFlite quantized SQUARE operator is not supported yet."
            )

        exp_type = self.get_tensor_type_str(output_tensor.tensor.Type())
        out = _op.power(in_expr, relay.const(2, exp_type))

        return out

    def _convert_elemwise(
        self,
        relay_op,
        op,
        ignore_qnn_params=False,
        comparison_op=False,
    ):
        """Generic method to Convert TFLite elemwise"""
        try:
            from tflite.AddOptions import AddOptions
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.DivOptions import DivOptions
            from tflite.MulOptions import MulOptions
            from tflite.SubOptions import SubOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        rhs_tensor = input_tensors[1]
        lhs_expr = self.get_tensor_expr(lhs_tensor)
        rhs_expr = self.get_tensor_expr(rhs_tensor)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        # TFLite format demands equal scale and zero_point tuple parameters for some operations
        # to allow us to use non-quantized operation instead of quantized if ignore_qnn_params=True
        if ignore_qnn_params and not comparison_op:
            assert (
                lhs_tensor.qnn_params
                and self.has_same_qnn_params(lhs_tensor, output_tensor)
                and self.has_same_qnn_params(rhs_tensor, output_tensor)
            ), "All tensors should be quantized with the same (scale,zero-point) tuple parameters"

        # If quantized, extracts qnn params and call QNN add operator.
        if not ignore_qnn_params and lhs_tensor.qnn_params:
            assert rhs_tensor.qnn_params, "Both tensors should be quantized."
            assert output_tensor.qnn_params, "Output tensor should be quantized."
            out = relay_op(
                lhs=lhs_expr,
                rhs=rhs_expr,
                lhs_scale=lhs_tensor.qnn_params["scale"],
                lhs_zero_point=lhs_tensor.qnn_params["zero_point"],
                rhs_scale=rhs_tensor.qnn_params["scale"],
                rhs_zero_point=rhs_tensor.qnn_params["zero_point"],
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
            )
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

            # Handle fused activations
            if not ignore_qnn_params and output_tensor.qnn_params:
                scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
                zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
                output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
                out = self.convert_qnn_fused_activation_function(
                    expr=out,
                    fused_activation_fn=fused_activation_fn,
                    scale=scale_val,
                    zero_point=zero_point_val,
                    dtype=output_tensor_type_str,
                )
            else:
                out = self.convert_fused_activation_function(out, fused_activation_fn)
        return out

    def convert_add(self, op):
        """Convert TFLite ADD"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            return self._convert_elemwise(_qnn.op.add, op)
        return self._convert_elemwise(_op.add, op)

    def convert_add_n(self, op):
        """Convert TFLite ADD_N"""
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        input_tensors = self.get_input_tensors(op)
        assert not input_tensors[0].qnn_params, "TFLite does not support quantized ADD_N."
        lhs_expr = self.get_tensor_expr(input_tensors[0])
        for rhs_tensor in input_tensors[1:]:
            assert not rhs_tensor.qnn_params, "TFLite does not support quantized ADD_N"
            rhs_expr = self.get_tensor_expr(rhs_tensor)
            lhs_expr = _op.add(lhs_expr, rhs_expr)
        return lhs_expr

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
            raise tvm.error.OpNotImplemented("TFlite quantized DIV operator is not supported yet.")
        return self._convert_elemwise(_op.divide, op)

    def convert_pow(self, op):
        """Convert TFLite POW"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented("TFlite quantized POW operator is not supported yet.")
        return self._convert_elemwise(_op.power, op)

    def convert_maximum(self, op):
        """Convert TFLite MAXIMUM"""
        return self._convert_elemwise(_op.maximum, op, self.is_quantized(op))

    def convert_minimum(self, op):
        """Convert TFLite MINIMUM"""
        return self._convert_elemwise(_op.minimum, op, self.is_quantized(op))

    def convert_greater(self, op):
        """Convert TFLite GREATER"""
        return self._convert_elemwise(_op.greater, op, self.is_quantized(op), comparison_op=True)

    def convert_squared_difference(self, op):
        """Convert TFLite SQUARED DIFFERENCE"""
        # Check if the input tensor is quantized, call QNN op
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized squared difference operator is not supported yet."
            )
        difference = self._convert_elemwise(_op.subtract, op)
        # _convert_elemwise has guaranteed only have one output tensor
        exp_type = self.get_tensor_type_str(self.get_output_tensors(op)[0].tensor.Type())
        out = _op.power(difference, relay.const(2, exp_type))
        return out

    def convert_greater_equal(self, op):
        """Convert TFLite GREATER_EQUAL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized GREATER_EQUAL operator is not supported yet."
            )
        return self._convert_elemwise(_op.greater_equal, op)

    def convert_less(self, op):
        """Convert TFLite LESS"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented("TFlite quantized LESS operator is not supported yet.")
        return self._convert_elemwise(_op.less, op)

    def convert_less_equal(self, op):
        """Convert TFLite LESS_EQUAL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized LESS_EQUAL operator is not supported yet."
            )
        return self._convert_elemwise(_op.less_equal, op)

    def convert_equal(self, op):
        """Convert TFLite EQUAL"""
        return self._convert_elemwise(_op.equal, op, self.is_quantized(op), comparison_op=True)

    def convert_not_equal(self, op):
        """Convert TFLite NOT_EQUAL"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized NOT_EQUAL operator is not supported yet."
            )
        return self._convert_elemwise(_op.not_equal, op)

    def _convert_logical_binary(self, relay_op, op):
        """Generic method to convert logical binary ops"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        lhs_tensor = input_tensors[0]
        lhs_expr = self.get_tensor_expr(lhs_tensor)
        rhs_tensor = input_tensors[1]
        rhs_expr = self.get_tensor_expr(rhs_tensor)
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

        data = self.get_tensor_expr(input_tensors[0])
        indices = input_tensors[1]
        indices_type = indices.tensor.Type()
        assert indices_type in (TensorType.INT32, TensorType.INT64)

        assert op.BuiltinOptionsType() == BuiltinOptions.GatherOptions
        op_options = op.BuiltinOptions()
        gather_options = GatherOptions()
        gather_options.Init(op_options.Bytes, op_options.Pos)
        axis = gather_options.Axis()

        # Check the indices are with in bounds.
        data_shape = to_int_list(self.get_tensor_shape(input_tensors[0]))
        data_dim = len(data_shape)

        axis = data_dim + axis if axis < 0 else axis
        assert axis >= 0, "Axis out of bounds"
        assert axis < data_dim, "Axis out of bounds"

        if self.has_expr(indices.tensor_idx):
            indices_expr = self.get_expr(indices.tensor_idx)
        else:
            indices_val = self.get_tensor_value(indices)
            indices_expr = self.exp_tab.new_const(
                indices_val, dtype=self.get_tensor_type_str(indices_type)
            )
            indices_shape = list(indices_val.shape)
            indices_len = len(indices_shape)

            out_shape = data_shape[:axis] + indices_shape[:] + data_shape[axis + 1 :]

            loopover = [range(s) for s in out_shape]
            for idx in list(itertools.product(*loopover)):
                real_indices = (
                    list(idx[:axis])
                    + [indices_val[idx[axis : axis + indices_len]]]
                    + list(idx[axis + indices_len :])
                )
                if np.any(np.subtract(data_shape, real_indices) < 0):
                    raise ValueError("TFLite out of bound indices are not supported.")

        # Use mode 'fast' since indices are already checked within bounds.
        out = _op.take(data, indices_expr, axis=axis, mode="fast")
        return out

    def convert_gather_nd(self, op):
        """Method to Convert TFLite GATHER_ND operator"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        for t in input_tensors:
            assert not t.qnn_params, "Quantized input is not expected."

        data = self.get_tensor_expr(input_tensors[0])
        indices = self.get_tensor_expr(input_tensors[1])

        indices_type = input_tensors[1].tensor.Type()
        assert indices_type in (TensorType.INT32, TensorType.INT64)

        indices_dims = len(_infer_shape(indices))
        indices_t = _op.transpose(indices, axes=[-1] + list(range(indices_dims - 1)))

        out = _op.gather_nd(data, indices_t)
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

        data_shape = to_int_list(self.get_tensor_shape(input_tensors[0]))
        data_dim = len(data_shape)
        stride_dim = len(stride)

        def _transform_mask(stride_dim, ellipsis_mask):
            """Handle mask inputs to create new begin, end, stride and output shape"""
            m_begin = [0] * data_dim
            m_end = [0] * data_dim
            m_stride = [0] * data_dim
            fshape_indices = []
            # Count new axis after ellipsis_mask, consider while applying ellipsis_mask.
            ellipsis_seen = False
            new_axes_after_ellipsis = 0
            for i in range(stride_dim):
                mask = 1 << i
                if ellipsis_seen and (mask & new_axis_mask) != 0:
                    new_axes_after_ellipsis += 1
                if (mask & ellipsis_mask) != 0:
                    ellipsis_seen = True
            if not ellipsis_seen:
                # Used later for extending the stride attributes in the below loop.
                ellipsis_mask |= 1 << stride_dim
                stride_dim += 1
            final_index = 0
            for index in range(stride_dim):
                mask = 1 << index
                if mask & ellipsis_mask:
                    # Identify the end index for applying ellipsis_mask
                    to_index = min(
                        ((data_dim - (stride_dim - index)) + 1 + new_axes_after_ellipsis), data_dim
                    )
                    for i in range(final_index, to_index):
                        m_begin[final_index] = 0
                        m_end[final_index] = data_shape[final_index]
                        m_stride[final_index] = 1
                        fshape_indices.append(final_index)
                        final_index += 1
                elif mask & new_axis_mask:
                    fshape_indices.append(-1)
                elif not mask & new_axis_mask:
                    if final_index == len(m_begin):
                        break
                    if mask & begin_mask:
                        m_begin[final_index] = data_shape[final_index] if stride[index] < 0 else 0
                    elif begin[index]:
                        m_begin[final_index] = begin[index]
                    if mask & end_mask:
                        m_end[final_index] = 0 if stride[index] < 0 else data_shape[final_index]
                    elif end[index]:
                        m_end[final_index] = end[index]
                    m_stride[final_index] = stride[index]
                    if mask & shrink_axis_mask:
                        # Tensorflow make axis with shrink_axis_mask as dimension 1
                        m_begin[final_index] = (
                            data_shape[final_index] + begin[index]
                            if begin[index] < 0
                            else begin[index]
                        )
                        m_end[final_index] = m_begin[final_index] + 1
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

        # Create final output shape.
        final_output = []
        final_len = len(fshape_indices)
        for gather_index in fshape_indices:
            if gather_index == -1:
                final_output.append(1)
                final_len += 1
            elif gather_index == -2:
                final_len -= 1
            else:
                final_output.append(out_shape[gather_index])

        if final_len == 0:
            return _op.squeeze(out, axis=tuple(range(len(fshape_indices))))

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
            raise tvm.error.OpNotImplemented(
                "For dims parameter of Fill operator," " only constant values are supported."
            )

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
        axis_value = self.get_tensor_value(input_tensors[1])
        axis = tuple(axis_value) if len(axis_value.shape) > 0 else tuple((axis_value.item(),))

        # Options - keep_dims (bool)
        # In case Options are not present, set keep_dims to False(default)
        if op.BuiltinOptionsType():
            assert op.BuiltinOptionsType() == BuiltinOptions.ReducerOptions
            reduce_options = ReducerOptions()
            op_options = op.BuiltinOptions()
            reduce_options.Init(op_options.Bytes, op_options.Pos)
            keep_dims = reduce_options.KeepDims()
        else:
            keep_dims = False

        if input_tensor.qnn_params:
            in_expr = _op.cast(in_expr, "int32")

        out = relay_op(in_expr, axis, keep_dims)

        # Finally if the reduce is quantized. Add a requantize at the end.
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())
        if output_tensor.qnn_params:
            out = _qnn.op.requantize(
                out,
                input_scale=input_tensor.qnn_params["scale"],
                input_zero_point=input_tensor.qnn_params["zero_point"],
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
            )

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

    def _convert_arg_min_max(self, relay_op, op):
        """Generic method converting TFLite arg_min_max"""
        try:
            from tflite.ArgMaxOptions import ArgMaxOptions
            from tflite.ArgMinOptions import ArgMinOptions
            from tflite.BuiltinOptions import BuiltinOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "two input tensor arguments expected"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "one output tensor expected"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)
        axis_tensor = input_tensors[1]
        # In Tensorflow, `axis` argument is a Tensor, not attribute. We
        # support the case where it inputs from a scalar constant.
        axis_value = self.get_tensor_value(axis_tensor)
        assert axis_value.size == 1
        axis_value = axis_value.item()

        if op.BuiltinOptionsType() == BuiltinOptions.ArgMinOptions:
            arg_min_max_options = ArgMinOptions()
        elif op.BuiltinOptionsType() == BuiltinOptions.ArgMaxOptions:
            arg_min_max_options = ArgMaxOptions()
        op_options = op.BuiltinOptions()
        arg_min_max_options.Init(op_options.Bytes, op_options.Pos)

        # set keepdims to True since tflite 1.13 removes all dims of size 1
        # WARNING: all other versions of tflite > 1.13 need keepdims=False
        out = relay_op(in_expr, axis=axis_value, keepdims=False, exclude=False)

        return out

    def convert_arg_min(self, op):
        """Convert TFLite ARG_MIN"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized ARG_MIN operator is not supported yet."
            )
        return self._convert_arg_min_max(_op.argmin, op)

    def convert_arg_max(self, op):
        """Convert TFLite ARG_MAX"""
        return self._convert_arg_min_max(_op.argmax, op)

    def convert_fully_connected(self, op):
        """Convert TFLite fully connected"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.FullyConnectedOptions import FullyConnectedOptions
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) in (2, 3), "input tensors length should be two or three"

        input_tensor = input_tensors[0]
        weight_tensor = input_tensors[1]

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type = output_tensor.tensor.Type()
        output_tensor_type_str = self.get_tensor_type_str(output_tensor_type)

        weight_tensor_shape = to_int_list(self.get_tensor_shape(weight_tensor))

        # Weight should have only 2 dimensions(TFLite convention)
        assert len(weight_tensor_shape) == 2, "Weight should be only 2-dim"

        # Input shape: [i_batch_size, ..., n_inputs]
        # Filter shape: [n_inputs, n_units]
        #
        # As we will transform Fully_Connected Input to Dense Op inputs as below
        # Dense expected Input shape: [batch_size, n_units]
        # Dense expected Weight shape: [out_dim, n_units]
        # Dense output shape: [batch_size, out_dim]
        target_shape = tuple((-1, weight_tensor_shape[1]))
        in_expr = self.get_tensor_expr(input_tensor)
        in_expr = _op.reshape(in_expr, target_shape)

        # TODO: Change the output shape calculation based on keep_dim option
        assert op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions
        op_options = op.BuiltinOptions()
        fully_connected_options = FullyConnectedOptions()
        fully_connected_options.Init(op_options.Bytes, op_options.Pos)
        fused_activation_fn = fully_connected_options.FusedActivationFunction()
        keep_num_dims = fully_connected_options.KeepNumDims()

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
        else:
            weight_value = self.get_tensor_value(weight_tensor)
            weight_expr = self.exp_tab.new_const(weight_value, dtype=weight_tensor_type_str)
        weight_shape = _infer_shape(weight_expr)

        if input_tensor.qnn_params:
            out = _qnn.op.dense(
                in_expr,
                weight_expr,
                input_zero_point=input_tensor.qnn_params["zero_point"],
                kernel_zero_point=weight_tensor.qnn_params["zero_point"],
                input_scale=input_tensor.qnn_params["scale"],
                kernel_scale=weight_tensor.qnn_params["scale"],
                units=weight_shape[0],
                out_dtype="int32",
            )
        else:
            out = _op.nn.dense(in_expr, weight_expr, units=weight_shape[0])

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            if bias_tensor.tensor_idx != -1:
                bias_tensor_type = bias_tensor.tensor.Type()
                # bias tensor type should be INT32 (quantization) or FLOAT32
                assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
                bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
                if self.has_expr(bias_tensor.tensor_idx):
                    bias_expr = self.get_expr(bias_tensor.tensor_idx)
                else:
                    bias_expr = self.exp_tab.new_const(
                        self.get_tensor_value(bias_tensor), dtype=bias_tensor_type_str
                    )
                out = _op.nn.bias_add(out, bias_expr)

        # Finally if the dense is quantized. Add a requantize at the end.
        if output_tensor.qnn_params:
            data_scale = input_tensor.qnn_params["scale"]
            weight_scale = weight_tensor.qnn_params["scale"]
            data_scale_val = get_scalar_from_constant(data_scale)
            weight_scale_val = get_scalar_from_constant(weight_scale)
            new_input_scale_val = data_scale_val * weight_scale_val
            new_input_scale = relay.const(new_input_scale_val, "float32")
            new_input_zero_point = relay.const(0, "int32")

            # Requantize
            out = _qnn.op.requantize(
                out,
                input_scale=new_input_scale,
                input_zero_point=new_input_zero_point,
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
            )

            # Call activation function
            output_scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            output_zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=output_scale_val,
                zero_point=output_zero_point_val,
                dtype=output_tensor_type_str,
            )

        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        # Change the output shape calculation based on keep_dim option
        if keep_num_dims:
            input_shape = _infer_shape(self.get_tensor_expr(input_tensor))
            output_shape = input_shape[:-1] + tuple([weight_tensor_shape[0]])
            out = _op.reshape(out, output_shape)

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

        if fused_activation_fn == ActivationFunctionType.NONE:
            return in_expr
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
            "Fused activation {} is not supported yet.".format(fused_activation_fn_str)
        )

    def convert_conv(self, op, conv_type):
        """convolution implementation."""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Conv2DOptions import Conv2DOptions
            from tflite.DepthwiseConv2DOptions import DepthwiseConv2DOptions
            from tflite.Padding import Padding
            from tflite.TensorType import TensorType
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
        if conv_type == "conv2d":
            assert op.BuiltinOptionsType() == BuiltinOptions.Conv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = Conv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
        elif conv_type == "depthwise":
            is_depthwise_conv = True
            assert op.BuiltinOptionsType() == BuiltinOptions.DepthwiseConv2DOptions
            op_options = op.BuiltinOptions()
            conv_options = DepthwiseConv2DOptions()
            conv_options.Init(op_options.Bytes, op_options.Pos)
            depth_multiplier = conv_options.DepthMultiplier()
        else:
            raise tvm.error.OpNotImplemented(
                "Operator {} is not supported for frontend TFLite.".format(conv_type)
            )

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        _, input_h, input_w, input_c = to_int_list(self.get_tensor_shape(input_tensor))

        if is_depthwise_conv:
            # TFLite depthwise convolution kernel layout is:
            # 1 KH KW C(input_c * depth_multiplier)
            _, kernel_h, kernel_w, in_channels = to_int_list(self.get_tensor_shape(weight_tensor))
            assert in_channels == input_c * depth_multiplier
        else:
            output_channels, kernel_h, kernel_w, _ = to_int_list(
                self.get_tensor_shape(weight_tensor)
            )

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {
            "kernel_size": [kernel_h, kernel_w],
            "strides": [stride_h, stride_w],
            "dilation": [dilation_h, dilation_w],
            "padding": [0, 0],
            "data_layout": "NHWC",
        }

        if is_depthwise_conv:
            params["channels"] = int(in_channels)
            params["groups"] = int(input_c)
            # If number of input channels is 1, treat as normal
            # convolution.
            params["kernel_layout"] = "HWIO" if input_c == 1 else "HWOI"
        else:
            params["channels"] = int(output_channels)
            params["kernel_layout"] = "HWIO"

        # weight tensor type should be INT8/UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        in_expr = self.get_expr(input_tensor_idx)

        # TFLite converts float32 models to float16 models by introducing
        # a Dequantize op in every op that contains a float32 values.
        # (weights, biases, and constants etc. )
        # So conv op may have weight and bias as tensors instead of values.
        if self.has_expr(weight_tensor.tensor_idx):
            weight_expr = self.get_expr(weight_tensor.tensor_idx)
            if is_depthwise_conv:
                weight_expr = _op.reshape(
                    weight_expr, (kernel_h, kernel_w, input_c, depth_multiplier)
                )
            else:
                weight_expr = _op.transpose(weight_expr, axes=(1, 2, 3, 0))
        else:
            if self.is_prefetched(weight_tensor.tensor_idx):
                weight_value = self.get_prefetched_node(weight_tensor.tensor_idx)
            else:
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
                params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]

        else:
            raise tvm.error.OpAttributeUnImplemented(
                "Padding format {} is not supported for operator Conv.".format(padding)
            )

        if input_tensor.qnn_params:
            qnn_conv2d_params = dict(params)
            qnn_conv2d_params["input_zero_point"] = input_tensor.qnn_params["zero_point"]
            qnn_conv2d_params["kernel_zero_point"] = weight_tensor.qnn_params["zero_point"]
            qnn_conv2d_params["out_dtype"] = (
                "int64" if output_tensor_type_str == "int16" else "int32"
            )
            qnn_conv2d_params["input_scale"] = input_tensor.qnn_params["scale"]
            qnn_conv2d_params["kernel_scale"] = weight_tensor.qnn_params["scale"]
            out = _qnn.op.conv2d(in_expr, weight_expr, **qnn_conv2d_params)
        else:
            out = _op.nn.conv2d(in_expr, weight_expr, **params)

        # if we have bias
        if len(input_tensors) == 3:
            bias_tensor = input_tensors[2]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (int8 qnn) or INT64 (int16 qnn) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.INT64, TensorType.FLOAT32)
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            if self.has_expr(bias_tensor.tensor_idx):
                bias_expr = self.get_expr(bias_tensor.tensor_idx)
            else:
                bias_expr = self.exp_tab.new_const(
                    self.get_tensor_value(bias_tensor), dtype=bias_tensor_type_str
                )
            channel_axis = 3
            out = _op.nn.bias_add(out, bias_expr, axis=channel_axis)

        # Handle fused activation.
        if output_tensor.qnn_params:
            # Calculate the intermediate scale and zero point of the int32 output.
            data_scale = input_tensor.qnn_params["scale"]
            data_scale_val = get_scalar_from_constant(data_scale)

            weight_scale = weight_tensor.qnn_params["scale"]
            # If weight scale is scalar, it is per-tensor quantization
            if isinstance(weight_scale, float):
                weight_scale_val = get_scalar_from_constant(weight_scale)
            else:
                weight_scale_val = get_tensor_from_constant(weight_scale)

            new_input_scale_val = data_scale_val * weight_scale_val
            new_input_scale = relay.const(new_input_scale_val, "float32")
            new_input_zero_point = relay.const(0, "int32")

            # Finally requantize
            out = _qnn.op.requantize(
                out,
                input_scale=new_input_scale,
                input_zero_point=new_input_zero_point,
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
                axis=3,
            )

            # Call activation function
            output_scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            output_zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=output_scale_val,
                zero_point=output_zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)
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
            raise tvm.error.OpNotImplemented(
                "For size_splits parameter of SPLIT_V operator, "
                "only constant values are supported."
            )
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
        input_tensor_shape = to_int_list(self.get_tensor_shape(input_tensor))
        input_tensor_rank = len(input_tensor_shape)
        for i in range(input_tensor_rank):
            if size[i] == -1:
                end[i] = input_tensor_shape[i]
            else:
                end[i] += begin[i]

        out = _op.strided_slice(in_expr, begin, end)

        return out

    def convert_select(self, op):
        """Convert TFLite SELECT"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be == 3"
        cond = self.get_tensor_expr(input_tensors[0])
        x = self.get_tensor_expr(input_tensors[1])
        y = self.get_tensor_expr(input_tensors[2])

        out = _op.where(cond, x, y)

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

    def convert_reverse_sequence(self, op):
        """Convert TFLite REVERSE_SEQUENCE"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.ReverseSequenceOptions import ReverseSequenceOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFLite does not support quantized REVERSE_SEQUENCE operator yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        in_expr = self.get_tensor_expr(input_tensors[0])
        length_expr = self.get_tensor_expr(input_tensors[1])

        assert op.BuiltinOptionsType() == BuiltinOptions.ReverseSequenceOptions
        op_options = op.BuiltinOptions()
        options = ReverseSequenceOptions()
        options.Init(op_options.Bytes, op_options.Pos)
        batch_axis = options.BatchDim()
        seq_axis = options.SeqDim()

        return _op.reverse_sequence(in_expr, length_expr, seq_axis, batch_axis)

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

        # MLIR-based converter outputs no BuiltinOptions for Cast operator. In this
        # case the output type can be derived from the Cast operator output tensor.
        # When TOCO converter is used there will be "normal" BuiltinOptions.CastOptions
        # with output type.
        if op.BuiltinOptions() is not None:
            assert op.BuiltinOptionsType() == BuiltinOptions.CastOptions
            op_options = op.BuiltinOptions()
            cast_options = CastOptions()
            cast_options.Init(op_options.Bytes, op_options.Pos)
            cast_dtype = cast_options.OutDataType()
        else:
            cast_dtype = self.get_output_tensors(op)[0].tensor.Type()

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
        """Convert TFLite TOPK_v2"""
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
            from tflite.Padding import Padding
            from tflite.Pool2DOptions import Pool2DOptions
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

        params = {
            "pool_size": (filter_h, filter_w),
            "strides": (stride_h, stride_w),
            "padding": [0, 0],
            "layout": "NHWC",
        }

        in_expr = self.get_expr(input_tensor_idx)

        _, input_h, input_w, _ = to_int_list(self.get_tensor_shape(input_tensor))

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, filter_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, filter_w, stride_w)
            params["padding"] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            raise tvm.error.OpAttributeUnImplemented(
                "Padding format {} for operator Pool2D is not supported.".format(padding)
            )

        if pool_type == "average":
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(input_tensor, output_tensor), (
                    "TFLite avg_pool2dreshape requires input and output scale"
                    "and zero points to be equal"
                )
                out = _op.cast(in_expr, dtype="int32")
                out = _op.nn.avg_pool2d(out, **params)
                out = _op.cast(out, dtype=output_tensor_type_str)
            else:
                out = _op.nn.avg_pool2d(in_expr, **params)
        elif pool_type == "max":
            if input_tensor.qnn_params:
                assert self.has_same_qnn_params(
                    input_tensor, output_tensor
                ), "qnn.op.max_pool2d requires input and output qnn params to be same"
            out = _op.nn.max_pool2d(in_expr, **params)
        elif pool_type == "l2":
            # L2_POOL_2D is equivalent to square_root(avg_pool(square(in_data)))
            # TFLite does not have support for quantised L2_POOL_2D op.
            assert (
                not input_tensor.qnn_params
            ), "As TFLite does not have support for quantized L2_POOL_2D, \
                Quantized input is not expected."
            exp_type = self.get_tensor_type_str(output_tensor.tensor.Type())
            square_exp = _op.power(in_expr, relay.const(2, exp_type))
            avg_pool_exp = _op.nn.avg_pool2d(square_exp, **params)
            out = _op.sqrt(avg_pool_exp)
        else:
            raise tvm.error.OpNotImplemented(
                "Operator {} is not supported for frontend TFLite.".format(pool_type + " pool")
            )

        # Handle fused activations
        if output_tensor.qnn_params:
            scale_val = get_scalar_from_constant(output_tensor.qnn_params["scale"])
            zero_point_val = get_scalar_from_constant(output_tensor.qnn_params["zero_point"])
            out = self.convert_qnn_fused_activation_function(
                expr=out,
                fused_activation_fn=fused_activation_fn,
                scale=scale_val,
                zero_point=zero_point_val,
                dtype=output_tensor_type_str,
            )
        else:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def convert_pad(self, op):
        """Convert TFLite PAD/PADV2 \
           TFLite treats PAD and PADV2 operators identically"""

        input_tensors = self.get_input_tensors(op)

        # TFLite PAD/PADV2 only supports CONSTANT mode
        assert (
            len(input_tensors) == 2 or len(input_tensors) == 3
        ), "input tensor's length should be 2 for PAD and 3 for PADV2"

        if len(input_tensors) == 3:
            assert (
                input_tensors[0].tensor.Type() == input_tensors[2].tensor.Type()
            ), "constant_values tensor must be of same type as input tensor"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])

        # convert list of lists to tuple of tuples
        paddings = tuple(tuple(l) for l in pad_list)

        # Set the pad value, by default 0, unless constant_values parameter is provided
        pad_value = 0

        if input_tensor.qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(
                input_tensor, output_tensor
            ), "TFLite PADV2 requires input and output scale and zero points to be equal"

            # The pad value for quantized pad is the input zero point by default.
            pad_value = float(input_tensor.qnn_params["zero_point"].data.numpy())

        if len(input_tensors) == 3:
            pad_value = self.get_tensor_value(input_tensors[2])
            if isinstance(pad_value, np.ndarray):
                pad_value = pad_value.tolist()
            if isinstance(pad_value, list):
                assert len(pad_value) == 1, "Only one constant value is expected."
                pad_value = pad_value[0]
            if input_tensor.qnn_params:
                # Check that input tensor and constant_values have same qnn params.
                assert self.has_same_qnn_params(
                    input_tensor, input_tensors[2]
                ), "TFLite PADV2 requires input and constant_values tensors' \
                        scale and zero points to be equal"

        out = _op.nn.pad(in_expr, pad_width=paddings, pad_value=pad_value)
        return out

    def convert_floor_div(self, op):
        """Convert TFLite FLOOR_DIV"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized FLOOR DIV operator is not supported yet."
            )
        return self._convert_elemwise(_op.floor_divide, op)

    def convert_floor_mod(self, op):
        """Convert TFLite FLOOR_MOD"""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized FLOOR MOD operator is not supported yet."
            )
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
                "TFlite quantized MIRROR_PAD operator is not supported yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        # tensor
        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # paddings
        pad_list = self.get_tensor_value(input_tensors[1])
        # convert list of lists to tuple of tuples
        paddings = tuple(tuple(l.astype(np.int32)) for l in pad_list)

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
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"

        if input_tensors[0].qnn_params:
            output_tensor = output_tensors[0]
            assert self.has_same_qnn_params(
                input_tensors[0], output_tensor
            ), "TFLite pack requires input and output scale and zero points to be equal"

            for input_tensor in input_tensors:
                assert self.has_same_qnn_params(
                    input_tensors[0], input_tensor
                ), "TFLite pack requires all input tensors to have same scale and zero point"

        assert op.BuiltinOptionsType() == BuiltinOptions.PackOptions
        op_options = op.BuiltinOptions()
        pack_options = PackOptions()
        pack_options.Init(op_options.Bytes, op_options.Pos)
        pack_axis = pack_options.Axis()
        pack_values_count = pack_options.ValuesCount()
        assert len(input_tensors) == pack_values_count, "Discordance in input values count"

        in_exprs = [self.get_tensor_expr(_) for _ in input_tensors]
        in_exprs_reshaped = [_op.expand_dims(_, axis=pack_axis, num_newaxis=1) for _ in in_exprs]
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
        # We have to do 'squeeze' along the split axis.
        # Relay expects squeeze_axis to be List.
        squeeze_axis = [unpack_axis]

        # Relay doesn't like TupleWrapper of 1 element so we isolate the case of unpacking
        # a tensor by an axis with len(axis) == 1. For reference see convert_split().
        # Such unpacking will result in the same tensor so we omit 'split' and only squeeze
        # along the axis of dim == 1.
        if num_unpacks == 1:
            squeezed = _op.squeeze(in_expr, axis=squeeze_axis)
            if isinstance(squeezed, _expr.TupleWrapper):
                squeezed = squeezed[0]
        else:
            splitted = _op.split(in_expr, indices_or_sections=num_unpacks, axis=unpack_axis)
            squeezed = _expr.TupleWrapper(
                _expr.Tuple(
                    [_op.squeeze(split_item, axis=squeeze_axis) for split_item in splitted]
                ),
                len(splitted),
            )

        return squeezed

    def convert_unidirectional_sequence_lstm(self, op):
        """Long Short Term Memory for TFLite implementation."""
        if self.is_quantized(op):
            raise tvm.error.OpNotImplemented(
                "TFlite quantized UNIDIRECTIONALSEQUENCELSTM operator is not supported yet."
            )

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 24, "input tensors length should be == 24"

        # Extract input tensor from saved model
        input_tensor = input_tensors[0]

        # Extract tensors from input tensors from saved model
        # Input weights
        input_input_weights = input_tensors[1]
        input_forget_weights = input_tensors[2]
        input_cell_weights = input_tensors[3]
        input_output_weights = input_tensors[4]
        # Recurrent weights
        recurrent_input_weights = input_tensors[5]
        recurrent_forget_weights = input_tensors[6]
        recurrent_cell_weights = input_tensors[7]
        recurrent_output_weights = input_tensors[8]
        # inputs 9, 10, 11, 16, 17, 20, 21, 22, 23 are not occupied
        # there locations are -1 in the flatbuffer
        # Bias weights
        input_gate_bias = input_tensors[12]
        forget_gate_bias = input_tensors[13]
        cell_gate_bias = input_tensors[14]
        output_gate_bias = input_tensors[15]

        # State input
        output_state_in = input_tensors[18]
        cell_state_in = input_tensors[19]

        # Extract output tensor from saved model
        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        X_steps = self.unbind(input_tensor, axis=1)
        weights_dict = {}

        # hidden_state_weights is equivalent to output_state_in in tflite model
        out_state_in_shape = tuple(self.get_tensor_shape(output_state_in))
        out_state_in_dtype = self.get_tensor_type_str(output_state_in.tensor.Type())
        out_state_in_expr = _op.zeros(out_state_in_shape, dtype=out_state_in_dtype)
        weights_dict["hidden_state"] = _op.split(out_state_in_expr, 1)[0]

        # cell_state_weights is equivalent to output_state_in tflite model
        cell_state_in_shape = tuple(self.get_tensor_shape(cell_state_in))
        cell_state_in_dtype = self.get_tensor_type_str(cell_state_in.tensor.Type())
        cell_state_in_expr = _op.zeros(cell_state_in_shape, dtype=cell_state_in_dtype)
        weights_dict["cell_state"] = _op.split(cell_state_in_expr, 1)[0]

        # Process weight matrix of input: w_inp
        # Concatenate of [input_input_weight, input_forget_weights,
        # input_cell_weights, input_output_weights]
        input_input_weights_default_values = self.get_tensor_value(input_input_weights)
        input_input_weights_op = _op.split(
            _op.const(input_input_weights_default_values.tolist()), 1
        )
        input_output_weights_default_values = self.get_tensor_value(input_output_weights)
        input_output_weights_op = _op.split(
            _op.const(input_output_weights_default_values.tolist()), 1
        )
        input_forget_weights_default_values = self.get_tensor_value(input_forget_weights)
        input_forget_weights_op = _op.split(
            _op.const(input_forget_weights_default_values.tolist()), 1
        )
        input_cell_weights_default_values = self.get_tensor_value(input_cell_weights)
        input_cell_weights_op = _op.split(_op.const(input_cell_weights_default_values.tolist()), 1)
        weights_dict["w_inp"] = _op.concatenate(
            [
                _op.squeeze(input_input_weights_op[0]),
                _op.squeeze(input_forget_weights_op[0]),
                _op.squeeze(input_cell_weights_op[0]),
                _op.squeeze(input_output_weights_op[0]),
            ],
            axis=0,
        )

        # Process weight matrix of hidden state:
        # w_hid to support lstm_cell function. Not used in tflite
        recurrent_input_weights_values = self.get_tensor_value(recurrent_input_weights)
        recurrent_input_weights_op = _op.split(
            _op.const(recurrent_input_weights_values.tolist()), 1
        )
        recurrent_output_weights_values = self.get_tensor_value(recurrent_output_weights)
        recurrent_output_weights_op = _op.split(
            _op.const(recurrent_output_weights_values.tolist()), 1
        )
        recurrent_forget_weights_values = self.get_tensor_value(recurrent_forget_weights)
        recurrent_forget_weights_op = _op.split(
            _op.const(recurrent_forget_weights_values.tolist()), 1
        )
        recurrent_cell_weights_values = self.get_tensor_value(recurrent_cell_weights)
        recurrent_cell_weights_op = _op.split(_op.const(recurrent_cell_weights_values.tolist()), 1)
        weights_dict["w_hid"] = _op.concatenate(
            [
                recurrent_input_weights_op[0],
                recurrent_forget_weights_op[0],
                recurrent_cell_weights_op[0],
                recurrent_output_weights_op[0],
            ],
            axis=0,
        )

        # Process weight matrix of bias: b_inp
        input_gate_bias_values = self.get_tensor_value(input_gate_bias)
        input_gate_bias_op = _op.split(_op.const(input_gate_bias_values.tolist()), 1)
        output_gate_bias_values = self.get_tensor_value(output_gate_bias)
        output_gate_bias_op = _op.split(_op.const(output_gate_bias_values.tolist()), 1)
        forget_gate_bias_values = self.get_tensor_value(forget_gate_bias)
        forget_gate_bias_op = _op.split(_op.const(forget_gate_bias_values.tolist()), 1)
        cell_gate_bias_values = self.get_tensor_value(cell_gate_bias)
        cell_gate_bias_op = _op.split(_op.const(cell_gate_bias_values.tolist()), 1)
        weights_dict["b_inp"] = _op.concatenate(
            [
                input_gate_bias_op[0],
                forget_gate_bias_op[0],
                cell_gate_bias_op[0],
                output_gate_bias_op[0],
            ],
            axis=0,
        )

        # Process weight matrix of hidden bias:
        # b_hid (with the same shape as b_inp)
        gate_bias_dtype = self.get_tensor_type_str(input_gate_bias.tensor.Type())
        weights_dict["b_hid"] = _op.split(
            _op.const(
                np.zeros(_infer_shape(weights_dict["b_inp"]), dtype=gate_bias_dtype),
                dtype=gate_bias_dtype,
            ),
            1,
        )[0]

        outputs, _, _ = lstm_cell(input_seqs=X_steps, **weights_dict)

        output = _op.stack(outputs, axis=1)
        return output

    def convert_batch_to_space_nd(self, op):
        """batch_to_space_nd implementation."""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        block_shape = list(self.get_tensor_value(input_tensors[1]))
        crops = self.get_tensor_value(input_tensors[2]).tolist()

        out = _op.nn.batch_to_space_nd(in_expr, block_shape, crops)

        return out

    def convert_space_to_batch_nd(self, op):
        """space_to_batch_nd implementation."""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 3, "input tensors length should be 3"

        input_tensor = input_tensors[0]
        input_tensor_idx = input_tensor.tensor_idx
        in_expr = self.get_expr(input_tensor_idx)

        block_shape = list(self.get_tensor_value(input_tensors[1]))
        paddings = self.get_tensor_value(input_tensors[2]).tolist()

        out = _op.nn.space_to_batch_nd(in_expr, block_shape, paddings)

        return out

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
        out = _op.nn.depth_to_space(in_expr, block_size, layout="NHWC")

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
        out = _op.nn.space_to_depth(in_expr, block_size, layout="NHWC")

        return out

    def convert_sparse_to_dense(self, op):
        """Convert TFLite SPARSE_TO_DENSE"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 4, "input tensors length should be 4"

        indices, values = input_tensors[0], input_tensors[2]
        default_value = input_tensors[3]
        output_shape = input_tensors[1]

        for t in input_tensors:
            assert not t.qnn_params, "Quantized input is not expected."

        for t in [indices, output_shape]:
            t_type = t.tensor.Type()
            assert t_type in (TensorType.INT32, TensorType.INT64)

        out = _op.sparse_to_dense(
            self.get_tensor_expr(indices),
            list(self.get_tensor_value(output_shape)),
            self.get_tensor_expr(values),
            self.get_tensor_expr(default_value),
        )

        return out

    def convert_prelu(self, op):
        """Convert TFLite PReLU"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        input_tensor = input_tensors[0]
        alpha_tensor = input_tensors[1]
        if self.has_expr(alpha_tensor.tensor_idx):
            alpha_expr = self.get_expr(alpha_tensor.tensor_idx)
        else:
            alpha_tensor_type = alpha_tensor.tensor.Type()
            alpha_tensor_type_str = self.get_tensor_type_str(alpha_tensor_type)
            alpha_expr = self.exp_tab.new_const(
                self.get_tensor_value(alpha_tensor), dtype=alpha_tensor_type_str
            )
        in_expr = self.get_expr(input_tensor.tensor_idx)
        data_shape = to_int_list(self.get_tensor_shape(input_tensor))

        alpha_expr = _op.broadcast_to(alpha_expr, data_shape)
        alpha_expr = _op.reshape(alpha_expr, [-1])
        out = _op.nn.prelu(_op.reshape(in_expr, [-1]), alpha_expr, axis=0)
        out = _op.reshape(out, data_shape)
        return out

    def convert_transpose_conv(self, op):
        """Convert TFLite TRANSPOSE_CONV"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.Padding import Padding
            from tflite.TensorType import TensorType
            from tflite.TransposeConvOptions import TransposeConvOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) >= 3, "input tensors length should be >= 3"

        # Input (data) Tensor. NHWC layout
        input_tensor = input_tensors[2]
        _, _, _, input_c = to_int_list(self.get_tensor_shape(input_tensor))
        # Weights tensor. TFLite uses OHWI layout
        weights_tensor = input_tensors[1]
        out_channels, kernel_h, kernel_w, in_channels = to_int_list(
            self.get_tensor_shape(weights_tensor)
        )

        assert (
            input_c == in_channels
        ), "Input channel in the filter should match to channel in the input"
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
        assert padding in (
            Padding.VALID,
            Padding.SAME,
        ), "Padding format {} is not supported for operator TRANSPOSE_CONV".format(padding)

        # Data
        in_expr = self.get_expr(input_tensor.tensor_idx)

        # Weights
        weights_tensor_type = weights_tensor.tensor.Type()
        # weights tensor type should be UINT8 (quantization) or FLOAT32
        assert weights_tensor_type in (TensorType.INT8, TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weights_tensor_type)

        if self.has_expr(weights_tensor.tensor_idx):
            weight_expr_iohw = self.get_expr(weights_tensor.tensor_idx)
            weight_expr_iohw = _op.transpose(weight_expr_iohw, axes=(3, 0, 1, 2))
        else:
            weight_value_ohwi = self.get_tensor_value(weights_tensor)
            # Relay kernel_layout should be OIHW
            # Relay weights layout should be different from kernel_layout - it should be IOHW
            weight_value_iohw = np.transpose(weight_value_ohwi, (3, 0, 1, 2))
            weight_expr_iohw = self.exp_tab.new_const(
                weight_value_iohw, dtype=weight_tensor_type_str
            )

        # Output shape value
        output_shape_value = self.get_tensor_value(output_shape_tensor)
        # Relay expects filter output channel to match to output tensor channel.
        assert (
            out_channels == output_shape_value[3]
        ), "Output channel in the filter should match to channel in the output_shape"

        if padding == Padding.SAME:
            output_h, output_w = output_shape_value[1], output_shape_value[2]
            pad_top, pad_bottom = get_pad_value(output_h, kernel_h, stride_h)
            pad_left, pad_right = get_pad_value(output_w, kernel_w, stride_w)
            padding = (pad_top, pad_left, pad_bottom, pad_right)
        else:
            padding = (0, 0, 0, 0)

        if input_tensor.qnn_params:
            input_zero_point = input_tensor.qnn_params["zero_point"]
            kernel_zero_point = weights_tensor.qnn_params["zero_point"]
            input_scale = input_tensor.qnn_params["scale"]
            kernel_scale = weights_tensor.qnn_params["scale"]
            out = _qnn.op.conv2d_transpose(
                in_expr,
                weight_expr_iohw,
                input_zero_point,
                kernel_zero_point,
                input_scale,
                kernel_scale,
                strides=(stride_h, stride_w),
                padding=padding,
                channels=int(out_channels),
                kernel_size=(int(kernel_h), int(kernel_w)),
                data_layout="NHWC",
                kernel_layout="IOHW",
                out_dtype="int32",
            )
        else:
            out = _op.nn.conv2d_transpose(
                in_expr,
                weight_expr_iohw,
                strides=(stride_h, stride_w),
                padding=padding,
                channels=int(out_channels),
                kernel_size=(int(kernel_h), int(kernel_w)),
                data_layout="NHWC",
                kernel_layout="IOHW",
                out_dtype=output_tensor_type_str,
            )

        # Checking if there is a fused bias
        if len(input_tensors) == 4:
            bias_tensor = input_tensors[3]
            bias_tensor_type = bias_tensor.tensor.Type()
            # bias tensor type should be INT32 (quantization) or FLOAT32
            assert bias_tensor_type in (TensorType.INT32, TensorType.FLOAT32)
            bias_tensor_type_str = self.get_tensor_type_str(bias_tensor_type)
            if self.has_expr(bias_tensor.tensor_idx):
                bias_expr = self.get_expr(bias_tensor.tensor_idx)
            else:
                bias_expr = self.exp_tab.new_const(
                    self.get_tensor_value(bias_tensor), dtype=bias_tensor_type_str
                )
            channel_axis = 3
            out = _op.nn.bias_add(out, bias_expr, axis=channel_axis)

        if output_tensor.qnn_params:
            # Calculate the intermediate scale and zero point of the int32 output.
            data_scale = input_tensor.qnn_params["scale"]
            data_scale_val = get_scalar_from_constant(data_scale)

            weight_scale = weights_tensor.qnn_params["scale"]
            # If weight scale is scalar, it is per-tensor quantization
            if isinstance(weight_scale, float):
                weight_scale_val = get_scalar_from_constant(weight_scale)
            else:
                weight_scale_val = get_tensor_from_constant(weight_scale)

            new_input_scale_val = data_scale_val * weight_scale_val
            new_input_scale = relay.const(new_input_scale_val, "float32")
            new_input_zero_point = relay.const(0, "int32")

            out = _qnn.op.requantize(
                out,
                input_scale=new_input_scale,
                input_zero_point=new_input_zero_point,
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
                axis=3,
            )
        return out

    def convert_quantize(self, op):
        """Convert TFLite Quantize"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]
        input_tensor_type_str = self.get_tensor_type_str(input_tensor.tensor.Type())
        in_expr = self.get_tensor_expr(input_tensor)

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]
        output_tensor_type_str = self.get_tensor_type_str(output_tensor.tensor.Type())

        # The output must be quantized
        assert output_tensor.qnn_params

        # TFLite Quantize op can also act as Requantize op
        if input_tensor_type_str == "float32":
            out = self.quantize(in_expr, output_tensor)
        else:
            out = _qnn.op.requantize(
                in_expr,
                input_scale=input_tensor.qnn_params["scale"],
                input_zero_point=input_tensor.qnn_params["zero_point"],
                output_scale=output_tensor.qnn_params["scale"],
                output_zero_point=output_tensor.qnn_params["zero_point"],
                out_dtype=output_tensor_type_str,
            )
        return out

    def convert_dequantize(self, op):
        """Convert TFLite Dequantize"""
        try:
            from tflite.TensorType import TensorType
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"
        input_tensor = input_tensors[0]

        if input_tensor.tensor.Type() == TensorType.FLOAT16:
            dtype = self.get_tensor_type_str(input_tensor.tensor.Type())
            input_value = self.get_tensor_value(input_tensor)
            in_expr = self.exp_tab.new_const(input_value, dtype=dtype)
            out = relay.cast(in_expr, dtype="float32")
            return out

        in_expr = self.get_expr(input_tensor.tensor_idx)

        # The input must be quantized
        assert input_tensor.qnn_params
        # Dequantize the input.
        out = self.dequantize(in_expr, input_tensor)

        return out

    def convert_detection_postprocess(self, op):
        """Convert TFLite_Detection_PostProcess"""
        flexbuffer = op.CustomOptionsAsNumpy().tobytes()
        custom_options = FlexBufferDecoder(flexbuffer).decode()

        if "use_regular_nms" in custom_options:
            if custom_options["use_regular_nms"]:
                raise tvm.error.OpAttributeUnImplemented(
                    "use_regular_nms=True is not yet supported for operator {}.".format(
                        "TFLite_Detection_PostProcess"
                    )
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
            loc_prob = _qnn.op.dequantize(
                data=loc_prob,
                input_scale=inputs[0].qnn_params["scale"],
                input_zero_point=inputs[0].qnn_params["zero_point"],
            )
        if inputs[1].qnn_params:
            cls_pred = _qnn.op.dequantize(
                data=cls_pred,
                input_scale=inputs[1].qnn_params["scale"],
                input_zero_point=inputs[1].qnn_params["zero_point"],
            )
        if inputs[2].qnn_params:
            anchor_expr = _qnn.op.dequantize(
                data=anchor_expr,
                input_scale=inputs[2].qnn_params["scale"],
                input_zero_point=inputs[2].qnn_params["zero_point"],
            )

        # reshape the cls_pred and loc_prob tensors so
        # they can be consumed by multibox_transform_loc
        cls_pred = _op.transpose(cls_pred, [0, 2, 1])
        # loc_prob coords are in yxhw format
        # need to convert to xywh
        loc_coords = _op.split(loc_prob, 4, axis=2)
        loc_prob = _op.concatenate(
            [loc_coords[1], loc_coords[0], loc_coords[3], loc_coords[2]], axis=2
        )
        loc_prob = _op.reshape(loc_prob, [batch_size, anchor_boxes * 4])

        # anchor coords are in yxhw format
        # need to convert to ltrb
        anchor_coords = _op.split(anchor_expr, 4, axis=1)
        anchor_y = anchor_coords[0]
        anchor_x = anchor_coords[1]
        anchor_h = anchor_coords[2]
        anchor_w = anchor_coords[3]
        plus_half = _expr.const(0.5, dtype="float32")
        minus_half = _expr.const(-0.5, dtype="float32")
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
        non_max_suppression_attrs["force_suppress"] = True
        non_max_suppression_attrs["top_k"] = anchor_boxes
        non_max_suppression_attrs["max_output_size"] = custom_options["max_detections"]
        non_max_suppression_attrs["invalid_to_bottom"] = False

        ret = _op.vision.multibox_transform_loc(
            cls_pred, loc_prob, anchor_expr, **multibox_transform_loc_attrs
        )
        ret = _op.vision.non_max_suppression(ret[0], ret[1], ret[1], **non_max_suppression_attrs)
        ret = _op.vision.get_valid_counts(ret, 0)
        valid_count = ret[0]
        # keep only the top 'max_detections' rows
        ret = _op.strided_slice(
            ret[1], [0, 0, 0], [batch_size, custom_options["max_detections"], 6]
        )
        # the output needs some reshaping to match tflite
        ret = _op.split(ret, 6, axis=2)
        cls_ids = _op.reshape(ret[0], [batch_size, -1])
        scores = _op.reshape(ret[1], [batch_size, -1])
        boxes = _op.concatenate([ret[3], ret[2], ret[5], ret[4]], axis=2)
        ret = _expr.TupleWrapper(_expr.Tuple([boxes, cls_ids, scores, valid_count]), size=4)
        return ret

    def convert_nms_v5(self, op):
        """Convert TFLite NonMaxSuppressionV5"""
        # https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/non-max-suppression-v5

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 6, "input tensor length should be 6"
        boxes = self.get_expr(input_tensors[0].tensor_idx)
        scores = self.get_expr(input_tensors[1].tensor_idx)
        max_output_size = self.get_tensor_value(input_tensors[2])
        iou_threshold = self.get_tensor_value(input_tensors[3])
        score_threshold = self.get_tensor_value(input_tensors[4])
        soft_nms_sigma = self.get_tensor_value(input_tensors[5])

        if isinstance(max_output_size, np.ndarray):
            assert max_output_size.size == 1, "only one value is expected."
            max_output_size = int(max_output_size)

        if isinstance(iou_threshold, np.ndarray):
            assert iou_threshold.size == 1, "only one value is expected."
            iou_threshold = float(iou_threshold)

        if isinstance(score_threshold, np.ndarray):
            assert score_threshold.size == 1, "only one value is expected."
            score_threshold = float(score_threshold)

        if isinstance(soft_nms_sigma, np.ndarray):
            assert soft_nms_sigma.size == 1, "only one value is expected."
            soft_nms_sigma = float(soft_nms_sigma)
        if soft_nms_sigma != 0.0:
            raise tvm.error.OpNotImplemented(
                "It is soft_nms when soft_nms_sigma != 0, which is not supported!"
            )

        scores_expand = _op.expand_dims(scores, axis=-1, num_newaxis=1)
        data = _op.concatenate([scores_expand, boxes], -1)
        data = _op.expand_dims(data, axis=0, num_newaxis=1)

        count, data, indices = _op.vision.get_valid_counts(
            data, score_threshold=score_threshold, id_index=-1, score_index=0
        )

        nms_ret = _op.vision.non_max_suppression(
            data=data,
            valid_count=count,
            indices=indices,
            max_output_size=max_output_size,
            iou_threshold=iou_threshold,
            force_suppress=True,
            top_k=-1,
            coord_start=1,
            score_index=0,
            id_index=-1,
            return_indices=True,
            invalid_to_bottom=False,
        )

        selected_indices = _op.squeeze(nms_ret[0], axis=[0])
        selected_indices = _op.strided_slice(selected_indices, [0], [max_output_size])
        valide_num = _op.squeeze(nms_ret[1], axis=[1])
        selected_scores = _op.take(scores, selected_indices, axis=0)
        out = _expr.TupleWrapper(_expr.Tuple([selected_indices, selected_scores, valide_num]), 3)
        return out

    def convert_expand_dims(self, op):
        """Convert TFLite EXPAND_DIMS"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensors length should be 2"

        if input_tensors[0].qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            assert self.has_same_qnn_params(
                input_tensors[0], output_tensors[0]
            ), "TFLite EXPAND_DIMS requires input and output tensors' \
                    scale and zero points to be equal"

        input_expr = self.get_tensor_expr(input_tensors[0])
        axis = self.get_tensor_value(input_tensors[1])
        if isinstance(axis, np.ndarray):
            assert axis.size == 1, "only one value is expected."
            axis = int(axis)

        ndims = len(input_tensors[0].tensor.ShapeAsNumpy())
        assert -1 - ndims <= axis <= ndims, "axis out of range"

        out = _op.expand_dims(input_expr, axis, 1)

        return out

    def convert_one_hot(self, op):
        """Convert TFLite ONE_HOT"""
        try:
            from tflite.BuiltinOptions import BuiltinOptions
            from tflite.OneHotOptions import OneHotOptions
        except ImportError:
            raise ImportError("The tflite package must be installed")

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 4, "Input tensor's length should be 4"

        # Ensuring input isn't quantized
        assert all(not i.qnn_params for i in input_tensors), "Quantized input is not expected."

        # TFlite ONE_HOT requires both on_value
        # and off_value, making dtype redundant.
        indices = input_tensors[0]
        depth = input_tensors[1]
        on_value = input_tensors[2]
        off_value = input_tensors[3]

        assert (
            on_value.tensor.Type() == off_value.tensor.Type()
        ), "on_value and off_value should be the same type"

        # Getting relay expr
        indices_expr = self.get_expr(indices.tensor_idx)
        on_value_expr = self.get_expr(on_value.tensor_idx)
        off_value_expr = self.get_expr(off_value.tensor_idx)

        # Getting depth value
        depth = self.get_tensor_value(depth)
        if isinstance(depth, np.ndarray):
            depth = int(depth)

        # Getting Axis from Option (Attributes)
        assert op.BuiltinOptionsType() == BuiltinOptions.OneHotOptions
        op_options = op.BuiltinOptions()
        one_hot_options = OneHotOptions()
        one_hot_options.Init(op_options.Bytes, op_options.Pos)
        axis = one_hot_options.Axis()

        # Setting dtype
        dtype = self.get_tensor_type_str(on_value.tensor.Type())

        out = _op.one_hot(indices_expr, on_value_expr, off_value_expr, depth, axis, dtype)

        return out

    def convert_reverse_v2(self, op):
        """Convert TFLite REVERSE_V2"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensor's length should be 2"

        input_expr = self.get_expr(input_tensors[0].tensor_idx)

        # Getting axis value
        axis = self.get_tensor_value(input_tensors[1])
        if isinstance(axis, np.ndarray):
            assert len(axis) == 1, "TFLite does not support multi-axis yet"
            axis = int(axis)

        out = _op.reverse(input_expr, axis)
        return out

    def convert_matrix_set_diag(self, op):
        """Convert TFLite MATRIX_SET_DIAG"""

        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 2, "input tensor's length should be 2"

        assert (
            input_tensors[0].tensor.Type() == input_tensors[1].tensor.Type()
        ), "input and diagonal should be the same type of tensors"

        if input_tensors[0].qnn_params:
            # Check that input and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            assert self.has_same_qnn_params(
                input_tensors[0], output_tensors[0]
            ), "TFLite MATRIX_SET_DIAG requires input and output tensors' \
                    scale and zero points to be equal"

            # Check that input and diagonal tensor have same qnn params.
            assert self.has_same_qnn_params(
                input_tensors[0], input_tensors[1]
            ), "TFLite MATRIX_SET_DIAG requires input and diagonal tensors' \
                    scale and zero points to be equal"

        input_expr = self.get_tensor_expr(input_tensors[0])
        diagonal_expr = self.get_tensor_expr(input_tensors[1])

        out = _op.matrix_set_diag(input_expr, diagonal_expr)
        return out

    def convert_matrix_diag(self, op):
        """Convert TFLite MATRIX_DIAG"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensor's length should be 1"

        diagonal = input_tensors[0]

        if diagonal.qnn_params:
            # Check that diagonal and output tensor have same qnn params.
            output_tensors = self.get_output_tensors(op)
            assert self.has_same_qnn_params(
                diagonal, output_tensors[0]
            ), "TFLite MATRIX_DIAG requires diagonal and output tensors' \
                    scale and zero points to be equal"

        shape = to_int_list(self.get_tensor_shape(diagonal))
        shape = np.append(shape, shape[-1])
        dtype = self.get_tensor_type_str(diagonal.tensor.Type())

        input_expr = _op.zeros(tuple(shape), dtype)
        diagonal_expr = self.get_tensor_expr(diagonal)

        out = _op.matrix_set_diag(input_expr, diagonal_expr)
        return out

    def convert_densify(self, op):
        """Convert TFLite DENSIFY"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        output_tensors = self.get_output_tensors(op)
        assert len(output_tensors) == 1, "output tensors length should be 1"
        output_tensor = output_tensors[0]

        sparse_weight_tensor = input_tensors[0]
        sparse_weight_tensor_type_str = self.get_tensor_type_str(sparse_weight_tensor.tensor.Type())

        # NOTE: With current implementation in TFLite, Densify Op does not need to be present
        # in runtime.
        # TODO(ANSHUMAN87): we need to use the sparse_indices output
        # from below function and use that in sparse_to_dense Op.
        # Once the stack corruption issue is resolved in sparse_to_dense Op.
        _, dense_weight = prepare_dense_matrix_from_sparse(
            sparse_weight_tensor.tensor,
            self.get_tensor_value(sparse_weight_tensor, is_sparse=True),
            sparse_weight_tensor_type_str,
        )

        self.set_prefetched_node(output_tensor.tensor_idx, dense_weight)

    def convert_fake_quant(self, op):
        """Convert TFLite FAKE_QUANT"""
        input_tensors = self.get_input_tensors(op)
        assert len(input_tensors) == 1, "input tensors length should be 1"

        input_tensor = input_tensors[0]
        in_expr = self.get_expr(input_tensor.tensor_idx)

        from tflite.BuiltinOptions import BuiltinOptions
        from tflite.FakeQuantOptions import FakeQuantOptions

        assert op.BuiltinOptionsType() == BuiltinOptions.FakeQuantOptions

        op_options = op.BuiltinOptions()
        fake_quant_options = FakeQuantOptions()
        fake_quant_options.Init(op_options.Bytes, op_options.Pos)

        opt_min = fake_quant_options.Min()
        opt_max = fake_quant_options.Max()
        narrow_range = fake_quant_options.NarrowRange()
        num_bits = fake_quant_options.NumBits()

        assert 2 <= num_bits <= 16

        quant_min = 1 if narrow_range else 0
        quant_max = (1 << num_bits) - 1
        scale = (opt_max - opt_min) / (quant_max - quant_min)

        zero_point_from_min = quant_min - opt_min / scale
        if zero_point_from_min <= quant_min:
            nudged_zero_point = quant_min
        elif zero_point_from_min >= quant_max:
            nudged_zero_point = quant_max
        else:
            nudged_zero_point = round(zero_point_from_min)

        nudged_min = (quant_min - nudged_zero_point) * scale
        nudged_max = (quant_max - nudged_zero_point) * scale

        nudged_min_expr = _op.const(nudged_min)
        clamped = _op.clip(in_expr, nudged_min, nudged_max)
        clamped_shifted = _op.subtract(clamped, nudged_min_expr)

        half = _op.const(0.5)
        one = _op.const(1.0)
        scale_expr = _op.const(scale)
        inv_scale = _op.divide(one, scale_expr)
        rounded = _op.floor(_op.add(_op.multiply(clamped_shifted, inv_scale), half))
        return _op.add(_op.multiply(rounded, scale_expr), nudged_min_expr)

    def get_expr(self, input_tensor_idx):
        return self.exp_tab.get_expr(get_tensor_name(self.subgraph, input_tensor_idx))

    def has_expr(self, input_tensor_idx):
        return self.exp_tab.has_expr(get_tensor_name(self.subgraph, input_tensor_idx))

    def is_prefetched(self, input_tensor_idx):
        return (
            self.prefetched_nodes.get(get_tensor_name(self.subgraph, input_tensor_idx)) is not None
        )

    def set_prefetched_node(self, input_tensor_idx, value):
        self.prefetched_nodes[get_tensor_name(self.subgraph, input_tensor_idx)] = value

    def get_prefetched_node(self, input_tensor_idx):
        return self.prefetched_nodes[get_tensor_name(self.subgraph, input_tensor_idx)]

    def get_tensor_expr(self, tensor, is_sparse=False):
        """Return the Relay expr for tensor."""
        if self.has_expr(tensor.tensor_idx):
            expr = self.get_expr(tensor.tensor_idx)
        else:
            type_str = self.get_tensor_type_str(tensor.tensor.Type())
            expr = self.exp_tab.new_const(self.get_tensor_value(tensor, is_sparse), dtype=type_str)
        return expr

    def get_tensor_shape(self, tensor_wrapper):
        """Returns tensor shape. Infers shape if the shape is empty."""
        assert isinstance(tensor_wrapper, TensorWrapper), "Expecting TensorWrapper here"
        return (
            tensor_wrapper.tensor.ShapeAsNumpy()
            if tensor_wrapper.tensor.ShapeLength() > 0
            else _infer_shape(self.get_tensor_expr(tensor_wrapper))
        )


# pylint: disable=no-else-return
def prepare_dense_matrix_from_sparse(sparse_tensor, sparse_tensor_value, sparse_tensor_type):
    """Prepare sparse indices and dense matrix from TFLite sparse parameters."""
    # The function is implemented based on TFLite sparse parameter specifications
    # Please refer
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs#L89
    # for details about each parameters
    sparsity = sparse_tensor.Sparsity()
    dense_shape = sparse_tensor.ShapeAsNumpy()
    orig_rank = len(dense_shape)

    # The traversal order of the dimensions defined in the `shape` field of the to be dense tensor.
    traversal_order = sparsity.TraversalOrderAsNumpy()

    # For an n-dimensional tensor with a k-dimensional block (0 <= k <= n),
    # stores how a block dimension in (dn, ..., dn+k-1) maps to the original
    # tensor dimension in (d0, ..., dn). It's stored in the order of (dn, ..., dn+k-1).
    # If not block-sparse, this field is NULL.
    block_map = sparsity.BlockMapAsNumpy()

    total_rank = sparsity.TraversalOrderLength()
    dense_mat = np.full(shape=dense_shape, fill_value=0, dtype=sparse_tensor_type).flatten()

    from enum import Enum

    # NOTE: Here the Vector term is borrowed from TFLite spec.
    class VectorType(Enum):
        Empty = 0
        Int32 = 1
        Uint16 = 2
        Uint8 = 3

    def _get_vector_flag(v_type):
        if VectorType(v_type) == VectorType.Int32:
            return N.Int32Flags
        elif VectorType(v_type) == VectorType.Uint16:
            return N.Uint16Flags
        elif VectorType(v_type) == VectorType.Uint8:
            return N.Uint8Flags
        else:
            raise tvm.error.OpNotImplemented("The provided type {} is not supported".format(v_type))

    def _get_flattened_index(indices, shape):
        index = 0
        sub_elements = 1
        for i in reversed(range(0, len(dense_shape))):
            index += indices[i] * sub_elements
            sub_elements *= shape[i]
        return index

    # DimensionMetadata per dimension: the metadata needed for
    #     each dimension to locate the non-zero values in the original dense tensor
    #     inline with traversal order parameter.
    #
    # sp_format has 2 possible values: {DENSE = 0, SPARSE_CSR = 1}
    # If format = DENSE{0} : DenseSize represents size of that dimension
    # If format = SPARSE_CSR{1} : array_segments represents how to segment the indices array,
    #      each segment corresponds to one element in the previous dimension. array_indices
    #      represents the index of the non-zero elements within this dimension
    #      (as those in the CSR matrix format, where the first array is row pointers
    #       and the second array is column indices).
    sp_format = np.zeros(sparsity.DimMetadataLength())
    dim_metadata = [None] * (2 * sparsity.DimMetadataLength())

    # Below loop will fetch all meta data per dimension based on format type
    # Dense or Sparse and will put it in an agnostic array for easy access
    # while preparing dense buffer or indices.
    for i in range(sparsity.DimMetadataLength()):
        sp_format[i] = sparsity.DimMetadata(i).Format()
        if sp_format[i] == 0:
            dim_metadata[2 * i] = [sparsity.DimMetadata(i).DenseSize()]
        else:
            from flatbuffers import number_types as N

            dim_metadata[2 * i] = (
                sparsity.DimMetadata(i)
                .ArraySegments()
                .GetVectorAsNumpy(
                    flags=_get_vector_flag(sparsity.DimMetadata(i).ArraySegmentsType()), off=4
                )
            )
            dim_metadata[2 * i + 1] = (
                sparsity.DimMetadata(i)
                .ArrayIndices()
                .GetVectorAsNumpy(
                    flags=_get_vector_flag(sparsity.DimMetadata(i).ArrayIndicesType()), off=4
                )
            )

    block_dim = 0
    block_size = np.zeros(sparsity.BlockMapLength())

    # Block size parameter if encoded in BSR format
    for i in range(orig_rank):
        if block_dim < sparsity.BlockMapLength() and block_map[block_dim] == i:
            orig_dim = traversal_order[orig_rank + block_dim]
            block_size[block_dim] = sparsity.DimMetadata(orig_dim).DenseSize()
            block_dim += 1

    indices_list = []

    # Below function iterates through each applicable indices per dimension
    # based on format type specified and finally produce the dense matrix and the NZ indices.
    def _def_prepare_dense_matrix_from_sparse(indices, level, prev_idx):
        if level == len(indices):
            start_pos = 0
            orig_idx = np.zeros(orig_rank, dtype="int32")
            while start_pos < orig_rank:
                orig_idx[traversal_order[start_pos]] = indices[start_pos]
                start_pos += 1
            while start_pos < len(indices):
                block_idx = traversal_order[start_pos] - orig_rank
                orig_dim = block_map[block_idx]
                orig_idx[orig_dim] = orig_idx[orig_dim] * block_size[block_idx] + indices[start_pos]
                start_pos += 1
            indices_list.append(orig_idx)
            nonlocal value_idx
            dense_mat[_get_flattened_index(orig_idx, dense_shape)] = sparse_tensor_value[value_idx]
            value_idx += 1
        else:
            metadata_idx = 2 * level
            if sp_format[level] == 0:
                shape_of_level = dim_metadata[metadata_idx][0]
                for idx in range(shape_of_level):
                    indices[level] = idx
                    _def_prepare_dense_matrix_from_sparse(
                        indices, level + 1, prev_idx * shape_of_level + idx
                    )
            else:
                array_segments = dim_metadata[metadata_idx]
                array_indices = dim_metadata[metadata_idx + 1]
                for idx in range(array_segments[prev_idx], array_segments[prev_idx + 1]):
                    indices[level] = array_indices[idx]
                    _def_prepare_dense_matrix_from_sparse(indices, level + 1, idx)

    indices = np.zeros(total_rank)
    value_idx = 0
    _def_prepare_dense_matrix_from_sparse(indices, 0, 0)
    return np.array(indices_list, dtype="int32"), dense_mat.reshape(dense_shape)


def get_scalar_from_constant(expr):
    """Returns scalar value from Relay constant scalar."""
    assert (
        isinstance(expr, _expr.Constant) and not expr.data.shape
    ), "Expr is not a constant scalar."
    value = expr.data.numpy()
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(
        np.float32
    ), "value must be float32/int32"
    return value.item(0)


def get_tensor_from_constant(expr):
    """Returns tensor of values from Relay constant node."""
    assert isinstance(expr, _expr.Constant)
    value = expr.data.numpy()
    assert value.dtype == np.dtype(np.int32) or value.dtype == np.dtype(
        np.float32
    ), "value must be float32/int32"
    return value


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
        if not field_name.startswith("_"):
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


def _decode_type(n):
    _tflite_m = {
        0: "float32",
        1: "float16",
        2: "int32",
        3: "uint8",
        4: "int64",
        5: "string",
        6: "bool",
        7: "int16",
        8: "complex64",
        9: "int8",
    }
    return _tflite_m[n]


def _input_type(model):
    subgraph_count = model.SubgraphsLength()
    assert subgraph_count > 0
    shape_dict = {}
    dtype_dict = {}
    for subgraph_index in range(subgraph_count):
        subgraph = model.Subgraphs(subgraph_index)
        inputs_count = subgraph.InputsLength()
        assert inputs_count >= 1
        for input_index in range(inputs_count):
            input_ = subgraph.Inputs(input_index)
            assert subgraph.TensorsLength() > input_
            tensor = subgraph.Tensors(input_)
            input_shape = tuple(tensor.ShapeAsNumpy())
            tensor_type = tensor.Type()
            input_name = tensor.Name().decode("utf8")
            shape_dict[input_name] = input_shape
            dtype_dict[input_name] = _decode_type(tensor_type)

    return shape_dict, dtype_dict


def from_tflite(model, shape_dict=None, dtype_dict=None, op_converter=OperatorConverter):
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
        import tflite.BuiltinOperator
        import tflite.SubGraph
    except ImportError:
        raise ImportError("The tflite package must be installed")

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite

        assert isinstance(model, tflite.Model)
    except TypeError:
        import tflite.Model

        assert isinstance(model, tflite.Model.Model)

    _shape_dict, _dtype_dict = _input_type(model)
    if shape_dict is not None:
        _shape_dict.update(shape_dict)
    if dtype_dict is not None:
        _dtype_dict.update(dtype_dict)

    # keep the same as tflite
    assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"
    subgraph = model.Subgraphs(0)

    # model inputs / outputs
    model_inputs = subgraph.InputsAsNumpy()
    model_outputs = subgraph.OutputsAsNumpy()

    exp_tab = ExprTable()
    for model_input in model_inputs:
        model_input_name = get_tensor_name(subgraph, model_input)
        shape = _shape_dict[model_input_name] if model_input_name in _shape_dict else None
        dtype = _dtype_dict[model_input_name] if model_input_name in _dtype_dict else "float32"
        exp_tab.set_expr(model_input_name, _expr.var(model_input_name, shape=shape, dtype=dtype))

    # op code in model
    op_converter = op_converter(model, subgraph, exp_tab)
    op_converter.check_unsupported_ops()
    op_converter.convert_op_to_relay()

    # params and outputs
    params = {k: _nd.array(np.array(v)) for k, v in exp_tab.params.items()}
    outputs = [exp_tab.get_expr(get_tensor_name(subgraph, i)) for i in model_outputs]
    outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
    attrs = tvm.ir.make_node(
        "DictAttrs",
        **{
            "output_tensor_names": [
                sanitize_name(get_tensor_name(subgraph, model_output))
                for model_output in model_outputs
            ]
        },
    )
    func = _function.Function(analysis.free_vars(outputs), outputs, attrs=attrs)
    mod = IRModule.from_expr(func)
    return mod, params
