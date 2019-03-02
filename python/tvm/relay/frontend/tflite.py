# pylint: disable=invalid-name, unused-argument
"""Tensorflow lite frontend."""
from __future__ import absolute_import as _abs
import math
import numpy as np
from .. import ir_pass
from .. import expr as _expr
from .. import op as _op
from ... import nd as _nd
from .common import ExprTable

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
            'SOFTMAX': self.convert_softmax,
            'SQUEEZE': self.convert_squeeze,
            'MAX_POOL_2D': self.convert_max_pool2d,
            "CONCATENATION": self.convert_concatenation
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
            raise NotImplementedError("Unsupported Ops: %s" % (
                ','.join(unsupported_ops_set)))

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
            raise NotImplementedError("Not Support Custom Operator Now")
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
        raise NotImplementedError("Not support tensor type {}"
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
        raise NotImplementedError("Not support tensor type {}".format(str(tensor_type)))

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
        input_shape_length = len(input_tensor.tensor.ShapeAsNumpy())

        in_expr = self.get_expr(input_tensor_idx)

        if input_shape_length in (1, 2):
            # The rule is channel first (after N but before H, W).
            # length of 1 means N*H*W*C, do nothing.
            # length of 2 means N*H*W, C, do nothing.
            pass
        elif input_shape_length == 3:
            # convert N C H*W to N H*W C
            in_expr = _op.transpose(in_expr, axes=(0, 2, 1))
        elif input_shape_length == 4:
            # convert input to N H W C, then reshape to target shape,
            # finally convert back if necessary
            in_expr = _op.transpose(in_expr, axes=(0, 2, 3, 1))
        else:
            raise NotImplementedError("Not support input shape length {} of reshape : "
                                      .format(str(input_shape_length)))

        out = _op.reshape(in_expr, newshape=tuple(target_shape))

        # The rule is channel first.
        # 1: N*H*W*C
        # 2: N*H*W, C
        # 3: N H W C, reshape to N H*W C, transpose to N C H*W
        # 4: N H W C, transpose to N C H W
        # add more if we need target shapes in future
        if len(target_shape) == 1 or len(target_shape) == 2:
            pass
        elif len(target_shape) == 3:
            out = _op.transpose(out, axes=(0, 2, 1))
        elif len(target_shape) == 4:
            out = _op.transpose(out, axes=(0, 3, 1, 2))
        else:
            raise NotImplementedError("Not support to reshape to shape length {}: "
                                      .format(str(len(target_shape))))

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

    def convert_concatenation(self, op):
        """ convert TFLite concatenation"""
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
        input_shape_length = len(input_tensors[0].tensor.ShapeAsNumpy())

        # TFLite is N H W C, our layout is N C H W
        if input_shape_length <= 4:
            axis_convert_map = [0] + list(range(2, input_shape_length)) + [1]
            concatenation_axis = axis_convert_map[concatenation_axis]
        else:
            raise NotImplementedError("Not support input shape length {} of concatenatio : "
                                      .format(str(input_shape_length)))

        # with axis in N H W C
        out = _op.concatenate(in_exprs, axis=concatenation_axis)

        # if we have activation fn
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
        input_shape_length = len(input_tensor.tensor.ShapeAsNumpy())
        output_shape_length = len(output_tensors[0].tensor.ShapeAsNumpy())

        in_expr = self.get_expr(input_tensor_idx)

        # TFLite is N H W C, our layout is N C H W
        if input_shape_length in (1, 2):
            # The rule is channel first (after N but before H, W).
            # length of 1 means N*H*W*C, do nothing.
            # length of 2 means N*H*W, C, do nothing.
            pass
        elif input_shape_length == 3:
            # convert N C H*W to N H*W C
            in_expr = _op.transpose(in_expr, axes=(0, 2, 1))
        elif input_shape_length == 4:
            # convert input to N H W C, then reshape to target shape,
            # finally convert back if necessary
            in_expr = _op.transpose(in_expr, axes=(0, 2, 3, 1))
        else:
            raise NotImplementedError("Not support input shape length {} of squeeze : "
                                      .format(str(input_shape_length)))

        out = _op.squeeze(in_expr, axis=tuple(squeeze_axis))

        # The rule is channel first.
        # 1: N*H*W*C
        # 2: N*H*W, C
        # 3: N H W C, reshape to N H*W C, transpose to N C H*W
        # 4: N H W C, transpose to N C H W
        # add more if we need target shapes in future
        if output_shape_length in (1, 2):
            pass
        elif output_shape_length == 3:
            out = _op.transpose(out, axes=(0, 2, 1))
        elif output_shape_length == 4:
            out = _op.transpose(out, axes=(0, 3, 1, 2))
        else:
            raise NotImplementedError("Not support to squeeze to length {} : "
                                      .format(str(output_shape_length)))

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
        raise NotImplementedError("Unsupported fused activation fn {}"
                                  .format(fused_activation_fn_str))

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
            assert depth_multiplier == 1, "TF frontend have transformed it be 1 " \
                                          "no matter original value be set by 0.25, 0.5 or any else"
        else:
            raise ValueError("Not support conv type: {}".format(conv_type))

        stride_h = conv_options.StrideH()
        stride_w = conv_options.StrideW()
        dilation_h = conv_options.DilationHFactor()
        dilation_w = conv_options.DilationWFactor()
        padding = conv_options.Padding()
        fused_activation_fn = conv_options.FusedActivationFunction()

        _, input_h, input_w, _ = input_tensor.tensor.ShapeAsNumpy()

        if is_depthwise_conv:
            multiplier, kernel_h, kernel_w, in_channels = weight_tensor.tensor.ShapeAsNumpy()
            assert multiplier == depth_multiplier
        else:
            output_channels, kernel_h, kernel_w, _ = weight_tensor.tensor.ShapeAsNumpy()

        dilated_kernel_h = dilation_h * (kernel_h - 1) + 1
        dilated_kernel_w = dilation_w * (kernel_w - 1) + 1

        params = {'kernel_size': [kernel_h, kernel_w],
                  'strides': [stride_h, stride_w],
                  'dilation': [dilation_h, dilation_w],
                  'padding': [0, 0]}

        if is_depthwise_conv:
            params['channels'] = int(in_channels * multiplier)
            params['groups'] = int(in_channels)
        else:
            params['channels'] = int(output_channels)

        # weight tensor type should be UINT8 (quantization) or FLOAT32
        weight_tensor_type = weight_tensor.tensor.Type()
        assert weight_tensor_type in (TensorType.UINT8, TensorType.FLOAT32)
        weight_tensor_type_str = self.get_tensor_type_str(weight_tensor_type)

        in_expr = self.get_expr(input_tensor_idx)
        weight_value = self.get_tensor_value(weight_tensor)

        if is_depthwise_conv:
            # TFLite is M KH KW IC, we require IC M KH KW
            weight_value = weight_value.transpose((3, 0, 1, 2))
        else:
            # TFLite is OC KH KW IC, we require OC IC KH kW
            weight_value = weight_value.transpose((0, 3, 1, 2))

        weight_expr = self.exp_tab.new_const(weight_value, dtype=weight_tensor_type_str)

        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, dilated_kernel_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, dilated_kernel_w, stride_w)
            in_expr = _op.nn.pad(data=in_expr, pad_width=((0, 0), (0, 0),
                                                          (pad_top, pad_bottom),
                                                          (pad_left, pad_right)))
        else:
            raise NotImplementedError("Not support padding format: {}".format(padding))

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
            out = _op.nn.bias_add(out, bias_expr)

        # If we have fused activations
        if fused_activation_fn != ActivationFunctionType.NONE:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

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
                  'padding': [0, 0]}

        in_expr = self.get_expr(input_tensor_idx)

        _, input_h, input_w, _ = input_tensor.tensor.ShapeAsNumpy()
        if padding == Padding.VALID:
            pass
        elif padding == Padding.SAME:
            pad_top, pad_bottom = get_pad_value(input_h, filter_h, stride_h)
            pad_left, pad_right = get_pad_value(input_w, filter_w, stride_w)
            params['padding'] = [pad_top, pad_left, pad_bottom, pad_right]
        else:
            raise NotImplementedError("Not support padding format: {}".format(padding))

        if pool_type == "average":
            out = _op.nn.avg_pool2d(in_expr, **params)
        elif pool_type == "max":
            out = _op.nn.max_pool2d(in_expr, **params)
        else:
            raise ValueError("Not support pool type: {}".format(pool_type))

        # If we have fused activations
        if fused_activation_fn != ActivationFunctionType.NONE:
            out = self.convert_fused_activation_function(out, fused_activation_fn)

        return out

    def get_expr(self, input_tensor_idx):
        return self.exp_tab.get_expr(get_tensor_name(self.subgraph, input_tensor_idx))

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
    func : tvm.relay.Function
        Compatible relay Function

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
    func = _expr.Function(ir_pass.free_vars(outputs), outputs)
    return func, params
