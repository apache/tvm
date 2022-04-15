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
# pylint: disable=invalid-name, import-outside-toplevel
""" Functions to convert quantized torch models to QNN """

import numpy as np
import tvm
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import op as _op
from tvm.relay.frontend.common import infer_shape

from .common import logger
from .pytorch_utils import is_version_greater_than, getattr_attr_name


class QNNParam(object):
    """A placeholder for weight quantization parameters"""

    def __init__(self, weight, bias, scale, zero_point):
        self.weight = weight
        self.bias = None if bias is None else bias.detach().numpy()
        self.scale = _expr.const(scale)
        self.zero_point = _expr.const(zero_point, dtype="int32")


class ConvPackedParam(QNNParam):
    """A placeholder for quantized conv2d op attributes
    As of PyTorch 1.6, attributes of quantized conv2d ops, like
    stride, padding etc are stored in ConvPackedParams objects,
    together with weights and quantization parameters
    """

    def __init__(
        self,
        weight_np,
        bias,
        scale,
        zero_point,
        stride,
        padding,
        dilation,
        groups,
        output_padding,
    ):
        super().__init__(weight_np, bias, scale, zero_point)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        # Used only for conv_transpose2d
        self.output_padding = output_padding


def _get_quant_params(qweight):
    import torch

    weight_np = qweight.dequantize().numpy()

    if qweight.qscheme() == torch.per_tensor_affine:
        return weight_np, qweight.q_scale(), int(qweight.q_zero_point())

    scales = qweight.q_per_channel_scales().numpy()
    zero_points = qweight.q_per_channel_zero_points().numpy()
    # This is an assumption posed by QNN
    msg = "The values of zero points should be all zero for per channel"
    assert np.all(zero_points == 0), msg
    return weight_np, scales, 0


def make_qnn_param(qweight, bias):
    weight_np, scale, zero_point = _get_quant_params(qweight)
    return QNNParam(weight_np, bias, scale, zero_point)


def make_conv_packed_param(qweight, bias, packed_params):
    weight_np, scale, zero_point = _get_quant_params(qweight)
    stride = packed_params.stride()
    padding = packed_params.padding()
    dilation = packed_params.dilation()
    groups = packed_params.groups()
    output_padding = packed_params.output_padding()
    return ConvPackedParam(
        weight_np,
        bias,
        scale,
        zero_point,
        stride,
        padding,
        dilation,
        groups,
        output_padding,
    )


def get_weight_quant_params(script_module, packed_param_names):
    """Retrive and unpack weight parameters from quantized modules"""
    import torch

    param_name = "_packed_params"
    quant_params = {}

    def filter_func(named_module):
        m = named_module[1]
        return isinstance(m, torch.jit.RecursiveScriptModule) and (
            ("Conv" in m.original_name) or (m.original_name == "LinearPackedParams")
        )

    for name, m in filter(filter_func, script_module.named_modules()):
        key = name + "." + param_name
        state_dict = m.state_dict()

        if key not in packed_param_names:
            continue

        if len(state_dict) == 0 and not hasattr(m, param_name):
            # for v1.6 and above
            # This case seems to happen if a model is serialized
            # and loaded back
            # This module can be safely ignored
            continue

        if len(state_dict) == 0 and hasattr(m, param_name):
            # for v1.6 and above
            packed_params = m._packed_params
        else:
            assert len(state_dict) == 1
            packed_params = list(state_dict.values())[0]

        if "Conv" in m.original_name and len(state_dict) == 0:
            qweight, bias = torch.ops.quantized.conv2d_unpack(packed_params)
            quant_params[key] = make_conv_packed_param(qweight, bias, packed_params)
        elif "Conv" in m.original_name:
            qweight, bias = torch.ops.quantized.conv2d_unpack(packed_params)
            quant_params[key] = make_qnn_param(qweight, bias)
        elif m.original_name == "LinearPackedParams":
            qweight, bias = torch.ops.quantized.linear_unpack(packed_params)
            quant_params[key] = make_qnn_param(qweight, bias)

    return quant_params


def quantize_numpy(weight, scale, zero_point, out_dtype_np):
    iinfo = np.iinfo(out_dtype_np)
    clip_min = iinfo.min
    clip_max = iinfo.max
    if len(scale.shape) > 0:
        scale = np.reshape(scale, [weight.shape[0]] + [1] * (len(weight.shape) - 1))
    transformed = zero_point + weight / scale
    return np.clip(np.round(transformed), clip_min, clip_max).astype(out_dtype_np)


def add_quant_params_to_outputs(
    outputs, packed_param_map, quant_params, input_scales_for_bias, keep_quantized_weight=False
):
    """
    Add quant params to outputs so that they can be referenced by other
    ops later. Weights are quantized here.
    """
    for node_name, packed_param_name in packed_param_map.items():
        qparam = quant_params[packed_param_name]
        weight_scale = _get_numpy(qparam.scale)
        param_prefix = packed_param_name[: -len("._packed_params")]

        if keep_quantized_weight:
            qparam.weight_var = _expr.var(
                param_prefix + "_weight", shape=qparam.weight.shape, dtype="int8"
            )
            qparam.weight = quantize_numpy(
                qparam.weight, weight_scale, _get_numpy(qparam.zero_point), np.int8
            )
            qweight = qparam.weight_var
        else:
            qparam.weight_var = _expr.var(
                param_prefix + "_weight", shape=qparam.weight.shape, dtype="float32"
            )
            qweight = relay.qnn.op.quantize(
                qparam.weight_var, qparam.scale, qparam.zero_point, out_dtype="int8", axis=0
            )

        if qparam.bias is not None:
            float_bias_var = _expr.var(
                param_prefix + "_bias", shape=qparam.bias.shape, dtype="float32"
            )
            if node_name not in input_scales_for_bias:
                # This case is for dynamic quantization, where the input activation scale is
                # unknown until runtime.
                qparam.bias_var = float_bias_var
                qbias = qparam.bias_var
            elif keep_quantized_weight:
                qparam.bias_var = _expr.var(
                    param_prefix + "_bias", shape=qparam.bias.shape, dtype="int32"
                )
                qparam.bias = quantize_numpy(
                    qparam.bias, input_scales_for_bias[node_name] * weight_scale, 0, np.int32
                )
                qbias = qparam.bias_var
            else:
                qparam.bias_var = float_bias_var
                qbias = relay.qnn.op.quantize(
                    qparam.bias_var,
                    _expr.const(input_scales_for_bias[node_name] * weight_scale),
                    _expr.const(0, "int32"),
                    out_dtype="int32",
                    axis=0,
                )
        else:
            qbias = None

        quant_params[packed_param_name] = qparam

        params = [qweight, qparam.scale, qparam.zero_point, qbias]

        if isinstance(quant_params[packed_param_name], ConvPackedParam):
            params += [
                qparam.stride,
                qparam.padding,
                qparam.dilation,
                qparam.groups,
                qparam.output_padding,
            ]

        outputs[node_name] = params


def _get_quant_param_for_input(input_value):
    """
    We want to know the input scale and zp of this input_value, since
    input quant params are not explicitly passed around in torch (they
    are embedded in a QTensor data structure, not visible statically).
    We know that it is quantized using output scale and zp
    of some previous quantized op. The purpose of this function
    is to find that pair of parameters.
    """
    # Indices for output scale and zp
    # For example, in quantized::conv2d(%input, %1, %2, %3, %4, %5, %6, %7),
    # 6th and 7th arg are output scale and zp respectively.

    # PyTorch 1.6 changed qconv API
    if is_version_greater_than("1.5.1"):
        qconv_indices = (2, 3)
    else:
        qconv_indices = (6, 7)

    output_quant_param_indices = {
        "aten::quantize_per_tensor": (1, 2),
        "quantized::conv2d": qconv_indices,
        "quantized::conv2d_relu": qconv_indices,
        "quantized::linear": (2, 3),
        "quantized::linear_relu": (2, 3),
        "quantized::add_relu": (2, 3),
        "quantized::add": (2, 3),
        "quantized::mul_relu": (2, 3),
        "quantized::mul": (2, 3),
        "quantized::cat": (2, 3),
        "quantized::mul_scalar": (2, 3),
        "quantized::add_scalar": (2, 3),
        "quantized::hardswish": (1, 2),
        "quantized::conv_transpose2d": qconv_indices,
    }

    def dfs(current_node):
        # trace back to find the producer of this input value
        current_op = current_node.kind()
        if current_op in output_quant_param_indices:
            indices = output_quant_param_indices[current_op]
            scale = current_node.inputsAt(indices[0])
            zp = current_node.inputsAt(indices[1])
            return scale, zp

        # Trace back eariler nodes, dfs order
        # Assume quantized tensor comes earlier in the args
        for arg in current_node.inputs():
            return dfs(arg.node())

        # If input_value is not quantized, we reach here.
        return None, None

    return dfs(input_value.node())


def _get_add_scalar_output_quant_param(input_scale, input_zero_point, scalar):
    """
    Determine the output scale and zp of quantized::add_scalar op
    This is used for mobilenet v3
    Refer to aten/src/ATen/native/quantized/cpu/qadd.cpp
    The names of variables are the same as torch impl
    """
    q_min = 0
    q_max = 255
    s = input_scale
    z = input_zero_point
    c = scalar
    c_q = round(c / s)

    if q_min > z - c_q:
        s_prime = (float(q_max) - (z - c_q)) / (float(q_max) - q_min) * s
        z_prime = q_min
    elif q_max < z - c_q:
        s_prime = (float(z - c_q) - q_min) / (float(q_max) - q_min) * s
        z_prime = q_max
    else:
        s_prime = s
        z_prime = z - c_q

    return s_prime, z_prime


def _get_mul_scalar_output_quant_param(input_scale, input_zero_point, scalar):
    """
    Determine the output scale and zp of quantized::mul_scalar op
    This is used for mobilenet v3
    Refer to aten/src/ATen/native/quantized/cpu/qmul.cpp
    The names of variables are the same as torch impl
    """
    q_min = 0
    q_max = 255
    self_scale = input_scale
    self_zero_point = input_zero_point
    other_val = scalar

    if other_val > 0.0:
        s_prime = other_val * self_scale
        z_prime = self_zero_point
    elif other_val == 0.0:
        s_prime = 1.0
        z_prime = 0
    else:
        s_prime = abs(other_val) * self_scale
        z_prime = q_max - (self_zero_point - q_min)

    return s_prime, z_prime


def _add_output_quant_params_to_scalar_op(node, graph, input_scale, input_zero_point, scalar):
    """
    The output scale and zp of {add,mul}_scalar op are not explicit in the IR
    They are required for _get_quant_param_for_input above to work correctly
    So calculate these params using the same way torch does, and make new
    constant nodes in the input IR. Also add these params to the inputs of
    scalar op.

    For example,
       %6 : float = prim::Constant[value=3.]()
       %input : QUInt8(1, 3, 224, 224) = quantized::add_scalar(%x.1, %6)
    becomes
       %6 : float = prim::Constant[value=3.]()
       %7 : float = prim::Constant[value=0.015686161816120148]()
       %8 : int = prim::Constant[value=0]()
       %input : UInt8(1, 3, 224, 224) = quantized::add_scalar(%x.1, %6, %7, %8)

    %7 and %8 are newly created output scale and zp constant nodes
    """
    # pylint: disable=c-extension-no-member
    import torch

    operator = node.kind()

    if operator == "quantized::mul_scalar":
        out_scale, out_zero_point = _get_mul_scalar_output_quant_param(
            input_scale, input_zero_point, scalar
        )
    elif operator == "quantized::add_scalar":
        out_scale, out_zero_point = _get_add_scalar_output_quant_param(
            input_scale, input_zero_point, scalar
        )
    else:
        raise NotImplementedError("unsupported scalar op: %s" % operator)

    # create new constant nodes and add them to graph
    out_scale_node = graph.create("prim::Constant")
    out_zero_point_node = graph.create("prim::Constant")
    out_scale_node.insertBefore(node)
    out_zero_point_node.insertBefore(node)
    out_scale_node.f_("value", out_scale)
    out_zero_point_node.i_("value", out_zero_point)
    out_scale_node.output().setType(torch._C.FloatType.get())
    out_zero_point_node.output().setType(torch._C.IntType.get())
    node.addInput(out_scale_node.output())
    node.addInput(out_zero_point_node.output())


def add_input_quant_params_to_op_inputs(graph):
    """
    In Torch, input quant params are not explicitly passed around
    Instead, they are stored in QTensor data structure, and retrieved
    at runtime by each quantized ops.
    However, they need to be known statically for QNN translation.
    To workaround and simplify the translation of inputs, we manually add
    input quant params to inputs of Torch quantized operators listed below.
    See _quantized_conv2d() below for example of why this is helpful.

    For example,
      %input : QUInt8(1, 512, 7, 7) = quantized::add(%x.8, %x.9, %434, %435)
    becomes
      %395 : float = prim::Constant[value=0.036212071776390076]()
      %396 : int = prim::Constant[value=0]()
      %430 : float = prim::Constant[value=0.16080744564533234]()
      %431 : int = prim::Constant[value=42]()
      %input : QUInt8(1, 512, 7, 7) = quantized::add(%x.8, %x.9, %434, %435,
                                                     %430, %431, %395, %396)

    %434, %435 are output scale and zp of quantized::add op
    %430, %431, %395, %396 are two pairs of input (scale, zp) for two tensors
    added by this function
    """
    # How many quantized tensors each op takes as inputs?
    # A pair of (scale, zp) for each input quantized tensor will be added
    # to the input nodes
    num_quantized_inputs = {
        "quantized::conv2d": 1,
        "quantized::conv2d_relu": 1,
        "quantized::linear": 1,
        "quantized::linear_relu": 1,
        "quantized::add_relu": 2,
        "quantized::add": 2,
        "quantized::mul_relu": 2,
        "quantized::mul": 2,
        "aten::dequantize": 1,
        "aten::mean": 1,
        "aten::sigmoid": 1,
        "aten::upsample_nearest2d": 1,
        "aten::upsample_bilinear2d": 1,
        "aten::relu_": 1,
        "aten::relu": 1,
        "quantized::add_scalar": 1,
        "quantized::mul_scalar": 1,
        "quantized::relu6": 1,
        "quantized::hardswish": 1,
        "aten::hardsigmoid": 1,
        "quantized::conv_transpose2d": 1,
    }

    need_input_quant_param = set(num_quantized_inputs.keys())
    need_input_quant_param.add("quantized::cat")

    input_scales_for_bias = {}

    for node in graph.nodes():
        operator = node.kind()
        if operator not in need_input_quant_param:
            continue

        input_scales = []
        input_zero_points = []

        if operator == "quantized::cat":
            # the number of inputs to concat is not constant
            # so handle it separately
            inputs = node.inputsAt(0).node().inputs()
            for inp in inputs:
                scale, zp = _get_quant_param_for_input(inp)
                input_scales.append(scale)
                input_zero_points.append(zp)
        else:
            for i in range(num_quantized_inputs[operator]):
                scale, zp = _get_quant_param_for_input(node.inputsAt(i))
                if scale is not None and zp is not None:
                    input_scales.append(scale)
                    input_zero_points.append(zp)

        if operator in ["quantized::add_scalar", "quantized::mul_scalar"]:
            scalar = node.inputsAt(1).node().f("value")
            inp_scale = input_scales[0].node().f("value")
            inp_zero_point = input_zero_points[0].node().i("value")

            # see the comments in this function above
            _add_output_quant_params_to_scalar_op(node, graph, inp_scale, inp_zero_point, scalar)

        for scale, zp in zip(input_scales, input_zero_points):
            node.addInput(scale)
            node.addInput(zp)

        if "quantized::conv" in operator or "quantized::linear" in operator:
            # This is required for quantizing the bias
            assert len(input_scales) == 1, "One quantized parameter expected for qconv or qlinear."
            input_scales_for_bias[node.inputsAt(1).debugName()] = input_scales[0].node().f("value")

    return input_scales_for_bias


def add_quant_params(params, quant_params):
    """Add quant parameters to TVM param map"""
    for qparam in quant_params.values():
        params[qparam.weight_var.name_hint] = tvm.nd.array(qparam.weight)
        if qparam.bias is not None:
            params[qparam.bias_var.name_hint] = tvm.nd.array(qparam.bias)


def inline_input_quant_params_for_fx(graph, params):
    """
    Canonicalize input scale and zero point access for FX-quantized graphs.
    We expect input qparams to aten::quantize_per_tensor to be prim::Constant, but that's
    not the case for FX-based quantized models as shown below.
    We replace prim::GetAttr with prim::Constant so that FX-based quantized models can be
    converted in the same way as eager-mode based quantized models.

    Before:
    %pan_input_zero_point_1 : Tensor = prim::GetAttr[name="pan_input_zero_point_1"](%backbone)
    %pan_input_scale_1 : Tensor = prim::GetAttr[name="pan_input_scale_1"](%backbone)
    ...
    %quantize_per_tensor_2 ... = aten::quantize_per_tensor(...,
                                       %pan_input_scale_1, %pan_input_zero_point_1, ...)

    After:
    %2402 : int = prim::Constant[value=0]()
    %2403 : float = prim::Constant[value=1.]()
    %quantize_per_tensor_2 ...  = aten::quantize_per_tensor(..., %2403, %2402, ...)
    """
    # pylint: disable=c-extension-no-member
    import torch

    def get_full_attr_name(current):
        current_attr = getattr_attr_name(current)
        inputs = list(current.inputs())
        if len(inputs) == 1 and inputs[0].node().kind() == "prim::GetAttr":
            return get_full_attr_name(inputs[0].node()) + "." + current_attr
        return current_attr

    for node in graph.findAllNodes("prim::GetAttr", recurse=True):
        out_name = node.output().debugName()

        if "_input_scale" in out_name or "_input_zero_point" in out_name:
            full_attr = get_full_attr_name(node)
            assert full_attr in params, "%s not found in param dict." % full_attr
            param_np = params[full_attr].numpy()
            new_const_node = graph.create("prim::Constant")
            new_const_node.insertBefore(node)

            if "_input_scale" in out_name:
                new_const_node.f_("value", param_np)
                new_const_node.output().setType(torch._C.FloatType.get())
            else:
                new_const_node.i_("value", param_np.item())
                new_const_node.output().setType(torch._C.IntType.get())

            node.replaceAllUsesWith(new_const_node)


def apply_with_upcast(data, func):
    inp = _op.cast(data, dtype="int32")
    out = func(inp)
    return _op.cast(out, "uint8")


def apply_with_fp32_fallback(data, input_scale, input_zero_point, func_fp32):
    dequantized = relay.qnn.op.dequantize(data, input_scale, input_zero_point)
    out = func_fp32(dequantized)
    return relay.qnn.op.quantize(out, input_scale, input_zero_point, out_dtype="uint8", axis=1)


def quantized_relu(data, input_zero_point):
    # refer to aten/src/ATen/native/quantized/cpu/qrelu.cpp
    zp = _op.cast(input_zero_point, dtype="uint8")
    return _op.tensor.maximum(data, zp)


def _quantize_per_tensor():
    def _impl(inputs, _):
        dim = len(infer_shape(inputs[0]))
        if dim > 1:
            axis = 1
        else:
            axis = 0

        return relay.qnn.op.quantize(
            inputs[0], _expr.const(inputs[1]), _expr.const(inputs[2]), out_dtype="uint8", axis=axis
        )

    return _impl


def _dequantize():
    def _impl(inputs, _):
        assert len(inputs) == 3, "Input quant params not found in op inputs"
        inp_scale = _expr.const(inputs[1])
        inp_zero_point = _expr.const(inputs[2])
        return relay.qnn.op.dequantize(inputs[0], inp_scale, inp_zero_point)

    return _impl


def _get_numpy(relay_const_scalar):
    return relay_const_scalar.data.numpy()


def _get_scalar(relay_const_scalar):
    return _get_numpy(relay_const_scalar).item(0)


def _do_bias_and_requantize(
    output, bias, input_scale, weight_scale, output_scale, output_zero_point, with_relu
):
    """Output processing for conv and linear"""
    # this is a vector for per channel case
    requant_input_scale = _expr.const(_get_numpy(input_scale) * _get_numpy(weight_scale))
    # Torch does bias add and requanize scale in fp32
    # refer to third_party/fbgemm/include/fbgemm/OutputProcessing-inl.h
    # Instead, we do bias add in int32 and use qnn requantize, which needs
    # integer input.
    # We observed no loss in accuracy in doing this way, and it is better
    # for tvm because bias quantization can be done at compile time
    # Instead, the torch way requires rounding of activation at runtime

    if bias is not None:
        requantize_input = _op.nn.bias_add(output, bias)
    else:
        requantize_input = output

    requantized = relay.qnn.op.requantize(
        requantize_input,
        requant_input_scale,
        relay.const(0, "int32"),
        output_scale,
        output_zero_point,
        out_dtype="int32",
        axis=1,
    )
    clip_min = 0
    if with_relu:
        clip_min = _get_scalar(output_zero_point)

    clip = _op.tensor.clip(requantized, clip_min, 255.0)
    return _op.cast(clip, dtype="uint8")


def _quantized_conv2d(with_relu=False):
    def _impl(inputs, _):
        # refer to src/ATen/native/quantized/cpu/qconv.cpp
        # inputs[0]: input tensor
        # inputs[1]: (weight, scale, zero_point, bias)
        # inputs[2-5]: stride, padding, dilation, groups
        # inputs[6]: output_scale
        # inputs[7]: output_zero_point
        # inputs[8]: input_scale (added manually by frontend)
        # inputs[9]: input_zero_point (added manually by frontend)
        conv_params = inputs[1]
        weight = conv_params[0]
        weight_scale = conv_params[1]
        weight_zero_point = conv_params[2]
        bias = conv_params[3]

        if len(conv_params) > 4:
            # Torch 1.6 or newer case
            strides = conv_params[4]
            padding = conv_params[5]
            dilation = conv_params[6]
            groups = conv_params[7]

            output_scale = _expr.const(inputs[2])
            output_zero_point = _expr.const(inputs[3])

            assert len(inputs) == 6, "Input quant params not found in op inputs"

            # These are manually added by add_input_quant_params_to_op_inputs above
            # In torch, they are retrieved from QTensor data structure at runtime
            input_scale = _expr.const(inputs[4])
            input_zero_point = _expr.const(inputs[5])
        else:
            strides = inputs[2]
            padding = inputs[3]
            dilation = inputs[4]
            groups = inputs[5]
            output_scale = _expr.const(inputs[6])
            output_zero_point = _expr.const(inputs[7])

            assert len(inputs) == 10, "Input quant params not found in op inputs"

            input_scale = _expr.const(inputs[8])
            input_zero_point = _expr.const(inputs[9])

        weight_shape = infer_shape(weight)
        kernel_size = (weight_shape[2], weight_shape[3])
        out_channels = weight_shape[0]

        if padding[0] != 0 or padding[1] != 0:
            pad_val = _get_scalar(input_zero_point)
            inp = _op.nn.pad(
                inputs[0],
                pad_width=((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                pad_value=float(pad_val),
            )
        else:
            inp = inputs[0]

        # padding is (0, 0) because we did explicit pad op with
        # pad value being zero point above
        conv_out = relay.qnn.op.conv2d(
            inp,
            weight,
            input_zero_point,
            weight_zero_point,
            input_scale,
            weight_scale,
            kernel_size=kernel_size,
            dilation=dilation,
            strides=strides,
            padding=(0, 0),
            groups=groups,
            channels=out_channels,
        )

        return _do_bias_and_requantize(
            conv_out, bias, input_scale, weight_scale, output_scale, output_zero_point, with_relu
        )

    return _impl


def _linear(with_relu=False):
    # similar to conv
    def _impl(inputs, _):
        weight = inputs[1][0]
        weight_scale = inputs[1][1]
        weight_zero_point = inputs[1][2]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        # Manually added by add_input_quant_params_to_op_inputs above
        input_scale = _expr.const(inputs[4])
        input_zero_point = _expr.const(inputs[5])

        weight_shape = infer_shape(weight)
        dense = relay.qnn.op.dense(
            inputs[0],
            weight,
            input_zero_point,
            weight_zero_point,
            input_scale,
            weight_scale,
            units=weight_shape[0],
        )
        bias_var = inputs[1][3]

        return _do_bias_and_requantize(
            dense, bias_var, input_scale, weight_scale, output_scale, output_zero_point, with_relu
        )

    return _impl


def _binop(relay_op, with_relu=False, fp32_piggy_back=False):
    def qnn_impl(
        lhs,
        rhs,
        input_scale_lhs,
        input_zero_point_lhs,
        input_scale_rhs,
        input_zero_point_rhs,
        output_scale,
        output_zero_point,
    ):
        qnn_out = relay_op(
            lhs,
            rhs,
            input_scale_lhs,
            input_zero_point_lhs,
            input_scale_rhs,
            input_zero_point_rhs,
            output_scale,
            output_zero_point,
        )
        if with_relu:
            clip_min = _get_scalar(output_zero_point)
            return _op.tensor.clip(qnn_out, clip_min, 255)
        return qnn_out

    # refer to aten/src/ATen/native/quantized/cpu/{qadd, qmul}.cpp
    # they piggy backs to fp32 math by dequantize -> fp32 math -> quantize
    def torch_impl(
        lhs,
        rhs,
        input_scale_lhs,
        input_zero_point_lhs,
        input_scale_rhs,
        input_zero_point_rhs,
        output_scale,
        output_zero_point,
    ):
        if isinstance(lhs, _expr.Call) and lhs.op.name == "qnn.quantize":
            lhs = lhs.args[0]
        else:
            lhs = relay.qnn.op.dequantize(lhs, input_scale_lhs, input_zero_point_lhs)

        if isinstance(rhs, _expr.Call) and rhs.op.name == "qnn.quantize":
            rhs = rhs.args[0]
        else:
            rhs = relay.qnn.op.dequantize(rhs, input_scale_rhs, input_zero_point_rhs)
        fp32_out = relay_op(lhs, rhs)

        if with_relu:
            fp32_out = _op.nn.relu(fp32_out)

        return relay.qnn.op.quantize(
            fp32_out, output_scale, output_zero_point, axis=-1, out_dtype="uint8"
        )

    def _impl(inputs, _):
        lhs = inputs[0]
        rhs = inputs[1]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        assert len(inputs) == 8, "Input quant params not found in op inputs"
        # Manually added by add_input_quant_params_to_op_inputs above
        input_scale_lhs = _expr.const(inputs[4])
        input_zero_point_lhs = _expr.const(inputs[5])
        input_scale_rhs = _expr.const(inputs[6])
        input_zero_point_rhs = _expr.const(inputs[7])

        if fp32_piggy_back:
            logger.info("Piggy backing to FP32 op (PyTorch way)")
            return torch_impl(
                lhs,
                rhs,
                input_scale_lhs,
                input_zero_point_lhs,
                input_scale_rhs,
                input_zero_point_rhs,
                output_scale,
                output_zero_point,
            )

        return qnn_impl(
            lhs,
            rhs,
            input_scale_lhs,
            input_zero_point_lhs,
            input_scale_rhs,
            input_zero_point_rhs,
            output_scale,
            output_zero_point,
        )

    return _impl


def _cat(fp32_piggy_back=False):
    # refer to aten/src/ATen/native/quantized/cpu/qconcat.cpp
    # for concat they also piggy backs to fp32(!)
    # dequantize -> fp32 math -> quantize
    def torch_impl(inputs, input_scales, input_zero_points, output_scale, output_zero_point, axis):
        dequantized = []
        for inp, inp_scale, inp_zp in zip(inputs, input_scales, input_zero_points):
            dequantized.append(relay.qnn.op.dequantize(inp, inp_scale, inp_zp))

        concat = _op.tensor.concatenate(dequantized, axis=axis)
        return relay.qnn.op.quantize(
            concat, output_scale, output_zero_point, axis=axis, out_dtype="uint8"
        )

    def _impl(inputs, _):
        axis = inputs[1]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        num_inputs = (len(inputs) - 4) // 2

        input_scales = []
        input_zero_points = []

        for i in range(0, num_inputs):
            input_scales.append(_expr.const(inputs[4 + i * 2]))
            input_zero_points.append(_expr.const(inputs[4 + i * 2 + 1]))

        if fp32_piggy_back:
            return torch_impl(
                inputs[0], input_scales, input_zero_points, output_scale, output_zero_point, axis
            )

        return relay.qnn.op.concatenate(
            inputs[0], input_scales, input_zero_points, output_scale, output_zero_point, axis
        )

    return _impl


def _add_scalar():
    # this is used for mobilenet v3
    def _impl(inputs, _):
        # refer to aten/src/ATen/native/quantized/cpu/qadd.cpp
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        s = inputs[4]
        z = inputs[5]
        c = inputs[1]
        c_q = round(c / s)
        q_min = 0
        q_max = 255

        # math for calculating output scale and zp are already done
        # during _add_output_quant_params_to_scalar_op above
        out_scale = _expr.const(inputs[2])
        out_zp = _expr.const(inputs[3])

        if q_min > z - c_q or q_max < z - c_q:
            # TODO(masahi): Replace this with integer only compute
            dequant = relay.qnn.op.dequantize(inputs[0], _expr.const(s), _expr.const(z))
            dequantized_add = _op.tensor.add(dequant, _expr.const(c_q * s))
            return relay.qnn.op.quantize(
                dequantized_add, out_scale, out_zp, axis=1, out_dtype="uint8"
            )
        # only scale change
        return inputs[0]

    return _impl


def quantize_scalar(data, scale, zero_point):
    # used to quantize 6., in mobilenet v3
    transformed = zero_point + data / scale
    return max(0, min(round(transformed), 255))


def _relu6():
    # refer to src/ATen/native/quantized/cpu/qrelu.cpp
    def _impl(inputs, _):
        assert len(inputs) == 4, "Input quant params not found in op inputs"
        input_scale = inputs[2]
        input_zero_point = inputs[3]
        six = quantize_scalar(6.0, input_scale, input_zero_point)
        return _op.tensor.clip(inputs[0], input_zero_point, six)

    return _impl


def _mul_scalar():
    # this is used for mobilenet v3
    def _impl(inputs, _):
        # refer to aten/src/ATen/native/quantized/cpu/qmul.cpp
        # math for calculating output scale and zp are already done
        # during _add_output_quant_params_to_scalar_op above
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        other_val = inputs[1]  # scalar

        if other_val > 0.0:
            # only scale change
            return inputs[0]
        if other_val == 0.0:
            shape = infer_shape(inputs[0])
            return _op.full(_expr.const(0), shape, dtype="uint8")

        # negative scale case
        q_min = 0
        q_max = 255
        bias = _expr.const(q_max + q_min, dtype="int8")
        int8 = bias - _op.cast(inputs[0], "int8")
        return _op.cast(int8, "uint8")

    return _impl


def _hswish():
    # refer to src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp
    # They fallback to fp32
    def _impl(inputs, _):
        assert len(inputs) == 5, "Input quant params not found in op inputs"
        # TODO(masahi): Replace this with integer only compute.
        # We do not have to strictly follow how PyTorch does it.

        def relu6(x):
            return _op.tensor.clip(x, 0.0, 6.0)

        def hardsigmoid(x):
            dtype = "float32"
            return relu6(x + _expr.const(3.0, dtype=dtype)) / _expr.const(6.0, dtype=dtype)

        output_scale = _expr.const(inputs[1])
        output_zero_point = _expr.const(inputs[2])
        input_scale = _expr.const(inputs[3])
        input_zero_point = _expr.const(inputs[4])

        dequant = relay.qnn.op.dequantize(inputs[0], input_scale, input_zero_point, axis=1)
        dequantized_hswish = dequant * hardsigmoid(dequant)
        return relay.qnn.op.quantize(
            dequantized_hswish, output_scale, output_zero_point, out_dtype="uint8"
        )

    return _impl


def _linear_dynamic():
    def _calculate_qparam(inp):
        # reference ATen/native/quantized/cpu/qlinear_dynamic.cpp
        # ChooseQuantizationParams function
        mn = _op.min(inp)
        mx = _op.max(inp)

        # Ensure that the interval contains 0
        mn = _op.minimum(mn, _op.const(0.0, dtype="float32"))
        mx = _op.maximum(mx, _op.const(0.0, dtype="float32"))

        qmax = 255

        # reduce_range became True in v1.6
        if is_version_greater_than("1.5.1"):
            qmax = 127

        scale = (mx - mn) / _expr.const(qmax, dtype="float32")

        zero_point_from_min = -(mn / scale)
        zero_point = _op.cast(_op.round(_op.clip(zero_point_from_min, 0.0, qmax)), "int32")

        return scale, zero_point

    def _impl(inputs, _):
        weight = inputs[1][0]
        weight_scale = inputs[1][1]
        weight_zero_point = inputs[1][2]

        inp = inputs[0]

        input_scale, input_zero_point = _calculate_qparam(inp)
        qinp = relay.qnn.op.quantize(inp, input_scale, input_zero_point, out_dtype="uint8")

        data_shape = infer_shape(inp)

        if len(data_shape) > 2:
            qinp = _op.reverse_reshape(qinp, [-1, 0])

        weight_shape = infer_shape(weight)
        units = weight_shape[0]
        dense = relay.qnn.op.dense(
            qinp,
            weight,
            input_zero_point,
            weight_zero_point,
            input_scale,
            weight_scale,
            units=units,
        )
        bias_var = inputs[1][3]

        dequant_scale = input_scale * weight_scale
        dense_out = relay.qnn.op.dequantize(
            dense, dequant_scale, input_zero_point=relay.const(0, "int32"), axis=1
        )

        if len(data_shape) > 2:
            new_shape = list(data_shape[:-1])
            new_shape.append(units)
            dense_out = _op.reshape(dense_out, new_shape)

        if bias_var is not None:
            return dense_out + bias_var

        return dense_out

    return _impl


def _quantized_conv_transpose2d(with_relu=False):
    def _impl(inputs, _):
        # Refer to aten/src/ATen/native/quantized/cpu/qconv.cpp
        # Supported in Torch 1.7 or newer
        conv_params = inputs[1]
        weight = conv_params[0]
        weight_scale = conv_params[1]
        weight_zero_point = conv_params[2]
        bias = conv_params[3]

        strides = conv_params[4]
        padding = conv_params[5]
        dilation = conv_params[6]
        groups = conv_params[7]
        output_padding = conv_params[8]

        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])

        assert len(inputs) == 6, "Input quant params not found in op inputs"

        # These are manually added by add_input_quant_params_to_op_inputs above
        # In torch, they are retrieved from QTensor data structure at runtime
        input_scale = _expr.const(inputs[4])
        input_zero_point = _expr.const(inputs[5])

        weight_shape = list(infer_shape(weight))

        kernel_size = (weight_shape[2], weight_shape[3])
        out_channels = weight_shape[1]

        conv_out = relay.qnn.op.conv2d_transpose(
            inputs[0],
            weight,
            input_zero_point,
            weight_zero_point,
            input_scale,
            weight_scale,
            kernel_size=kernel_size,
            dilation=dilation,
            strides=strides,
            padding=padding,
            groups=groups,
            channels=out_channels,
            output_padding=output_padding,
            out_dtype="int32",
            kernel_layout="IOHW",
        )

        return _do_bias_and_requantize(
            conv_out, bias, input_scale, weight_scale, output_scale, output_zero_point, with_relu
        )

    return _impl


convert_map = {
    "aten::quantize_per_tensor": _quantize_per_tensor(),
    "quantized::conv2d_relu": _quantized_conv2d(with_relu=True),
    "aten::dequantize": _dequantize(),
    "quantized::conv2d": _quantized_conv2d(),
    "quantized::add_relu": _binop(relay.qnn.op.add, with_relu=True),
    "quantized::add": _binop(relay.qnn.op.add),
    "quantized::mul_relu": _binop(relay.qnn.op.mul, with_relu=True),
    "quantized::mul": _binop(relay.qnn.op.mul),
    "quantized::linear": _linear(),
    "quantized::linear_relu": _linear(with_relu=True),
    "quantized::cat": _cat(),
    "quantized::add_scalar": _add_scalar(),
    "quantized::mul_scalar": _mul_scalar(),
    "quantized::relu6": _relu6(),
    "quantized::linear_dynamic": _linear_dynamic(),
    "quantized::hardswish": _hswish(),
    "quantized::conv_transpose2d": _quantized_conv_transpose2d(),
}
