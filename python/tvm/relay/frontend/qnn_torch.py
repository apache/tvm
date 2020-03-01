import torch
import tvm
import numpy as np
from tvm import relay
from tvm.relay import expr as _expr
from tvm.relay import op as _op
from tvm.relay.frontend.common import infer_shape


class QuantParam:
    def __init__(self, weight, bias, scale, zero_point, param_key):
        param_prefix = param_key[:-len("._packed_params")]
        self.weight_var = _expr.var(param_prefix + "_weight",
                                    shape=weight.shape)
        self.weight = weight

        if bias is not None:
            self.bias_var = _expr.var(param_prefix + "_bias",
                                      shape=bias.shape)
            self.bias = bias.detach().numpy()
        else:
            self.bias_var = None
            self.bias = None

        self.scale = _expr.const(scale)
        self.zero_point = _expr.const(zero_point, dtype="int32")


def unpack_quant_params(param_name, packed_params, unpack_func):
    qweight, bias = unpack_func(packed_params)
    weight_np = qweight.dequantize().numpy()

    if qweight.qscheme() == torch.per_tensor_affine:
        param = QuantParam(weight_np, bias, qweight.q_scale(),
                           int(qweight.q_zero_point()), param_name)
    else:
        scales = qweight.q_per_channel_scales().numpy()
        zero_points = qweight.q_per_channel_zero_points().numpy()
        assert np.all(zero_points == 0)
        param = QuantParam(weight_np, bias, scales, 0, param_name)

    return param


def get_weight_quant_params(script_module):
    conv_packed_params = []
    linear_packed_params = []

    for name, m in script_module.named_modules():
        if isinstance(m, torch.jit.RecursiveScriptModule):
            if "Conv" in m.original_name:
                conv_packed_params.append((name, m.state_dict()))
            elif m.original_name == "LinearPackedParams":
                linear_packed_params.append((name, m.state_dict()))

    pairs = [(torch.ops.quantized.conv2d_unpack, conv_packed_params),
             (torch.ops.quantized.linear_unpack, linear_packed_params)]

    quant_params = {}
    param_name = "_packed_params"
    for unpack_func, params in pairs:
        for name, state_dict in params:
            assert len(state_dict) == 1
            assert param_name in state_dict
            key = name + "." + param_name
            packed_param = state_dict[param_name]
            quant_params[key] = unpack_quant_params(key, packed_param,
                                                    unpack_func)

    return quant_params


def add_quant_params_to_outputs(outputs, output_index_map,
                                packed_param_map, quant_params):
    for node_name, packed_param_name in packed_param_map.items():
        qparam = quant_params[packed_param_name]
        output_index_map[node_name] = len(outputs)
        qweight = relay.qnn.op.quantize(qparam.weight_var, qparam.scale,
                                        qparam.zero_point, out_dtype="int8",
                                        axis=0)
        param_tup = (qweight, qparam.scale, qparam.zero_point, qparam.bias_var)
        outputs.append(param_tup)


def get_quant_param_for_input(input_value):
    output_quant_param_indices = {
        "aten::quantize_per_tensor": (1, 2),
        "quantized::conv2d": (6, 7),
        "quantized::conv2d_relu": (6, 7),
        "quantized::linear": (2, 3),
        "quantized::linear_relu": (2, 3),
        "quantized::add_relu": (2, 3),
        "quantized::add": (2, 3),
        "quantized::mul_relu": (2, 3),
        "quantized::mul": (2, 3),
        "quantized::cat": (2, 3),
        "quantized::mul_scalar": (2, 3),
        "quantized::add_scalar": (2, 3)
    }

    def dfs(current_node):
        # trace back to find the producer of this input value
        current_op = current_node.kind()
        if current_op in output_quant_param_indices:
            indices = output_quant_param_indices[current_op]
            scale = current_node.inputsAt(indices[0])
            zp = current_node.inputsAt(indices[1])
            return scale, zp
        else:
            # Assume quantized tensor comes earlier in the args
            for arg in current_node.inputs():
                return dfs(arg.node())

        assert False, "No producer for %s" % (str(current_node))

    return dfs(input_value.node())


def get_add_scalar_output_quant_param(input_scale, input_zero_point,
                                      scalar):
    # refer to aten/src/ATen/native/quantized/cpu/qadd.cpp
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


def get_mul_scalar_output_quant_param(input_scale, input_zero_point,
                                      scalar):
    # refer to aten/src/ATen/native/quantized/cpu/qmul.cpp
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


def add_output_quant_params_to_scalar_op(node, graph,
                                         input_scale, input_zero_point,
                                         scalar):
    operator = node.kind()

    if operator == "quantized::mul_scalar":
        out_scale, out_zero_point = \
          get_mul_scalar_output_quant_param(input_scale, input_zero_point,
                                            scalar)
    elif operator == "quantized::add_scalar":
        out_scale, out_zero_point = \
          get_add_scalar_output_quant_param(input_scale, input_zero_point,
                                            scalar)
    else:
        assert False, "unsupported scalar op: %s" % operator

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
    # Quantized operators in PyTorch do not take input quant params as
    # arguments. But QNN expects them to be passed in as arguements.
    # To simplify the translation of inputs, we add input quant params
    # to inputs of PyTorch quantized operator nodes. See _impl in
    #  _quantized_conv2d() below for example of why this is helpful.
    num_quantized_inputs = {"quantized::conv2d": 1,
                            "quantized::conv2d_relu": 1,
                            "quantized::linear": 1,
                            "quantized::linear_relu": 1,
                            "quantized::add_relu": 2,
                            "quantized::add": 2,
                            "quantized::mul_relu": 2,
                            "quantized::mul": 2,
                            "aten::dequantize": 1,
                            "aten::mean": 1,
                            "aten::upsample_bilinear2d": 1,
                            "aten::relu_": 1,
                            "aten::relu": 1,
                            "quantized::add_scalar": 1,
                            "quantized::mul_scalar": 1,
                            'quantized::relu6': 1}

    need_input_quant_param = set(num_quantized_inputs.keys())
    need_input_quant_param.add("quantized::cat")

    for node in graph.nodes():
        operator = node.kind()
        if operator not in need_input_quant_param:
            continue

        input_scales = []
        input_zero_points = []

        if operator == "quantized::cat":
            inputs = node.inputsAt(0).node().inputs()
            for inp in inputs:
                scale, zp = get_quant_param_for_input(inp)
                input_scales.append(scale)
                input_zero_points.append(zp)
        else:
            for i in range(num_quantized_inputs[operator]):
                scale, zp = get_quant_param_for_input(node.inputsAt(i))
                input_scales.append(scale)
                input_zero_points.append(zp)

        if operator in ["quantized::add_scalar", "quantized::mul_scalar"]:
            scalar = node.inputsAt(1).node().f("value")
            inp_scale = input_scales[0].node().f("value")
            inp_zero_point = input_zero_points[0].node().i("value")

            add_output_quant_params_to_scalar_op(node, graph,
                                                 inp_scale, inp_zero_point,
                                                 scalar)

        for scale, zp in zip(input_scales, input_zero_points):
            node.addInput(scale)
            node.addInput(zp)


def add_quant_params(params, quant_params):
    for qparam in quant_params.values():
        params[qparam.weight_var.name_hint] = tvm.nd.array(qparam.weight)
        if qparam.bias is not None:
            params[qparam.bias_var.name_hint] = tvm.nd.array(qparam.bias)


def quantized_adaptive_avg_2d(data, func):
    inp = _op.cast(data, dtype="int32")
    out = func(inp)
    return _op.cast(out, "uint8")


def quantized_mean(data, input_scale, input_zero_point, func):
    dequantized = relay.qnn.op.dequantize(data, input_scale, input_zero_point)
    out = func(dequantized)
    return relay.qnn.op.quantize(out, input_scale, input_zero_point,
                                 out_dtype="uint8", axis=1)


def quantized_upsample(data, input_scale, input_zero_point, func):
    data = relay.qnn.op.dequantize(data, input_scale, input_zero_point)
    out = func(data)
    return relay.qnn.op.quantize(out, input_scale, input_zero_point,
                                 out_dtype="uint8", axis=1)


def quantized_relu(data, input_zero_point):
    zp = _op.cast(input_zero_point, dtype="uint8")
    return _op.tensor.maximum(data, zp)


def _quantize_per_tensor():
    def _impl(inputs, input_type):
        return relay.qnn.op.quantize(inputs[0], _expr.const(inputs[1]),
                                     _expr.const(inputs[2]), out_dtype="uint8",
                                     axis=1)
    return _impl


def _dequantize():
    def _impl(inputs, input_type):
        inp_scale = _expr.const(inputs[1])
        inp_zero_point = _expr.const(inputs[2])
        return relay.qnn.op.dequantize(inputs[0], inp_scale, inp_zero_point)
    return _impl


def get_numpy(relay_const_scalar):
    return relay_const_scalar.data.asnumpy()


def get_scalar(relay_const_scalar):
    return np.asscalar(get_numpy(relay_const_scalar))


def _quantized_conv2d(with_relu=False):
    def _impl(inputs, input_type):
        # refer to src/ATen/native/quantized/cpu/qconv.cpp
        # inputs[0]: input tensor
        # inputs[1]: (weight, scale, zero_point, bias)
        # inputs[2-5]: stride, padding, dilation, groups
        # inputs[6]: output_scale
        # inputs[7]: output_zero_point
        # inputs[8]: input_scale (added manually by frontend)
        # inputs[9]: input_zero_point (added manually by frontend)
        weight = inputs[1][0]
        weight_scale = inputs[1][1]
        weight_zero_point = inputs[1][2]

        output_scale = _expr.const(inputs[6])
        output_zero_point = _expr.const(inputs[7])

        assert len(inputs) == 10, "Input quant params not found in op inputs"
        input_scale = _expr.const(inputs[8])
        input_zero_point = _expr.const(inputs[9])

        strides, padding, dilation = inputs[2], inputs[3], inputs[4]
        strides = infer_shape(inputs[2])
        padding = infer_shape(inputs[3])
        dilation = infer_shape(inputs[4])
        groups = inputs[5]

        weight_shape = infer_shape(weight)
        kernel_size = (weight_shape[2], weight_shape[3])
        out_channels = weight_shape[0]

        if padding[0] != 0 or padding[1] != 0:
            pad_val = get_scalar(input_zero_point)
            inp = _op.nn.pad(inputs[0], pad_width=((0, 0),
                                                   (0, 0),
                                                   (padding[0], padding[0]),
                                                   (padding[1], padding[1])),
                             pad_value=float(pad_val))
        else:
            inp = inputs[0]

        conv_out = relay.qnn.op.conv2d(inp, weight,
                                       input_zero_point, weight_zero_point,
                                       input_scale, weight_scale,
                                       kernel_size=kernel_size,
                                       dilation=dilation, strides=strides,
                                       padding=(0, 0), groups=groups,
                                       channels=out_channels)

        # input scale * weight scale
        requant_input_scale = _expr.const(inputs[8] * get_numpy(weight_scale))
        bias_var = inputs[1][3]

        if bias_var is not None:
            qbias = relay.qnn.op.quantize(bias_var, requant_input_scale,
                                          _expr.const(0, "int32"),
                                          out_dtype="int32", axis=0)
            conv_res = _op.nn.bias_add(conv_out, qbias)
        else:
            conv_res = conv_out

        requantized = relay.qnn.op.requantize(conv_res,
                                              requant_input_scale,
                                              _expr.const(0, "int32"),
                                              output_scale, output_zero_point,
                                              out_dtype="int32", axis=1)
        clip_min = 0
        if with_relu:
            clip_min = get_scalar(output_zero_point)

        clip = _op.tensor.clip(requantized, clip_min, 255.)
        return _op.cast(clip, dtype="uint8")

    return _impl


def _binop(relay_op, with_relu=False):
    def _impl(inputs, input_type):
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        assert len(inputs) == 8, "Input quant params not found in op inputs"
        input_scale_lhs = _expr.const(inputs[4])
        input_zero_point_lhs = _expr.const(inputs[5])
        input_scale_rhs = _expr.const(inputs[6])
        input_zero_point_rhs = _expr.const(inputs[7])
        lhs = inputs[0]
        rhs = inputs[1]

        if isinstance(lhs, _expr.Call) and lhs.op.name == 'qnn.quantize':
            lhs = lhs.args[0]
        else:
            lhs = relay.qnn.op.dequantize(lhs,
                                          input_scale_lhs,
                                          input_zero_point_lhs)

        if isinstance(rhs, _expr.Call) and rhs.op.name == 'qnn.quantize':
            rhs = rhs.args[0]
        else:
            rhs = relay.qnn.op.dequantize(rhs,
                                          input_scale_rhs,
                                          input_zero_point_rhs)
        fp32_out = relay_op(lhs, rhs)

        if with_relu:
            fp32_out = _op.nn.relu(fp32_out)

        return relay.qnn.op.quantize(fp32_out,
                                     output_scale,
                                     output_zero_point,
                                     axis=-1,
                                     out_dtype="uint8")
    return _impl


def _linear(with_relu=False):
    def _impl(inputs, input_type):
        weight = inputs[1][0]
        weight_scale = inputs[1][1]
        weight_zero_point = inputs[1][2]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        input_scale = _expr.const(inputs[4])
        input_zero_point = _expr.const(inputs[5])

        weight_shape = infer_shape(weight)
        dense = relay.qnn.op.dense(inputs[0], weight,
                                   input_zero_point, weight_zero_point,
                                   input_scale, weight_scale,
                                   units=weight_shape[0])

        requant_input_scale = _expr.const(inputs[4] * get_numpy(weight_scale))
        bias_var = inputs[1][3]

        if bias_var is not None:
            qbias = relay.qnn.op.quantize(bias_var, requant_input_scale,
                                          _expr.const(0, "int32"),
                                          out_dtype="int32", axis=0)
            dense_res = _op.nn.bias_add(dense, qbias)
        else:
            dense_res = dense

        requantized = relay.qnn.op.requantize(dense_res,
                                              requant_input_scale,
                                              relay.const(0, 'int32'),
                                              output_scale, output_zero_point,
                                              out_dtype="int32", axis=1)
        clip_min = 0
        if with_relu:
            clip_min = get_scalar(output_zero_point)

        clip = _op.tensor.clip(requantized, clip_min, 255.)
        return _op.cast(clip, dtype="uint8")

    return _impl


def _cat():
    def _impl(inputs, input_type):
        axis = inputs[1]
        output_scale = _expr.const(inputs[2])
        output_zero_point = _expr.const(inputs[3])
        num_inputs = (len(inputs) - 4) // 2
        dequantized = []

        for i in range(0, num_inputs):
            inp_scale = _expr.const(inputs[4+i*2])
            inp_zp = _expr.const(inputs[4+i*2+1])
            dequantized.append(relay.qnn.op.dequantize(inputs[0][i],
                                                       inp_scale, inp_zp))

        concat = _op.tensor.concatenate(dequantized, axis=axis)
        return relay.qnn.op.quantize(concat, output_scale, output_zero_point,
                                     axis=1, out_dtype="uint8")

    return _impl


def _add_scalar():
    def _impl(inputs, input_type):
        # refer to aten/src/ATen/native/quantized/cpu/qadd.cpp
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        s = inputs[4]
        z = inputs[5]
        c = inputs[1]
        c_q = round(c / s)
        q_min = 0
        q_max = 255

        out_scale = _expr.const(inputs[2])
        out_zp = _expr.const(inputs[3])

        if q_min > z - c_q or q_max < z - c_q:
            dequant = relay.qnn.op.dequantize(inputs[0],
                                              _expr.const(s), _expr.const(z))
            dequantized_add = _op.tensor.add(dequant, _expr.const(c_q * s))
            return relay.qnn.op.quantize(dequantized_add, out_scale, out_zp,
                                         axis=1, out_dtype="uint8")
        else:
            # only scale change
            return inputs[0]

    return _impl


def quantize_scalar(data, scale, zero_point):
    transformed = zero_point + data / scale
    return max(0, min(round(transformed), 255))


def _relu6():
    def _impl(inputs, input_type):
        assert len(inputs) == 4, "Input quant params not found in op inputs"
        input_scale = inputs[2]
        input_zero_point = inputs[3]
        six = quantize_scalar(6., input_scale, input_zero_point)
        return _op.tensor.clip(inputs[0], input_zero_point, six)
    return _impl


def _mul_scalar():
    def _impl(inputs, input_type):
        # refer to aten/src/ATen/native/quantized/cpu/qmul.cpp
        assert len(inputs) == 6, "Input quant params not found in op inputs"
        other_val = inputs[1]  # scalar

        if other_val > 0.0:
            # only scale change
            return inputs[0]
        elif other_val == 0.0:
            shape = infer_shape(inputs[0])
            return _op.full(_expr.const(0), shape, dtype="uint8")
        else:
            q_min = 0
            q_max = 255
            bias = _expr.const(q_max + q_min, dtype="int8")
            int8 = bias - _op.cast(inputs[0], "int8")
            return _op.cast(int8, "uint8")

    return _impl


convert_map = {
    'aten::quantize_per_tensor': _quantize_per_tensor(),
    'quantized::conv2d_relu': _quantized_conv2d(True),
    'aten::dequantize': _dequantize(),
    'quantized::conv2d': _quantized_conv2d(),
    'quantized::add_relu': _binop(relay.add, True),
    'quantized::add': _binop(relay.add),
    'quantized::mul_relu': _binop(relay.multiply, True),
    'quantized::mul': _binop(relay.multiply),
    'quantized::linear': _linear(),
    'quantized::linear_relu': _linear(True),
    'quantized::cat': _cat(),
    'quantized::add_scalar': _add_scalar(),
    'quantized::mul_scalar': _mul_scalar(),
    'quantized::relu6': _relu6()
}
