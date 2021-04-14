import numpy as np
import tvm
import tvm.testing as testing
from tvm import relay
from tvm.relay import testing


# This is how quantization works for
def quantize_everything_helper(relay_call: tvm.relay.Call):
    if isinstance(relay_call, relay.Function):
        relay_call = relay_call.body

    if isinstance(relay_call, relay.TupleGetItem):
        op_name = relay_call.tuple_value.op.name
    else:
        op_name = relay_call.op.name

    # TODO: blacklist certain ops from being used e.g. softmax

    # TODO: insert other fields
    func_factory = tvm.ir.op.Op.get(op_name)

    args_op = []
    quantization_info = []
    # TODO: insert simulated quantize/dequantize ops
    for sub_node in relay_call.args:
        if isinstance(sub_node, tvm.relay.Call):
            node_to_quantize = quantize_everything_helper(sub_node)
            var_name = node_to_quantize.op.name + "_output"
        elif isinstance(sub_node, tvm.relay.Var) or isinstance(sub_node, tvm.relay.Constant):
            node_to_quantize = sub_node
            var_name = sub_node.name_hint
        elif isinstance(sub_node, relay.TupleGetItem):
            node_to_quantize = quantize_everything_helper(sub_node)
            var_name = node_to_quantize.tuple_value.op.name + "_output"
        else:
            raise ValueError(f"Unknown type {type(sub_node)}")

        scale = relay.Var(f"{op_name}.{var_name}.scale")
        zero_point = relay.Var(f"{op_name}.{var_name}.zero_point")
        quantized_node = relay.qnn.op.quantize(node_to_quantize, scale, zero_point)
        quantization_info.append((scale, zero_point))
        args_op.append(quantized_node)

    out = tvm.relay.Call(
        func_factory, args_op, attrs=relay_call.attrs, type_args=relay_call.type_args, span=None
    )

    # TODO: calculate final output scales for each op the correct way
    final_scale = relay.Constant(tvm.nd.array(1))
    final_zero_point = relay.Constant(tvm.nd.array(0))
    for scale, zero_point in quantization_info:
        final_scale = final_scale * scale
    return relay.qnn.op.dequantize(out, final_scale, final_zero_point)


if __name__ == "__main__":
    """
    data = relay.var("data")
    weight = relay.var("weight")
    out = relay.nn.conv2d(data, weight)
    bias = relay.var("bias")
    out = relay.add(out, bias)
    print(relay.Function([], out))
    print(relay.Function([], quantize_everything_helper(out)))


    # TODO fuse b
    print(
        relay.Function(
            [],
            relay.testing.resnet.get_net(
                num_layers=18, num_classes=2, batch_size=1, image_shape=(3, 100, 100)
            ).body,
        )
    )

    print(
        relay.Function(
            [],
            quantize_everything_helper(
                relay.testing.resnet.get_net(
                    num_layers=18, num_classes=2, batch_size=1, image_shape=(3, 100, 100)
                )
            ),
        )
    )
    """

    d = relay.var("d")
    e = relay.var("e")
    a = relay.add(d, e)
    f = relay.Function([d, e], a)
    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    op_res = intrp.evaluate(f)(np.int8(120), np.int16(10))
    print("*" * 10, op_res)
