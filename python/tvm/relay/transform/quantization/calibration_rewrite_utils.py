from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import tvm
import tvm.testing as testing
from tvm import relay
from tvm.relay import testing

QParams = NamedTuple(
    "QParams", [("scale_factor", tvm.relay.Expr), ("zero_point", tvm.relay.Expr), ("dtype", str)]
)


def quantize_var_or_const_node(
    relay_node: Union[tvm.relay.Constant, tvm.relay.Var], axis=-1, q_dtype: str = "int8"
) -> Tuple[tvm.relay.Expr, Optional[QParams], Optional[QParams]]:
    """Takes a var or const node and inserts a quantization operation with specified qbits.

    Returns the call node to the quantization op and returns the scale and zero_point variables
    for the operation.
    """
    if not isinstance(relay_node, tvm.relay.Constant) and not isinstance(relay_node, tvm.relay.Var):
        raise ValueError(f"Node is not a constant or var, has type {type(relay_node)}")

    var_name = relay_node.name_hint
    quantized_node, output_qparams = quantize_node(relay_node, var_name, q_dtype, axis=axis)

    return quantized_node, None, output_qparams


def get_name_of_node(relay_node: tvm.relay.Expr) -> str:
    if isinstance(relay_node, tvm.relay.Constant) or isinstance(relay_node, tvm.relay.Var):
        return relay_node.name_hint
    elif isinstance(relay_node, tvm.relay.Call):
        return relay_node.op.name
    else:
        raise ValueError(f"Don't know how to get name for {type(relay_node)}. Please implement!")


def quantize_call_node(
    relay_node: tvm.relay.Call, axis=-1, q_dtype: str = "int8", recursive: bool = False
) -> Tuple[tvm.relay.Expr, Optional[QParams], Optional[QParams]]:
    call_op_name = relay_node.op.name
    operator = tvm.ir.op.Op.get(call_op_name)

    q_args = []
    qparams_args = []
    for arg in relay_node.args:
        quantization_node, qparams_out = quantize_node(
            arg, get_name_of_node(arg), q_dtype, axis=axis
        )
        q_args.append(quantization_node)
        qparams_args.append(qparams_out)

    # TODO: copy relay_node attrs, type_args, better
    out = tvm.relay.Call(
        operator, q_args, attrs=relay_node.attrs, type_args=relay_node.type_args, span=None
    )

    # TODO: calculate output qparams better
    return (
        out,
        qparams_args,
        QParams(tvm.relay.Constant(tvm.nd.array(1)), tvm.relay.Constant(tvm.nd.array(0)), "int32"),
    )


def dequantize_node(
    relay_node: tvm.relay.Expr,
    scale_factor: tvm.relay.Expr,
    zero_point: tvm.relay.Expr,
    in_dtype: str,
    axis: int = -1,
) -> tvm.relay.Expr:
    return relay.qnn.op.simulated_dequantize(
        out, scale_factor, zero_point, axis=axis, in_dtype=in_dtype
    )


def quantize_node(relay_node: tvm.relay.Expr, name: str, q_dtype: str, axis=-1):
    # Return a version of the
    scale = relay.Var(f"{name}.scale")
    zero_point = relay.Var(f"{name}.zero_point")
    return (
        relay.qnn.op.simulated_quantize(
            relay_node, scale, zero_point, axis=axis, out_dtype=q_dtype
        ),
        QParams(scale, zero_point, q_dtype),
    )


if __name__ == "__main__":
    data = relay.var("data")
    weight = relay.var("weight")
    out = relay.nn.conv2d(data, weight)
    bias = relay.var("bias")
    out = relay.add(out, bias)
    print(relay.Function([], out))

    out, qparams_ins, qparam_out = quantize_call_node(out)
    print(relay.Function([], out))

    """
    # TODO fuse bn op.
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

    """
    d = relay.var("d")
    a = relay.add(d, d)
    f = relay.Function([d], a)
    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    op_res = intrp.evaluate(f)(1)
    print("*" * 10, op_res)
    """
