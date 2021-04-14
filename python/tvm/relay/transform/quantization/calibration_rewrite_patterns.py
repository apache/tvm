from typing import Callable, List, NamedTuple, Optional, Tuple

import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    _DFPatternCallback,
    is_constant,
    is_op,
    wildcard,
)
from tvm.relay.transform.quantization import calibration_rewrite_utils

QParams = NamedTuple(
    "QParams", [("scale_factor", tvm.relay.Expr), ("zero_point", tvm.relay.Expr), ("dtype", str)]
)


def get_affine_quantized_op(op_name: str):
    raise NotImplementedError("Stub!")


def get_affine_simulated_quantized_op(op_name: str):
    raise NotImplementedError("Stub!")


def get_symmetric_zero_point(dtype: str) -> tvm.relay.Constant:
    # TODO: implement for dtypes
    return tvm.relay.Constant(tvm.nd.array(0))


class AffineQuantizationVarCreator:
    def __init__(self):
        self.ref_count = 0
        self.qparams = []

    def get_qparams(self, name_hint: str, dtype: str = "int8") -> QParams:
        scale = relay.Var(f"{name_hint}.scale")
        zero_point = relay.Var(f"{name_hint}.zero_point")
        qparam = QParams(scale, zero_point, dtype)
        self.qparams.append(qparam)
        self.ref_count += 1
        return qparam


def insert_simulated_quantize(
    relay_node: tvm.relay.Expr,
    qparam: Optional[QParams],
    axis=-1,
) -> tvm.relay.Expr:
    scale, zero_point, out_dtype = qparam
    return relay.qnn.op.simulated_quantize(
        relay_node, scale, zero_point, axis=axis, out_dtype=out_dtype
    )


def insert_simulated_dequantize(
    relay_node: tvm.relay.Expr,
    qparam: Optional[QParams],
    axis=-1,
) -> tvm.relay.Expr:
    scale, zero_point, out_dtype = qparam
    return relay.qnn.op.simulated_dequantize(relay_node, scale, zero_point, axis=axis)


def get_simulated_conv2d(*args):
    operator = tvm.ir.op.Op.get("nn.conv2d")
    return tvm.relay.Call(operator, args)


def quantize_all_inputs(
    name_hint: str,
    dtype: str,
    *args: List[tvm.relay.Expr],
    axis=-1,
) -> Tuple[List[tvm.relay.Expr], List[QParams]]:
    out = []
    qparams = []
    for i, arg in enumerate(args):
        qparam = qparam_manager.get_qparams(name_hint=f"{name_hint}.input_{i}", dtype=dtype)
        qparams.append(qparam)
        out.append(insert_simulated_quantize(arg, qparam, axis=axis))
    return out, qparams


def get_affine_simulated_quantized_conv(
    input_tensor: tvm.relay.Expr,
    conv_weight: tvm.relay.Expr,
    conv2d: tvm.relay.Call,
    qparam_manager: AffineQuantizationVarCreator,
    input_dtype: str = "int8",
    axis=-1,
):
    quantized_inputs, qparams_inputs = quantize_all_inputs(
        "nn.conv2d_sim", input_dtype, input_tensor, conv_weight
    )

    qparam_input_tensor, qparam_weight_tensor = qparams_inputs
    input_tensor, conv_weight = quantized_inputs
    conv_sim = get_simulated_conv2d(input_tensor, conv_weight)

    # The simulated conv2d assumes a symmetric zero point for the output.
    # It also assumes an int32 accumulation buffer.
    return insert_simulated_dequantize(
        conv_sim,
        QParams(
            qparam_weight_tensor.scale_factor * qparam_input_tensor.scale_factor,
            get_symmetric_zero_point(input_dtype),
            "int32",
        ),
        axis=axis,
    )


if __name__ == "__main__":
    data = relay.var("data")
    weight = relay.var("weight")
    out = relay.nn.conv2d(data, weight)
    qparam_manager = AffineQuantizationVarCreator()

    sim_quantized = get_affine_simulated_quantized_conv(data, weight, out, qparam_manager)

    import pdb

    pdb.set_trace()
    print(out)
    print()
    print(sim_quantized)
