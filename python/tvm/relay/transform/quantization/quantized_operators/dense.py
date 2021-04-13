gfrom typing import *

import numpy as np
import tvm
from tvm import relay
from tvm.relay.dataflow_pattern import (
    DFPatternCallback,
    _DFPatternCallback,
    is_constant,
    is_op,
    wildcard,
)
from tvm.relay.op import nn, tensor
from tvm.relay.transform.quantization.quantized_operators import utils


def generate_generic_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: utils.QParams,
    weight_qparams: utils.QParams,
    internal_accumulation_dtype: str = "float32",
    simulated_accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
    dequantize: bool = False,
    bias: Optional[tvm.relay.Expr] = None,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    """TODO"""

    # TODO: figure out whether we need this or we can always have the
    # callee pass it in
    if in_units is None:
        units_in = weight.checked_type.shape[-1]
    if out_units is None:
        out_units = weight.checked_type.shape[-2]

    data, weight = utils.quantize_inputs(
        internal_accumulation_dtype,
        data,
        data_qparams,
        weight,
        weight_qparams,
    )

    # Assume this casting is a no-op in the case we are casting back to itself
    weight_zero_point, data_zero_point, data_casted, weight_casted = utils.cast_all(
        internal_accumulation_dtype,
        weight_qparams.zero_point,
        data_qparams.zero_point,
        data,
        weight,
    )

    first_term = nn.dense(data, weight, units=out_units, out_dtype=internal_accumulation_dtype)
    if weight_zero_point == relay.const(0):
        second_term = relay.const(0)
    else:
        second_term = (
            relay.op.sum(data_casted, axis=1, keepdims=True, exclude=False) * weight_zero_point
        )

    if data_zero_point == relay.const(0):
        third_term = relay.const(0)
    else:
        third_term = (
            relay.op.sum(weight_casted, axis=1, keepdims=False, exclude=False) * data_zero_point
        )

    if weight_zero_point == relay.const(0) or data_zero_point == relay.const(0):
        fourth_term = 0
    else:
        fourth_term = (
            relay.const(np.array(in_units, dtype=internal_accumulation_dtype))
            * data_zero_point
            * weight_zero_point
        )

    # TODO: simulate overflow for other data types

    output_qparams = utils.QParams(
        data_qparams.scale_factor * weight_qparams.scale_factor,
        relay.const(0, dtype=simulated_accumulation_dtype),
        simulated_accumulation_dtype,
    )

    # Make graph more parallizable by manually creating tree of computation
    output_term = (first_term - second_term) - (third_term - fourth_term)

    if bias is not None:
        bias = utils.quantize_inputs(internal_accumulation_dtype, bias, output_qparams)
        output_term += bias

    if dequantize:
        output_term = utils.dequantize_expr(
            internal_accumulation_dtype, output_term, output_qparams
        )

    return output_term, output_qparams


def generate_static_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: utils.QParams,
    weight_qparams: utils.QParams,
    accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
    dequantize: bool = True,
    bias: Optional[tvm.relay.Expr] = None,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_dense(
        data,
        weight,
        data_qparams,
        weight_qparams,
        internal_accumulation_dtype=accumulation_dtype,
        simulated_accumulation_dtype=accumulation_dtype,
        out_units=out_units,
        in_units=in_units,
        dequantize=dequantize,
        bias=bias,
    )


def generate_simulated_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: utils.QParams,
    weight_qparams: utils.QParams,
    simulated_accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
    dequantize: bool = True,
    bias: Optional[tvm.relay.Expr] = None,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_dense(
        data,
        weight,
        data_qparams,
        weight_qparams,
        internal_accumulation_dtype="float32",
        simulated_accumulation_dtype=simulated_accumulation_dtype,
        out_units=out_units,
        in_units=in_units,
        dequantize=dequantize,
        bias=bias,
    )


def example_dense_simulated(n, in_units, out_units, seed=42):
    np.random.seed(seed=seed)
    data_arr = np.random.uniform(-10, 10, size=(n, in_units)).astype("float32")
    weight_arr = np.random.uniform(-10, 10, size=(out_units, in_units)).astype("float32")
    bias_arr = np.random.uniform(-100, 100, size=(n, out_units)).astype("float32")

    var_creator = utils.AffineQuantizationVarCreator()
    data = relay.var("data")
    weight = relay.var("weight")
    bias = relay.var("bias")
    data_qparams = var_creator.get_qparams("dense_data")
    weight_qparams = var_creator.get_qparams("dense_weight")
    dense_output, output_qparams = generate_simulated_quantized_dense(
        data,
        weight,
        data_qparams,
        weight_qparams,
        in_units=in_units,
        out_units=out_units,
        bias=bias,
    )
    f = relay.Function(
        [
            data,
            weight,
            data_qparams.scale_factor,
            data_qparams.zero_point,
            weight_qparams.scale_factor,
            weight_qparams.zero_point,
            bias,
        ],
        dense_output,
    )
    print(f)

    actual_data_qparams = utils.get_quantization_parameters(data_arr, True, 8)
    actual_weight_qparams = utils.get_quantization_parameters(weight_arr, True, 8)

    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        data_arr,
        weight_arr,
        actual_data_qparams.scale_factor,
        actual_data_qparams.zero_point,
        actual_weight_qparams.scale_factor,
        actual_weight_qparams.zero_point,
        bias_arr,
    ).asnumpy()

    print("Quantized result:")
    print(result)
    print()
    print("FP32 result:")
    print(data_arr @ weight_arr.T + bias_arr)


if __name__ == "__main__":
    # Test that the sim_q and static_q get the same results
    example_dense_simulated(5, 5, 10, seed=42)
