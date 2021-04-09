from typing import *

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
from tvm.relay.transform.quantization.quantized_operators import common


def generate_generic_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: common.QParams,
    weight_qparams: common.QParams,
    internal_accumulation_dtype: str = "float32",
    simulated_accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
    dequantize: bool = False,
) -> Tuple[tvm.relay.Expr, common.QParams]:
    """TODO"""

    # TODO: figure out whether we need this or we can always have the
    # callee pass it in
    if in_units is None:
        units_in = weight.checked_type.shape[-1]
    if out_units is None:
        out_units = weight.checked_type.shape[-2]

    data_scale, data_zero_point, data_dtype = data_qparams
    weight_scale, weight_zero_point, weight_dtype = weight_qparams

    # This means use simulated operations
    if internal_accumulation_dtype == "float32":
        quantize_op = relay.qnn.op.simulated_quantize
        dequantize_op = relay.qnn.op.simulated_dequantize
    elif "int" in internal_accumulation_dtype:
        quantize_op = relay.qnn.op.quantize
        dequantize_op = relay.qnn.op.dequantize
    else:
        raise ValueError(
            f"Unknown quantization from specified internal accumulation dtype {internal_accumulation_dtype}"
        )
    data = quantize_op(
        data=data, output_scale=data_scale, output_zero_point=data_zero_point, out_dtype=data_dtype
    )
    weight = quantize_op(
        data=weight,
        output_scale=weight_scale,
        output_zero_point=weight_zero_point,
        out_dtype=weight_dtype,
    )

    # Assume this casting is a no-op in the case we are casting back to itself
    weight_zero_point = relay.cast(weight_zero_point, internal_accumulation_dtype)
    data_zero_point = relay.cast(data_zero_point, internal_accumulation_dtype)

    data = relay.cast(data, internal_accumulation_dtype)
    weight = relay.cast(weight, internal_accumulation_dtype)

    first_term = nn.dense(data, weight, units=out_units, out_dtype=internal_accumulation_dtype)

    # The fields for the reduction operations to make things clear
    axis = [1]
    keep_dims = True
    exclude = False

    second_term = tensor._make.sum(data, axis, keep_dims, exclude) * weight_zero_point

    # The fields for the reduction operations to make things clear
    axis = [1]
    keep_dims = False
    exclude = False
    third_term = tensor._make.sum(weight, axis, keep_dims, exclude) * data_zero_point

    fourth_term = (
        relay.Constant(tvm.nd.array(np.array(in_units, dtype=internal_accumulation_dtype)))
        * data_zero_point
        * weight_zero_point
    )

    # TODO: simulate overflow for other data types
    output_qparams = common.QParams(
        data_scale * weight_scale, relay.const(0), simulated_accumulation_dtype
    )

    # Make graph more parallizable by being exact with order of operations
    output_term = (first_term - second_term) - (third_term - fourth_term)

    if dequantize:
        quantization_axis = -1
        output_term = dequantize_op(
            data=output_term,
            input_scale=output_qparams.scale_factor,
            input_zero_point=output_qparams.zero_point,
            in_dtype=output_qparams.dtype,
        )

    return output_term, output_qparams


def generate_static_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: common.QParams,
    weight_qparams: common.QParams,
    accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, common.QParams]:
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
    )


def generate_simulated_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: common.QParams,
    weight_qparams: common.QParams,
    simulated_accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, common.QParams]:
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
    )


def example_dense_simulated(n, in_units, out_units, seed=42):
    np.random.seed(seed=seed)
    data_arr = np.random.uniform(-10, 10, size=(n, in_units)).astype("float32")
    weight_arr = np.random.uniform(-10, 10, size=(out_units, in_units)).astype("float32")

    var_creator = common.AffineQuantizationVarCreator()
    data = relay.var("data")
    weight = relay.var("weight")
    data_qparams = var_creator.get_qparams("dense_data")
    weight_qparams = var_creator.get_qparams("dense_weight")
    dense_output, output_qparams = generate_simulated_quantized_dense(
        data, weight, data_qparams, weight_qparams, in_units=in_units, out_units=out_units
    )
    f = relay.Function(
        [
            data,
            weight,
            data_qparams.scale_factor,
            data_qparams.zero_point,
            weight_qparams.scale_factor,
            weight_qparams.zero_point,
        ],
        dense_output,
    )
    print(f)

    actual_data_qparams = common.get_quantization_parameters(data_arr, True, 8)
    actual_weight_qparams = common.get_quantization_parameters(data_arr, True, 8)

    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        data_arr,
        weight_arr,
        actual_data_qparams.scale_factor,
        actual_data_qparams.zero_point,
        actual_weight_qparams.scale_factor,
        actual_weight_qparams.zero_point,
    ).asnumpy()

    print("Quantized result:")
    print(result)
    print()
    print("FP32 result:")
    print(data_arr @ weight_arr.T)


if __name__ == "__main__":
    # Test that the sim_q and static_q get the same results
    example_dense_simulated(5, 5, 10, seed=42)
