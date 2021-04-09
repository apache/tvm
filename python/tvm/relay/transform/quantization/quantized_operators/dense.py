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
from tvm.relay.transform.quantization import calibration_rewrite_utils
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

    # return first_term, second_term, third_term, fourth_term
    return (first_term - second_term - third_term + fourth_term), common.QParams(
        data_scale * weight_scale, 0, simulated_accumulation_dtype
    )


def generate_static_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: common.QParams,
    weight_qparams: common.QParams,
    accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
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
    )


def generate_simulated_quantized_dense(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: common.QParams,
    weight_qparams: common.QParams,
    simulated_accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
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
    )


def example_dense_simulated(n, in_units, out_units, seed=42):
    np.random.seed(seed=seed)
    data_arr = np.random.randint(-10, 10, size=(n, in_units)).astype("float32")
    weight_arr = np.random.randint(-10, 10, size=(out_units, in_units)).astype("float32")

    var_creator = common.AffineQuantizationVarCreator()
    data = relay.Var("data")
    weight = relay.Var("weight")
    data_qparams = var_creator.get_qparams("dense_data")
    weight_qparams = var_creator.get_qparams("dense_weight")
    dense_output, output_qparams = generate_simulated_quantized_dense(
        data, weight, data_qparams, weight_qparams, in_units=in_units, out_units=out_units
    )
    f = relay.Function(
        [data, weight, data_qparams.zero_point, weight_qparams.zero_point], dense_output
    )
    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    return intrp.evaluate(f)(data_arr, weight_arr, 0.0, 0.0).asnumpy()


def example_dense_static_quantized(n, in_units, out_units, seed=42):
    np.random.seed(seed=seed)
    data_arr = np.random.randint(-10, 10, size=(n, in_units)).astype("int8")
    weight_arr = np.random.randint(-10, 10, size=(out_units, in_units)).astype("int8")

    var_creator = common.AffineQuantizationVarCreator()
    data = relay.Var("data")
    weight = relay.Var("weight")
    data_qparams = var_creator.get_qparams("dense_data")
    weight_qparams = var_creator.get_qparams("dense_weight")
    dense_output, output_qparams = generate_static_quantized_dense(
        data, weight, data_qparams, weight_qparams, in_units=in_units, out_units=out_units
    )
    f = relay.Function(
        [data, weight, data_qparams.zero_point, weight_qparams.zero_point], dense_output
    )
    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    return intrp.evaluate(f)(data_arr, weight_arr, 0, 0).asnumpy()


if __name__ == "__main__":
    # Test that the sim_q and static_q get the same results
    dense_sim_q_example = example_dense_simulated(5, 100, 10, seed=42)
    dense_static_q_example = example_dense_static_quantized(5, 100, 10, seed=42)
    print((dense_sim_q_example == dense_static_q_example).all())
