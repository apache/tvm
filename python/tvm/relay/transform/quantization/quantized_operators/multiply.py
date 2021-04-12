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
from tvm.relay.transform.quantization.quantized_operators import utils


def generate_generic_quantized_multiply(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    internal_accumulation_dtype: str = "float32",
    simulated_accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    if output_qparams is None and (input1_qparams is None or input2_qparams is None):
        raise ValueError(
            "Must give either the output qparams or both input qparams to infer output qparams!"
        )

    if output_qparams is None:
        output_qparams = utils.QParams(
            (input1_qparams.scale_factor * input2_qparams.scale_factor),
            relay.const(0, dtype=simulated_accumulation_dtype),
            simulated_accumulation_dtype,
        )
        input1, input2 = utils.quantize_inputs(
            internal_accumulation_dtype, input1, input1_qparams, input2, input2_qparams
        )
        input1_zero_point, input2_zero_point = utils.cast_all(
            internal_accumulation_dtype, input1_qparams.zero_point, input2_qparams.zero_point
        )
        output_term = (input1 - input1_zero_point) * (input2 - input2_zero_point)
    else:
        input_qparams = utils.QParams(
            relay.sqrt(output_term.scale_factor),
            output_term.zero_point,
            simulated_accumulation_dtype,
        )
        input1, input2 = utils.quantize_inputs(
            internal_accumulation_dtype, input1, input_qparams, input2, input_qparams
        )
        output_term = (input1 - input_qparams.zero_point) * (
            input2 - input_qparams.zero_point
        ) + output_term.zero_point

    if dequantize:
        output_term = utils.dequantize_expr(
            internal_accumulation_dtype, output_term, output_qparams
        )

    # TODO: simulate the effects of overflow
    return output_term, output_qparams


def generate_static_quantized_multiply(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_multiply(
        input1,
        input2,
        output_qparams,
        input1_qparams=input1_qparams,
        input2_qparams=input2_qparams,
        internal_accumulation_dtype=accumulation_dtype,
        simulated_accumulation_dtype=accumulation_dtype,
        dequantize=dequantize,
    )


def generate_simulated_quantized_multiply(
    input1: tvm.relay.Expr,
    input2: tvm.relay.Expr,
    output_qparams: Optional[utils.QParams],
    input1_qparams: Optional[utils.QParams] = None,
    input2_qparams: Optional[utils.QParams] = None,
    accumulation_dtype: str = "int32",
    dequantize: bool = True,
) -> Tuple[tvm.relay.Expr, utils.QParams]:
    return generate_generic_quantized_multiply(
        input1,
        input2,
        output_qparams,
        input1_qparams=input1_qparams,
        input2_qparams=input2_qparams,
        internal_accumulation_dtype="float32",
        simulated_accumulation_dtype=accumulation_dtype,
        dequantize=dequantize,
    )


def example_multiply_simulated(seed=42):
    np.random.seed(seed=seed)
    a_arr = np.random.uniform(-10, 10, size=(5, 10)).astype("float32")
    b_arr = np.random.uniform(-10, 10, size=(5, 10)).astype("float32")

    var_creator = utils.AffineQuantizationVarCreator()
    a = relay.var("a")
    b = relay.var("b")
    a_qparams = var_creator.get_qparams("a")
    b_qparams = var_creator.get_qparams("b")
    mul_output, output_qparams = generate_simulated_quantized_multiply(
        a, b, None, a_qparams, b_qparams, dequantize=True
    )
    f = relay.Function(
        [
            a,
            b,
            a_qparams.scale_factor,
            a_qparams.zero_point,
            b_qparams.scale_factor,
            b_qparams.zero_point,
        ],
        mul_output,
    )
    print(f)

    actual_a_qparams = utils.get_quantization_parameters(a_arr, True, 8)
    actual_b_qparams = utils.get_quantization_parameters(b_arr, True, 8)

    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    result = intrp.evaluate(f)(
        a_arr,
        b_arr,
        actual_a_qparams.scale_factor,
        actual_a_qparams.zero_point,
        actual_b_qparams.scale_factor,
        actual_b_qparams.zero_point,
    ).asnumpy()

    print("Quantized result:")
    print(result)
    print()
    print("FP32 result:")
    print(a_arr * b_arr)


if __name__ == "__main__":
    # Test that the sim_q and static_q get the same results
    example_multiply_simulated(seed=42)
