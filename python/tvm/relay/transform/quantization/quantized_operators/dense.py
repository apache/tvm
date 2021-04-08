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

QParams = NamedTuple(
    "QParams", [("scale_factor", tvm.relay.Expr), ("zero_point", tvm.relay.Expr), ("dtype", str)]
)


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


def dense_simulated(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: QParams,
    weight_qparams: QParams,
    internal_accumulation_dtype: str = "float32",
    simulated_accumulation_dtype: str = "int32",
    out_units: Optional[int] = None,
    in_units: Optional[int] = None,
) -> Tuple[tvm.relay.Expr, QParams]:
    if in_units is None:
        units_in = weight.checked_type.shape[-1]
    if out_units is None:
        out_units = weight.checked_type.shape[-2]

    data_scale, data_zero_point, data_dtype = data_qparams
    weight_scale, weight_zero_point, weight_dtype = weight_qparams

    weight_zero_point = relay.cast(weight_zero_point, internal_accumulation_dtype)
    data_zero_point = relay.cast(data_zero_point, internal_accumulation_dtype)

    first_term = nn.dense(data, weight, units=out_units)

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
    return (first_term - second_term - third_term + fourth_term), QParams(
        data_scale * weight_scale, 0, simulated_accumulation_dtype
    )


if __name__ == "__main__":
    var_creator = AffineQuantizationVarCreator()

    IN_UNITS = 1000
    OUT_UNITS = 10
    N = 5

    data_arr = np.random.randint(-10, 10, size=(N, IN_UNITS)).astype("float32")
    weight_arr = np.random.randint(-10, 10, size=(OUT_UNITS, IN_UNITS)).astype("float32")

    data = relay.Var("data")
    weight = relay.Var("weight")
    data_qparams = var_creator.get_qparams("dense_data")
    weight_qparams = var_creator.get_qparams("dense_weight")
    dense_output, output_qparams = dense_simulated(
        data, weight, data_qparams, weight_qparams, in_units=IN_UNITS, out_units=OUT_UNITS
    )

    # typed_expr = relay.testing.run_infer_type(data)

    f = relay.Function(
        [data, weight, data_qparams.zero_point, weight_qparams.zero_point], dense_output
    )
    mod = tvm.ir.IRModule.from_expr(f)
    intrp = relay.create_executor(kind="debug", mod=mod)
    output = intrp.evaluate(f)(data_arr, weight_arr, 0, 0).asnumpy()
    print(output)
    print(f)
    print(relay.cast)


"""
def dense_simulated(
    data: tvm.relay.Expr,
    weight: tvm.relay.Expr,
    data_qparams: QParams,
    weight_qparams: QParams,
    in_dimension: int,
    out_dimension: int,
    out_dtype: Optional[str] = "int32",
):

Expr DenseFirstTerm(const Expr& quantized_data, const Expr& quantized_kernel,
                    const DenseAttrs* attrs) {
  return Dense(quantized_data, quantized_kernel, attrs->units, attrs->out_dtype);
}

Expr DenseSecondTerm(const Expr& quantized_data, const Expr& kernel_zero_point) {
  Array<Integer> axes = {1};
  return Multiply(kernel_zero_point,
                  Sum(Cast(quantized_data, DataType::Int(32)), axes, true, false));
}

Expr DenseThirdTerm(const Expr& quantized_kernel, const Expr& input_zero_point) {
  Array<Integer> axes = {1};
  return Multiply(input_zero_point,
                  Sum(Cast(quantized_kernel, DataType::Int(32)), axes, false, false));
}

Expr DenseFourthTerm(int input_zero_point_int, int kernel_zero_point_int, int reduction_dim_size) {
  int32_t scalar_term = input_zero_point_int * kernel_zero_point_int * reduction_dim_size;
  return MakeConstantScalar(DataType::Int(32), scalar_term);
}

Expr DenseFourthTerm(const Expr& input_zero_point, const Expr& kernel_zero_point,
                     int reduction_dim_size) {
  auto reduction_dim = MakeConstantScalar(DataType::Int(32), reduction_dim_size);
  return Multiply(Multiply(input_zero_point, kernel_zero_point), reduction_dim);
}
"""
