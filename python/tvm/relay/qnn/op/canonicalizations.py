from typing import Callable

import numpy as np
import tvm
from tvm import relay


# TODO: replace with constant folding
def run_const_expr(expr: "relay.Expr") -> np.ndarray:
    mod = tvm.IRModule.from_expr(expr)
    vm_exe = relay.create_executor("vm", mod=mod)
    return vm_exe.evaluate()().asnumpy()


def create_integer_lookup_table(
    floating_point_func: Callable[[np.ndarray], np.ndarray],
    input_scale: "relay.Expr",
    input_zero_point: "relay.Expr",
    output_scale: "relay.Expr",
    output_zero_point: "relay.Expr",
    in_axis: int = -1,
    out_axis: int = -1,
    in_dtype: str = "uint8",
    out_dtype: str = "uint8",
) -> np.ndarray:
    """
    TODO
    """
    if not np.issubdtype(np.dtype(in_dtype), np.integer) or not np.issubdtype(
        np.dtype(out_dtype), np.integer
    ):
        raise ValueError(
            f"Only integer dtypes allowed got {in_dtype} and {out_dtype} for in and out dtypes."
        )

    dtype_info = np.iinfo(in_dtype)

    num_bits = dtype_info.bits

    # Use TVMs quantization methods via relay to be consistent
    # inputs_quantized = np.array(range(dtype_info.min, dtype_info.max + 1)).astype(in_dtype)

    # First generate a list of all num_bit integer patterns
    inputs_quantized = np.array(range(0, 2 ** num_bits), dtype=f"uint{num_bits}")

    # Reinterpret bits as the real datatype
    # Note what we are doing here is a bit tricky, the canonical view of our lookup table
    # is using the uintX version. When we run the lookup in the relay graph, we cast the
    # bit pattern back into this form.
    inputs_quantized = inputs_quantized.view(in_dtype)
    inputs_quantized = relay.const(inputs_quantized, dtype=in_dtype)
    inputs_dequantized = run_const_expr(
        relay.qnn.op.dequantize(
            inputs_quantized,
            input_scale=input_scale,
            input_zero_point=input_zero_point,
            axis=in_axis,
        )
    )

    output_dequantized = relay.const(floating_point_func(inputs_dequantized))
    output_quantized = run_const_expr(
        relay.qnn.op.quantize(
            output_dequantized, output_scale, output_zero_point, out_axis, out_dtype
        )
    )

    return output_quantized


def create_integer_lookup_op(
    input_arg: "relay.Expr",
    floating_point_func: Callable[[np.array], np.array],
    in_scale: "relay.Expr",
    in_zero_point: "relay.Expr",
    out_scale: "relay.Expr",
    out_zero_point: "relay.Expr",
    in_axis: int = -1,
    out_axis: int = -1,
    in_dtype: str = "uint8",
    out_dtype: str = "uint8",
) -> "relay.Expr":
    """
    TODO
    """
    # TODO: handle multi-channel q
    in_scale = in_scale.data.numpy().item()
    in_zero_point = in_zero_point.data.numpy().item()
    out_scale = out_scale.data.numpy().item()
    out_zero_point = out_zero_point.data.numpy().item()

    lookup_table = create_integer_lookup_table(
        floating_point_func,
        relay.const(in_scale),
        relay.const(in_zero_point, dtype="int32"),
        relay.const(out_scale),
        relay.const(out_zero_point, dtype="int32"),
        in_axis=in_axis,
        in_dtype=in_dtype,
        out_axis=out_axis,
        out_dtype=out_dtype,
    )

    in_dtype_info = np.iinfo(in_dtype)
    in_dtype_num_bits = in_dtype_info.bits

    lookup_table = relay.const(lookup_table)
    index_tensor = relay.reshape(input_arg, [-1])
    index_tensor = relay.reinterpret(index_tensor, f"uint{in_dtype_num_bits}")
    result = relay.gather(lookup_table, -1, index_tensor)
    result = relay.reshape_like(result, input_arg)
    return result


"""
# TODO: better error messages if reference functions fail in FQ2I pass
register_unary_elementwise_table_lookup_op("tanh", np.tanh)
register_unary_elementwise_table_lookup_op("erf", special.erf)
register_unary_elementwise_table_lookup_op("exp", np.exp)
register_unary_elementwise_table_lookup_op("sigmoid", lambda x: 1 / (1 + np.exp(-x)))
"""
