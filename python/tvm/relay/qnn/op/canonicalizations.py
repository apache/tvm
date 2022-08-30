# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Consist of utilities and methods for lowering QNN into mainline relay."""
from typing import Callable

import numpy as np
import tvm
from tvm import relay


def run_const_expr(expr: "relay.Expr") -> np.ndarray:
    """Evaluate a const expression, receiving result as np array."""
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
    Return a table where each input indexes to the output quantizing the given function.

    Note this also supports mapping unsigned and signed integers to each other.

    Args:
      floating_point_func: The numpy function which this table is to approximate
      input_scale: The scale of the quantized input tensor.
      input_zero_point: The zero point of the quantized input tensor.
      output_scale: The scale of the quantized output tensor.
      output_zero_point: The zero point of the quantized output tensor.
      in_axis: The axis for multi-channel quantization of the input if applicable.
      out_axis: The axis for multi-channel quantization of the output if applicable.
      in_dtype: The dtype of the input tensor.
      out_dtype: The wanted dtype of the output tensor.

    Returns:
      A numpy array where values in quantized space will index to the output in quantized space
      approximating the given function.
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
    inputs_quantized = np.array(range(0, 2**num_bits), dtype=f"uint{num_bits}")

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
    Create a quantized version of the given floating point unary operation using table lookup.

    Args:
      input_arg: The quantized input to the final function.
      floating_point_func: The numpy function which this table is to approximate
      in_scale: The scale of the quantized input tensor.
      in_zero_point: The zero point of the quantized input tensor.
      out_scale: The scale of the quantized output tensor.
      out_zero_point: The zero point of the quantized output tensor.
      in_axis: The axis for multi-channel quantization of the input if applicable.
      out_axis: The axis for multi-channel quantization of the output if applicable.
      in_dtype: The dtype of the input tensor.
      out_dtype: The wanted dtype of the output tensor.

    Returns:
      A Relay expression representing a quantized version of the given function.
    """

    # TODO: handle multi-channel q, below will fail with multi-channel q
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
    index_tensor = relay.reinterpret(input_arg, f"uint{in_dtype_num_bits}")
    result = relay.take(lookup_table, index_tensor, axis=0, mode="fast")
    return result
