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
import tvm
import tvm.script
import tvm.testing
from tvm.target import Target
from tvm.script import tir as T
from tvm.tir.transform.transform import BindTarget

# pylint: disable=no-member,invalid-name,unused-variable


def get_before(dtype: str):
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def main(Aptr: T.handle(dtype), Bptr: T.handle(dtype), Dptr: T.handle(dtype)):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), dtype, data=Aptr)
            B = T.decl_buffer((100,), dtype, data=Bptr)
            D = T.decl_buffer((100,), dtype, data=Dptr)
            C = T.decl_buffer((100,), dtype)
            for i in T.grid(100):
                C[i] = A[i] + B[i]
                D[i] = T.exp(C[i])

    return Before


def promote_f8(f8_dtype: str, promote_dtype: str, v):
    return promote_uint8(f8_dtype, promote_dtype, T.reinterpret("uint8", v))


def cast_to_f8(f8_dtype: str, promote_dtype: str, v):
    return T.reinterpret(f8_dtype, cast_to_uint8(f8_dtype, promote_dtype, v))


def get_after_compute_legalize(dtype: str, promote_dtype: str):
    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(Aptr: T.handle(dtype), Bptr: T.handle(dtype), Dptr: T.handle(dtype)):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), dtype, data=Aptr)
            B = T.decl_buffer((100,), dtype, data=Bptr)
            D = T.decl_buffer((100,), dtype, data=Dptr)
            C = T.decl_buffer((100,), promote_dtype)
            for i in T.grid(100):
                C[i] = promote_f8(dtype, promote_dtype, A[i]) + promote_f8(
                    dtype, promote_dtype, B[i]
                )
                D[i] = cast_to_f8(dtype, promote_dtype, T.exp(C[i]))

    return After


def promote_uint8(f8_dtype: str, promote_dtype: str, v):
    if f8_dtype == "e4m3_float8":
        if promote_dtype == "float16":
            mantissa = T.bitwise_and(
                T.shift_left(T.Cast("uint16", v), T.uint16(7)), T.uint16(0x3FF)
            )
            exponent = T.shift_left(
                T.Cast(
                    "uint16",
                    T.shift_right(T.shift_left(v, T.uint8(1)), T.uint8(4)) + T.uint8(8),
                ),
                T.uint16(10),
            )
            sign = T.shift_left(T.Cast("uint16", T.shift_right(v, T.uint8(7))), T.uint16(15))
            return T.reinterpret("float16", T.bitwise_or(T.bitwise_or(mantissa, exponent), sign))
        else:  # promote_dtype == "float32"
            mantissa = T.bitwise_and(
                T.shift_left(T.Cast("uint32", v), T.uint32(20)), T.uint32(0x7FFFFF)
            )
            exponent = T.shift_left(
                T.Cast(
                    "uint32",
                    T.shift_right(T.shift_left(v, T.uint8(1)), T.uint8(4)) + T.uint8(120),
                ),
                T.uint32(23),
            )
            sign = T.shift_left(T.Cast("uint32", T.shift_right(v, T.uint8(7))), T.uint32(31))
            return T.reinterpret("float32", T.bitwise_or(T.bitwise_or(mantissa, exponent), sign))
    else:  # f8_dtype == "e5m2_float8"
        if promote_dtype == "float16":
            return T.reinterpret("float16", T.shift_left(T.Cast("uint16", v), T.uint16(8)))
        else:  # promote_dtype == "float32"
            mantissa = T.bitwise_and(
                T.shift_left(T.Cast("uint32", v), T.uint32(21)), T.uint32(0x7FFFFF)
            )
            exponent = T.shift_left(
                T.Cast(
                    "uint32",
                    T.shift_right(T.shift_left(v, T.uint8(1)), T.uint8(3)) + T.uint8(112),
                ),
                T.uint32(23),
            )
            sign = T.shift_left(T.Cast("uint32", T.shift_right(v, T.uint8(7))), T.uint32(31))
            return T.reinterpret("float32", T.bitwise_or(T.bitwise_or(mantissa, exponent), sign))


def cast_to_uint8(f8_dtype: str, promote_dtype: str, v):
    if f8_dtype == "e4m3_float8":
        if promote_dtype == "float16":
            uint16_v = T.reinterpret("uint16", v)
            rounding_bias = T.bitwise_and(
                T.shift_right(uint16_v, T.uint16(7)),
                T.uint16(1),
            ) + T.uint16(0x3F)
            uint16_v = uint16_v + rounding_bias
            mantissa = T.bitwise_and(
                T.Cast("uint8", T.shift_right(uint16_v, T.uint8(7))), T.uint8(0x7)
            )
            exponent_before_delta = T.shift_right(T.shift_left(uint16_v, T.uint16(1)), T.uint16(11))
            round_to_zero = exponent_before_delta < T.uint16(8)
            exponent = T.shift_left(
                T.Cast("uint8", exponent_before_delta - T.uint16(8)),
                T.uint8(3),
            )
            sign = T.shift_left(T.Cast("uint8", T.shift_right(uint16_v, T.uint16(15))), T.uint8(7))
            return T.if_then_else(
                round_to_zero, T.uint8(0), T.bitwise_or(T.bitwise_or(mantissa, exponent), sign)
            )
        else:  # promote_dtype == "float32"
            uint32_v = T.reinterpret("uint32", v)
            rounding_bias = T.bitwise_and(
                T.shift_right(uint32_v, T.uint32(20)), T.uint32(1)
            ) + T.uint32(0x7FFFF)
            uint32_v = uint32_v + rounding_bias
            mantissa = T.bitwise_and(
                T.Cast("uint8", T.shift_right(uint32_v, T.uint8(20))), T.uint8(0x7)
            )
            exponent_before_delta = T.shift_right(T.shift_left(uint32_v, T.uint32(1)), T.uint32(24))
            round_to_zero = exponent_before_delta < T.uint32(120)
            exponent = T.shift_left(
                T.Cast("uint8", exponent_before_delta - T.uint32(120)), T.uint8(3)
            )
            sign = T.shift_left(T.Cast("uint8", T.shift_right(uint32_v, T.uint32(31))), T.uint8(7))
            return T.if_then_else(
                round_to_zero, T.uint8(0), T.bitwise_or(T.bitwise_or(mantissa, exponent), sign)
            )
    else:  # f8_dtype == "e5m2_float8"
        if promote_dtype == "float16":
            uint16_v = T.reinterpret("uint16", v)
            rounding_bias = T.bitwise_and(
                T.shift_right(uint16_v, T.uint16(8)), T.uint16(1)
            ) + T.uint16(0x7F)
            uint16_v = uint16_v + rounding_bias
            return T.Cast("uint8", T.shift_right(uint16_v, T.uint16(8)))
        else:  # promote_dtype == "float32"
            uint32_v = T.reinterpret("uint32", v)
            rounding_bias = T.bitwise_and(
                T.shift_right(uint32_v, T.uint32(21)), T.uint32(1)
            ) + T.uint32(0xFFFFF)
            uint32_v = uint32_v + rounding_bias
            mantissa = T.bitwise_and(
                T.Cast("uint8", T.shift_right(uint32_v, T.uint8(21))), T.uint8(0x3)
            )
            exponent_before_delta = T.shift_right(T.shift_left(uint32_v, T.uint32(1)), T.uint32(24))
            round_to_zero = exponent_before_delta < T.uint32(112)
            exponent = T.shift_left(
                T.Cast("uint8", exponent_before_delta - T.uint32(112)), T.uint8(2)
            )
            sign = T.shift_left(T.Cast("uint8", T.shift_right(uint32_v, T.uint32(31))), T.uint8(7))
            return T.if_then_else(
                round_to_zero, T.uint8(0), T.bitwise_or(T.bitwise_or(mantissa, exponent), sign)
            )


def get_after_storage_legalize(dtype: str, promote_dtype: str):
    @tvm.script.ir_module
    class After:
        @T.prim_func
        def main(Aptr: T.handle("uint8"), Bptr: T.handle("uint8"), Dptr: T.handle("uint8")):
            T.func_attr({"global_symbol": "main"})
            A = T.decl_buffer((100,), "uint8", data=Aptr)
            B = T.decl_buffer((100,), "uint8", data=Bptr)
            D = T.decl_buffer((100,), "uint8", data=Dptr)
            C = T.decl_buffer((100,), promote_dtype)
            for i in T.grid(100):
                C[i] = promote_uint8(dtype, promote_dtype, A[i]) + promote_uint8(
                    dtype, promote_dtype, B[i]
                )
                D[i] = cast_to_uint8(dtype, promote_dtype, T.exp(C[i]))

    return After


dtype = tvm.testing.parameter("e4m3_float8", "e5m2_float8")
promote_dtype = tvm.testing.parameter("float16", "float32")


def test_fp8_compute_legalize(dtype, promote_dtype):
    target = Target("cuda")
    before = BindTarget(target)(get_before(dtype))
    expected = BindTarget(target)(get_after_compute_legalize(dtype, promote_dtype))
    # run the transform twice to ensure we can afford to deal
    # with this repeative optimizations
    after = tvm.tir.transform.FP8ComputeLegalize(promote_dtype)(before)
    after = tvm.tir.transform.FP8ComputeLegalize(promote_dtype)(after)
    tvm.ir.assert_structural_equal(after, expected)


def test_fp8_storage_legalize(dtype, promote_dtype):
    target = Target("cuda")
    before = BindTarget(target)(get_after_compute_legalize(dtype, promote_dtype))
    after = tvm.tir.transform.FP8StorageLegalize()(before)
    expected = BindTarget(target)(get_after_storage_legalize(dtype, promote_dtype))
    tvm.ir.assert_structural_equal(after, expected)


if __name__ == "__main__":
    tvm.testing.main()
