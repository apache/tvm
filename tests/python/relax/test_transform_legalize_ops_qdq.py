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
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R, tir as T
import tvm.testing


def test_quantize_fp32_to_int8():
    @tvm.script.ir_module
    class Quantize:
        @R.function
        def main(
            data: R.Tensor((2, 4), "float32"),
            scale: R.Tensor((2,), "float32"),
            zp: R.Tensor((2,), "int8"),
        ) -> R.Tensor((2, 4), "int8"):
            out = R.quantize(data, scale, zp, axis=0, out_dtype="int8")
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def quantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "float32"),
            B: T.Buffer((T.int64(2),), "float32"),
            C: T.Buffer((T.int64(2),), "int8"),
            quantized: T.Buffer((T.int64(2), T.int64(4)), "int8"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("quantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i0], C[v_i0])
                    T.writes(quantized[v_i0, v_i1])
                    quantized[v_i0, v_i1] = T.Cast(
                        "int8",
                        T.max(
                            T.min(
                                T.round(A[v_i0, v_i1] / B[v_i0]) + T.Cast("float32", C[v_i0]),
                                T.float32(127),
                            ),
                            T.float32(-128),
                        ),
                    )

        @R.function
        def main(
            data: R.Tensor((2, 4), dtype="float32"),
            scale: R.Tensor((2,), dtype="float32"),
            zp: R.Tensor((2,), dtype="int8"),
        ) -> R.Tensor((2, 4), dtype="int8"):
            out = R.call_tir(
                Expected.quantize, (data, scale, zp), out_sinfo=R.Tensor((2, 4), dtype="int8")
            )
            return out

    mod = LegalizeOps()(Quantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_quantize_fp16_to_uint8():
    @tvm.script.ir_module
    class Quantize:
        @R.function
        def main(
            data: R.Tensor((2, 4), "float16"),
            scale: R.Tensor((2,), "float16"),
            zp: R.Tensor((2,), "int8"),
        ) -> R.Tensor((2, 4), "uint8"):
            out = R.quantize(data, scale, zp, axis=0, out_dtype="uint8")
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def quantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "float16"),
            B: T.Buffer((T.int64(2),), "float16"),
            C: T.Buffer((T.int64(2),), "int8"),
            quantized: T.Buffer((T.int64(2), T.int64(4)), "uint8"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("quantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i0], C[v_i0])
                    T.writes(quantized[v_i0, v_i1])
                    quantized[v_i0, v_i1] = T.Cast(
                        "uint8",
                        T.max(
                            T.min(
                                T.round(A[v_i0, v_i1] / B[v_i0]) + T.Cast("float16", C[v_i0]),
                                T.float16(255),
                            ),
                            T.float16(0),
                        ),
                    )

        @R.function
        def main(
            data: R.Tensor((2, 4), dtype="float16"),
            scale: R.Tensor((2,), dtype="float16"),
            zp: R.Tensor((2,), dtype="int8"),
        ) -> R.Tensor((2, 4), dtype="uint8"):
            out = R.call_tir(
                Expected.quantize, (data, scale, zp), out_sinfo=R.Tensor((2, 4), dtype="uint8")
            )
            return out

    mod = LegalizeOps()(Quantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_quantize_fp32_to_int8_symbolic():
    @tvm.script.ir_module
    class Quantize:
        @R.function
        def main(
            data: R.Tensor((4, "n"), "float32"),
            scale: R.Tensor(("n",), "float32"),
            zp: R.Tensor(("n",), "int8"),
        ) -> R.Tensor((4, "n"), "int8"):
            out = R.quantize(data, scale, zp, axis=-1, out_dtype="int8")
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def quantize(var_A: T.handle, var_B: T.handle, var_C: T.handle, var_quantized: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(4), n))
            B = T.match_buffer(var_B, (n,))
            C = T.match_buffer(var_C, (n,), "int8")
            quantized = T.match_buffer(var_quantized, (T.int64(4), n), "int8")
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(4), n):
                with T.block("quantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i1], C[v_i1])
                    T.writes(quantized[v_i0, v_i1])
                    quantized[v_i0, v_i1] = T.Cast(
                        "int8",
                        T.max(
                            T.min(
                                T.round(A[v_i0, v_i1] / B[v_i1]) + T.Cast("float32", C[v_i1]),
                                T.float32(127),
                            ),
                            T.float32(-128),
                        ),
                    )

        @R.function
        def main(
            data: R.Tensor((4, "n"), dtype="float32"),
            scale: R.Tensor(("n",), dtype="float32"),
            zp: R.Tensor(("n",), dtype="int8"),
        ) -> R.Tensor((4, "n"), dtype="int8"):
            n = T.int64()
            out = R.call_tir(
                Expected.quantize, (data, scale, zp), out_sinfo=R.Tensor((4, n), "int8")
            )
            return out

    mod = LegalizeOps()(Quantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_quantize_fp32_to_int8_scalar_param():
    @tvm.script.ir_module
    class Quantize:
        @R.function
        def main(data: R.Tensor((2, 4), "float32")) -> R.Tensor((2, 4), "int8"):
            out = R.quantize(
                data, R.const(2.0, "float32"), R.const(1, "int8"), axis=-1, out_dtype="int8"
            )
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def quantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "float32"),
            quantized: T.Buffer((T.int64(2), T.int64(4)), "int8"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("quantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1])
                    T.writes(quantized[v_i0, v_i1])
                    quantized[v_i0, v_i1] = T.Cast(
                        "int8",
                        T.max(
                            T.min(
                                T.round(A[v_i0, v_i1] * T.float32(0.5)) + T.float32(1),
                                T.float32(127),
                            ),
                            T.float32(-128),
                        ),
                    )

        @R.function
        def main(data: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="int8"):
            out = R.call_tir(Expected.quantize, (data,), out_sinfo=R.Tensor((2, 4), dtype="int8"))
            return out

    mod = LegalizeOps()(Quantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_quantize_fp32_to_int8_scalar_1d_param():
    @tvm.script.ir_module
    class Quantize:
        @R.function
        def main(data: R.Tensor((2, 4), "float32")) -> R.Tensor((2, 4), "int8"):
            out = R.quantize(
                data,
                R.const([2.0, 1.0], "float32"),
                R.const([4, 5], "int8"),
                axis=0,
                out_dtype="int8",
            )
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def quantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "float32"),
            B: T.Buffer((T.int64(2),), "float32"),
            C: T.Buffer((T.int64(2),), "int8"),
            quantized: T.Buffer((T.int64(2), T.int64(4)), "int8"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("quantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], B[v_i0], C[v_i0])
                    T.writes(quantized[v_i0, v_i1])
                    quantized[v_i0, v_i1] = T.Cast(
                        "int8",
                        T.max(
                            T.min(
                                T.round(A[v_i0, v_i1] / B[v_i0]) + T.Cast("float32", C[v_i0]),
                                T.float32(127),
                            ),
                            T.float32(-128),
                        ),
                    )

        @R.function
        def main(data: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="int8"):
            cls = Expected
            out = R.call_tir(
                cls.quantize,
                (data, R.const([2.0, 1.0], "float32"), R.const([4, 5], "int8")),
                out_sinfo=R.Tensor((2, 4), dtype="int8"),
            )
            return out

    mod = LegalizeOps()(Quantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_quantize_fp16_to_int8_scalar_param():
    @tvm.script.ir_module
    class Quantize:
        @R.function
        def main(data: R.Tensor((2, 4), "float16")) -> R.Tensor((2, 4), "int8"):
            out = R.quantize(
                data, R.const(2.0, "float16"), R.const(1, "int8"), axis=-1, out_dtype="int8"
            )
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def quantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "float16"),
            quantized: T.Buffer((T.int64(2), T.int64(4)), "int8"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("quantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1])
                    T.writes(quantized[v_i0, v_i1])
                    quantized[v_i0, v_i1] = T.Cast(
                        "int8",
                        T.max(
                            T.min(
                                T.round(A[v_i0, v_i1] * T.float16(0.5)) + T.float16(1),
                                T.float16(127),
                            ),
                            T.float16(-128),
                        ),
                    )

        @R.function
        def main(data: R.Tensor((2, 4), dtype="float16")) -> R.Tensor((2, 4), dtype="int8"):
            out = R.call_tir(Expected.quantize, (data,), out_sinfo=R.Tensor((2, 4), dtype="int8"))
            return out

    mod = LegalizeOps()(Quantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dequantize_int8_to_fp32():
    @tvm.script.ir_module
    class Dequantize:
        @R.function
        def main(
            data: R.Tensor((2, 4), "int8"),
            scale: R.Tensor((2,), "float32"),
            zp: R.Tensor((2,), "int8"),
        ) -> R.Tensor((2, 4), "float32"):
            out = R.dequantize(data, scale, zp, axis=0, out_dtype="float32")
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def dequantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "int8"),
            B: T.Buffer((T.int64(2),), "float32"),
            C: T.Buffer((T.int64(2),), "int8"),
            dequantized: T.Buffer((T.int64(2), T.int64(4)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("dequantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], C[v_i0], B[v_i0])
                    T.writes(dequantized[v_i0, v_i1])
                    dequantized[v_i0, v_i1] = (
                        T.Cast("float32", T.Cast("int32", A[v_i0, v_i1]) - T.Cast("int32", C[v_i0]))
                        * B[v_i0]
                    )

        @R.function
        def main(
            data: R.Tensor((2, 4), dtype="int8"),
            scale: R.Tensor((2,), dtype="float32"),
            zp: R.Tensor((2,), dtype="int8"),
        ) -> R.Tensor((2, 4), dtype="float32"):
            out = R.call_tir(
                Expected.dequantize, (data, scale, zp), out_sinfo=R.Tensor((2, 4), dtype="float32")
            )
            return out

    mod = LegalizeOps()(Dequantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dequantize_int8_to_fp32_scalar_param():
    @tvm.script.ir_module
    class Dequantize:
        @R.function
        def main(data: R.Tensor((2, 4), "int8")) -> R.Tensor((2, 4), "float32"):
            out = R.dequantize(
                data, R.const(2.0, "float32"), R.const(1, "int8"), axis=0, out_dtype="float32"
            )
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def dequantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "int8"),
            dequantized: T.Buffer((T.int64(2), T.int64(4)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("dequantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1])
                    T.writes(dequantized[v_i0, v_i1])
                    dequantized[v_i0, v_i1] = T.Cast(
                        "float32", T.Cast("int32", A[v_i0, v_i1]) - 1
                    ) * T.float32(2)

        @R.function
        def main(data: R.Tensor((2, 4), dtype="int8")) -> R.Tensor((2, 4), dtype="float32"):
            cls = Expected
            out = R.call_tir(cls.dequantize, (data,), out_sinfo=R.Tensor((2, 4), dtype="float32"))
            return out

    mod = LegalizeOps()(Dequantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dequantize_int8_to_fp32_symbolic():
    @tvm.script.ir_module
    class Dequantize:
        @R.function
        def main(
            data: R.Tensor((2, "n"), "int8"),
            scale: R.Tensor(("n",), "float32"),
            zp: R.Tensor(("n",), "int8"),
        ) -> R.Tensor((2, "n"), "float32"):
            out = R.dequantize(data, scale, zp, axis=-1, out_dtype="float32")
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def dequantize(
            var_A: T.handle, var_B: T.handle, var_C: T.handle, var_dequantized: T.handle
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            n = T.int64()
            A = T.match_buffer(var_A, (T.int64(2), n), "int8")
            B = T.match_buffer(var_B, (n,))
            C = T.match_buffer(var_C, (n,), "int8")
            dequantized = T.match_buffer(var_dequantized, (T.int64(2), n))
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), n):
                with T.block("dequantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], C[v_i1], B[v_i1])
                    T.writes(dequantized[v_i0, v_i1])
                    dequantized[v_i0, v_i1] = (
                        T.Cast("float32", T.Cast("int32", A[v_i0, v_i1]) - T.Cast("int32", C[v_i1]))
                        * B[v_i1]
                    )

        @R.function
        def main(
            data: R.Tensor((2, "n"), dtype="int8"),
            scale: R.Tensor(("n",), dtype="float32"),
            zp: R.Tensor(("n",), dtype="int8"),
        ) -> R.Tensor((2, "n"), dtype="float32"):
            n = T.int64()
            out = R.call_tir(
                Expected.dequantize, (data, scale, zp), out_sinfo=R.Tensor((2, n), dtype="float32")
            )
            return out

    mod = LegalizeOps()(Dequantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dequantize_int8_to_fp16():
    @tvm.script.ir_module
    class Dequantize:
        @R.function
        def main(
            data: R.Tensor((2, 4), "int8"),
            scale: R.Tensor((2,), "float16"),
            zp: R.Tensor((2,), "int8"),
        ) -> R.Tensor((2, 4), "float16"):
            out = R.dequantize(data, scale, zp, axis=0, out_dtype="float16")
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def dequantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "int8"),
            B: T.Buffer((T.int64(2),), "float16"),
            C: T.Buffer((T.int64(2),), "int8"),
            dequantized: T.Buffer((T.int64(2), T.int64(4)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("dequantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1], C[v_i0], B[v_i0])
                    T.writes(dequantized[v_i0, v_i1])
                    dequantized[v_i0, v_i1] = T.Cast(
                        "float16",
                        T.max(
                            T.min(
                                T.Cast(
                                    "float32",
                                    T.Cast("int32", A[v_i0, v_i1]) - T.Cast("int32", C[v_i0]),
                                )
                                * T.Cast("float32", B[v_i0]),
                                T.float32(65504),
                            ),
                            T.float32(-65504),
                        ),
                    )

        @R.function
        def main(
            data: R.Tensor((2, 4), dtype="int8"),
            scale: R.Tensor((2,), dtype="float16"),
            zp: R.Tensor((2,), dtype="int8"),
        ) -> R.Tensor((2, 4), dtype="float16"):
            out = R.call_tir(
                Expected.dequantize, (data, scale, zp), out_sinfo=R.Tensor((2, 4), dtype="float16")
            )
            return out

    mod = LegalizeOps()(Dequantize)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dequantize_int8_to_fp16_scalar_param():
    @tvm.script.ir_module
    class Dequantize:
        @R.function
        def main(data: R.Tensor((2, 4), "int8")) -> R.Tensor((2, 4), "float16"):
            out = R.dequantize(
                data, R.const(2.0, "float16"), R.const(1, "int8"), axis=0, out_dtype="float16"
            )
            return out

    @tvm.script.ir_module
    class Expected:
        @T.prim_func(private=True)
        def dequantize(
            A: T.Buffer((T.int64(2), T.int64(4)), "int8"),
            dequantized: T.Buffer((T.int64(2), T.int64(4)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1 in T.grid(T.int64(2), T.int64(4)):
                with T.block("dequantized"):
                    v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                    T.reads(A[v_i0, v_i1])
                    T.writes(dequantized[v_i0, v_i1])
                    dequantized[v_i0, v_i1] = T.Cast(
                        "float16",
                        T.max(
                            T.min(
                                T.Cast("float32", T.Cast("int32", A[v_i0, v_i1]) - 1)
                                * T.float32(2),
                                T.float32(65504),
                            ),
                            T.float32(-65504),
                        ),
                    )

        @R.function
        def main(data: R.Tensor((2, 4), dtype="int8")) -> R.Tensor((2, 4), dtype="float16"):
            cls = Expected
            out = R.call_tir(cls.dequantize, (data,), out_sinfo=R.Tensor((2, 4), dtype="float16"))
            return out

    mod = LegalizeOps()(Dequantize)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
