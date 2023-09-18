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
import tvm.testing
from tvm import relax
from tvm.script import relax as R, ir as I, tir as T
from tvm.ir.base import assert_structural_equal


def test_lower_qnn_quantize():
    @I.ir_module
    class Before:
        @R.function
        def main(data: R.Tensor((2, 4), "float32")) -> R.Tensor((2, 4), "int8"):
            with R.dataflow():
                out = R.qnn.quantize(data, R.const(0.7, "float32"), R.const(2, "int32"), -1, "int8")
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((2, 4), "float32")) -> R.Tensor((2, 4), "int8"):
            with R.dataflow():
                lv0 = R.divide(data, R.const(0.7, dtype="float32"))
                lv1 = R.round(lv0)
                lv2 = R.astype(R.const(2, dtype="int32"), dtype="float32")
                lv3 = R.add(lv1, lv2)
                lv4 = R.clip(lv3, -128.0, 127.0)
                out = R.astype(lv4, dtype="int8")
                R.output(out)
            return out

    assert_structural_equal(relax.transform.QnnCanonicalize()(Before), Expected)


def test_lower_qnn_quantize_non_scalar_params():
    @I.ir_module
    class Before:
        @R.function
        def main(data: R.Tensor((2, 4), "float32")) -> R.Tensor((2, 4), "int8"):
            with R.dataflow():
                out = R.qnn.quantize(
                    data,
                    R.const([0.1, 0.2], dtype="float32"),
                    R.const([0, 1], dtype="int32"),
                    axis=0,
                    out_dtype="int8",
                )
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((2, 4), "float32")) -> R.Tensor((2, 4), "int8"):
            with R.dataflow():
                scale = R.expand_dims(R.const([0.1, 0.2], dtype="float32"), axis=[1])
                lv0 = R.divide(data, scale)
                lv1 = R.round(lv0)
                zero_point = R.expand_dims(R.const([0, 1], dtype="int32"), axis=[1])
                lv2 = R.astype(zero_point, dtype="float32")
                lv3 = R.add(lv1, lv2)
                lv4 = R.clip(lv3, -128.0, 127.0)
                out = R.astype(lv4, dtype="int8")
                R.output(out)
            return out

    assert_structural_equal(relax.transform.QnnCanonicalize()(Before), Expected)


def test_lower_qnn_quantize_fp16_to_int8():
    @I.ir_module
    class Before:
        @R.function
        def main(data: R.Tensor((2, 4), "float16")) -> R.Tensor((2, 4), "int8"):
            with R.dataflow():
                out = R.qnn.quantize(data, R.const(0.7, "float16"), R.const(2, "int32"), -1, "int8")
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((2, 4), "float16")) -> R.Tensor((2, 4), "int8"):
            with R.dataflow():
                lv: R.Tensor((2, 4), dtype="float16") = R.divide(data, R.const(0.7, "float16"))
                lv1: R.Tensor((2, 4), dtype="float16") = R.round(lv)
                lv2: R.Tensor((), dtype="float16") = R.astype(R.const(2, "int32"), dtype="float16")
                lv3: R.Tensor((2, 4), dtype="float16") = R.add(lv1, lv2)
                lv4: R.Tensor((2, 4), dtype="float16") = R.clip(
                    lv3, R.prim_value(T.float16(-128)), R.prim_value(T.float16(127))
                )
                out: R.Tensor((2, 4), dtype="int8") = R.astype(lv4, dtype="int8")
                R.output(out)
            return out

    assert_structural_equal(relax.transform.QnnCanonicalize()(Before), Expected)


def test_lower_qnn_dequantize_int8_to_fp32():
    @I.ir_module
    class Before:
        @R.function
        def main(data: R.Tensor((2, 4), "int8")) -> R.Tensor((2, 4), "float32"):
            with R.dataflow():
                out = R.qnn.dequantize(
                    data, R.const(0.7, "float32"), R.const(2, "int32"), axis=-1, out_dtype="float32"
                )
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((2, 4), "int8")) -> R.Tensor((2, 4), "float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4), dtype="int32") = R.astype(data, dtype="int32")
                lv1: R.Tensor((2, 4), dtype="int32") = R.subtract(lv, R.const(2, "int32"))
                lv2: R.Tensor((2, 4), dtype="float32") = R.astype(lv1, dtype="float32")
                out: R.Tensor((2, 4), dtype="float32") = R.multiply(lv2, R.const(0.7, "float32"))
                R.output(out)
            return out

    assert_structural_equal(relax.transform.QnnCanonicalize()(Before), Expected)


def test_lower_qnn_dequantize_int32_to_fp32():
    @I.ir_module
    class Before:
        @R.function
        def main(data: R.Tensor((2, 4), "int32")) -> R.Tensor((2, 4), "float32"):
            with R.dataflow():
                out = R.qnn.dequantize(
                    data, R.const(0.7, "float32"), R.const(2, "int32"), axis=-1, out_dtype="float32"
                )
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((2, 4), "int32")) -> R.Tensor((2, 4), "float32"):
            with R.dataflow():
                lv: R.Tensor((2, 4), dtype="int32") = R.subtract(data, R.const(2, "int32"))
                lv1: R.Tensor((2, 4), dtype="float32") = R.astype(lv, dtype="float32")
                out: R.Tensor((2, 4), dtype="float32") = R.multiply(lv1, R.const(0.7, "float32"))
                R.output(out)
            return out

    assert_structural_equal(relax.transform.QnnCanonicalize()(Before), Expected)


def test_lower_qnn_dequantize_int8_to_fp16():
    @I.ir_module
    class Before:
        @R.function
        def main(data: R.Tensor((2, 4), "int8")) -> R.Tensor((2, 4), "float16"):
            with R.dataflow():
                out = R.qnn.dequantize(
                    data, R.const(0.7, "float16"), R.const(2, "int32"), axis=-1, out_dtype="float16"
                )
                R.output(out)
            return out

    @I.ir_module
    class Expected:
        @R.function
        def main(data: R.Tensor((2, 4), "int8")) -> R.Tensor((2, 4), "float16"):
            with R.dataflow():
                lv: R.Tensor((2, 4), dtype="int32") = R.astype(data, dtype="int32")
                lv1: R.Tensor((2, 4), dtype="int32") = R.subtract(lv, R.const(2, "int32"))
                lv2: R.Tensor((2, 4), dtype="float32") = R.astype(lv1, dtype="float32")
                lv3: R.Tensor((), dtype="float32") = R.astype(R.const(0.7, "float16"), "float32")
                lv4: R.Tensor((2, 4), dtype="float32") = R.multiply(lv2, lv3)
                lv5: R.Tensor((2, 4), dtype="float32") = R.clip(
                    lv4, R.prim_value(T.float32(-65504)), R.prim_value(T.float32(65504))
                )
                out: R.Tensor((2, 4), dtype="float16") = R.astype(lv5, dtype="float16")
                R.output(out)
            return out

    assert_structural_equal(relax.transform.QnnCanonicalize()(Before), Expected)


if __name__ == "__main__":
    tvm.testing.main()
