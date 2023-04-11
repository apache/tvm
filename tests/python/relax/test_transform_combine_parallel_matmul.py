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
import tvm.testing

from tvm import relax, tir
from tvm.script import relax as R, tir as T
from tvm.relax.transform import CombineParallelMatmul
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder


def get_parallel_matmul(
    num_branches,
    lhs_shape=(640, 640),
    rhs_shape=(640, 640),
    with_bias=None,
    activation=None,
):
    dtype = "float32"

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            x = R.arg("x", R.Tensor(lhs_shape, dtype))

            rhs = []
            bias = []

            for i in range(num_branches):
                rhs.append(R.arg("y", R.Tensor(rhs_shape, dtype)))

                if with_bias and with_bias[i]:
                    bias.append(R.arg("bias", R.Tensor((rhs_shape[1],), dtype)))
                else:
                    bias.append(None)

            with R.dataflow() as frame:
                branches = []

                for i, r in enumerate(rhs):
                    result = R.emit(R.matmul(x, r, out_dtype=dtype))
                    if bias[i]:
                        result = R.emit(result + bias[i])
                    if activation is not None:
                        result = R.emit(activation(result))

                    branches.append(result)

                R.output(R.emit(R.concat(branches, axis=1)))

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def test_simple():
    mod = get_parallel_matmul(3)
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected1(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv_1 = R.strided_slice(lv1, axes=[1], begin=[0], end=[640], strides=[1])
            lv1_1 = R.strided_slice(lv1, axes=[1], begin=[640], end=[1280], strides=[1])
            lv2 = R.strided_slice(lv1, axes=[1], begin=[1280], end=[1920], strides=[1])
            lv3 = R.concat((lv_1, lv1_1, lv2), axis=1)
            R.output(lv3)
        return lv3

    tvm.ir.assert_structural_equal(mod["main"], expected1)

    # Test a batched LHS case, slicing is done on the axis 2
    mod = get_parallel_matmul(3, lhs_shape=(2, 1024, 640))
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected2(
        x: R.Tensor((2, 1024, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
    ) -> R.Tensor((2, 3072, 640), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv_1 = R.strided_slice(lv1, axes=[2], begin=[0], end=[640], strides=[1])
            lv1_1 = R.strided_slice(lv1, axes=[2], begin=[640], end=[1280], strides=[1])
            lv2 = R.strided_slice(lv1, axes=[2], begin=[1280], end=[1920], strides=[1])
            lv3 = R.concat((lv_1, lv1_1, lv2), axis=1)
            R.output(lv3)
        return lv3

    tvm.ir.assert_structural_equal(mod["main"], expected2)


def test_bias():
    mod = get_parallel_matmul(3, with_bias=[True, True, True])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected1(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        bias: R.Tensor((640,), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        bias_1: R.Tensor((640,), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
        bias_2: R.Tensor((640,), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv2 = R.concat((bias, bias_1, bias_2), axis=0)
            lv3 = R.add(lv1, lv2)
            lv1_1 = R.strided_slice(lv3, axes=[1], begin=[0], end=[640], strides=[1])
            lv3_1 = R.strided_slice(lv3, axes=[1], begin=[640], end=[1280], strides=[1])
            lv5 = R.strided_slice(lv3, axes=[1], begin=[1280], end=[1920], strides=[1])
            lv6 = R.concat((lv1_1, lv3_1, lv5), axis=1)
            R.output(lv6)
        return lv6

    tvm.ir.assert_structural_equal(mod["main"], expected1)

    mod = get_parallel_matmul(3, with_bias=[True, False, True])
    mod = CombineParallelMatmul()(mod)

    @R.function
    def expected2(
        x: R.Tensor((640, 640), dtype="float32"),
        y: R.Tensor((640, 640), dtype="float32"),
        bias: R.Tensor((640,), dtype="float32"),
        y_1: R.Tensor((640, 640), dtype="float32"),
        y_2: R.Tensor((640, 640), dtype="float32"),
        bias_1: R.Tensor((640,), dtype="float32"),
    ) -> R.Tensor((640, 1920), dtype="float32"):
        with R.dataflow():
            lv = R.concat((y, y_1, y_2), axis=1)
            lv1 = R.matmul(x, lv, out_dtype="float32")
            lv_1 = R.strided_slice(lv1, axes=[1], begin=[0], end=[640], strides=[1])
            lv1_1 = R.add(lv_1, bias)
            lv2 = R.strided_slice(lv1, axes=[1], begin=[640], end=[1280], strides=[1])
            lv3 = R.strided_slice(lv1, axes=[1], begin=[1280], end=[1920], strides=[1])
            lv4 = R.add(lv3, bias_1)
            lv5 = R.concat((lv1_1, lv2, lv4), axis=1)
            R.output(lv5)
        return lv5

    tvm.ir.assert_structural_equal(mod["main"], expected2)


if __name__ == "__main__":
    # tvm.testing.main()
    test_bias()
