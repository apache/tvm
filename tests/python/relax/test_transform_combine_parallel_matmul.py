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
    with_bias=False,
    activation=None,
):
    shape = (640, 640)
    dtype = "float32"

    with IRBuilder() as builder:
        with relax_builder.function():
            R.func_name("main")
            x = R.arg("x", R.Tensor(shape, dtype))

            rhs = []
            bias = []

            for _ in range(num_branches):
                rhs.append(R.arg("y", R.Tensor(shape, dtype)))

                if with_bias:
                    bias.append(R.arg("bias", R.Tensor((shape[1],), dtype)))

            with R.dataflow() as frame:
                branches = []

                for i, r in enumerate(rhs):
                    result = R.emit(R.matmul(x, r, out_dtype=dtype))
                    if with_bias:
                        result = R.emit(result + bias[i])
                    if activation is not None:
                        result = R.emit(activation(result))

                    branches.append(result)

                R.output(R.emit(R.concat(branches, axis=1)))

            R.func_ret_value(frame.output_vars[0])

    func = builder.get()
    return tvm.IRModule({"main": func})


def test_attention_qkv():
    @tvm.script.ir_module
    class QKV_proj:
        @R.function
        def main(
            x: R.Tensor((2, 1024, 640), "float32"),
            w0: R.Tensor((640, 640), "float32"),
            w1: R.Tensor((640, 640), "float32"),
            w2: R.Tensor((640, 640), "float32"),
        ) -> R.Tensor:
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.matmul(x, w1)
                lv2 = R.matmul(x, w2)
                out = (lv0, lv1, lv2)
                R.output(out)
            return out

    mod = get_parallel_matmul(3)

    # tvm.ir.assert_structural_equal(mod, QKV_proj)
    mod = CombineParallelMatmul()(mod)


    print(mod)


if __name__ == "__main__":
    # tvm.testing.main()
    test_attention_qkv()
