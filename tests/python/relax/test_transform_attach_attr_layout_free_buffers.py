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
import numpy as np
import tvm.testing

from tvm import relax, tir
from tvm.script import relax as R, tir as T, ir as I
from tvm.relax.transform import CombineParallelMatmul
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder


def test_param():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def matmul(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), y: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(cls.matmul, (x, y), out_sinfo=R.Tensor((32, 32), "float32"))
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def matmul1(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_buffers": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32"), y: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.matmul1, (x, y), out_sinfo=R.Tensor((32, 32), "float32"))
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeBuffers()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_const():
    const_value = np.ones((32, 32), dtype="float32")

    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def matmul(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.matmul,
                    (x, relax.const(const_value)),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def matmul1(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_buffers": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(x: R.Tensor((32, 32), "float32")):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.matmul1,
                    (x, relax.const(const_value)),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeBuffers()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_multiple_same_func():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def matmul(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul,
                    (x, w1),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                gv = R.call_tir(
                    cls.matmul,
                    (lv1, w2),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def matmul1(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_buffers": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul1,
                    (x, w1),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                gv = R.call_tir(
                    cls.matmul1,
                    (lv1, w2),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeBuffers()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_multiple_same_func_with_different_free_buffers():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def matmul(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul,
                    (x, w1),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                gv = R.call_tir(
                    cls.matmul,
                    (w2, lv1),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def matmul1(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_buffers": [1]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @T.prim_func(private=True)
        def matmul2(
            A: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            B: T.Buffer((T.int64(32), T.int64(32)), "float32"),
            C: T.Buffer((T.int64(32), T.int64(32)), "float32"),
        ):
            T.func_attr({"layout_free_buffers": [0]})
            for i, j, k in T.grid(T.int64(32), T.int64(32), T.int64(32)):
                with T.block("C"):
                    with T.init():
                        C[i, j] = T.float32(0)
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

        @R.function
        def main(
            x: R.Tensor((32, 32), "float32"),
            w1: R.Tensor((32, 32), "float32"),
            w2: R.Tensor((32, 32), "float32"),
        ):
            R.func_attr({"num_input": 1})
            cls = Expected
            with R.dataflow():
                lv1 = R.call_tir(
                    cls.matmul1,
                    (x, w1),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                gv = R.call_tir(
                    cls.matmul2,
                    (w2, lv1),
                    out_sinfo=R.Tensor((32, 32), "float32"),
                )
                R.output(gv)
            return gv

    after = relax.transform.AttachAttrLayoutFreeBuffers()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
