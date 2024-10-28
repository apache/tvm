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
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_single_buffer():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def tir_func(
            X: T.Buffer((224, 224), "float32"),
            W: T.Buffer((224, 224), "float32"),
            Out: T.Buffer((224, 224), "float32"),
        ):
            T.func_attr({"layout_free_buffers": [1]})
            W_rewrite = T.alloc_buffer((4, 4, 56, 56))
            for i, j in T.grid(224, 224):
                with T.block("W_rewrite"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.block_attr({"meta_schedule.layout_rewrite_preproc": T.bool(True)})
                    W_rewrite[vi // 56, vj // 56, vi % 56, vj % 56] = W[vi, vj]
            for i0, j0, i1, j1 in T.grid(4, 4, 56, 56):
                with T.block("Out"):
                    vi = T.axis.spatial(224, i0 * 56 + i1)
                    vj = T.axis.spatial(224, j0 * 56 + j1)
                    Out[vi, vj] = X[vi, vj] + W_rewrite[vi // 56, vj // 56, vi % 56, vj % 56]

        @R.function
        def forward(
            x: R.Tensor((224, 224), dtype="float32"),
            w: R.Tensor((224, 224), dtype="float32"),
        ) -> R.Tensor((224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.tir_func, (x, w), out_sinfo=R.Tensor((224, 224), dtype="float32")
                )
                R.output(gv)
            return gv

    @I.ir_module
    class After:
        @T.prim_func(private=True)
        def tir_func_prepacked(
            X: T.Buffer((224, 224), "float32"),
            W_rewrite: T.Buffer((4, 4, 56, 56), "float32"),
            Out: T.Buffer((224, 224), "float32"),
        ):
            for i0, j0, i1, j1 in T.grid(4, 4, 56, 56):
                with T.block("Out"):
                    vi = T.axis.spatial(224, i0 * 56 + i1)
                    vj = T.axis.spatial(224, j0 * 56 + j1)
                    Out[vi, vj] = X[vi, vj] + W_rewrite[vi // 56, vj // 56, vi % 56, vj % 56]

        @T.prim_func(private=True)
        def tir_func_weight_prepack(
            W: T.Buffer((224, 224), "float32"),
            W_rewrite: T.Buffer((4, 4, 56, 56), "float32"),
        ):
            for i, j in T.grid(224, 224):
                with T.block("W_rewrite"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    W_rewrite[vi // 56, vj // 56, vi % 56, vj % 56] = W[vi, vj]

        @R.function
        def forward(
            x: R.Tensor((224, 224), dtype="float32"),
            w: R.Tensor((224, 224), dtype="float32"),
        ) -> R.Tensor((224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = After
            with R.dataflow():
                lv = R.call_tir(
                    cls.tir_func_weight_prepack, (w,), out_sinfo=R.Tensor((4, 4, 56, 56), "float32")
                )
                lv1 = R.call_tir(
                    cls.tir_func_prepacked, (x, lv), out_sinfo=R.Tensor((224, 224), "float32")
                )
                gv: R.Tensor((224, 224), dtype="float32") = lv1
                R.output(gv)
            return gv

    mod = relax.transform.SplitLayoutRewritePreproc()(Before)
    tvm.ir.assert_structural_equal(mod, After)


def test_multiple_buffers():
    @I.ir_module
    class Before:
        @T.prim_func(private=True)
        def tir_func(
            X: T.Buffer((224, 224), "float32"),
            W1: T.Buffer((224, 224), "float32"),
            W2: T.Buffer((224, 224), "float32"),
            Out: T.Buffer((224, 224), "float32"),
        ):
            W1_rewrite = T.alloc_buffer((4, 4, 56, 56))
            W2_rewrite = T.alloc_buffer((4, 4, 56, 56))
            for i, j in T.grid(224, 224):
                with T.block("W1_rewrite"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.block_attr({"meta_schedule.layout_rewrite_preproc": T.bool(True)})
                    W1_rewrite[vi // 56, vj // 56, vi % 56, vj % 56] = W1[vi, vj]
            for i, j in T.grid(224, 224):
                with T.block("W2_rewrite"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    T.block_attr({"meta_schedule.layout_rewrite_preproc": T.bool(True)})
                    W2_rewrite[vi // 56, vj // 56, vi % 56, vj % 56] = W2[vi, vj]
            for i0, j0, i1, j1 in T.grid(4, 4, 56, 56):
                with T.block("Out"):
                    vi = T.axis.spatial(224, i0 * 56 + i1)
                    vj = T.axis.spatial(224, j0 * 56 + j1)
                    Out[vi, vj] = (
                        X[vi, vj]
                        + W1_rewrite[vi // 56, vj // 56, vi % 56, vj % 56]
                        + W2_rewrite[vi // 56, vj // 56, vi % 56, vj % 56]
                    )

        @R.function
        def forward(
            x: R.Tensor((224, 224), dtype="float32"),
            w1: R.Tensor((224, 224), dtype="float32"),
            w2: R.Tensor((224, 224), dtype="float32"),
        ) -> R.Tensor((224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = Before
            with R.dataflow():
                gv = R.call_tir(
                    cls.tir_func, (x, w1, w2), out_sinfo=R.Tensor((224, 224), dtype="float32")
                )
                R.output(gv)
            return gv

    @I.ir_module
    class After:
        @T.prim_func(private=True)
        def tir_func_prepacked(
            X: T.Buffer((224, 224), "float32"),
            W1_rewrite: T.Buffer((4, 4, 56, 56), "float32"),
            W2_rewrite: T.Buffer((4, 4, 56, 56), "float32"),
            Out: T.Buffer((224, 224), "float32"),
        ):
            for i0, j0, i1, j1 in T.grid(4, 4, 56, 56):
                with T.block("Out"):
                    vi = T.axis.spatial(224, i0 * 56 + i1)
                    vj = T.axis.spatial(224, j0 * 56 + j1)
                    Out[vi, vj] = (
                        X[vi, vj]
                        + W1_rewrite[vi // 56, vj // 56, vi % 56, vj % 56]
                        + W2_rewrite[vi // 56, vj // 56, vi % 56, vj % 56]
                    )

        @T.prim_func(private=True)
        def tir_func_weight_prepack(
            W1: T.Buffer((224, 224), "float32"),
            W2: T.Buffer((224, 224), "float32"),
            W1_rewrite: T.Buffer((4, 4, 56, 56), "float32"),
            W2_rewrite: T.Buffer((4, 4, 56, 56), "float32"),
        ):
            for i, j in T.grid(224, 224):
                with T.block("W1_rewrite"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    W1_rewrite[vi // 56, vj // 56, vi % 56, vj % 56] = W1[vi, vj]
            for i, j in T.grid(224, 224):
                with T.block("W2_rewrite"):
                    vi, vj = T.axis.remap("SS", [i, j])
                    W2_rewrite[vi // 56, vj // 56, vi % 56, vj % 56] = W2[vi, vj]

        @R.function
        def forward(
            x: R.Tensor((224, 224), dtype="float32"),
            w1: R.Tensor((224, 224), dtype="float32"),
            w2: R.Tensor((224, 224), dtype="float32"),
        ) -> R.Tensor((224, 224), dtype="float32"):
            R.func_attr({"num_input": 1})
            cls = After
            with R.dataflow():
                lv0 = R.call_tir(
                    cls.tir_func_weight_prepack,
                    (w1, w2),
                    out_sinfo=[
                        R.Tensor((4, 4, 56, 56), "float32"),
                        R.Tensor((4, 4, 56, 56), "float32"),
                    ],
                )
                lv1 = R.call_tir(
                    cls.tir_func_prepacked,
                    (x, lv0[0], lv0[1]),
                    out_sinfo=R.Tensor((224, 224), "float32"),
                )
                gv: R.Tensor((224, 224), dtype="float32") = lv1
                R.output(gv)
            return gv

    mod = relax.transform.SplitLayoutRewritePreproc()(Before)
    tvm.ir.assert_structural_equal(mod, After)


if __name__ == "__main__":
    tvm.testing.main()
