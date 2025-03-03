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
# pylint: disable=invalid-name, missing-docstring

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


def test_transform_fuse_transpose_matmul():
    @I.ir_module
    class Before:
        @R.function
        def main(
            x: R.Tensor((128, 256), "float32"),
            w: R.Tensor((128, 256), "float32"),
        ) -> R.Tensor((128, 128), "float32"):
            with R.dataflow():
                wT = R.permute_dims(w, [1, 0])
                o = R.matmul(x, wT)
                R.output(o)
            return o

    @I.ir_module
    class Expected:
        @T.prim_func(private=True)
        def NT_matmul(
            x: T.Buffer((T.int64(128), T.int64(256)), "float32"),
            w: T.Buffer((T.int64(128), T.int64(256)), "float32"),
            NT_matmul: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(256)):
                with T.block("NT_matmul"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(x[v_i0, v_k], w[v_i1, v_k])
                    T.writes(NT_matmul[v_i0, v_i1])
                    with T.init():
                        NT_matmul[v_i0, v_i1] = T.float32(0)
                    NT_matmul[v_i0, v_i1] = NT_matmul[v_i0, v_i1] + x[v_i0, v_k] * w[v_i1, v_k]

        @R.function
        def main(
            x: R.Tensor((128, 256), dtype="float32"), w: R.Tensor((128, 256), dtype="float32")
        ) -> R.Tensor((128, 128), dtype="float32"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.NT_matmul, (x, w), out_sinfo=R.Tensor((128, 128), dtype="float32")
                )
                R.output(gv)
            return gv

    after = tvm.ir.transform.Sequential(
        [
            relax.transform.FuseTransposeMatmul(),
            relax.transform.FuseTIR(),  # Only used for remove unused primitive function
        ]
    )(Before)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
