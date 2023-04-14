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
# pylint: disable=invalid-name,,missing-function-docstring
import tvm
from tvm.tir.tensor_intrin.cuda import *
from tvm.tir.tensor_intrin.x86 import *
from tvm.relax.transform import FewShotTuning
from tvm.script import tir as T
import tvm.testing


def test_matmul():
    # pylint: disable=no-self-argument,missing-class-docstring,line-too-long
    # fmt: off
    @tvm.script.ir_module
    class Before:
        @T.prim_func
        def matmul(
            A: T.Buffer((32, 32), "float16"),
            B: T.Buffer((32, 32), "float16"),
            C: T.Buffer((32, 32), "float16"),
        ):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            # with T.block("root"):
            for i, j, k in T.grid(32, 32, 32):
                with T.block("C"):
                    v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                    T.reads(A[v_i, v_k], B[v_k, v_j])
                    T.writes(C[v_i, v_j])
                    with T.init():
                        C[v_i, v_j] = T.float16(0)
                    C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
    # fmt: on
    # pylint: enable=no-self-argument,missing-class-docstring,line-too-long
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    with target, tvm.transform.PassContext(opt_level=3):
        After = FewShotTuning()(Before)
        After.show()


if __name__ == "__main__":
    tvm.testing.main()
