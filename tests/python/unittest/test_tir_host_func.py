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
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.meta_schedule.testing import te_workload

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,missing-class-docstring,missing-function-docstring
# fmt: off


@I.ir_module
class Module:
    @T.prim_func
    def main(
        A: T.Buffer((729, 729), "float32"),
        B: T.Buffer((729, 729), "float32"),
        C: T.Buffer((729, 729), "float32"),
    ):
        T.func_attr(
            {
                "global_symbol": "test",
                "target": tvm.target.Target("llvm", host="llvm"),
                "tir.noalias": True,
            }
        )
        # with T.block("root"):
        for i, j, k in T.grid(729, 729, 729):
            with T.block("C"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(C[v_i, v_j])
                with T.init():
                    C[v_i, v_j] = T.float32(0)
                C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,missing-class-docstring,missing-function-docstring


def test_host_func():
    """Test that host functions are not split."""
    # te schedule copied from test_tir_transform_split_host_device.py

    func = tvm.te.create_prim_func(
        te_workload.matmul(729, 729, 729, in_dtype="float32", out_dtype="float32")
    )
    mod = tvm.ir.IRModule({"main": func})
    target = tvm.target.Target("cuda")
    mod = tvm.tir.transform.Apply(
        lambda f: f.with_attr(
            {
                "global_symbol": "test",
                "tir.is_host_func": 1,
            }
        )
    )(mod)
    mod = tvm.tir.transform.BindTarget(target)(mod)
    tvm.ir.assert_structural_equal(mod, Module)
    assert (
        "tir.is_host_func" not in mod["main"].attrs
    ), """Target and is_host_func attributes should be mutually exclusive"""


if __name__ == "__main__":
    test_host_func()
