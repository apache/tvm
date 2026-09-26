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
from tvm.script import tirx as T

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
                "tirx.noalias": True,
            }
        )
        for i, j, k in T.grid(729, 729, 729):
            if k == 0:
                C[i, j] = T.float32(0)
            C[i, j] = C[i, j] + A[i, k] * B[k, j]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,missing-class-docstring,missing-function-docstring


def test_host_func():
    """Test that host functions are not split."""
    func = Module["main"].without_attr("target")
    mod = tvm.ir.IRModule({"main": func})
    target = tvm.target.Target("cuda")
    mod = tvm.tirx.transform.Apply(
        lambda f: f.with_attr(
            {
                "global_symbol": "test",
                "tirx.is_host_func": True,
            }
        )
    )(mod)
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    tvm.ir.assert_structural_equal(mod, Module)
    assert "tirx.is_host_func" not in mod["main"].attrs, (
        """Target and is_host_func attributes should be mutually exclusive"""
    )


if __name__ == "__main__":
    test_host_func()
