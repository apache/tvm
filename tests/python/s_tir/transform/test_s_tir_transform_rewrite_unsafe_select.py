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
from tvm import s_tir
from tvm.script import ir as I, tir as T


def test_rewrite_Select():
    @I.ir_module
    class ModuleY:
        @T.prim_func
        def main(i: T.int32):
            A_data = T.allocate([100], "float32", "global")
            A = T.Buffer(100, "float32", data=A_data)
            T.evaluate(T.Select(i > 1, A[i - 1], T.float32(1.0)))

    yy = tvm.s_tir.transform.RewriteUnsafeSelect()(ModuleY)["main"].body.body.value

    @I.ir_module
    class ModuleZ:
        @T.prim_func
        def main(i: T.int32):
            A_data = T.allocate([100], "float32", "global")
            A = T.Buffer(100, "float32", data=A_data)
            T.evaluate(
                T.Select(
                    T.Select(i > 1, A[i - 1], T.float32(1.0)) > T.float32(0.0), A[i], T.float32(0.1)
                )
            )

    zz = tvm.s_tir.transform.RewriteUnsafeSelect()(ModuleZ)["main"].body.body.value

    @I.ir_module
    class ModuleA:
        @T.prim_func
        def main(i: T.int32):
            A_data = T.allocate([100], "float32", "global")
            A = T.Buffer(100, "float32", data=A_data)
            # Inline y and z to avoid Let bindings - outer Select condition is safe (no buffer access)
            T.evaluate(
                T.Select(
                    T.floordiv(i, 4) > 10,
                    T.Select(i > 1, A[i - 1], T.float32(1.0)),
                    T.Select(
                        T.Select(i > 1, A[i - 1], T.float32(1.0)) > T.float32(0.0),
                        A[i],
                        T.float32(0.1),
                    ),
                )
            )

    aa = tvm.s_tir.transform.RewriteUnsafeSelect()(ModuleA)["main"].body.body.value
    builtin_if_then_else = tvm.ir.Op.get("tir.if_then_else")

    assert yy.op.same_as(builtin_if_then_else)
    assert yy.op.same_as(builtin_if_then_else)
    assert isinstance(aa, tvm.tir.Select)


if __name__ == "__main__":
    test_rewrite_Select()
