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
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.stmt_functor import substitute


def _apply_substitute(mod):
    """Apply substitute transform to replace the first parameter with 16."""
    func = mod["main"]
    vmap = {func.params[0]: 16}
    new_func = tvm.tir.PrimFunc(params=[], body=substitute(func.body, vmap)).with_attr(
        "global_symbol", func.attrs["global_symbol"]
    )
    return tvm.IRModule.from_expr(new_func)


def test_basic_substitute():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(n: T.int32):
            for i in range(n):
                T.evaluate(i)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            for i in range(16):
                T.evaluate(i)

    After = _apply_substitute(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_substitute_allocate():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(n: T.int32):
            A_data = T.allocate([n], "float32")
            T.evaluate(A_data)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            A_data = T.allocate([16], "float32")
            T.evaluate(A_data)

    After = _apply_substitute(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_substitute_buffer_load():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(n: T.int32):
            A_data = T.allocate([n], "float32")
            A = T.Buffer(n, "float32", data=A_data)
            for i in range(n):
                T.evaluate(A[i])

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            A_data = T.allocate([16], "float32")
            A = T.Buffer(16, "float32", data=A_data)
            for i in range(16):
                T.evaluate(A[i])

    After = _apply_substitute(Before)
    tvm.ir.assert_structural_equal(After, Expected)


def test_substitute_decl_buffer():
    @I.ir_module
    class Before:
        @T.prim_func
        def main(n: T.int32):
            A_data = T.allocate([n], "float32")
            A = T.decl_buffer(n, "float32", data=A_data)
            T.evaluate(A.data)

    @I.ir_module
    class Expected:
        @T.prim_func
        def main():
            A_data = T.allocate([16], "float32")
            A = T.decl_buffer(16, "float32", data=A_data)
            T.evaluate(A.data)

    After = _apply_substitute(Before)
    tvm.ir.assert_structural_equal(After, Expected)


if __name__ == "__main__":
    tvm.testing.main()
