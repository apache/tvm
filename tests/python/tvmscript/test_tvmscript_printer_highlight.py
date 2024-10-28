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

import pytest

import tvm
import tvm.testing
from tvm import relay
from tvm.script import tir as T
from tvm.script.highlight import cprint, _format


def test_highlight_script():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(  # type: ignore
            a: T.handle,
            b: T.handle,
            c: T.handle,
        ) -> None:  # pylint: disable=no-self-argument
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [16, 128, 128])
            B = T.match_buffer(b, [16, 128, 128])
            C = T.match_buffer(c, [16, 128, 128])
            for n, i, j, k in T.grid(16, 128, 128, 128):
                with T.block("matmul"):
                    vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                    with T.init():
                        C[vn, vi, vj] = 0.0  # type: ignore
                    C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]

    Module.show()
    Module["main"].show()
    Module["main"].show(style="light")
    Module["main"].show(style="dark")
    Module["main"].show(style="ansi")


def test_cprint():
    # Print string
    cprint("a + 1")

    # Print nodes with `script` method, e.g. PrimExpr
    cprint(tvm.tir.Var("v", "int32") + 1)

    # Cannot print non-Python-style codes when using the black
    # formatter.  This error comes from `_format`, used internally by
    # `cprint`, and doesn't occur when using the `ruff` formatter.
    try:
        import black

        with pytest.raises(ValueError):
            _format("if (a == 1) { a +=1; }", formatter="black")
    except ImportError:
        pass

    # Cannot print unsupported nodes (nodes without `script` method)
    with pytest.raises(TypeError):
        cprint(relay.const(1))


if __name__ == "__main__":
    tvm.testing.main()
