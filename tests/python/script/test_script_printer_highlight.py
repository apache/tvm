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
# ruff: noqa: F401

import pytest

import tvm
import tvm.testing
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.script.highlight import _format, cprint


def test_highlight_script():
    @tvm.script.ir_module
    class Module:
        @Ts.prim_func
        def main(  # type: ignore
            A: T.Buffer([16, 128, 128]),
            B: T.Buffer([16, 128, 128]),
            C: T.Buffer([16, 128, 128]),
        ) -> None:  # pylint: disable=no-self-argument
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})

            for n, i, j, k in T.grid(16, 128, 128, 128):
                with Ts.sblock("matmul"):
                    vn, vi, vj, vk = Ts.axis.remap("SSSR", [n, i, j, k])
                    with Ts.init():
                        C[vn, vi, vj] = 0.0  # type: ignore
                    C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]

    Module.show()
    Module["main"].show()
    Module["main"].show(style="light")
    Module["main"].show(style="dark")
    Module["main"].show(style="ansi")


if __name__ == "__main__":
    tvm.testing.main()
