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

"""TIRx script printer highlight."""

import tvm
import tvm.testing
from tvm.script import tirx as T


def test_highlight_script():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def main(  # type: ignore
            A: T.Buffer([16, 128, 128]),
            B: T.Buffer([16, 128, 128]),
            C: T.Buffer([16, 128, 128]),
        ) -> None:  # pylint: disable=no-self-argument
            T.func_attr({"global_symbol": "main", "tirx.noalias": True})
            for n, i, j in T.grid(16, 128, 128):
                C[n, i, j] = 0.0  # type: ignore
                for k in T.serial(128):
                    C[n, i, j] = C[n, i, j] + A[n, i, k] * B[n, j, k]

    Module.show()
    Module["main"].show()
    Module["main"].show(style="light")
    Module["main"].show(style="dark")
    Module["main"].show(style="ansi")
