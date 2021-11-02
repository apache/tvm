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
# pylint: disable=missing-function-docstring,missing-module-docstring
import sys
import pytest
import tvm
from tvm import tir
from tvm.script import tir as T


@pytest.mark.mypy_testing
@tvm.script.ir_module
class Module:
    @T.prim_func
    def func(a: T.handle, b: T.handle, c: T.handle) -> None:
        A = T.match_buffer(a, [128, 128], dtype="float32")
        B = T.match_buffer(b, [128, 128], dtype="float32")
        C = T.match_buffer(c, [128, 128], dtype="float32")
        reveal_type(A)  # R: <Buffer>
        for i, j, k in T.grid(128, 128, T.reduce_axis(0, 128)):
            with T.block("C"):
                C[i, j] = T.if_then_else(
                    i == 0 and j == 0 and k == 0,
                    0.0,
                    C[i, j] + A[i, k] * B[k, j],
                    dtype="float32",
                )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
