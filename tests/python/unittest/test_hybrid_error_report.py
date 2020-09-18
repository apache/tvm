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
from tvm import tir
from tvm.hybrid import ty
from tvm.hybrid.parser import HybridParserError


@tvm.hybrid.script
class Module1:
    def buffer_bind_missing_args(a: ty.handle) -> None:
        A = tir.match_buffer((16, 16), "float32")


@tvm.hybrid.script
class Module2:
    def range_missing_args(a: ty.handle) -> None:
        A = tir.match_buffer(a, (16, 16), "float32")

        tir.attr(A, "realize_scope", "")
        tir.realize(A[0:16, 0:16])
        for i in tir.serial(16):
            for j in tir.serial(0, 16):
                A[i, j] = 0.0


@tvm.hybrid.script
class Module3:
    def undefined_buffer(a: ty.handle) -> None:
        A = tir.match_buffer(a, (16, 16), "float32")

        tir.attr(A, "realize_scope", "")
        tir.realize(C[0:16, 0:16])
        for i in tir.serial(16):
            for j in tir.serial(0, 16):
                A[i, j] = 0.0


@tvm.hybrid.script
class Module4:
    def unsupported_stmt(a: ty.int32) -> None:
        if a > 0:
            print("I love tvm")


@tvm.hybrid.script
class Module5:
    def unsupported_function_call(a: ty.handle) -> None:
        A = tir.match_buffer(a, (16, 16), "float32")

        tir.attr(A, "realize_scope", "")
        tir.realize(A[0:16, 0:16])
        for i in tir.const_range(16):
            for j in tir.serial(0, 16):
                A[i, j] = 0.0


@tvm.hybrid.script
class Module6:
    def missing_type_annotation(a) -> None:
        pass


@tvm.hybrid.script
class Module7:
    def invalid_concise_scoping() -> None:
        tir.Assert(1.0 > 0.0, "aaaa")
        tir.evaluate(0.0)


@tvm.hybrid.script
class Module8:
    def invalid_expr_stmt() -> None:
        tir.max(1, 2)


@tvm.hybrid.script
class Module9:
    def invalid_for_function(a: ty.handle) -> None:
        A = tir.match_buffer(a, (16, 16), "float32")

        for i in tir.evaluate(0.0):
            for j in tir.serial(0, 16):
                A[i, j] = 0.0


@tvm.hybrid.script
class Module10:
    def invalid_block_function(a: ty.handle) -> None:
        A = tir.match_buffer(a, (16, 16), "float32")

        with tir.evaluate(0.0):
            pass


def wrap_error(module, lineno):
    with pytest.raises(HybridParserError) as error:
        mod = module()
    assert error is not None
    e = error.value
    print(e)
    msg = str(e).split("\n")[-1].split(":", maxsplit=1)[0].strip().split(" ")[-1].strip()
    assert int(msg) == lineno


if __name__ == "__main__":
    wrap_error(Module1, 29)
    wrap_error(Module2, 39)
    wrap_error(Module3, 50)
    wrap_error(Module4, 60)
    wrap_error(Module5, 70)
    wrap_error(Module6, 77)
    wrap_error(Module7, 84)
    wrap_error(Module8, 91)
    wrap_error(Module9, 99)
    wrap_error(Module10, 109)
