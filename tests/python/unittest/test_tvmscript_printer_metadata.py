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
# pylint: disable=missing-docstring
import tvm.testing
from tvm.script.parser import ir as I
from tvm.script.parser import tir as T


def _assert_print(obj, expected):
    assert obj.script(verbose_expr=True).strip() == expected.strip()


def test_str_metadata():
    str_imm = T.StringImm("aaa\nbbb\n")

    @I.ir_module
    class Module:
        @T.prim_func
        def foo() -> None:
            A = str_imm
            B = str_imm

        @T.prim_func
        def foo1() -> None:
            A = str_imm

    _assert_print(
        Module,
        """
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def foo():
        A: T.handle = metadata["tir.StringImm"][0]
        B: T.handle = metadata["tir.StringImm"][0]
        T.evaluate(0)

    @T.prim_func
    def foo1():
        A: T.handle = metadata["tir.StringImm"][0]
        T.evaluate(0)


# Metadata omitted. Use show_meta=True in script() method to show it.""",
    )


if __name__ == "__main__":
    tvm.testing.main()
