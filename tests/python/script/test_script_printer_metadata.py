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
# ruff: noqa: F841
import tvm.testing
from tvm.script import s_tir as Ts
from tvm.script.parser import ir as I


def test_str_metadata():
    # This test is to check we reuse the existing metadata element for the same tvm.ir.StringImm
    # So metadata["ir.StringImm"][0] will occur in the printed script for three times
    str_imm = tvm.ir.StringImm("aaa\nbbb\n")

    @I.ir_module
    class Module:
        @Ts.prim_func
        def foo() -> None:
            A = str_imm
            B = str_imm

        @Ts.prim_func
        def foo1() -> None:
            A = str_imm

    printed_str = Module.script(verbose_expr=True)
    assert (
        printed_str.count('metadata["ir.StringImm"][0]') == 3
        and printed_str.count('metadata["ir.StringImm"][1]') == 0
    )


if __name__ == "__main__":
    tvm.testing.main()
