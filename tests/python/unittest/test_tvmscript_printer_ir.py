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

import pytest

from tvm import IRModule, TVMError
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tir as T


def _assert_print(obj, expected):
    assert str(obj).strip() == expected.strip()
    assert repr(obj).strip() == expected.strip()
    if isinstance(obj, IRModule):
        assert obj.script().strip() == expected.strip()


def test_ir_module():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module():
            with T.prim_func():
                T.func_name("foo")
    mod = ib.get()
    _assert_print(
        mod,
        """
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def foo():
        T.evaluate(0)""",
    )


def test_failed_invalid_prefix():
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module():
            with T.prim_func():
                T.func_name("foo")
    mod = ib.get()

    with pytest.raises(TVMError):
        mod.script(ir_prefix="2I")


if __name__ == "__main__":
    test_ir_module()
    test_failed_invalid_prefix()
