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
from tvm.ir import assert_structural_equal
from tvm.relay.op.transform import split
from tvm.runtime import ObjectPath
from tvm.script import ir as I, tir as T


def _error_message(exception):
    splitter = "ValueError: StructuralEqual"
    return splitter + str(exception).split(splitter)[1]


def _expected_result(func1, func2, objpath1, objpath2):
    return f"""ValueError: StructuralEqual check failed, caused by lhs at {objpath1}:
{func1.script(path_to_underline=[objpath1], syntax_sugar=False)}
and rhs at {objpath2}:
{func2.script(path_to_underline=[objpath2], syntax_sugar=False)}"""


def test_prim_func_buffer_map():
    @T.prim_func
    def func1(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (128, 128))
        B = T.match_buffer(b, (128, 128))

    @T.prim_func
    def func2(a: T.handle, b: T.handle):
        A = T.match_buffer(a, (128, 128))
        B = T.match_buffer(b, (128, 256))

    with pytest.raises(ValueError) as ve:
        assert_structural_equal(func1, func2)
    assert _error_message(ve.value) == _expected_result(
        func1,
        func2,
        ObjectPath.root()
        .attr("buffer_map")
        .map_value(func1.params[1])
        .attr("shape")
        .array_index(1)
        .attr("value"),
        ObjectPath.root()
        .attr("buffer_map")
        .map_value(func2.params[1])
        .attr("shape")
        .array_index(1)
        .attr("value"),
    )


def test_evaluate():
    @I.ir_module
    class module1:
        @T.prim_func
        def func():
            T.evaluate(0)

    @I.ir_module
    class module2:
        @T.prim_func
        def func():
            T.evaluate(1)

    with pytest.raises(ValueError) as ve:
        assert_structural_equal(module1, module2)
    assert _error_message(ve.value) == _expected_result(
        module1,
        module2,
        ObjectPath.root()
        .attr("functions")
        .map_value(module1.get_global_var("func"))
        .attr("body")
        .attr("value")
        .attr("value"),
        ObjectPath.root()
        .attr("functions")
        .map_value(module2.get_global_var("func"))
        .attr("body")
        .attr("value")
        .attr("value"),
    )


def test_allocate():
    @T.prim_func
    def func1():
        a_data = T.allocate((128, 128), dtype="float32")
        a = T.decl_buffer((128, 128), dtype="float32", data=a_data)

    @T.prim_func
    def func2():
        a_data = T.allocate((256, 128), dtype="float32")
        a = T.decl_buffer((256, 128), dtype="float32", data=a_data)

    with pytest.raises(ValueError) as ve:
        assert_structural_equal(func1, func2)
    assert _error_message(ve.value) == _expected_result(
        func1,
        func2,
        ObjectPath.root().attr("body").attr("extents").array_index(0).attr("value"),
        ObjectPath.root().attr("body").attr("extents").array_index(0).attr("value"),
    )


def test_for():
    @T.prim_func
    def func1():
        for i, j in T.grid(128, 128):
            with T.block():
                pass

    @T.prim_func
    def func2():
        for i, j, k in T.grid(128, 128, 128):
            with T.block():
                pass

    with pytest.raises(ValueError) as ve:
        assert_structural_equal(func1, func2)
    assert _error_message(ve.value) == _expected_result(
        func1,
        func2,
        ObjectPath.root().attr("body").attr("block").attr("body").attr("body").attr("body"),
        ObjectPath.root().attr("body").attr("block").attr("body").attr("body").attr("body"),
    )


if __name__ == "__main__":
    tvm.testing.main()
