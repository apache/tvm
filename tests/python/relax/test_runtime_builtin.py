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
import pytest
import numpy as np

from tvm.ir import assert_structural_equal
from tvm.relax.testing.runtime_builtin import MatchShapeCode, MakeShapeCode


def test_make_shape():
    MK = MakeShapeCode
    make_shape = tvm.get_global_func("vm.builtin.make_shape")
    heap = tvm.nd.array(np.arange(10).astype("int64"))
    s = make_shape(heap, 3, MK.USE_IMM, 10, MK.LOAD_SHAPE, 0, MK.LOAD_SHAPE, 2)

    assert s == tvm.runtime.container.ShapeTuple([10, 0, 2])


def test_match_shape():
    MS = MatchShapeCode
    match_shape = tvm.get_global_func("vm.builtin.match_shape")
    heap = tvm.nd.array(np.zeros(10).astype("int64"))

    assert heap.numpy()[2] == 0

    s = tvm.runtime.container.ShapeTuple([1, 2, 3])
    x = tvm.nd.array(np.zeros([1, 2, 3]))

    match_shape(s, heap, 3, MS.ASSERT_EQUAL_TO_IMM, 1, MS.STORE_TO_HEAP, 2, MS.NO_OP, 0, "")

    assert heap.numpy()[2] == 2

    match_shape(
        x,
        heap,
        3,
        MS.ASSERT_EQUAL_TO_IMM,
        1,
        MS.ASSERT_EQUAL_TO_LOAD,
        2,
        MS.ASSERT_EQUAL_TO_IMM,
        3,
        "",
    )

    with pytest.raises(RuntimeError):
        match_shape(s, heap, 2, MS.ASSERT_EQUAL_TO_IMM, 1, MS.STORE_TO_HEAP, 2, "")

    with pytest.raises(RuntimeError):
        match_shape(s, heap, 3, MS.ASSERT_EQUAL_TO_IMM, 2, MS.STORE_TO_HEAP, 2, MS.NO_OP, 0, "")


def test_check_shape_info():
    check_shape_info = tvm.get_global_func("vm.builtin.check_shape_info")
    s = tvm.runtime.container.ShapeTuple([1, 2, 3])

    check_shape_info(s, 3, "")
    check_shape_info(s, -1, "")

    # wrong ndim
    with pytest.raises(ValueError):
        check_shape_info(s, 2, "")

    # wrong type
    with pytest.raises(TypeError):
        check_shape_info([], 2, "")


def test_check_tensor_info():
    check_tensor_info = tvm.get_global_func("vm.builtin.check_tensor_info")
    x = tvm.nd.array(np.zeros((2, 3)).astype("int32"))

    check_tensor_info(x, 2, "int32", "")
    check_tensor_info(x, -1, "int32", "")
    check_tensor_info(x, 2, "", "")
    check_tensor_info(x, -1, "", "")

    # allow not passing in dtype
    check_tensor_info(x, 2, "")
    check_tensor_info(x, -1, "")

    # ndim mismatch
    with pytest.raises(ValueError, match=r".* ndim .*"):
        check_tensor_info(x, 3, "int32", "")

    # dtype mismatch
    with pytest.raises(ValueError, match=r"myerror.* dtype .*"):
        check_tensor_info(x, 2, "float32", "myerror")

    # error with context
    with pytest.raises(ValueError, match=r".* myerror .*"):
        check_tensor_info(x, 3, "myerror")

    # wrong type
    with pytest.raises(TypeError):
        check_tensor_info([], 2, "", "")


def test_check_tuple_info():
    check_tuple_info = tvm.get_global_func("vm.builtin.check_tuple_info")
    x = tvm.nd.array(np.zeros((2, 3)).astype("int32"))
    t = tvm.runtime.convert([x, x, x])

    check_tuple_info(t, 3, "")

    # size
    with pytest.raises(ValueError, match=r".*elements.*"):
        check_tuple_info(t, 2, "")

    # wrong type
    with pytest.raises(TypeError):
        check_tuple_info(x, 2, "")


def test_check_func_info():
    check_func_info = tvm.get_global_func("vm.builtin.check_func_info")
    f = tvm.runtime.convert(lambda x: x)
    x = tvm.nd.array(np.zeros((2, 3)).astype("int32"))

    check_func_info(f, "")

    # wrong type
    with pytest.raises(TypeError, match=".*myerror.*"):
        check_func_info(x, "myerror")


def test_tuple_getitem():
    tuple_getitem = tvm.get_global_func("vm.builtin.tuple_getitem")
    x = tvm.nd.array(np.zeros((2, 3)).astype("int32"))
    y = tvm.nd.array(np.zeros((2, 3)).astype("int32"))
    t = tvm.runtime.convert([x, y])

    assert tuple_getitem(t, 0) == x
    assert tuple_getitem(t, 1) == y


if __name__ == "__main__":
    tvm.testing.main()
