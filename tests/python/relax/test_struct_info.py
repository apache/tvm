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
import tvm.testing
import pytest

from tvm import relax as rx, TVMError, tir


def _check_equal(x, y, map_free_vars=False):
    tvm.ir.assert_structural_equal(x, y, map_free_vars)
    tvm.ir.assert_structural_equal(y, x, map_free_vars)

    xhash = tvm.ir.structural_hash(x, map_free_vars)
    yhash = tvm.ir.structural_hash(y, map_free_vars)

    assert xhash == yhash


def _check_json_roundtrip(x):
    xret = tvm.ir.load_json(tvm.ir.save_json(x))
    _check_equal(x, xret, map_free_vars=True)
    return xret


def test_object_struct_info():
    s0 = rx.ObjectStructInfo()
    s1 = rx.ObjectStructInfo()

    # can turn into str
    str(s0)
    _check_equal(s0, s1)

    assert isinstance(s0, rx.ObjectStructInfo)
    _check_json_roundtrip(s0)


def test_shape_type():
    t0 = rx.ShapeType()
    t1 = rx.ShapeType()
    assert t0 == t1


def test_dyn_tensor_type():
    t0 = rx.DynTensorType()
    assert t0.ndim == -1
    t1 = rx.DynTensorType(3, "int32")
    assert t1.ndim == 3
    assert t1.dtype == "int32"


def test_prim_struct_info():
    s0 = rx.PrimStructInfo("float32")
    s1 = rx.PrimStructInfo("float32")
    s2 = rx.PrimStructInfo("int32")

    _check_equal(s0, s1)

    # can turn into str
    str(s0)

    assert s0 == s1
    assert s0 != s2

    assert isinstance(s0, rx.PrimStructInfo)
    _check_json_roundtrip(s0)
    _check_json_roundtrip(s1)

    assert s1.dtype == "float32"
    assert s2.dtype == "int32"

    # wrong API constructors
    with pytest.raises(TVMError):
        rx.PrimStructInfo([1])


def test_prim_struct_info_with_expr():
    n = tir.Var("n", "int64")
    sinfo = rx.PrimStructInfo(value=n + 1)

    _check_equal(sinfo, rx.PrimStructInfo(value=n + 1))
    assert not tvm.ir.structural_equal(sinfo, rx.PrimStructInfo(dtype=n.dtype))

    # can turn into str
    str(sinfo)

    assert isinstance(sinfo, rx.PrimStructInfo)
    _check_json_roundtrip(sinfo)

    assert sinfo.dtype == "int64"


def test_shape_struct_info():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    s0 = rx.ShapeStructInfo([1, n + 1, m])
    s1 = rx.ShapeStructInfo([1, n + 1, m])

    _check_equal(s0, s1)

    assert s0 == s1
    assert s0.ndim == 3
    assert s1.ndim == 3

    assert s0.values[2] == m

    assert isinstance(s0, rx.ShapeStructInfo)
    _check_json_roundtrip(s0)
    _check_json_roundtrip(s1)

    s2 = rx.ShapeStructInfo(ndim=2)

    assert s2.ndim == 2
    assert s2.values is None
    _check_json_roundtrip(s2)
    assert s0 != s2

    # can turn into str
    str(s0)

    # wrong argument type
    with pytest.raises(TVMError):
        rx.ShapeStructInfo(1)

    # cannot pass both ndim and values
    with pytest.raises(ValueError):
        rx.ShapeStructInfo([1, 2], ndim=3)

    # cannot pass both ndim and values even if they are consistent
    with pytest.raises(ValueError):
        rx.ShapeStructInfo([1, 2], ndim=2)


def test_tensor_struct_info():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    s0 = rx.TensorStructInfo([1, n + 1, m], "float32")
    s1 = rx.TensorStructInfo(rx.ShapeExpr([1, n + 1, m]), "float32")

    _check_equal(s0, s1)

    assert s0 == s1
    assert s0.ndim == 3
    assert s1.ndim == 3

    assert isinstance(s0, rx.TensorStructInfo)
    _check_json_roundtrip(s0)
    _check_json_roundtrip(s1)

    s2 = rx.TensorStructInfo(ndim=2, dtype="int32")

    assert s2.ndim == 2
    assert s2.dtype == "int32"
    assert s2.shape is None
    _check_json_roundtrip(s2)
    assert s0 != s2

    # take in opaque var
    rshape = rx.Var("shape", rx.ShapeStructInfo(ndim=2))

    s3 = rx.TensorStructInfo(rshape, dtype="int32")
    assert s3.dtype == "int32"
    assert s3.shape == rshape
    assert s3.ndim == 2
    _check_json_roundtrip(s3)

    # can turn into str
    str(s0)

    # cannot pass both ndim and values
    with pytest.raises(ValueError):
        rx.TensorStructInfo([1, 2], ndim=3)

    # cannot pass both ndim and values even if they are consistent
    with pytest.raises(ValueError):
        rx.TensorStructInfo([1, 2], ndim=2)


def test_tuple_struct_info():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    s0 = rx.TensorStructInfo([1, 2, m + n], "float32")
    s1 = rx.ObjectStructInfo()

    t0 = rx.TupleStructInfo([s0, s1])
    t1 = rx.TupleStructInfo([s0, rx.ObjectStructInfo()])
    t2 = rx.TupleStructInfo([s0, s0])

    _check_equal(t0, t1)

    assert t0 == t1

    assert isinstance(t0, rx.TupleStructInfo)
    t0 = _check_json_roundtrip(t0)
    t1 = _check_json_roundtrip(t1)
    t2 = _check_json_roundtrip(t2)

    # can turn into str
    str(t0)

    # wrong argument type
    with pytest.raises(TVMError):
        rx.TupleStructInfo(1)


def test_func_struct_info():
    def fn_info(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n, m], "float32")
        return rx.FuncStructInfo([x, y], z)

    f0 = fn_info(1)
    f1 = fn_info(1)
    f2 = fn_info(2)
    f3 = rx.FuncStructInfo.opaque_func()

    _check_equal(f0, f1)

    assert f0 == f1
    assert f0 != f2

    assert len(f0.params) == 2
    assert isinstance(f0.ret, rx.TensorStructInfo)
    assert f2.derive_func is None
    assert f3.params is None
    assert f3.derive_func is None
    _check_equal(f3.ret, rx.ObjectStructInfo())

    assert isinstance(f0, rx.FuncStructInfo)
    f0 = _check_json_roundtrip(f0)
    f1 = _check_json_roundtrip(f1)
    f2 = _check_json_roundtrip(f2)
    f3 = _check_json_roundtrip(f3)

    # can turn into str
    str(f3)


if __name__ == "__main__":
    tvm.testing.main()
