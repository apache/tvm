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

from typing import Union

import tvm
import tvm.testing
from tvm import TVMError, relax, tir
from tvm.ir import Op, VDevice
from tvm.script import relax as R
from tvm.script import tir as T

from tvm.script.parser.relax.entry import StructInfoProxy


def test_op_correctness():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    fill_value = relax.Var("fill_value", R.Tensor((), "float32"))
    assert R.full((2, 3), fill_value).op == Op.get("relax.full")
    assert R.full_like(x, fill_value).op == Op.get("relax.full_like")
    assert R.ones((2, 3), "float32").op == Op.get("relax.ones")
    assert R.ones_like(x).op == Op.get("relax.ones_like")
    assert R.zeros((2, 3), "float32").op == Op.get("relax.zeros")
    assert R.zeros_like(x).op == Op.get("relax.zeros_like")
    assert R.arange(3, 4, 1, "float32").op == Op.get("relax.arange")
    assert R.tril(x).op == Op.get("relax.tril")
    assert R.triu(x).op == Op.get("relax.triu")


def _get_inference_checker(bb: relax.BlockBuilder, normalize_before_check: bool = True):
    def _check(call: relax.Call, expected_sinfo: Union[relax.StructInfo, StructInfoProxy]):
        if isinstance(expected_sinfo, StructInfoProxy):
            expected_sinfo = expected_sinfo.as_struct_info()

        if normalize_before_check:
            call = bb.normalize(call)

        tvm.ir.assert_structural_equal(call.struct_info, expected_sinfo)

    return _check


normalize_before_check = tvm.testing.parameter(True, False)


def test_full_infer_struct_info(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    vdev0 = VDevice("llvm")
    v0 = relax.Var("v", R.Tensor((), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=0))
    v2 = relax.Var("v", R.Tensor(()))
    v3 = relax.Var("v", R.Tensor(ndim=0))
    v4 = relax.Var("v", R.Tensor((), "float32", vdev0))
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    inference_checker(R.full((2, 3), v0, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full((2, 3), v0), R.Tensor((2, 3), "float32"))
    inference_checker(R.full(s0, v0, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full(s0, v0), R.Tensor((2, 3), "float32"))
    inference_checker(R.full(s0, v4), R.Tensor((2, 3), "float32", vdev0))
    inference_checker(R.full(s1, v0, "float16"), R.Tensor(s1, "float16"))
    inference_checker(R.full(s1, v0), R.Tensor(s1, "float32"))
    inference_checker(R.full(s2, v0, "float16"), R.Tensor(s2, "float16"))
    inference_checker(R.full(s2, v0), R.Tensor(s2, "float32"))
    inference_checker(R.full(s3, v0, "float16"), R.Tensor(s3, "float16"))
    inference_checker(R.full(s3, v0), R.Tensor(s3, "float32"))
    inference_checker(R.full((2, 3), v1, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full((2, 3), v1), R.Tensor((2, 3), "float32"))
    inference_checker(R.full(s0, v1, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full(s0, v1), R.Tensor((2, 3), "float32"))
    inference_checker(R.full(s1, v1, "float16"), R.Tensor(s1, "float16"))
    inference_checker(R.full(s1, v1), R.Tensor(s1, "float32"))
    inference_checker(R.full(s2, v1, "float16"), R.Tensor(s2, "float16"))
    inference_checker(R.full(s2, v1), R.Tensor(s2, "float32"))
    inference_checker(R.full(s3, v1, "float16"), R.Tensor(s3, "float16"))
    inference_checker(R.full(s3, v1), R.Tensor(s3, "float32"))
    inference_checker(R.full((2, 3), v2, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full((2, 3), v2), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full(s0, v2, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full(s0, v2), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full(s1, v2, "float16"), R.Tensor(s1, "float16"))
    inference_checker(R.full(s1, v2), R.Tensor(s1, dtype=""))
    inference_checker(R.full(s2, v2, "float16"), R.Tensor(s2, "float16"))
    inference_checker(R.full(s2, v2), R.Tensor(s2, dtype=""))
    inference_checker(R.full(s3, v2, "float16"), R.Tensor(s3, "float16"))
    inference_checker(R.full(s3, v2), R.Tensor(s3, dtype=""))
    inference_checker(R.full((2, 3), v3, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full((2, 3), v3), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full(s0, v3, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full(s0, v3), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full(s1, v3, "float16"), R.Tensor(s1, "float16"))
    inference_checker(
        R.full(
            s1,
            v3,
        ),
        R.Tensor(s1, dtype=""),
    )
    inference_checker(R.full(s2, v3, "float16"), R.Tensor(s2, "float16"))
    inference_checker(
        R.full(
            s2,
            v3,
        ),
        R.Tensor(s2, dtype=""),
    )
    inference_checker(R.full(s3, v3, "float16"), R.Tensor(s3, "float16"))
    inference_checker(R.full(s3, v3), R.Tensor(s3, dtype=""))


def test_full_infer_struct_info_shape_symbolic(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    a = tir.Var("a", "int64")
    v = relax.Var("v", R.Tensor((), "float32"))
    s0 = relax.ShapeExpr((a, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((a, 3)))

    inference_checker(R.full((a, 3), v, "float16"), R.Tensor((a, 3), "float16"))
    inference_checker(R.full((a, 3), v), R.Tensor((a, 3), "float32"))
    inference_checker(R.full(s0, v, "float16"), R.Tensor((a, 3), "float16"))
    inference_checker(R.full(s0, v), R.Tensor((a, 3), "float32"))
    inference_checker(R.full(s1, v, "float16"), R.Tensor(s1, "float16"))
    inference_checker(R.full(s1, v), R.Tensor(s1, "float32"))


def test_full_infer_struct_info_shape_var(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    s0 = relax.Var("s", relax.ShapeStructInfo(()))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    v0 = relax.Var("v", R.Tensor(s0, "float32"))
    v1 = relax.Var("v", R.Tensor(s1, "float32"))

    inference_checker(R.full((2, 3), v0, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full((2, 3), v1, "float16"), R.Tensor((2, 3), "float16"))


def test_full_infer_struct_info_more_input_dtype(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    v0 = relax.Var("v", R.Tensor((), "float16"))
    v1 = relax.Var("v", R.Tensor((), "int8"))
    v2 = relax.Var("v", R.Tensor((), "int32"))

    inference_checker(R.full((2, 3), v0, "float32"), R.Tensor((2, 3), "float32"))
    inference_checker(R.full((2, 3), v0), R.Tensor((2, 3), "float16"))
    inference_checker(R.full((2, 3), v1, "int32"), R.Tensor((2, 3), "int32"))
    inference_checker(R.full((2, 3), v1), R.Tensor((2, 3), "int8"))
    inference_checker(R.full((2, 3), v2, "int8"), R.Tensor((2, 3), "int8"))
    inference_checker(R.full((2, 3), v2), R.Tensor((2, 3), "int32"))


def test_full_infer_struct_info_fill_value_not_scalar_tensor(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    s0 = relax.Var("s", relax.ShapeStructInfo((1,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", R.Tensor(s0, "float32"))
    v4 = relax.Var("v", R.Tensor(s1, "float32"))
    v5 = relax.Var("v", R.Tensor(s2, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v0))
    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v1))
    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v2))
    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v3))
    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v4))
    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v5))


def test_full_shape_not_tuple():
    m = tir.Var("m", "int64")
    v = relax.Var("v", R.Tensor((), "float32"))

    with pytest.raises(TVMError):
        R.full(4, v)
    with pytest.raises(TVMError):
        R.full(m, v)


def test_full_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float32"))
    v1 = relax.Var("v", relax.ShapeStructInfo(()))
    v2 = relax.Var("v", relax.FuncStructInfo([], R.Tensor((), "float32")))
    s = relax.Var("s", R.Tensor((2, 3)))

    with pytest.raises(TVMError):
        bb.normalize(R.full(s, v0))
    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v1))
    with pytest.raises(TVMError):
        bb.normalize(R.full((2, 3), v2))


def test_full_like_infer_struct_info(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())
    v0 = relax.Var("v", R.Tensor((), "float16"))
    v1 = relax.Var("v", R.Tensor("float16", ndim=0))
    v2 = relax.Var("v", R.Tensor(()))
    v3 = relax.Var("v", R.Tensor(ndim=0))

    inference_checker(R.full_like(x0, v0), R.Tensor((2, 3), "float32"))
    inference_checker(R.full_like(x0, v1), R.Tensor((2, 3), "float32"))
    inference_checker(R.full_like(x0, v2), R.Tensor((2, 3), "float32"))
    inference_checker(R.full_like(x0, v3), R.Tensor((2, 3), "float32"))
    inference_checker(R.full_like(x1, v0), R.Tensor(dtype="float32", ndim=2))
    inference_checker(R.full_like(x1, v1), R.Tensor(dtype="float32", ndim=2))
    inference_checker(R.full_like(x1, v2), R.Tensor(dtype="float32", ndim=2))
    inference_checker(R.full_like(x1, v3), R.Tensor(dtype="float32", ndim=2))
    inference_checker(R.full_like(x2, v0), R.Tensor(dtype="float32"))
    inference_checker(R.full_like(x2, v1), R.Tensor(dtype="float32"))
    inference_checker(R.full_like(x2, v2), R.Tensor(dtype="float32"))
    inference_checker(R.full_like(x2, v3), R.Tensor(dtype="float32"))
    inference_checker(R.full_like(x3, v0), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full_like(x3, v1), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full_like(x3, v2), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full_like(x3, v3), R.Tensor((2, 3), dtype=""))
    inference_checker(R.full_like(x4, v0), R.Tensor(dtype="", ndim=2))
    inference_checker(R.full_like(x4, v1), R.Tensor(dtype="", ndim=2))
    inference_checker(R.full_like(x4, v2), R.Tensor(dtype="", ndim=2))
    inference_checker(R.full_like(x4, v3), R.Tensor(dtype="", ndim=2))
    inference_checker(R.full_like(x5, v0), R.Tensor(dtype=""))
    inference_checker(R.full_like(x5, v1), R.Tensor(dtype=""))
    inference_checker(R.full_like(x5, v2), R.Tensor(dtype=""))
    inference_checker(R.full_like(x5, v3), R.Tensor(dtype=""))
    inference_checker(R.full_like(x0, v0, dtype="float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full_like(x0, v2, dtype="float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full_like(x3, v0, dtype="float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.full_like(x3, v2, dtype="float16"), R.Tensor((2, 3), "float16"))


def test_full_like_infer_struct_info_shape_symbolic(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))
    v = relax.Var("v", R.Tensor((), "float16"))

    inference_checker(R.full_like(x0, v), R.Tensor((m, n), "float32"))
    inference_checker(R.full_like(x1, v), R.Tensor((m, n), dtype=""))


def test_full_like_infer_struct_info_shape_var(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    vdev0 = VDevice("llvm")
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor(s0, "float32"))
    x1 = relax.Var("x", R.Tensor(s1, "float32"))
    x2 = relax.Var("x", R.Tensor(s2, "float32"))
    x3 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x4 = relax.Var("x", R.Tensor((2, 3), "float32", vdev0))
    sv0 = relax.Var("sv", relax.ShapeStructInfo(()))
    sv1 = relax.Var("sv", relax.ShapeStructInfo(ndim=0))
    v0 = relax.Var("v", R.Tensor(sv0, "float16"))
    v1 = relax.Var("v", R.Tensor(sv1, "float16"))
    v2 = relax.Var("v", R.Tensor((), "float16"))
    v3 = relax.Var("v", R.Tensor(sv1, "float16", vdev0))

    inference_checker(R.full_like(x0, v0), R.Tensor(s0, "float32"))
    inference_checker(R.full_like(x0, v1), R.Tensor(s0, "float32"))
    inference_checker(R.full_like(x0, v2), R.Tensor(s0, "float32"))
    inference_checker(R.full_like(x1, v0), R.Tensor(s1, "float32"))
    inference_checker(R.full_like(x1, v1), R.Tensor(s1, "float32"))
    inference_checker(R.full_like(x1, v2), R.Tensor(s1, "float32"))
    inference_checker(R.full_like(x2, v0), R.Tensor(s2, "float32"))
    inference_checker(R.full_like(x2, v1), R.Tensor(s2, "float32"))
    inference_checker(R.full_like(x2, v2), R.Tensor(s2, "float32"))
    inference_checker(R.full_like(x3, v0), R.Tensor((2, 3), "float32"))
    inference_checker(R.full_like(x3, v1), R.Tensor((2, 3), "float32"))
    inference_checker(R.full_like(x4, v3), R.Tensor((2, 3), "float32", vdev0))


def test_full_like_infer_struct_info_more_input_dtype(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    v0 = relax.Var("v", R.Tensor((), "int32"))
    v1 = relax.Var("v", R.Tensor((), "float64"))

    inference_checker(R.full_like(x0, v0), R.Tensor((2, 3), "float16"))
    inference_checker(R.full_like(x0, v1), R.Tensor((2, 3), "float16"))
    inference_checker(R.full_like(x1, v0), R.Tensor((2, 3), "int8"))
    inference_checker(R.full_like(x1, v1), R.Tensor((2, 3), "int8"))


def test_full_like_infer_struct_info_fill_value_not_scalar_tensor(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    s0 = relax.Var("s", relax.ShapeStructInfo((1,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", R.Tensor(s0, "float32"))
    v4 = relax.Var("v", R.Tensor(s1, "float32"))
    v5 = relax.Var("v", R.Tensor(s2, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x, v0))
    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x, v1))
    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x, v2))
    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x, v3))
    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x, v4))
    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x, v5))


def test_full_like_infer_struct_info_wrong_input_type(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3)))
    v0 = relax.Var("v", R.Tensor(()))
    v1 = relax.Var("v", relax.ShapeStructInfo(()))

    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x0, v0))
    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x1, v0))
    with pytest.raises(TVMError):
        bb.normalize(R.full_like(x2, v1))


def test_ones_zeros_infer_struct_info(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    inference_checker(R.ones((2, 3), "float32"), R.Tensor((2, 3), "float32"))
    inference_checker(R.ones(s0, "float32"), R.Tensor((2, 3), "float32"))
    inference_checker(R.ones(s1, "float32"), R.Tensor(s1, "float32"))
    inference_checker(R.ones(s2, "float32"), R.Tensor(s2, "float32"))
    inference_checker(R.ones(s3, "float32"), R.Tensor(s3, "float32"))
    inference_checker(R.zeros((2, 3), "float32"), R.Tensor((2, 3), "float32"))
    inference_checker(R.zeros(s0, "float32"), R.Tensor((2, 3), "float32"))
    inference_checker(R.zeros(s1, "float32"), R.Tensor(s1, "float32"))
    inference_checker(R.zeros(s2, "float32"), R.Tensor(s2, "float32"))
    inference_checker(R.zeros(s3, "float32"), R.Tensor(s3, "float32"))


def test_ones_zeros_infer_struct_info_shape_symbolic(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    s0 = relax.ShapeExpr((m, n))
    s1 = relax.Var("s", relax.ShapeStructInfo((m, n)))

    inference_checker(R.ones((m, n), "float32"), R.Tensor((m, n), "float32"))
    inference_checker(R.ones(s0, "float32"), R.Tensor((m, n), "float32"))
    inference_checker(R.ones(s1, "float32"), R.Tensor(s1, "float32"))
    inference_checker(R.zeros((m, n), "float32"), R.Tensor((m, n), "float32"))
    inference_checker(R.zeros(s0, "float32"), R.Tensor((m, n), "float32"))
    inference_checker(R.zeros(s1, "float32"), R.Tensor(s1, "float32"))


def test_ones_zeros_infer_struct_info_more_input_dtype(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    inference_checker(R.ones(s0, "float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.ones(s1, "int8"), R.Tensor(s1, "int8"))
    inference_checker(R.zeros(s2, "int32"), R.Tensor(s2, "int32"))
    inference_checker(R.zeros(s3, "float64"), R.Tensor(s3, "float64"))


def test_ones_zeros_shape_not_tuple():
    m = tir.Var("m", "int64")

    with pytest.raises(TVMError):
        R.ones(10, "float32")
    with pytest.raises(TVMError):
        R.zeros(m, "float32")


def test_ones_zeros_wrong_dtype():
    with pytest.raises(TypeError):
        R.ones((2, 3))
    with pytest.raises(TVMError):
        R.ones((2, 3), "")
    with pytest.raises(TypeError):
        R.zeros((2, 3))
    with pytest.raises(TVMError):
        R.zeros((2, 3), "")


def test_ones_zeros_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", R.Tensor((2, 3)))
    s1 = relax.Var("s", relax.FuncStructInfo([], R.Tensor((2, 3))))

    with pytest.raises(TVMError):
        bb.normalize(R.ones(s0, "float32"))
    with pytest.raises(TVMError):
        bb.normalize(R.zeros(s1, "float32"))


def test_ones_like_zeros_like_infer_struct_info(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())

    inference_checker(R.ones_like(x0), R.Tensor((2, 3), "float32"))
    inference_checker(R.zeros_like(x1), R.Tensor(dtype="float32", ndim=2))
    inference_checker(R.ones_like(x2), R.Tensor(dtype="float32"))
    inference_checker(R.zeros_like(x3), R.Tensor((2, 3), dtype=""))
    inference_checker(R.ones_like(x4), R.Tensor(dtype="", ndim=2))
    inference_checker(R.zeros_like(x5), R.Tensor(dtype=""))
    inference_checker(R.ones_like(x0, dtype="float16"), R.Tensor((2, 3), "float16"))
    inference_checker(R.zeros_like(x3, dtype="float16"), R.Tensor((2, 3), "float16"))


def test_ones_like_zeros_like_infer_struct_info_shape_symbolic(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    inference_checker(R.ones_like(x0), R.Tensor((m, n), "float32"))
    inference_checker(R.zeros_like(x1), R.Tensor((m, n), dtype=""))


def test_ones_like_zeros_like_infer_struct_info_shape_var(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor(s0, "float32"))
    x1 = relax.Var("x", R.Tensor(s1, "float32"))
    x2 = relax.Var("x", R.Tensor(s2, "float32"))

    inference_checker(R.ones_like(x0), R.Tensor(s0, "float32"))
    inference_checker(R.zeros_like(x1), R.Tensor(s1, "float32"))
    inference_checker(R.zeros_like(x2), R.Tensor(s2, "float32"))


def test_ones_like_zeros_like_infer_struct_info_more_input_dtype(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))

    inference_checker(R.ones_like(x0), R.Tensor((2, 3), "float64"))
    inference_checker(R.zeros_like(x1), R.Tensor((2, 3), "int8"))


def test_ones_like_zeros_like_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(R.ones_like(x0))
    with pytest.raises(TVMError):
        bb.normalize(R.zeros_like(x1))


def test_arange_infer_struct_info(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)

    inference_checker(R.arange(10), R.Tensor((10,), "int64"))
    inference_checker(R.arange(1, 10), R.Tensor((9,), "int64"))
    inference_checker(R.arange(0, 10, 2), R.Tensor((5,), "int64"))
    inference_checker(R.arange(1, 10, 2), R.Tensor((5,), "int64"))

    inference_checker(R.arange(10.0), R.Tensor((10,), "float32"))
    inference_checker(R.arange(1.0, 10), R.Tensor((9,), "float32"))
    inference_checker(R.arange(0, 20, 2.5), R.Tensor((8,), "float32"))
    inference_checker(R.arange(1, 10, 2.3), R.Tensor((4,), "float32"))


def test_arange_infer_struct_info_shape_var(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    start = tir.Var("start", "int64")
    stop = tir.Var("stop", "int64")
    step = tir.Var("step", "int64")

    inference_checker(R.arange(stop), R.Tensor((stop,), "int64"))
    inference_checker(R.arange(1, stop), R.Tensor((stop - 1,), "int64"))
    inference_checker(R.arange(start, stop), R.Tensor((stop - start,), "int64"))
    inference_checker(
        R.arange(start, stop, 2),
        R.Tensor(((stop + 1 - start) // 2,), "int64"),
    )
    inference_checker(
        R.arange(start, stop, step),
        R.Tensor(((stop + step - start - 1) // step,), "int64"),
    )

    start = tir.Var("start", "float32")
    stop = tir.Var("stop", "float32")
    step = tir.Var("step", "float32")

    inference_checker(
        R.arange(stop),
        R.Tensor((T.cast(T.ceil(stop), "int64"),), "float32"),
    )
    inference_checker(
        R.arange(1, stop),
        R.Tensor((T.cast(T.ceil(stop - 1.0), "int64"),), "float32"),
    )
    inference_checker(
        R.arange(start, stop),
        R.Tensor((T.cast(T.ceil(stop - start), "int64"),), "float32"),
    )
    inference_checker(
        R.arange(start, stop, 2),
        R.Tensor((T.cast(T.ceil((stop - start) * 0.5), "int64"),), "float32"),
    )
    inference_checker(
        R.arange(start, stop, step),
        R.Tensor((T.cast(T.ceil((stop - start) / step), "int64"),), "float32"),
    )


def test_tril_triu_infer_struct_info(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))

    inference_checker(R.tril(x0, k=1), R.Tensor((2, 3, 4), "float32"))
    inference_checker(R.triu(x0, k=0), R.Tensor((2, 3, 4), "float32"))
    inference_checker(R.tril(x0), R.Tensor((2, 3, 4), "float32"))
    inference_checker(R.triu(x1), R.Tensor(dtype="float32", ndim=3))
    inference_checker(R.tril(x2), R.Tensor(dtype="float32"))
    inference_checker(R.triu(x3), R.Tensor((2, 3, 4), dtype=""))
    inference_checker(R.tril(x4), R.Tensor(dtype="", ndim=3))
    inference_checker(R.triu(x5), R.Tensor(dtype=""))
    inference_checker(R.tril(x6), R.Tensor((2, 3, 4), "float32", vdev0))


def test_tril_triu_infer_struct_info_shape_symbolic(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    vdev0 = VDevice("llvm")
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((a, b, c), "float32"))
    x1 = relax.Var("x", R.Tensor((a, b, c)))
    x2 = relax.Var("x", R.Tensor((a, b, c), "float32", vdev0))
    x3 = relax.Var("x", R.Tensor((16, 32, 64)))

    # Dynamic tensor, static offset
    inference_checker(R.tril(x0), R.Tensor((a, b, c), "float32"))
    inference_checker(R.triu(x1), R.Tensor((a, b, c), dtype=""))
    inference_checker(R.tril(x2), R.Tensor((a, b, c), "float32", vdev0))

    # Static tensor, dynamic offset
    inference_checker(R.tril(x3, a), R.Tensor((16, 32, 64), dtype=""))

    # Dynamic tensor, dynamic offset
    inference_checker(R.tril(x0, a), R.Tensor((a, b, c), "float32"))


def test_tril_triu_infer_struct_info_shape_var(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor(s0, "float32"))
    x1 = relax.Var("x", R.Tensor(s1, "float32"))
    x2 = relax.Var("x", R.Tensor(s2, "float32"))

    inference_checker(R.tril(x0), R.Tensor(s0, "float32"))
    inference_checker(R.triu(x1), R.Tensor(s1, "float32"))
    inference_checker(R.tril(x2), R.Tensor(s2, "float32"))


def test_tril_triu_infer_struct_info_more_input_dtype(normalize_before_check):
    bb = relax.BlockBuilder()
    inference_checker = _get_inference_checker(bb, normalize_before_check=normalize_before_check)
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 4), "int32"))

    inference_checker(R.triu(x0), R.Tensor((2, 3, 4), "float16"))
    inference_checker(R.tril(x1), R.Tensor((2, 3, 4), "int8"))
    inference_checker(R.triu(x2), R.Tensor((2, 3, 4), "int32"))


def test_tril_triu_infer_struct_info_less_than_two_ndim():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(()))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s3 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    x0 = relax.Var("x", R.Tensor((2,), "float32"))
    x1 = relax.Var("x", R.Tensor((), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=1))
    x3 = relax.Var("x", R.Tensor("float32", ndim=0))
    x4 = relax.Var("x", R.Tensor(s0, "float32"))
    x5 = relax.Var("x", R.Tensor(s1, "float32"))
    x6 = relax.Var("x", R.Tensor(s2, "float32"))
    x7 = relax.Var("x", R.Tensor(s3, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(R.tril(x0))
    with pytest.raises(TVMError):
        bb.normalize(R.triu(x1))
    with pytest.raises(TVMError):
        bb.normalize(R.tril(x2))
    with pytest.raises(TVMError):
        bb.normalize(R.triu(x3))
    with pytest.raises(TVMError):
        bb.normalize(R.tril(x4))
    with pytest.raises(TVMError):
        bb.normalize(R.triu(x5))
    with pytest.raises(TVMError):
        bb.normalize(R.tril(x6))
    with pytest.raises(TVMError):
        bb.normalize(R.triu(x7))


def test_tril_triu_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(R.tril(x0))
    with pytest.raises(TVMError):
        bb.normalize(R.triu(x1))


if __name__ == "__main__":
    tvm.testing.main()
