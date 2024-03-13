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
import tvm.testing
from tvm import TVMError, relax, tir
from tvm.ir import Op, VDevice
from tvm.script import relax as R
from tvm.script import tir as T


def test_op_correctness():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    fill_value = relax.Var("fill_value", R.Tensor((), "float32"))
    assert relax.op.full((2, 3), fill_value).op == Op.get("relax.full")
    assert relax.op.full_like(x, fill_value).op == Op.get("relax.full_like")
    assert relax.op.ones((2, 3), "float32").op == Op.get("relax.ones")
    assert relax.op.ones_like(x).op == Op.get("relax.ones_like")
    assert relax.op.zeros((2, 3), "float32").op == Op.get("relax.zeros")
    assert relax.op.zeros_like(x).op == Op.get("relax.zeros_like")
    assert relax.op.arange(3, 4, 1, "float32").op == Op.get("relax.arange")
    assert relax.op.tril(x).op == Op.get("relax.tril")
    assert relax.op.triu(x).op == Op.get("relax.triu")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_full_infer_struct_info():
    bb = relax.BlockBuilder()
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

    _check_inference(
        bb, relax.op.full((2, 3), v0, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full((2, 3), v0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.full(s0, v0, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(s0, v0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full(s0, v4), relax.TensorStructInfo((2, 3), "float32", vdev0))
    _check_inference(bb, relax.op.full(s1, v0, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v0), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full(s2, v0, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(bb, relax.op.full(s2, v0), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full(s3, v0, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v0), relax.TensorStructInfo(s3, "float32"))
    _check_inference(
        bb, relax.op.full((2, 3), v1, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full((2, 3), v1), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.full(s0, v1, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(s0, v1), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full(s1, v1, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full(s2, v1, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(bb, relax.op.full(s2, v1), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full(s3, v1, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v1), relax.TensorStructInfo(s3, "float32"))
    _check_inference(
        bb, relax.op.full((2, 3), v2, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full((2, 3), v2), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(
        bb, relax.op.full(s0, v2, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(s0, v2), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full(s1, v2, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v2), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.full(s2, v2, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(bb, relax.op.full(s2, v2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.full(s3, v2, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v2), relax.TensorStructInfo(s3, dtype=""))
    _check_inference(
        bb, relax.op.full((2, 3), v3, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full((2, 3), v3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(
        bb, relax.op.full(s0, v3, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(bb, relax.op.full(s0, v3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full(s1, v3, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(
        bb,
        relax.op.full(
            s1,
            v3,
        ),
        relax.TensorStructInfo(s1, dtype=""),
    )
    _check_inference(bb, relax.op.full(s2, v3, "float16"), relax.TensorStructInfo(s2, "float16"))
    _check_inference(
        bb,
        relax.op.full(
            s2,
            v3,
        ),
        relax.TensorStructInfo(s2, dtype=""),
    )
    _check_inference(bb, relax.op.full(s3, v3, "float16"), relax.TensorStructInfo(s3, "float16"))
    _check_inference(bb, relax.op.full(s3, v3), relax.TensorStructInfo(s3, dtype=""))


def test_full_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    v = relax.Var("v", R.Tensor((), "float32"))
    s0 = relax.ShapeExpr((a, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((a, 3)))

    _check_inference(
        bb, relax.op.full((a, 3), v, "float16"), relax.TensorStructInfo((a, 3), "float16")
    )
    _check_inference(bb, relax.op.full((a, 3), v), relax.TensorStructInfo((a, 3), "float32"))
    _check_inference(bb, relax.op.full(s0, v, "float16"), relax.TensorStructInfo((a, 3), "float16"))
    _check_inference(bb, relax.op.full(s0, v), relax.TensorStructInfo((a, 3), "float32"))
    _check_inference(bb, relax.op.full(s1, v, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.full(s1, v), relax.TensorStructInfo(s1, "float32"))


def test_full_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(()))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    v0 = relax.Var("v", relax.TensorStructInfo(s0, "float32"))
    v1 = relax.Var("v", relax.TensorStructInfo(s1, "float32"))

    _check_inference(
        bb, relax.op.full((2, 3), v0, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full((2, 3), v1, "float16"), relax.TensorStructInfo((2, 3), "float16")
    )


def test_full_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float16"))
    v1 = relax.Var("v", R.Tensor((), "int8"))
    v2 = relax.Var("v", R.Tensor((), "int32"))

    _check_inference(
        bb, relax.op.full((2, 3), v0, "float32"), relax.TensorStructInfo((2, 3), "float32")
    )
    _check_inference(bb, relax.op.full((2, 3), v0), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(
        bb, relax.op.full((2, 3), v1, "int32"), relax.TensorStructInfo((2, 3), "int32")
    )
    _check_inference(bb, relax.op.full((2, 3), v1), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.full((2, 3), v2, "int8"), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.full((2, 3), v2), relax.TensorStructInfo((2, 3), "int32"))


def test_full_infer_struct_info_fill_value_not_scalar_tensor():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((1,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", relax.TensorStructInfo(s0, "float32"))
    v4 = relax.Var("v", relax.TensorStructInfo(s1, "float32"))
    v5 = relax.Var("v", relax.TensorStructInfo(s2, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v5))


def test_full_shape_not_tuple():
    m = tir.Var("m", "int64")
    v = relax.Var("v", R.Tensor((), "float32"))

    with pytest.raises(TVMError):
        relax.op.full(4, v)
    with pytest.raises(TVMError):
        relax.op.full(m, v)


def test_full_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    v0 = relax.Var("v", R.Tensor((), "float32"))
    v1 = relax.Var("v", relax.ShapeStructInfo(()))
    v2 = relax.Var("v", relax.FuncStructInfo([], R.Tensor((), "float32")))
    s = relax.Var("s", R.Tensor((2, 3)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full(s, v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full((2, 3), v2))


def test_full_like_infer_struct_info():
    bb = relax.BlockBuilder()
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

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v2), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x0, v3), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.full_like(x1, v0), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.full_like(x1, v1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.full_like(x1, v2), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.full_like(x1, v3), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.full_like(x2, v0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x2, v3), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.full_like(x3, v0), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x3, v1), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x3, v2), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x3, v3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.full_like(x4, v0), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v2), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x4, v3), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.full_like(x5, v0), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.full_like(x5, v1), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.full_like(x5, v2), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.full_like(x5, v3), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.full_like(x0, v0, dtype="float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full_like(x0, v2, dtype="float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full_like(x3, v0, dtype="float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.full_like(x3, v2, dtype="float16"), relax.TensorStructInfo((2, 3), "float16")
    )


def test_full_like_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))
    v = relax.Var("v", R.Tensor((), "float16"))

    _check_inference(bb, relax.op.full_like(x0, v), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.full_like(x1, v), relax.TensorStructInfo((m, n), dtype=""))


def test_full_like_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x3 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x4 = relax.Var("x", R.Tensor((2, 3), "float32", vdev0))
    sv0 = relax.Var("sv", relax.ShapeStructInfo(()))
    sv1 = relax.Var("sv", relax.ShapeStructInfo(ndim=0))
    v0 = relax.Var("v", relax.TensorStructInfo(sv0, "float16"))
    v1 = relax.Var("v", relax.TensorStructInfo(sv1, "float16"))
    v2 = relax.Var("v", R.Tensor((), "float16"))
    v3 = relax.Var("v", relax.TensorStructInfo(sv1, "float16", vdev0))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x0, v2), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v0), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x1, v2), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v0), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v1), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x2, v2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.full_like(x3, v0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.full_like(x3, v1), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.full_like(x4, v3), relax.TensorStructInfo((2, 3), "float32", vdev0)
    )


def test_full_like_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    v0 = relax.Var("v", R.Tensor((), "int32"))
    v1 = relax.Var("v", R.Tensor((), "float64"))

    _check_inference(bb, relax.op.full_like(x0, v0), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.full_like(x0, v1), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.full_like(x1, v0), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.full_like(x1, v1), relax.TensorStructInfo((2, 3), "int8"))


def test_full_like_infer_struct_info_fill_value_not_scalar_tensor():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    s0 = relax.Var("s", relax.ShapeStructInfo((1,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    v0 = relax.Var("v", R.Tensor((1,), "float32"))
    v1 = relax.Var("v", R.Tensor("float32", ndim=1))
    v2 = relax.Var("v", R.Tensor("float32"))
    v3 = relax.Var("v", relax.TensorStructInfo(s0, "float32"))
    v4 = relax.Var("v", relax.TensorStructInfo(s1, "float32"))
    v5 = relax.Var("v", relax.TensorStructInfo(s2, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x, v5))


def test_full_like_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3)))
    v0 = relax.Var("v", R.Tensor(()))
    v1 = relax.Var("v", relax.ShapeStructInfo(()))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x0, v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x1, v0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.full_like(x2, v1))


def test_ones_zeros_infer_struct_info():
    bb = relax.BlockBuilder()
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    _check_inference(
        bb, relax.op.ones((2, 3), "float32"), relax.TensorStructInfo((2, 3), "float32")
    )
    _check_inference(bb, relax.op.ones(s0, "float32"), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.ones(s1, "float32"), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.ones(s2, "float32"), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.ones(s3, "float32"), relax.TensorStructInfo(s3, "float32"))
    _check_inference(
        bb, relax.op.zeros((2, 3), "float32"), relax.TensorStructInfo((2, 3), "float32")
    )
    _check_inference(bb, relax.op.zeros(s0, "float32"), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.zeros(s1, "float32"), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.zeros(s2, "float32"), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.zeros(s3, "float32"), relax.TensorStructInfo(s3, "float32"))


def test_ones_zeros_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    s0 = relax.ShapeExpr((m, n))
    s1 = relax.Var("s", relax.ShapeStructInfo((m, n)))

    _check_inference(
        bb, relax.op.ones((m, n), "float32"), relax.TensorStructInfo((m, n), "float32")
    )
    _check_inference(bb, relax.op.ones(s0, "float32"), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.ones(s1, "float32"), relax.TensorStructInfo(s1, "float32"))
    _check_inference(
        bb, relax.op.zeros((m, n), "float32"), relax.TensorStructInfo((m, n), "float32")
    )
    _check_inference(bb, relax.op.zeros(s0, "float32"), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.zeros(s1, "float32"), relax.TensorStructInfo(s1, "float32"))


def test_ones_zeros_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    s0 = relax.ShapeExpr((2, 3))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s", relax.ShapeStructInfo())

    _check_inference(bb, relax.op.ones(s0, "float16"), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.ones(s1, "int8"), relax.TensorStructInfo(s1, "int8"))
    _check_inference(bb, relax.op.zeros(s2, "int32"), relax.TensorStructInfo(s2, "int32"))
    _check_inference(bb, relax.op.zeros(s3, "float64"), relax.TensorStructInfo(s3, "float64"))


def test_ones_zeros_shape_not_tuple():
    m = tir.Var("m", "int64")

    with pytest.raises(TVMError):
        relax.op.ones(10, "float32")
    with pytest.raises(TVMError):
        relax.op.zeros(m, "float32")


def test_ones_zeros_wrong_dtype():
    with pytest.raises(TypeError):
        relax.op.ones((2, 3))
    with pytest.raises(TVMError):
        relax.op.ones((2, 3), "")
    with pytest.raises(TypeError):
        relax.op.zeros((2, 3))
    with pytest.raises(TVMError):
        relax.op.zeros((2, 3), "")


def test_ones_zeros_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", R.Tensor((2, 3)))
    s1 = relax.Var("s", relax.FuncStructInfo([], R.Tensor((2, 3))))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.ones(s0, "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.zeros(s1, "float32"))


def test_ones_like_zeros_like_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.ones_like(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.zeros_like(x3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.ones_like(x4), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.zeros_like(x5), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.ones_like(x0, dtype="float16"), relax.TensorStructInfo((2, 3), "float16")
    )
    _check_inference(
        bb, relax.op.zeros_like(x3, dtype="float16"), relax.TensorStructInfo((2, 3), "float16")
    )


def test_ones_like_zeros_like_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorStructInfo((m, n), dtype=""))


def test_ones_like_zeros_like_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.zeros_like(x2), relax.TensorStructInfo(s2, "float32"))


def test_ones_like_zeros_like_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))

    _check_inference(bb, relax.op.ones_like(x0), relax.TensorStructInfo((2, 3), "float64"))
    _check_inference(bb, relax.op.zeros_like(x1), relax.TensorStructInfo((2, 3), "int8"))


def test_ones_like_zeros_like_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.ones_like(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.zeros_like(x1))


def test_arange_infer_struct_info():
    bb = relax.BlockBuilder()

    _check_inference(bb, relax.op.arange(10), relax.TensorStructInfo((10,), "int64"))
    _check_inference(bb, relax.op.arange(1, 10), relax.TensorStructInfo((9,), "int64"))
    _check_inference(bb, relax.op.arange(0, 10, 2), relax.TensorStructInfo((5,), "int64"))
    _check_inference(bb, relax.op.arange(1, 10, 2), relax.TensorStructInfo((5,), "int64"))

    _check_inference(bb, relax.op.arange(10.0), relax.TensorStructInfo((10,), "float32"))
    _check_inference(bb, relax.op.arange(1.0, 10), relax.TensorStructInfo((9,), "float32"))
    _check_inference(bb, relax.op.arange(0, 20, 2.5), relax.TensorStructInfo((8,), "float32"))
    _check_inference(bb, relax.op.arange(1, 10, 2.3), relax.TensorStructInfo((4,), "float32"))


def test_arange_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    start = tir.Var("start", "int64")
    stop = tir.Var("stop", "int64")
    step = tir.Var("step", "int64")

    _check_inference(bb, relax.op.arange(stop), relax.TensorStructInfo((stop,), "int64"))
    _check_inference(bb, relax.op.arange(1, stop), relax.TensorStructInfo((stop - 1,), "int64"))
    _check_inference(
        bb, relax.op.arange(start, stop), relax.TensorStructInfo((stop - start,), "int64")
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop, 2),
        relax.TensorStructInfo(((stop + 1 - start) // 2,), "int64"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop, step),
        relax.TensorStructInfo(((stop + step - start - 1) // step,), "int64"),
    )

    start = tir.Var("start", "float32")
    stop = tir.Var("stop", "float32")
    step = tir.Var("step", "float32")

    _check_inference(
        bb,
        relax.op.arange(stop),
        relax.TensorStructInfo((T.cast(T.ceil(stop), "int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(1, stop),
        relax.TensorStructInfo((T.cast(T.ceil(stop - 1.0), "int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop),
        relax.TensorStructInfo((T.cast(T.ceil(stop - start), "int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop, 2),
        relax.TensorStructInfo((T.cast(T.ceil((stop - start) * 0.5), "int64"),), "float32"),
    )
    _check_inference(
        bb,
        relax.op.arange(start, stop, step),
        relax.TensorStructInfo((T.cast(T.ceil((stop - start) / step), "int64"),), "float32"),
    )


def test_tril_triu_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))

    _check_inference(bb, relax.op.tril(x0, k=1), relax.TensorStructInfo((2, 3, 4), "float32"))
    _check_inference(bb, relax.op.triu(x0, k=0), relax.TensorStructInfo((2, 3, 4), "float32"))
    _check_inference(bb, relax.op.tril(x0), relax.TensorStructInfo((2, 3, 4), "float32"))
    _check_inference(bb, relax.op.triu(x1), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.tril(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.triu(x3), relax.TensorStructInfo((2, 3, 4), dtype=""))
    _check_inference(bb, relax.op.tril(x4), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.triu(x5), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.tril(x6), relax.TensorStructInfo((2, 3, 4), "float32", vdev0))


def test_tril_triu_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((a, b, c), "float32"))
    x1 = relax.Var("x", R.Tensor((a, b, c)))
    x2 = relax.Var("x", R.Tensor((a, b, c), "float32", vdev0))
    x3 = relax.Var("x", R.Tensor((16, 32, 64)))

    # Dynamic tensor, static offset
    _check_inference(bb, relax.op.tril(x0), relax.TensorStructInfo((a, b, c), "float32"))
    _check_inference(bb, relax.op.triu(x1), relax.TensorStructInfo((a, b, c), dtype=""))
    _check_inference(bb, relax.op.tril(x2), relax.TensorStructInfo((a, b, c), "float32", vdev0))

    # Static tensor, dynamic offset
    _check_inference(bb, relax.op.tril(x3, a), relax.TensorStructInfo((16, 32, 64), dtype=""))

    # Dynamic tensor, dynamic offset
    _check_inference(bb, relax.op.tril(x0, a), relax.TensorStructInfo((a, b, c), "float32"))


def test_tril_triu_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.tril(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.triu(x1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.tril(x2), relax.TensorStructInfo(s2, "float32"))


def test_tril_triu_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 4), "int32"))

    _check_inference(bb, relax.op.triu(x0), relax.TensorStructInfo((2, 3, 4), "float16"))
    _check_inference(bb, relax.op.tril(x1), relax.TensorStructInfo((2, 3, 4), "int8"))
    _check_inference(bb, relax.op.triu(x2), relax.TensorStructInfo((2, 3, 4), "int32"))


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
    x4 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x6 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x7 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.tril(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.triu(x1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tril(x2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.triu(x3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tril(x4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.triu(x5))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tril(x6))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.triu(x7))


def test_tril_triu_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.tril(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.triu(x1))


if __name__ == "__main__":
    tvm.testing.main()
