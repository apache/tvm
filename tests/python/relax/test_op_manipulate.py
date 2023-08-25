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
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op, VDevice
from tvm.script import relax as R


def test_op_correctness():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    assert relax.op.broadcast_to(x, (3, 3, 4, 5)).op == Op.get("relax.broadcast_to")
    assert relax.op.concat([x]).op == Op.get("relax.concat")
    assert relax.op.expand_dims(x, axis=[]).op == Op.get("relax.expand_dims")
    assert relax.op.flatten(x).op == Op.get("relax.flatten")
    assert relax.op.permute_dims(x).op == Op.get("relax.permute_dims")
    assert relax.op.reshape(x, (4, 5, 3)).op == Op.get("relax.reshape")
    assert relax.op.split(x, indices_or_sections=1).op == Op.get("relax.split")
    assert relax.op.tile(x, (2, 2, 2)).op == Op.get("relax.tile")
    assert relax.op.repeat(x, 2, 0).op == Op.get("relax.repeat")
    assert relax.op.squeeze(x).op == Op.get("relax.squeeze")
    assert relax.op.layout_transform(x, index_map=lambda a, b, c: (b, c, a)).op == Op.get(
        "relax.layout_transform"
    )
    assert relax.op.collapse_sum_to(x, (4, 5)).op == Op.get("relax.collapse_sum_to")
    y = relax.Var("x", R.Tensor((4, 5), "float32"))
    assert relax.op.collapse_sum_like(x, y).op == Op.get("relax.collapse_sum_like")
    assert relax.op.cumsum(x, axis=1, dtype="int32").op == Op.get("relax.cumsum")
    assert relax.op.einsum(x, subscripts="ii").op == Op.get("relax.einsum")
    assert relax.op.flip(x, axis=1).op == Op.get("relax.flip")
    assert relax.op.scatter_elements(x, x, x).op == Op.get("relax.scatter_elements")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_reshape_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32", vdev0))
    s0 = relax.Var("s", R.Shape((3, 8, 5)))
    s1 = relax.Var("s", R.Shape(ndim=3))
    s2 = relax.Var("s", R.Shape())
    s3 = relax.ShapeExpr((3, 8, 5))

    _check_inference(
        bb, relax.op.reshape(x0, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x6, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32", vdev0)
    )
    _check_inference(
        bb, relax.op.reshape(x0, (3, -1, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(bb, relax.op.reshape(x0, (-1,)), relax.TensorStructInfo((120,), "float32"))
    _check_inference(
        bb, relax.op.reshape(x0, relax.ShapeExpr([-1])), relax.TensorStructInfo((120,), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x1, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x2, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x3, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    _check_inference(
        bb, relax.op.reshape(x3, (3, -1, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    _check_inference(
        bb, relax.op.reshape(x4, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    _check_inference(
        bb, relax.op.reshape(x5, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), dtype="")
    )
    # Remove Var from StructInfo when we can
    _check_inference(bb, relax.op.reshape(x0, s0), relax.TensorStructInfo((3, 8, 5), "float32"))
    _check_inference(bb, relax.op.reshape(x1, s0), relax.TensorStructInfo((3, 8, 5), "float32"))
    _check_inference(bb, relax.op.reshape(x2, s0), relax.TensorStructInfo((3, 8, 5), "float32"))
    _check_inference(bb, relax.op.reshape(x3, s0), relax.TensorStructInfo((3, 8, 5), dtype=""))
    _check_inference(bb, relax.op.reshape(x4, s0), relax.TensorStructInfo((3, 8, 5), dtype=""))
    _check_inference(bb, relax.op.reshape(x5, s0), relax.TensorStructInfo((3, 8, 5), dtype=""))
    _check_inference(bb, relax.op.reshape(x0, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.reshape(x1, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.reshape(x2, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.reshape(x3, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.reshape(x4, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.reshape(x5, s1), relax.TensorStructInfo(s1, dtype=""))
    _check_inference(bb, relax.op.reshape(x0, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.reshape(x1, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.reshape(x2, s2), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.reshape(x3, s2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.reshape(x4, s2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.reshape(x5, s2), relax.TensorStructInfo(s2, dtype=""))
    _check_inference(bb, relax.op.reshape(x0, s3), relax.TensorStructInfo(s3, "float32"))
    _check_inference(bb, relax.op.reshape(x1, s3), relax.TensorStructInfo(s3, "float32"))
    _check_inference(bb, relax.op.reshape(x2, s3), relax.TensorStructInfo(s3, "float32"))
    _check_inference(bb, relax.op.reshape(x3, s3), relax.TensorStructInfo(s3, dtype=""))
    _check_inference(bb, relax.op.reshape(x4, s3), relax.TensorStructInfo(s3, dtype=""))
    _check_inference(bb, relax.op.reshape(x5, s3), relax.TensorStructInfo(s3, dtype=""))


def test_reshape_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))
    s0 = relax.Var("s", R.Shape((c, a, d, b)))
    s1 = relax.Var("s", R.Shape())
    s2 = relax.ShapeExpr((c, a, d, b))

    _check_inference(
        bb, relax.op.reshape(x, (c, a, d, b)), relax.TensorStructInfo((c, a, d, b), "float32")
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (d, c, b, -1)),
        relax.TensorStructInfo((d, c, b, a), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (1, -1, 1)),
        relax.TensorStructInfo((1, a * b * c * d, 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (2, -1, a)),
        relax.TensorStructInfo((2, tir.floordiv(b * c * d, 2), a), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (c, -1, d, b)),
        relax.TensorStructInfo((c, a, d, b), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (c, a * d, b)),
        relax.TensorStructInfo((c, a * d, b), "float32"),
    )
    _check_inference(
        bb,
        relax.op.reshape(x, (c, a * b * d, -1)),
        relax.TensorStructInfo((c, a * b * d, 1), "float32"),
    )
    # Remove Var from StructInfo when we can
    _check_inference(bb, relax.op.reshape(x, s0), relax.TensorStructInfo((c, a, d, b), "float32"))
    _check_inference(bb, relax.op.reshape(x, s1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.reshape(x, s2), relax.TensorStructInfo(s2, "float32"))


def test_reshape_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4, 5)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    ns0 = relax.Var("ns", relax.ShapeStructInfo((3, 8, 5)))
    ns1 = relax.Var("ns", relax.ShapeStructInfo())

    _check_inference(
        bb, relax.op.reshape(x0, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x0, (2, 3, 0, 5)), relax.TensorStructInfo((2, 3, 4, 5), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x0, (1, 3, 0, -1)), relax.TensorStructInfo((1, 3, 4, 10), "float32")
    )
    _check_inference(
        bb, relax.op.reshape(x0, (3, -1, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    # Remove Var from StructInfo when we can
    _check_inference(bb, relax.op.reshape(x0, ns0), relax.TensorStructInfo((3, 8, 5), "float32"))
    _check_inference(bb, relax.op.reshape(x0, ns1), relax.TensorStructInfo(ns1, "float32"))
    _check_inference(
        bb, relax.op.reshape(x1, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    # Remove Var from StructInfo when we can
    _check_inference(bb, relax.op.reshape(x1, ns0), relax.TensorStructInfo((3, 8, 5), "float32"))
    _check_inference(bb, relax.op.reshape(x1, ns1), relax.TensorStructInfo(ns1, "float32"))
    _check_inference(
        bb, relax.op.reshape(x2, (3, 8, 5)), relax.TensorStructInfo((3, 8, 5), "float32")
    )
    # Remove Var from StructInfo when we can
    _check_inference(bb, relax.op.reshape(x2, ns0), relax.TensorStructInfo((3, 8, 5), "float32"))
    _check_inference(bb, relax.op.reshape(x2, ns1), relax.TensorStructInfo(ns1, "float32"))


def test_reshape_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))

    _check_inference(bb, relax.op.reshape(x0, (120,)), relax.TensorStructInfo((120,), "float16"))
    _check_inference(bb, relax.op.reshape(x1, (120,)), relax.TensorStructInfo((120,), "int8"))


def test_reshape_infer_struct_info_unequal_shape_prod():
    bb = relax.BlockBuilder()
    s = relax.Var("s", relax.ShapeStructInfo((2, 3, 4, 5)))
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s, "float32"))
    ns = relax.Var("ns", relax.ShapeStructInfo((4, 4, 1, 5)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (4, 4, 1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (4, 4, 1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (4, 4, -1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (4, 4, -1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, ns))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, ns))


def test_reshape_infer_struct_info_inference_not_deducible():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor("float32", ndim=4))
    x1 = relax.Var("x", R.Tensor("float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (2, 3, -1)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (2, 3, -1)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x2, (2, 3, -1)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x3, (2, 3, -1)))


def test_reshape_new_shape_not_tuple():
    m = tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        relax.op.reshape(x, 120)
    with pytest.raises(TVMError):
        relax.op.reshape(x, m)


def test_reshape_infer_struct_info_new_shape_not_integer():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2.0, 3, 4, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2, 3, -1.0)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2, 3, 4.0, -1)))


def test_reshape_infer_struct_info_multiple_dim_inference():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (2, -1, -1, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (-1, -1, -1, -1)))


def test_reshape_infer_struct_info_non_positive_new_shape():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x, (-2, -3, -4, -5)))


def test_reshape_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    ns = relax.Var("ns", relax.TensorStructInfo((120,), "float32"))
    pv = relax.Var("pv", relax.PrimStructInfo("int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x0, (2, 3, 4, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x1, (2, 3, 4, 5)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x2, ns))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.reshape(x2, [pv]))


def test_permute_dims_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((1, 2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((1,), "float32"))
    x7 = relax.Var("x", R.Tensor((), "float32"))
    x8 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32", vdev0))

    _check_inference(
        bb, relax.op.permute_dims(x0, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "float32")
    )
    _check_inference(
        bb,
        relax.op.permute_dims(x8, [2, 3, 1, 0]),
        relax.TensorStructInfo((3, 4, 2, 1), "float32", vdev0),
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, axes=None), relax.TensorStructInfo((4, 3, 2, 1), "float32")
    )
    _check_inference(
        bb,
        relax.op.permute_dims(x0, [-2, -3, 3, -4]),
        relax.TensorStructInfo((3, 2, 4, 1), "float32"),
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [2, 3, 1, 0]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, axes=None), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x2, axes=None), relax.TensorStructInfo(dtype="float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x3, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), dtype="")
    )
    _check_inference(
        bb, relax.op.permute_dims(x3, axes=None), relax.TensorStructInfo((4, 3, 2, 1), dtype="")
    )
    _check_inference(
        bb,
        relax.op.permute_dims(x3, [-2, -3, 3, -4]),
        relax.TensorStructInfo((3, 2, 4, 1), dtype=""),
    )
    _check_inference(
        bb, relax.op.permute_dims(x4, [2, 3, 1, 0]), relax.TensorStructInfo(dtype="", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x4, axes=None), relax.TensorStructInfo(dtype="", ndim=4)
    )
    _check_inference(bb, relax.op.permute_dims(x5, axes=None), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.permute_dims(x6, axes=None), relax.TensorStructInfo((1,), "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x7, axes=None), relax.TensorStructInfo((), "float32")
    )


def test_permute_dims_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))

    _check_inference(
        bb, relax.op.permute_dims(x, [2, 3, 1, 0]), relax.TensorStructInfo((c, d, b, a), "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x, axes=None), relax.TensorStructInfo((d, c, b, a), "float32")
    )
    _check_inference(
        bb,
        relax.op.permute_dims(x, [-2, -3, 3, -4]),
        relax.TensorStructInfo((c, b, d, a), "float32"),
    )


def test_permute_dims_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((1, 2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.permute_dims(x0, [0, 1, 2, 3]), relax.TensorStructInfo(s0, "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, [-4, -3, -2, -1]), relax.TensorStructInfo(s0, "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, [2, 3, 0, 1]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x0, axes=None), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [0, 1, 2, 3]), relax.TensorStructInfo(s1, "float32")
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [2, 3, 0, 1]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, axes=None), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.permute_dims(x2, axes=None), relax.TensorStructInfo(dtype="float32")
    )


def test_permute_dims_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((1, 2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((1, 2, 3, 4), "int32"))

    _check_inference(
        bb, relax.op.permute_dims(x0, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "float16")
    )
    _check_inference(
        bb, relax.op.permute_dims(x1, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "int8")
    )
    _check_inference(
        bb, relax.op.permute_dims(x2, [2, 3, 1, 0]), relax.TensorStructInfo((3, 4, 2, 1), "int32")
    )


def test_permute_dims_infer_struct_info_unknown_ndim_with_axes():
    bb = relax.BlockBuilder()
    s = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor("float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [2, 3, 1, 0]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [2, 3, 1, 0]))


def test_permute_dims_infer_struct_info_wrong_number_axes():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((1, 2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [1, 2, 4, 0, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [1, 2, 4, 0, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x2, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x2, [1, 2, 4, 0, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x3, [0, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x3, [1, 2, 4, 0, 3]))


def test_permute_dims_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 3, 4, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, -5, 1, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 3, 4, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, -5, 1, 3]))


def test_permute_dims_infer_struct_info_repetitive_axes():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((1, 2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 2, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0, [0, 2, -2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 2, 2, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1, [0, 2, -2, 1]))


def test_permute_dims_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((1, 2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((1, 2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.permute_dims(x1))


def test_expand_dims_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))

    _check_inference(
        bb, relax.op.expand_dims(x0, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "float32")
    )
    _check_inference(
        bb,
        relax.op.expand_dims(x6, [1, 3]),
        relax.TensorStructInfo((2, 1, 3, 1, 4), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.expand_dims(x0, [-1, 1, -6, 3, 5]),
        relax.TensorStructInfo((2, 1, 1, 1, 3, 1, 4, 1), "float32"),
    )
    _check_inference(bb, relax.op.expand_dims(x0, []), relax.TensorStructInfo((2, 3, 4), "float32"))
    _check_inference(
        bb, relax.op.expand_dims(x1, [1, 3]), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.expand_dims(x1, []), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.expand_dims(x2, [1, 3]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.expand_dims(x2, []), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.expand_dims(x3, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), dtype="")
    )
    _check_inference(
        bb,
        relax.op.expand_dims(x3, [-1, 1, -6, 3, 5]),
        relax.TensorStructInfo((2, 1, 1, 1, 3, 1, 4, 1), dtype=""),
    )
    _check_inference(bb, relax.op.expand_dims(x3, []), relax.TensorStructInfo((2, 3, 4), dtype=""))
    _check_inference(bb, relax.op.expand_dims(x4, [1, 3]), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.expand_dims(x4, []), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.expand_dims(x5, [1, 3]), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.expand_dims(x5, []), relax.TensorStructInfo(dtype=""))


def test_expand_dims_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x = relax.Var("x", R.Tensor((a, 4, b), "float32"))

    _check_inference(
        bb, relax.op.expand_dims(x, [1, 3]), relax.TensorStructInfo((a, 1, 4, 1, b), "float32")
    )
    _check_inference(
        bb,
        relax.op.expand_dims(x, [-1, 1, -6, 3, 5]),
        relax.TensorStructInfo((a, 1, 1, 1, 4, 1, b, 1), "float32"),
    )
    _check_inference(bb, relax.op.expand_dims(x, []), relax.TensorStructInfo((a, 4, b), "float32"))


def test_expand_dims_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.expand_dims(x0, [1, 3]), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.expand_dims(x0, []), relax.TensorStructInfo(s0, "float32"))
    _check_inference(
        bb, relax.op.expand_dims(x1, [1, 3]), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.expand_dims(x1, []), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.expand_dims(x2, [1, 3]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.expand_dims(x2, []), relax.TensorStructInfo(s2, "float32"))


def test_expand_dims_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 4), "int32"))

    _check_inference(
        bb, relax.op.expand_dims(x0, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "float16")
    )
    _check_inference(
        bb, relax.op.expand_dims(x1, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "int8")
    )
    _check_inference(
        bb, relax.op.expand_dims(x2, [1, 3]), relax.TensorStructInfo((2, 1, 3, 1, 4), "int32")
    )


def test_expand_dims_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", relax.TensorStructInfo(s0))
    x3 = relax.Var("x", relax.TensorStructInfo(s1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [-6, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [-6, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [-6, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [1, 5]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [-6, 1]))


def test_expand_dims_infer_struct_info_repetitive_axes():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", relax.TensorStructInfo(s0))
    x3 = relax.Var("x", relax.TensorStructInfo(s1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, [1, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, [1, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x2, [1, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x3, [1, -4]))


def test_expand_dims_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x0, axis=[]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.expand_dims(x1, axis=[]))


def test_layout_transform_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x = relax.Var("x", R.Tensor((10, 20, 30), "float32"))
    x1 = relax.Var("x", R.Tensor((10, 20, 30), "float32", vdev0))

    transpose_transform = lambda a, b, c: (a, c, b)
    _check_inference(
        bb,
        relax.op.layout_transform(x, index_map=transpose_transform),
        relax.TensorStructInfo((10, 30, 20), "float32"),
    )
    _check_inference(
        bb,
        relax.op.layout_transform(x1, index_map=transpose_transform),
        relax.TensorStructInfo((10, 30, 20), "float32", vdev0),
    )

    tiling_transform = lambda a, b, c: (a, b // 2, c, b % 2)
    _check_inference(
        bb,
        relax.op.layout_transform(x, index_map=tiling_transform),
        relax.TensorStructInfo((10, 10, 30, 2), "float32"),
    )

    implicit_padding_transform = lambda a, b, c: (a, c, b // 3, b % 3)
    _check_inference(
        bb,
        relax.op.layout_transform(x, index_map=implicit_padding_transform, pad_value=2),
        relax.TensorStructInfo((10, 30, 7, 3), "float32"),
    )

    flatten_transform = lambda a, b, c: (a * 600 + b * 30 + c)
    _check_inference(
        bb,
        relax.op.layout_transform(x, index_map=flatten_transform),
        relax.TensorStructInfo((6000,), "float32"),
    )


def test_layout_transform_infer_struct_info_mismatch_dtype():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((10, 20, 30), "int32"))

    transpose_transform = lambda a, b, c: (a, c, b)
    with pytest.raises(TVMError):
        bb.normalize(relax.op.layout_transform(x, index_map=transpose_transform, pad_value=2.2))


def test_layout_transform_infer_struct_info_unknown_shape():
    bb = relax.BlockBuilder()
    tiling_transform = lambda a, b: (a, b // 2, b % 2)

    x_unknown_shape = relax.Var("x", R.Tensor("float32", ndim=2))
    _check_inference(
        bb,
        relax.op.layout_transform(x_unknown_shape, index_map=tiling_transform),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )

    x_unknown_rank_dtype = relax.Var("x", R.Tensor())
    _check_inference(
        bb,
        relax.op.layout_transform(x_unknown_rank_dtype, index_map=tiling_transform),
        relax.TensorStructInfo(dtype="", ndim=3),
    )


def test_layout_transform_infer_struct_info_symbolic_shape():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x0 = relax.Var("x", R.Tensor((a, b), "float32"))

    tiling_transform = lambda a, b: (a, b // 3, b % 3)
    _check_inference(
        bb,
        relax.op.layout_transform(x0, index_map=tiling_transform),
        relax.TensorStructInfo((a, (b - b % (-3)) // 3, 3), "float32"),
    )


def test_layout_transform_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()

    s = relax.Var("s", relax.ShapeStructInfo((30, 20)))
    x = relax.Var("x", relax.TensorStructInfo(s, "float32"))
    tiling_padding_transform = lambda a, b: (a, b // 3, b % 3)
    _check_inference(
        bb,
        relax.op.layout_transform(x, index_map=tiling_padding_transform),
        relax.TensorStructInfo((30, 7, 3), "float32"),
    )

    s_unknown_shape = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    x_unknown_shape = relax.Var("x", relax.TensorStructInfo(s_unknown_shape, "float32"))
    _check_inference(
        bb,
        relax.op.layout_transform(x_unknown_shape, index_map=tiling_padding_transform),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )

    s_unknown_rank = relax.Var("s", relax.ShapeStructInfo())
    x_unknown_rank = relax.Var("x", relax.TensorStructInfo(s_unknown_rank, "float32"))
    _check_inference(
        bb,
        relax.op.layout_transform(x_unknown_rank, index_map=tiling_padding_transform),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )

    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    s_symbolic_shape = relax.Var("s", relax.ShapeStructInfo((a, b)))
    x_symbolic_shape = relax.Var("x", relax.TensorStructInfo(s_symbolic_shape, "float32"))
    _check_inference(
        bb,
        relax.op.layout_transform(x_symbolic_shape, index_map=tiling_padding_transform),
        relax.TensorStructInfo((a, (b - b % (-3)) // 3, 3), "float32"),
    )


def test_layout_transform_infer_struct_info_invalid_index_map():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((10, 20, 30), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.layout_transform(x, index_map=lambda a, b: (b, a)))


def test_squeeze_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=6))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=6))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32", vdev0))

    _check_inference(
        bb, relax.op.squeeze(x0, [1, 4]), relax.TensorStructInfo((2, 3, 1, 4), "float32")
    )
    _check_inference(
        bb, relax.op.squeeze(x6, [1, 4]), relax.TensorStructInfo((2, 3, 1, 4), "float32", vdev0)
    )
    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo((2, 3, 4), "float32"))
    _check_inference(
        bb, relax.op.squeeze(x1, [1, 4]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x2, [1, 4]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.squeeze(x3, [1, 4]), relax.TensorStructInfo((2, 3, 1, 4), dtype="")
    )
    _check_inference(bb, relax.op.squeeze(x3), relax.TensorStructInfo((2, 3, 4), dtype=""))
    _check_inference(bb, relax.op.squeeze(x4, [1, 4]), relax.TensorStructInfo(dtype="", ndim=4))
    _check_inference(bb, relax.op.squeeze(x4), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.squeeze(x5, [1, 4]), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.squeeze(x5), relax.TensorStructInfo(dtype=""))


def test_squeeze_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x0 = relax.Var("x", R.Tensor((a, 1, b), "float32"))
    x1 = relax.Var("x", R.Tensor((a, 1, b)))

    _check_inference(bb, relax.op.squeeze(x0, [1]), relax.TensorStructInfo((a, b), "float32"))
    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x1, [1]), relax.TensorStructInfo((a, b), dtype=""))
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo(dtype=""))


def test_squeeze_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1, 3, 1, 1, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s2 = relax.Var("s", relax.ShapeStructInfo((a, 1, b)))
    s3 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    s4 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s4, "float32"))

    _check_inference(
        bb, relax.op.squeeze(x0, [1, 4]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.squeeze(x0, []), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x1, []), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo(s1, dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x2, [1]), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.squeeze(x2, []), relax.TensorStructInfo(s2, "float32"))
    _check_inference(bb, relax.op.squeeze(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.squeeze(x3, [1, 4]), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(bb, relax.op.squeeze(x3, []), relax.TensorStructInfo(s3, "float32"))
    _check_inference(bb, relax.op.squeeze(x3), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x4, [1, 4]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.squeeze(x4, []), relax.TensorStructInfo(s4, "float32"))
    _check_inference(bb, relax.op.squeeze(x4), relax.TensorStructInfo(dtype="float32"))


def test_squeeze_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "int32"))

    _check_inference(bb, relax.op.squeeze(x0), relax.TensorStructInfo((2, 3, 4), "float16"))
    _check_inference(bb, relax.op.squeeze(x1), relax.TensorStructInfo((2, 3, 4), "int8"))
    _check_inference(bb, relax.op.squeeze(x2), relax.TensorStructInfo((2, 3, 4), "int32"))


def test_squeeze_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1, 3, 1, 1, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=6))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [-7]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [-7]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [-7]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [6]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [-7]))


def test_squeeze_infer_struct_info_repetitive_axes():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1, 3, 1, 1, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=6))
    x0 = relax.Var("x", R.Tensor((2, 1, 3, 1, 1, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=6))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [1, 1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [3, -3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x3, [1, 1]))


def test_squeeze_infer_struct_info_axis_length_not_one():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo((a, 3, 4)))
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((a, 3, 4), "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0, [0]))
    _check_inference(bb, relax.op.squeeze(x1, [0]), relax.TensorStructInfo((3, 4), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x2, [0]))
    _check_inference(bb, relax.op.squeeze(x3, [0]), relax.TensorStructInfo(dtype="float32", ndim=2))


def test_squeeze_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.squeeze(x1))


def test_flatten_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor((3,), "float32"))
    x2 = relax.Var("x", R.Tensor((), "float32"))
    x3 = relax.Var("x", R.Tensor("float32", ndim=3))
    x4 = relax.Var("x", R.Tensor("float32", ndim=1))
    x5 = relax.Var("x", R.Tensor("float32", ndim=0))
    x6 = relax.Var("x", R.Tensor("float32"))
    x7 = relax.Var("x", R.Tensor((3, 4, 5)))
    x8 = relax.Var("x", R.Tensor((3,)))
    x9 = relax.Var("x", R.Tensor(()))
    x10 = relax.Var("x", R.Tensor(ndim=3))
    x11 = relax.Var("x", R.Tensor(ndim=1))
    x12 = relax.Var("x", R.Tensor(ndim=0))
    x13 = relax.Var("x", R.Tensor())
    x14 = relax.Var("x", R.Tensor((3, 4, 5), "float32", vdev0))

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo((60,), "float32"))
    _check_inference(bb, relax.op.flatten(x14), relax.TensorStructInfo((60,), "float32", vdev0))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo((3,), "float32"))
    _check_inference(bb, relax.op.flatten(x2), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x3), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x4), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x5), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x6), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x7), relax.TensorStructInfo((60,), dtype=""))
    _check_inference(bb, relax.op.flatten(x8), relax.TensorStructInfo((3,), dtype=""))
    _check_inference(bb, relax.op.flatten(x9), relax.TensorStructInfo((1,), dtype=""))
    _check_inference(bb, relax.op.flatten(x10), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.flatten(x11), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.flatten(x12), relax.TensorStructInfo((1,), dtype=""))
    _check_inference(bb, relax.op.flatten(x13), relax.TensorStructInfo(dtype="", ndim=1))


def test_flatten_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x0 = relax.Var("x", R.Tensor((a, b), "float32"))
    x1 = relax.Var("x", R.Tensor((a, b)))

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo((a * b,), "float32"))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo((a * b,), dtype=""))


def test_flatten_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((3, 4, 5)))
    s1 = relax.Var("s", relax.ShapeStructInfo((3,)))
    s2 = relax.Var("s", relax.ShapeStructInfo(()))
    s3 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s4 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s5 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    s6 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s4, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s5, "float32"))
    x6 = relax.Var("x", relax.TensorStructInfo(s6, "float32"))

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.flatten(x2), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x3), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.flatten(x4), relax.TensorStructInfo(s4, "float32"))
    _check_inference(bb, relax.op.flatten(x5), relax.TensorStructInfo((1,), "float32"))
    _check_inference(bb, relax.op.flatten(x6), relax.TensorStructInfo(dtype="float32", ndim=1))


def test_flatten_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((3, 4, 5), "int8"))
    x2 = relax.Var("x", R.Tensor((3, 4, 5), "int32"))

    _check_inference(bb, relax.op.flatten(x0), relax.TensorStructInfo((60,), "float16"))
    _check_inference(bb, relax.op.flatten(x1), relax.TensorStructInfo((60,), "int8"))
    _check_inference(bb, relax.op.flatten(x2), relax.TensorStructInfo((60,), "int32"))


def test_flatten_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((3, 4, 5), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.flatten(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.flatten(x1))


def test_flatten_wrong_input_number():
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))

    with pytest.raises(TypeError):
        relax.op.flatten(x, y)


def test_concat_infer_struct_info_with_axis():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))
    y0 = relax.Var("y", R.Tensor((2, 4, 4), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=3))
    y2 = relax.Var("y", R.Tensor("float32"))
    y3 = relax.Var("y", R.Tensor((2, 4, 4)))
    y4 = relax.Var("y", R.Tensor(ndim=3))
    y5 = relax.Var("y", R.Tensor())
    y6 = relax.Var("y", R.Tensor((2, 4, 4), "float32", vdev0))
    z0 = relax.Var("z", R.Tensor((2, 5, 4), "float32"))
    z1 = relax.Var("z", R.Tensor("float32", ndim=3))
    z2 = relax.Var("z", R.Tensor("float32"))
    z3 = relax.Var("z", R.Tensor((2, 5, 4)))
    z4 = relax.Var("z", R.Tensor(ndim=3))
    z5 = relax.Var("z", R.Tensor())
    z6 = relax.Var("z", R.Tensor((2, 5, 4), "float32", vdev0))

    _check_inference(
        bb, relax.op.concat([x0, y0, z0], axis=1), relax.TensorStructInfo((2, 12, 4), "float32")
    )
    _check_inference(
        bb,
        relax.op.concat([x6, y6, z6], axis=1),
        relax.TensorStructInfo((2, 12, 4), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.concat([x6, y0, z0], axis=1),
        relax.TensorStructInfo((2, 12, 4), "float32", vdev0),
    )
    _check_inference(
        bb, relax.op.concat([x0, y0, z0], axis=-2), relax.TensorStructInfo((2, 12, 4), "float32")
    )
    _check_inference(
        bb, relax.op.concat([x1, y0, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x2, y0, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x3, y0, z0], axis=1), relax.TensorStructInfo((2, 12, 4), dtype="")
    )
    _check_inference(
        bb, relax.op.concat([x3, y0, z0], axis=-2), relax.TensorStructInfo((2, 12, 4), dtype="")
    )
    _check_inference(
        bb, relax.op.concat([x4, y0, z0], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x5, y0, z0], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x1, y1, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x2, y1, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x3, y1, z0], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x5, y1, z0], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x2, y2, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x3, y2, z0], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x5, y5, z0], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x1, y1, z1], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x2, y2, z1], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x3, y1, z1], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x2, y2, z2], axis=1), relax.TensorStructInfo(dtype="float32", ndim=-1)
    )
    _check_inference(
        bb, relax.op.concat([x3, y2, z2], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x4, y4, z2], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x5, y5, z2], axis=1), relax.TensorStructInfo(dtype="", ndim=-1)
    )
    _check_inference(
        bb, relax.op.concat([x3, y3, z3], axis=1), relax.TensorStructInfo((2, 12, 4), dtype="")
    )
    _check_inference(
        bb, relax.op.concat([x3, y3, z3], axis=-2), relax.TensorStructInfo((2, 12, 4), dtype="")
    )
    _check_inference(
        bb, relax.op.concat([x4, y3, z3], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x5, y5, z3], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x4, y4, z4], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x5, y5, z4], axis=1), relax.TensorStructInfo(dtype="", ndim=3)
    )
    _check_inference(bb, relax.op.concat([x5, y5, z5], axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb,
        relax.op.concat(relax.Tuple([x0, y0, z0]), axis=1),
        relax.TensorStructInfo((2, 12, 4), "float32"),
    )


def test_concat_infer_struct_info_with_axis_shape_symbolic():
    bb = relax.BlockBuilder()
    a0 = tir.Var("a0", "int64")
    a1 = tir.Var("a1", "int64")
    b0 = tir.Var("b0", "int64")
    b1 = tir.Var("b1", "int64")
    b2 = tir.Var("b2", "int64")
    c = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((a0, b0, c), "float32"))
    x1 = relax.Var("x", R.Tensor((a1, b0, c), "float32"))
    x2 = relax.Var("x", R.Tensor((a0, b0, c), "float32"))
    y = relax.Var("y", R.Tensor((a0, b1, c), "float32"))
    z = relax.Var("z", R.Tensor((a0, b2, c), "float32"))

    _check_inference(
        bb,
        relax.op.concat([x0, y, z], axis=1),
        relax.TensorStructInfo((a0, b0 + b1 + b2, c), "float32"),
    )
    _check_inference(
        bb,
        relax.op.concat([x0, y, z], axis=-2),
        relax.TensorStructInfo((a0, b0 + b1 + b2, c), "float32"),
    )
    _check_inference(
        bb, relax.op.concat([x1, y, z], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb,
        relax.op.concat(relax.Tuple([x0, y, z]), axis=1),
        relax.TensorStructInfo((a0, b0 + b1 + b2, c), "float32"),
    )
    _check_inference(
        bb,
        relax.op.concat(relax.Tuple([x0, x2]), axis=1),
        relax.TensorStructInfo((a0, b0 * 2, c), "float32"),
    )


def test_concat_infer_struct_info_with_axis_shape_var():
    bb = relax.BlockBuilder()
    a0 = tir.Var("a0", "int64")
    a1 = tir.Var("a1", "int64")
    b0 = tir.Var("b0", "int64")
    b1 = tir.Var("b1", "int64")
    b2 = tir.Var("b2", "int64")
    c = tir.Var("c", "int64")
    sx0 = relax.Var("sx", relax.ShapeStructInfo((2, 3, 4)))
    sx1 = relax.Var("sx", relax.ShapeStructInfo((a0, b0, c)))
    sx2 = relax.Var("sx", relax.ShapeStructInfo((a1, b0, c)))
    sx3 = relax.Var("sx", relax.ShapeStructInfo(ndim=3))
    sx4 = relax.Var("sx", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(sx0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(sx1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(sx2, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(sx3, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(sx4, "float32"))
    y0 = relax.Var("y", R.Tensor((2, 4, 4), "float32"))
    y1 = relax.Var("y", R.Tensor((a0, b1, c), "float32"))
    z0 = relax.Var("z", R.Tensor((2, 5, 4), "float32"))
    z1 = relax.Var("z", R.Tensor((a0, b2, c), "float32"))

    _check_inference(
        bb, relax.op.concat([x0, y0, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x1, y1, z1], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x2, y1, z1], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x3, y0, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x4, y0, z0], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb,
        relax.op.concat(relax.Tuple([x0, y0, z0]), axis=1),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )


def test_concat_infer_struct_info_without_axis():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3,), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=1))
    x2 = relax.Var("x", R.Tensor((3,)))
    x3 = relax.Var("x", R.Tensor(ndim=1))
    y0 = relax.Var("y", R.Tensor((4,), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=1))
    z0 = relax.Var("z", R.Tensor((5,), "float32"))
    z1 = relax.Var("z", R.Tensor("float32", ndim=1))

    _check_inference(
        bb, relax.op.concat([x0, y0, z0], axis=None), relax.TensorStructInfo((12,), "float32")
    )
    _check_inference(
        bb,
        relax.op.concat([x1, y0, z0], axis=None),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb, relax.op.concat([x2, y0, z0], axis=None), relax.TensorStructInfo((12,), dtype="")
    )
    _check_inference(
        bb, relax.op.concat([x3, y0, z0], axis=None), relax.TensorStructInfo(dtype="", ndim=1)
    )
    _check_inference(
        bb,
        relax.op.concat([x1, y1, z0], axis=None),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb, relax.op.concat([x2, y1, z0], axis=None), relax.TensorStructInfo(dtype="", ndim=1)
    )
    _check_inference(
        bb,
        relax.op.concat([x1, y1, z1], axis=None),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.concat(relax.Tuple([x0, y0, z0]), axis=None),
        relax.TensorStructInfo((12,), "float32"),
    )


def test_concat_infer_struct_info_without_axis_shape_symbolic():
    bb = relax.BlockBuilder()
    a0 = tir.Var("a0", "int64")
    a1 = tir.Var("a1", "int64")
    x0 = relax.Var("x", R.Tensor((a0,), "float32"))
    x1 = relax.Var("x", R.Tensor((a0,), ""))
    y0 = relax.Var("y", R.Tensor((a1,), "float32"))
    y1 = relax.Var("y", R.Tensor((a1,), ""))

    _check_inference(
        bb, relax.op.concat([x0, y0], axis=None), relax.TensorStructInfo((a0 + a1,), "float32")
    )
    _check_inference(
        bb, relax.op.concat([x0, y1], axis=None), relax.TensorStructInfo((a0 + a1,), dtype="")
    )
    _check_inference(
        bb, relax.op.concat([x1, y0], axis=None), relax.TensorStructInfo((a0 + a1,), dtype="")
    )
    _check_inference(
        bb, relax.op.concat([x1, y1], axis=None), relax.TensorStructInfo((a0 + a1,), dtype="")
    )
    _check_inference(
        bb,
        relax.op.concat(relax.Tuple([x0, y0]), axis=None),
        relax.TensorStructInfo((a0 + a1,), "float32"),
    )


def test_concat_infer_struct_info_without_axis_shape_var():
    bb = relax.BlockBuilder()
    sx0 = relax.Var("sx", relax.ShapeStructInfo((3,)))
    sx1 = relax.Var("sx", relax.ShapeStructInfo(ndim=1))
    sy0 = relax.Var("sy", relax.ShapeStructInfo((4,)))
    x0 = relax.Var("x", relax.TensorStructInfo(sx0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(sx1, "float32"))
    y0 = relax.Var("y", relax.TensorStructInfo(sy0, "float32"))

    _check_inference(
        bb, relax.op.concat([x0, y0], axis=None), relax.TensorStructInfo(dtype="float32", ndim=1)
    )
    _check_inference(
        bb, relax.op.concat([x1, y0], axis=None), relax.TensorStructInfo(dtype="float32", ndim=1)
    )
    _check_inference(
        bb,
        relax.op.concat(relax.Tuple([x0, y0]), axis=None),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )


def test_concat_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3,), "float16"))
    y0 = relax.Var("y", R.Tensor((4,), "float16"))
    x1 = relax.Var("x", R.Tensor((3,), "int8"))
    y1 = relax.Var("y", R.Tensor((4,), "int8"))
    x2 = relax.Var("x", R.Tensor((3,), "int32"))
    y2 = relax.Var("y", R.Tensor((4,), "int32"))

    _check_inference(
        bb, relax.op.concat([x0, y0], axis=None), relax.TensorStructInfo((7,), "float16")
    )
    _check_inference(bb, relax.op.concat([x1, y1], axis=None), relax.TensorStructInfo((7,), "int8"))
    _check_inference(
        bb, relax.op.concat([x2, y2], axis=None), relax.TensorStructInfo((7,), "int32")
    )


def test_concat_infer_struct_info_tuple_var():
    bb = relax.BlockBuilder()
    a = tir.Var("a0", "int64")
    b0 = tir.Var("b0", "int64")
    b1 = tir.Var("b1", "int64")
    t0 = relax.Var(
        "t",
        relax.TupleStructInfo(
            [relax.TensorStructInfo((a, b0), "float32"), relax.TensorStructInfo((a, b1), "float32")]
        ),
    )
    t1 = relax.Var(
        "t",
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((a, b0), "float32"),
                relax.TensorStructInfo(dtype="float32", ndim=2),
            ]
        ),
    )
    t2 = relax.Var(
        "t",
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="float32", ndim=2),
            ]
        ),
    )
    t3 = relax.Var(
        "t",
        relax.TupleStructInfo(
            [relax.TensorStructInfo(dtype="float32"), relax.TensorStructInfo(dtype="float32")]
        ),
    )
    t4 = relax.Var(
        "t",
        relax.TupleStructInfo(
            [relax.TensorStructInfo((a, b0), "float32"), relax.TensorStructInfo((a, b1))]
        ),
    )
    t5 = relax.Var(
        "t",
        relax.TupleStructInfo(
            [relax.TensorStructInfo((a, b0), dtype=""), relax.TensorStructInfo((a, b1), dtype="")]
        ),
    )
    t6 = relax.Var(
        "t",
        relax.TupleStructInfo(
            [relax.TensorStructInfo(dtype="", ndim=2), relax.TensorStructInfo(dtype="")]
        ),
    )
    t7 = relax.Var(
        "t",
        relax.TupleStructInfo([relax.TensorStructInfo(dtype=""), relax.TensorStructInfo(dtype="")]),
    )

    _check_inference(
        bb, relax.op.concat(t0, axis=1), relax.TensorStructInfo((a, b0 + b1), "float32")
    )
    _check_inference(
        bb, relax.op.concat(t1, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.concat(t2, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.concat(t3, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.concat(t4, axis=1), relax.TensorStructInfo((a, b0 + b1), "float32")
    )
    _check_inference(
        bb, relax.op.concat(t5, axis=1), relax.TensorStructInfo((a, b0 + b1), dtype="")
    )
    _check_inference(bb, relax.op.concat(t6, axis=1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.concat(t7, axis=1), relax.TensorStructInfo(dtype=""))


def test_concat_infer_struct_info_single_input_tensor():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((3, a)))
    s1 = relax.Var("s", relax.ShapeStructInfo((a,)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s3 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s4 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor((3, a), "float32"))
    x1 = relax.Var("x", R.Tensor((a,), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    x3 = relax.Var("x", R.Tensor("float32", ndim=1))
    x4 = relax.Var("x", R.Tensor("float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x6 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x7 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x8 = relax.Var("x", relax.TensorStructInfo(s3, "float32"))
    x9 = relax.Var("x", relax.TensorStructInfo(s4, "float32"))

    _check_inference(bb, relax.op.concat([x0], axis=1), relax.TensorStructInfo((3, a), "float32"))
    _check_inference(bb, relax.op.concat([x1], axis=0), relax.TensorStructInfo((a,), "float32"))
    _check_inference(bb, relax.op.concat([x1], axis=None), relax.TensorStructInfo((a,), "float32"))
    _check_inference(
        bb, relax.op.concat([x2], axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.concat([x3], axis=0), relax.TensorStructInfo(dtype="float32", ndim=1)
    )
    _check_inference(
        bb, relax.op.concat([x3], axis=None), relax.TensorStructInfo(dtype="float32", ndim=1)
    )
    _check_inference(bb, relax.op.concat([x4], axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.concat([x5], axis=1), relax.TensorStructInfo(s0, dtype="float32"))
    _check_inference(bb, relax.op.concat([x6], axis=0), relax.TensorStructInfo(s1, dtype="float32"))
    _check_inference(
        bb, relax.op.concat([x6], axis=None), relax.TensorStructInfo(s1, dtype="float32")
    )
    _check_inference(bb, relax.op.concat([x7], axis=1), relax.TensorStructInfo(s2, dtype="float32"))
    _check_inference(bb, relax.op.concat([x8], axis=0), relax.TensorStructInfo(s3, dtype="float32"))
    _check_inference(
        bb, relax.op.concat([x8], axis=None), relax.TensorStructInfo(s3, dtype="float32")
    )
    _check_inference(bb, relax.op.concat([x9], axis=1), relax.TensorStructInfo(s4, dtype="float32"))


def test_concat_infer_struct_info_zero_rank_input_tensor():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(()))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    x0 = relax.Var("x", R.Tensor((), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=0))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x0], axis=0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x1], axis=0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x2], axis=None))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x3], axis=None))


def test_concat_infer_struct_info_no_input_tensor():
    bb = relax.BlockBuilder()
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([], axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([], axis=None))


def test_concat_infer_struct_info_without_axis_but_tensor_not_one_dimensional():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x0], axis=None))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x1], axis=None))
    _check_inference(bb, relax.op.concat([x2], axis=None), relax.TensorStructInfo(dtype="float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x3], axis=None))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x4], axis=None))
    _check_inference(bb, relax.op.concat([x5], axis=None), relax.TensorStructInfo(s2, "float32"))


def test_concat_infer_struct_info_inconsistent_dtype():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((3,)))
    y = relax.Var("y", R.Tensor((4,), "float32"))
    z = relax.Var("z", R.Tensor((5,), "int8"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x, y, z], axis=0))


def test_concat_infer_struct_info_inconsistent_ndim():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((4, 5)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    x = relax.Var("x", R.Tensor((3,), "float32"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=2))
    y2 = relax.Var("y", relax.TensorStructInfo(s0, "float32"))
    y3 = relax.Var("y", relax.TensorStructInfo(s1, "float32"))
    z = relax.Var("z", R.Tensor((5,), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x, y0, z], axis=0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x, y1, z], axis=0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x, y2, z], axis=0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x, y3, z], axis=0))


def test_concat_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((3,)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    x0 = relax.Var("x", R.Tensor((3,), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=1))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x0], axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x1], axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x2], axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x3], axis=1))


def test_concat_infer_struct_info_unequal_shape():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo((3, a + 2)))
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((3, a + 2), "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    y0 = relax.Var("y", R.Tensor((3, 3), "float32"))
    y1 = relax.Var("y", R.Tensor((3, a), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x0, y0]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x2, y0]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x1, y1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([x3, y1]))


def test_concat_infer_struct_info_input_not_tuple():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((3,), "float32"))
    s = relax.Var("s", relax.ShapeStructInfo((3,)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat(x))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat(s))


def test_concat_infer_struct_info_input_tuple_field_not_tensor():
    bb = relax.BlockBuilder()
    s = relax.Var("s", relax.ShapeStructInfo((3,)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.concat([s]))


def test_split_infer_struct_info_by_indices():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.split(x0, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 4), "float32"),
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 3, 4), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x6, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 4), "float32", vdev0),
                relax.TensorStructInfo((2, 4, 4), "float32", vdev0),
                relax.TensorStructInfo((2, 3, 4), "float32", vdev0),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x0, [3, 7], axis=-2),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 4), "float32"),
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 3, 4), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x1, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x2, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x3, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 4), dtype=""),
                relax.TensorStructInfo((2, 4, 4), dtype=""),
                relax.TensorStructInfo((2, 3, 4), dtype=""),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x4, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=3),
                relax.TensorStructInfo(dtype="", ndim=3),
                relax.TensorStructInfo(dtype="", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x5, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype=""),
                relax.TensorStructInfo(dtype=""),
                relax.TensorStructInfo(dtype=""),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x0, [-2, 2, 6, 4, 8, 12, 9], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 0, 4), "float32"),
                relax.TensorStructInfo((2, 2, 4), "float32"),
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 0, 4), "float32"),
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 2, 4), "float32"),
                relax.TensorStructInfo((2, 0, 4), "float32"),
                relax.TensorStructInfo((2, 1, 4), "float32"),
            ]
        ),
    )


def test_split_infer_struct_info_by_indices_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x = relax.Var("x", R.Tensor((a, b), "float32"))

    _check_inference(
        bb,
        relax.op.split(x, [10, 20], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=2),
                relax.TensorStructInfo(dtype="float32", ndim=2),
                relax.TensorStructInfo(dtype="float32", ndim=2),
            ]
        ),
    )


def test_split_infer_struct_info_by_indices_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 10, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb,
        relax.op.split(x0, [3], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x1, [3], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x2, [3], axis=1),
        relax.TupleStructInfo(
            [relax.TensorStructInfo(dtype="float32"), relax.TensorStructInfo(dtype="float32")]
        ),
    )


def test_split_infer_struct_info_by_n_section():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(
        bb,
        relax.op.split(x0, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 2, 4), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x0, 2, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 5, 4), "float32"),
                relax.TensorStructInfo((2, 5, 4), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x0, 3, axis=-2),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 4, 4), "float32"),
                relax.TensorStructInfo((2, 2, 4), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x1, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x2, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x3, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 4, 4), dtype=""),
                relax.TensorStructInfo((2, 4, 4), dtype=""),
                relax.TensorStructInfo((2, 2, 4), dtype=""),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x4, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=3),
                relax.TensorStructInfo(dtype="", ndim=3),
                relax.TensorStructInfo(dtype="", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x5, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype=""),
                relax.TensorStructInfo(dtype=""),
                relax.TensorStructInfo(dtype=""),
            ]
        ),
    )


def test_split_infer_struct_info_by_n_section_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x = relax.Var("x", R.Tensor((a, b), "float32"))

    _check_inference(
        bb,
        relax.op.split(x, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((a, tir.ceildiv(b, 3)), "float32"),
                relax.TensorStructInfo((a, tir.ceildiv(b, 3)), "float32"),
                relax.TensorStructInfo((a, b - tir.ceildiv(b, 3) * 2), "float32"),
            ]
        ),
    )


def test_split_infer_struct_info_by_n_section_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 10, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb,
        relax.op.split(x0, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x1, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x2, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="float32"),
            ]
        ),
    )


def test_split_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 10, 4), "int8"))

    _check_inference(
        bb,
        relax.op.split(x0, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 4), "float16"),
                relax.TensorStructInfo((2, 4, 4), "float16"),
                relax.TensorStructInfo((2, 3, 4), "float16"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x1, [3, 7], axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 4), "int8"),
                relax.TensorStructInfo((2, 4, 4), "int8"),
                relax.TensorStructInfo((2, 3, 4), "int8"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x0, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 4, 4), "float16"),
                relax.TensorStructInfo((2, 4, 4), "float16"),
                relax.TensorStructInfo((2, 2, 4), "float16"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.split(x1, 3, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 4, 4), "int8"),
                relax.TensorStructInfo((2, 4, 4), "int8"),
                relax.TensorStructInfo((2, 2, 4), "int8"),
            ]
        ),
    )


def test_split_infer_struct_info_single_output():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((a, b)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor((a, b), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb,
        relax.op.split(x0, [], axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo((a, b), "float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x1, [], axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(dtype="float32", ndim=2)]),
    )
    _check_inference(
        bb,
        relax.op.split(x2, [], axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(dtype="float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x3, [], axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(s0, "float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x4, [], axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(s1, "float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x5, [], axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(s2, "float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x0, 1, axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo((a, b), "float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x1, 1, axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(dtype="float32", ndim=2)]),
    )
    _check_inference(
        bb,
        relax.op.split(x2, 1, axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(dtype="float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x3, 1, axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(s0, "float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x4, 1, axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(s1, "float32")]),
    )
    _check_inference(
        bb,
        relax.op.split(x5, 1, axis=1),
        relax.TupleStructInfo([relax.TensorStructInfo(s2, "float32")]),
    )


def test_split_indices_or_sections_int64():
    x = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    split0 = relax.op.split(x, [3, 6], axis=1)
    split1 = relax.op.split(x, 4, axis=1)

    assert split0.attrs.indices_or_sections[0].dtype == "int64"
    assert split0.attrs.indices_or_sections[1].dtype == "int64"
    assert split1.attrs.indices_or_sections.dtype == "int64"


def test_split_infer_struct_info_non_integer_indices():
    bb = relax.BlockBuilder()
    a = tir.Var("c", "int64")
    b = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((3, 4), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x, [a, b], axis=1))


def test_split_invalid_n_section():
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor((3, 4), "float32"))

    with pytest.raises(TVMError):
        relax.op.split(x, 0, axis=1)
    with pytest.raises(TVMError):
        relax.op.split(x, -1, axis=1)
    with pytest.raises(TVMError):
        relax.op.split(x, n, axis=1)


def test_split_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x0, [], axis=2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x0, [], axis=-3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x1, 1, axis=2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x1, 1, axis=-3))


def test_split_infer_invalid_struct_info_indices():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    v = relax.Var("v", relax.PrimStructInfo("int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x0, [v], axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x0, v, axis=1))


def test_split_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x0, 1, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.split(x1, 1, axis=1))


def test_broadcast_to_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 1, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 1, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 1, 3), "float32", vdev0))

    _check_inference(
        bb, relax.op.broadcast_to(x0, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "float32")
    )
    _check_inference(
        bb,
        relax.op.broadcast_to(x6, (4, 2, 5, 3)),
        relax.TensorStructInfo((4, 2, 5, 3), "float32", vdev0),
    )
    _check_inference(
        bb, relax.op.broadcast_to(x1, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x2, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x3, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), dtype="")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x4, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), dtype="")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x5, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), dtype="")
    )


def test_broadcast_to_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x0 = relax.Var("x", R.Tensor((b, 1, 1, d), "float32"))
    x1 = relax.Var("x", R.Tensor((b, 1, 1, d)))

    _check_inference(
        bb,
        relax.op.broadcast_to(x0, (a, b, 1, c, d)),
        relax.TensorStructInfo((a, b, 1, c, d), "float32"),
    )
    _check_inference(
        bb,
        relax.op.broadcast_to(x1, (a, b, 1, c, d)),
        relax.TensorStructInfo((a, b, 1, c, d), dtype=""),
    )


def test_broadcast_to_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.broadcast_to(x0, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x1, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x2, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "float32")
    )


def test_broadcast_to_infer_struct_info_tgt_shape_var():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((b, 1, 1, d)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor((b, 1, 1, d), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    stgt0 = relax.Var("stgt", relax.ShapeStructInfo((a, b, 1, c, d)))
    stgt1 = relax.Var("stgt", relax.ShapeStructInfo(ndim=5))
    stgt2 = relax.Var("stgt", relax.ShapeStructInfo())

    _check_inference(bb, relax.op.broadcast_to(x0, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x1, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x2, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x3, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x4, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x5, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x0, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x1, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x2, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x3, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x4, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x5, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x0, stgt2), relax.TensorStructInfo(stgt2, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x1, stgt2), relax.TensorStructInfo(stgt2, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x2, stgt2), relax.TensorStructInfo(stgt2, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x3, stgt2), relax.TensorStructInfo(stgt2, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x4, stgt2), relax.TensorStructInfo(stgt2, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x5, stgt2), relax.TensorStructInfo(stgt2, "float32"))


def test_broadcast_to_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 1, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 1, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 1, 3), "int32"))

    _check_inference(
        bb, relax.op.broadcast_to(x0, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "float16")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x1, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "int8")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x2, (4, 2, 5, 3)), relax.TensorStructInfo((4, 2, 5, 3), "int32")
    )


def test_broadcast_to_infer_struct_info_tgt_ndim_less_than_old_ndim():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 1)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    x0 = relax.Var("x", R.Tensor((2, 1), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    stgt0 = relax.Var("stgt", relax.ShapeStructInfo((2,)))
    stgt1 = relax.Var("stgt", relax.ShapeStructInfo(ndim=1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x0, (2,)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x0, stgt0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x0, stgt1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x1, (2,)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x1, stgt0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x1, stgt1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x2, (2,)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x2, stgt0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x2, stgt1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x3, (2,)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x3, stgt0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x3, stgt1))


def test_broadcast_to_infer_struct_info_not_broadcastable_static():
    bb = relax.BlockBuilder()
    s = relax.Var("s", relax.ShapeStructInfo((2, 1, 3)))
    x0 = relax.Var("x", R.Tensor((2, 1, 3), "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s, "float32"))
    stgt = relax.Var("stgt", relax.ShapeStructInfo((2, 1, 6)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x0, (2, 1, 6)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x0, stgt))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x1, (2, 1, 6)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x1, stgt))


def test_broadcast_to_infer_struct_info_not_broadcastable_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    s = relax.Var("s", relax.ShapeStructInfo((2, a)))
    x0 = relax.Var("x", R.Tensor((2, a), "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s, "float32"))
    stgt0 = relax.Var("stgt", relax.ShapeStructInfo((2, b)))
    stgt1 = relax.Var("stgt", relax.ShapeStructInfo((2, 1)))
    stgt2 = relax.Var("stgt", relax.ShapeStructInfo((b, a)))

    _check_inference(
        bb, relax.op.broadcast_to(x0, (2, b)), relax.TensorStructInfo((2, b), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x0, (2, 1)), relax.TensorStructInfo((2, 1), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x0, (b, a)), relax.TensorStructInfo((b, a), "float32")
    )
    _check_inference(bb, relax.op.broadcast_to(x0, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x0, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x0, stgt2), relax.TensorStructInfo(stgt2, "float32"))
    _check_inference(
        bb, relax.op.broadcast_to(x1, (2, b)), relax.TensorStructInfo((2, b), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x1, (2, 1)), relax.TensorStructInfo((2, 1), "float32")
    )
    _check_inference(
        bb, relax.op.broadcast_to(x1, (b, a)), relax.TensorStructInfo((b, a), "float32")
    )
    _check_inference(bb, relax.op.broadcast_to(x1, stgt0), relax.TensorStructInfo(stgt0, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x1, stgt1), relax.TensorStructInfo(stgt1, "float32"))
    _check_inference(bb, relax.op.broadcast_to(x1, stgt2), relax.TensorStructInfo(stgt2, "float32"))


def test_broadcast_to_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 1, 3)))
    x1 = relax.Var("x", R.Tensor((2, 1, 3), "float32"))
    stgt = relax.Var("stgt", relax.TensorStructInfo((4, 2, 5, 3), dtype=""))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x0, (4, 2, 5, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.broadcast_to(x1, stgt))


def test_collapse_sum_like_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    y0 = relax.Var("y", R.Tensor((3, 4), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=2))
    y2 = relax.Var("y", R.Tensor("float32"))
    y3 = relax.Var("y", R.Tensor((3, 4)))
    y4 = relax.Var("y", R.Tensor(ndim=2))
    y5 = relax.Var("y", R.Tensor((1, 4)))
    y6 = relax.Var("y", R.Tensor((3, 4), "float32", vdev0))

    _check_inference(
        bb, relax.op.collapse_sum_like(x0, y0), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x3, y6), relax.TensorStructInfo((3, 4), "float32", vdev0)
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x1, y1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x0, y1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x0, y2), relax.TensorStructInfo(dtype="float32", ndim=-1)
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x0, y3), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x2, y0), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x2, y4), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x4, y1), relax.TensorStructInfo(dtype="", ndim=2)
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x5, y3), relax.TensorStructInfo((3, 4), dtype="")
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x0, y5), relax.TensorStructInfo((1, 4), "float32")
    )


def test_collapse_sum_like_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x0 = relax.Var("x", R.Tensor((3, 4, a), "float32"))
    y0 = relax.Var("y", R.Tensor((4, a), "float32"))
    x1 = relax.Var("x", R.Tensor((3, 4, b + a), "float32"))
    y1 = relax.Var("x", R.Tensor((1, a + b), "float32"))

    _check_inference(
        bb, relax.op.collapse_sum_like(x0, y0), relax.TensorStructInfo((4, a), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_like(x1, y1), relax.TensorStructInfo((1, a + b), "float32")
    )


def test_collapse_sum_like_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s1", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s2", relax.ShapeStructInfo())
    s3 = relax.Var("s3", relax.ShapeStructInfo((3, 4)))
    s4 = relax.Var("s4", relax.ShapeStructInfo(ndim=2))
    s5 = relax.Var("s5", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    y0 = relax.Var("y", relax.TensorStructInfo(s3, "float32"))
    y1 = relax.Var("y", relax.TensorStructInfo(s4, "float32"))
    y2 = relax.Var("y", relax.TensorStructInfo(s5, "float32"))

    _check_inference(bb, relax.op.collapse_sum_like(x0, y0), relax.TensorStructInfo(s3, "float32"))
    _check_inference(bb, relax.op.collapse_sum_like(x1, y1), relax.TensorStructInfo(s4, "float32"))
    _check_inference(bb, relax.op.collapse_sum_like(x2, y2), relax.TensorStructInfo(s5, "float32"))


def test_collapse_sum_like_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    y0 = relax.Var("y", R.Tensor((3, 4), "float16"))
    y1 = relax.Var("y", R.Tensor((3, 4), "int8"))

    _check_inference(
        bb, relax.op.collapse_sum_like(x0, y0), relax.TensorStructInfo((3, 4), "float16")
    )
    _check_inference(bb, relax.op.collapse_sum_like(x1, y1), relax.TensorStructInfo((3, 4), "int8"))


def test_collapse_sum_like_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((4, 5)))
    x2 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_like(x0, x1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_like(x2, x0))


def test_collapse_sum_like_infer_struct_info_shape_mismatch():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    y0 = relax.Var("y", R.Tensor((3, 6, 5), "float32"))
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x1 = relax.Var("z", R.Tensor((3, a, 5), "float32"))
    y1 = relax.Var("w", R.Tensor((3, b, 5), "float32"))

    s0 = relax.Var("s0", relax.ShapeStructInfo((3, 4, 5)))
    s1 = relax.Var("s1", relax.ShapeStructInfo((3, 6, 5)))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    y2 = relax.Var("y", relax.TensorStructInfo(s1, "float32"))

    s2 = relax.Var("s2", relax.ShapeStructInfo((3, a, 5)))
    s3 = relax.Var("s3", relax.ShapeStructInfo((3, b, 5)))
    x3 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    y3 = relax.Var("y", relax.TensorStructInfo(s3, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_like(x0, y0))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_like(x1, y1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_like(x2, y2))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_like(x3, y3))


def test_collapse_sum_to_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(
        bb, relax.op.collapse_sum_to(x0, (3, 4)), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, (3, 4)), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x2, (3, 4)), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(bb, relax.op.collapse_sum_to(x3, (3, 4)), relax.TensorStructInfo((3, 4), ""))
    _check_inference(bb, relax.op.collapse_sum_to(x4, (3, 4)), relax.TensorStructInfo((3, 4), ""))
    _check_inference(bb, relax.op.collapse_sum_to(x5, (3, 4)), relax.TensorStructInfo((3, 4), ""))


def test_collapse_sum_to_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x0 = relax.Var("x", R.Tensor((3, 4, a), "float32"))
    x1 = relax.Var("x", R.Tensor((3, 4, b + a), "float32"))

    _check_inference(
        bb, relax.op.collapse_sum_to(x0, (4, a)), relax.TensorStructInfo((4, a), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, (1, a + b)), relax.TensorStructInfo((1, a + b), "float32")
    )


def test_collapse_sum_to_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s1", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s2", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    _check_inference(
        bb, relax.op.collapse_sum_to(x0, (3, 4)), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, (3, 4)), relax.TensorStructInfo((3, 4), "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, (3, 4)), relax.TensorStructInfo((3, 4), "float32")
    )


def test_collapse_sum_to_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))

    _check_inference(
        bb, relax.op.collapse_sum_to(x0, (3, 4)), relax.TensorStructInfo((3, 4), "float16")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, (3, 4)), relax.TensorStructInfo((3, 4), "int8")
    )


def test_collapse_sum_to_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((4, 5)))
    x2 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_to(x0, x0))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_to(x0, x2))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_to(x1, x1))


def test_collapse_sum_to_infer_struct_info_shape_mismatch():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x1 = relax.Var("x", R.Tensor((3, a, 5), "float32"))

    s0 = relax.Var("s0", relax.ShapeStructInfo((3, 4, 5)))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))

    s1 = relax.Var("s1", relax.ShapeStructInfo((3, a, 5)))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_to(x0, (4, 4, 5)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_to(x1, (3, b, 5)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_to(x2, (4, 4, 5)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.collapse_sum_to(x3, (3, b, 5)))


def test_collapse_sum_to_infer_struct_info_struct_info_tgt_shape_var():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    s0 = relax.Var("s0", relax.ShapeStructInfo((3, a, b)))
    s1 = relax.Var("s1", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s2", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor((3, a, b), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor(""))
    x3 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    stgt0 = relax.Var("stgt0", relax.ShapeStructInfo((a, b)))
    stgt1 = relax.Var("stgt1", relax.ShapeStructInfo(ndim=2))
    stgt2 = relax.Var("stgt2", relax.ShapeStructInfo())

    _check_inference(
        bb, relax.op.collapse_sum_to(x0, stgt0), relax.TensorStructInfo(stgt0, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, stgt0), relax.TensorStructInfo(stgt0, "float32")
    )
    _check_inference(bb, relax.op.collapse_sum_to(x2, stgt0), relax.TensorStructInfo(stgt0, ""))
    _check_inference(
        bb, relax.op.collapse_sum_to(x3, stgt0), relax.TensorStructInfo(stgt0, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x4, stgt0), relax.TensorStructInfo(stgt0, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x5, stgt0), relax.TensorStructInfo(stgt0, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x0, stgt1), relax.TensorStructInfo(stgt1, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, stgt1), relax.TensorStructInfo(stgt1, "float32")
    )
    _check_inference(bb, relax.op.collapse_sum_to(x2, stgt1), relax.TensorStructInfo(stgt1, ""))
    _check_inference(
        bb, relax.op.collapse_sum_to(x3, stgt1), relax.TensorStructInfo(stgt1, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x4, stgt1), relax.TensorStructInfo(stgt1, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x5, stgt1), relax.TensorStructInfo(stgt1, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x0, stgt2), relax.TensorStructInfo(stgt2, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x1, stgt2), relax.TensorStructInfo(stgt2, "float32")
    )
    _check_inference(bb, relax.op.collapse_sum_to(x2, stgt2), relax.TensorStructInfo(stgt2, ""))
    _check_inference(
        bb, relax.op.collapse_sum_to(x3, stgt2), relax.TensorStructInfo(stgt2, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x4, stgt2), relax.TensorStructInfo(stgt2, "float32")
    )
    _check_inference(
        bb, relax.op.collapse_sum_to(x5, stgt2), relax.TensorStructInfo(stgt2, "float32")
    )


def test_repeat_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.repeat(x0, 2, axis=0),
        relax.TensorStructInfo((4, 10, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.repeat(x6, 2, axis=0),
        relax.TensorStructInfo((4, 10, 4), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.repeat(x0, 2, axis=-2),
        relax.TensorStructInfo((2, 20, 4), "float32"),
    )
    _check_inference(
        bb,
        relax.op.repeat(x0, 2),
        relax.TensorStructInfo((160,), "float32"),
    )
    _check_inference(
        bb,
        relax.op.repeat(x1, 2, axis=0),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.repeat(x1, 2),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(bb, relax.op.repeat(x2, 2, axis=0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.repeat(x2, 2), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(
        bb,
        relax.op.repeat(x3, 2, axis=0),
        relax.TensorStructInfo((4, 10, 4), dtype=""),
    )
    _check_inference(bb, relax.op.repeat(x4, 2, axis=0), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.repeat(x5, 2, axis=0), relax.TensorStructInfo(dtype=""))


def test_repeat_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(bb, relax.op.repeat(x, 2, 0), relax.TensorStructInfo((a * 2, b, c), "float32"))
    _check_inference(
        bb,
        relax.op.repeat(x, 2, -1),
        relax.TensorStructInfo((a, b, c * 2), "float32"),
    )
    _check_inference(
        bb,
        relax.op.repeat(x, 2),
        relax.TensorStructInfo((a * b * c * 2,), "float32"),
    )


def test_repeat_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))

    _check_inference(bb, relax.op.repeat(x0, 2, 0), relax.TensorStructInfo((4, 3, 4), "float16"))
    _check_inference(bb, relax.op.repeat(x1, 2, 0), relax.TensorStructInfo((4, 3, 4), "int8"))


def test_repeat_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x0, 2, 3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x0, 2, -4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x1, 2, 3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x1, 2, -4))
    # okay
    bb.normalize(relax.op.repeat(x2, 2, 3))
    bb.normalize(relax.op.repeat(x2, 2, -4))


def test_repeat_return_data_sinfo():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))

    _check_inference(bb, relax.op.repeat(x0, 1, 0), x0.struct_info)
    _check_inference(bb, relax.op.repeat(x0, 1, -1), x0.struct_info)
    _check_inference(bb, relax.op.repeat(x1, 1, 0), x1.struct_info)
    _check_inference(bb, relax.op.repeat(x2, 1, 0), x2.struct_info)


def test_repeat_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    r1 = tir.Var("r", "float32")
    r2 = tir.StringImm("abc")

    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x0, 2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x1, 2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x2, 1.5))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x2, r1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.repeat(x2, r2))


def test_tile_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.tile(x0, 2),
        relax.TensorStructInfo((2, 10, 8), "float32"),
    )
    _check_inference(
        bb,
        relax.op.tile(x6, 2),
        relax.TensorStructInfo((2, 10, 8), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.tile(x0, (3, 2)),
        relax.TensorStructInfo((2, 30, 8), "float32"),
    )
    _check_inference(
        bb,
        relax.op.tile(x0, (4, 3, 2)),
        relax.TensorStructInfo((8, 30, 8), "float32"),
    )
    _check_inference(
        bb,
        relax.op.tile(x0, (5, 4, 3, 2)),
        relax.TensorStructInfo((5, 8, 30, 8), "float32"),
    )
    _check_inference(
        bb,
        relax.op.tile(x1, 2),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.tile(x1, (5, 4, 3, 2)),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.tile(x2, (5, 4, 3, 2)), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb,
        relax.op.tile(x3, 2),
        relax.TensorStructInfo((2, 10, 8), dtype=""),
    )
    _check_inference(
        bb,
        relax.op.tile(x3, (5, 4, 3, 2)),
        relax.TensorStructInfo((5, 8, 30, 8), dtype=""),
    )
    _check_inference(bb, relax.op.tile(x4, 2), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.tile(x4, (5, 4, 3, 2)), relax.TensorStructInfo(dtype="", ndim=4))
    _check_inference(bb, relax.op.tile(x5, (5, 4, 3, 2)), relax.TensorStructInfo(dtype=""))


def test_tile_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(bb, relax.op.tile(x, 2), relax.TensorStructInfo((a, b, c * 2), "float32"))
    _check_inference(
        bb, relax.op.tile(x, (3, 2)), relax.TensorStructInfo((a, b * 3, c * 2), "float32")
    )
    _check_inference(
        bb, relax.op.tile(x, (4, 3, 2)), relax.TensorStructInfo((a * 4, b * 3, c * 2), "float32")
    )
    _check_inference(
        bb,
        relax.op.tile(x, (5, 4, 3, 2)),
        relax.TensorStructInfo((5, a * 4, b * 3, c * 2), "float32"),
    )


def test_tile_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))

    _check_inference(bb, relax.op.tile(x0, (3, 2)), relax.TensorStructInfo((2, 9, 8), "float16"))
    _check_inference(bb, relax.op.tile(x1, (3, 2)), relax.TensorStructInfo((2, 9, 8), "int8"))


def test_tile_return_data_sinfo():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))

    _check_inference(bb, relax.op.tile(x0, 1), x0.struct_info)
    _check_inference(bb, relax.op.tile(x0, (1, 1)), x0.struct_info)
    _check_inference(bb, relax.op.tile(x0, (1, 1, 1)), x0.struct_info)
    _check_inference(bb, relax.op.tile(x1, 1), x1.struct_info)
    _check_inference(bb, relax.op.tile(x2, 1), x2.struct_info)


def test_tile_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    r1 = tir.Var("a", "float32")
    r2 = tir.StringImm("abc")

    with pytest.raises(TVMError):
        bb.normalize(relax.op.tile(x0, 2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tile(x1, 2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tile(x2, (2, 1.5, 2)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tile(x2, (2, r1)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.tile(x2, r2))


def test_flip_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float16", ndim=3))
    x2 = relax.Var("x", R.Tensor("int32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(bb, relax.op.flip(x0, axis=1), relax.TensorStructInfo((2, 10, 4), "float32"))
    _check_inference(
        bb, relax.op.flip(x5, axis=1), relax.TensorStructInfo((2, 10, 4), "float32", vdev0)
    )
    _check_inference(bb, relax.op.flip(x1, axis=0), R.Tensor("float16", ndim=3))
    _check_inference(bb, relax.op.flip(x2, axis=0), R.Tensor("int32"))
    _check_inference(bb, relax.op.flip(x3, axis=2), R.Tensor((2, 10, 4)))
    _check_inference(bb, relax.op.flip(x4, axis=2), R.Tensor(ndim=3))


def test_flip_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    x = relax.Var("x", R.Tensor((a, b), "float32"))

    _check_inference(bb, relax.op.flip(x, axis=0), relax.TensorStructInfo((a, b), "float32"))


def test_flip_infer_struct_info_wrong_inputs():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.flip(x0, axis=3))


def test_scatter_elements_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    d0 = relax.Var("data", R.Tensor((4, 4), "float32"))
    d1 = relax.Var("data", R.Tensor(dtype="float32", ndim=2))
    d2 = relax.Var("data", R.Tensor("float32"))
    d3 = relax.Var("data", R.Tensor((4, 4), "float32", vdev0))
    i0 = relax.Var("indices", R.Tensor((2, 2), "int64"))
    i1 = relax.Var("indices", R.Tensor((2, 2)))
    i2 = relax.Var("indices", R.Tensor(dtype="int64", ndim=2))
    i3 = relax.Var("indices", R.Tensor(ndim=2))
    i4 = relax.Var("indices", R.Tensor((2, 2), "int64", vdev0))
    u0 = relax.Var("updates", R.Tensor((2, 2), "float32"))
    u1 = relax.Var("updates", R.Tensor((2, 2), "float32", vdev0))
    _check_inference(
        bb,
        relax.op.scatter_elements(d0, i0, u0, 0, "updates"),
        relax.TensorStructInfo((4, 4), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d3, i4, u1, 0, "updates"),
        relax.TensorStructInfo((4, 4), dtype="float32", vdevice=vdev0),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d1, i0, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d2, i0, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=-1),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d0, i1, u0, 0, "updates"),
        relax.TensorStructInfo((4, 4), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d1, i1, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d2, i1, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=-1),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d0, i2, u0, 0, "updates"),
        relax.TensorStructInfo((4, 4), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d1, i2, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d2, i2, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=-1),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d0, i3, u0, 0, "updates"),
        relax.TensorStructInfo((4, 4), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d1, i3, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d2, i3, u0, 0, "updates"),
        relax.TensorStructInfo(dtype="float32", ndim=-1),
    )


def test_scatter_elements_infer_struct_info_symbolic_shape():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    e = tir.Var("e", "int64")
    f = tir.Var("f", "int64")

    d0 = relax.Var("data", R.Tensor((a, b), "float32"))
    i0 = relax.Var("indices", R.Tensor((c, d), "int64"))
    u0 = relax.Var("updates", R.Tensor((c, d), "float32"))
    u1 = relax.Var("updates", R.Tensor((e, f), "float32"))

    _check_inference(
        bb,
        relax.op.scatter_elements(d0, i0, u0, 0, "updates"),
        relax.TensorStructInfo((a, b), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.scatter_elements(d0, i0, u1, 0, "updates"),
        relax.TensorStructInfo((a, b), dtype="float32"),
    )


def test_scatter_elements_infer_struct_info_wrong_indices_type():
    bb = relax.BlockBuilder()
    d0 = relax.Var("data", R.Tensor((4, 4), "float32"))
    i0 = relax.Var("indices", R.Tensor((2, 2), "float32"))
    u0 = relax.Var("updates", R.Tensor((2, 2), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i0, u0))


def test_scatter_elements_infer_struct_info_rank_shape_mismatch():
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")

    bb = relax.BlockBuilder()
    d0 = relax.Var("data", R.Tensor((4, 4), "float32"))
    i0 = relax.Var("indices", R.Tensor((3, 3), "int64"))
    i1 = relax.Var("indices", R.Tensor((3, 3, 3), "int64"))
    i2 = relax.Var("indices", R.Tensor((a, b), "int64"))
    u0 = relax.Var("updates", R.Tensor((3, 2), "float32"))
    u1 = relax.Var("updates", R.Tensor((3, 2, 3), "float32"))
    u2 = relax.Var("updates", R.Tensor((3, 3, 3), "float32"))
    u3 = relax.Var("updates", R.Tensor((a + 1, b), "float32"))
    u4 = relax.Var("updates", R.Tensor((3, 3), "float16"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i0, u0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i1, u0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i0, u1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i1, u1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i1, u2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i2, u3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.scatter_elements(d0, i0, u4))


if __name__ == "__main__":
    tvm.testing.main()
