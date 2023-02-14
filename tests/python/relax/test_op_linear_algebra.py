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
from tvm.ir import Op
from tvm.script import relax as R


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((3, 4), "float32"))
    assert relax.op.matmul(x, y).op == Op.get("relax.matmul")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_matmul_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((4,), "float32"))
    x2 = relax.Var("x", R.Tensor((2, 3, 5, 4), "float32"))
    x3 = relax.Var("x", R.Tensor((2, 1, 4, 5), "float32"))
    x4 = relax.Var("x", R.Tensor((2, 1, 4, 5)))
    x5 = relax.Var("x", R.Tensor("float32"))
    x6 = relax.Var("x", R.Tensor((2, 1, 4, 5), "float16"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float32"))
    y1 = relax.Var("y", R.Tensor((4,), "float32"))
    y2 = relax.Var("y", R.Tensor((2, 3, 4, 5), "float32"))
    y3 = relax.Var("y", R.Tensor((6, 1, 3, 5, 7), "float32"))
    y4 = relax.Var("y", R.Tensor("float32", ndim=5))
    y5 = relax.Var("y", R.Tensor())

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorStructInfo((3, 5), "float32"))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorStructInfo((), "float32"))
    _check_inference(bb, relax.op.matmul(x1, y2), relax.TensorStructInfo((2, 3, 5), "float32"))
    _check_inference(bb, relax.op.matmul(x2, y1), relax.TensorStructInfo((2, 3, 5), "float32"))
    _check_inference(
        bb, relax.op.matmul(x3, y3), relax.TensorStructInfo((6, 2, 3, 4, 7), "float32")
    )
    _check_inference(bb, relax.op.matmul(x4, y3), relax.TensorStructInfo((6, 2, 3, 4, 7), ""))
    _check_inference(bb, relax.op.matmul(x3, y4), relax.TensorStructInfo(dtype="float32", ndim=5))
    _check_inference(bb, relax.op.matmul(x5, y3), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.matmul(x3, y5), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb,
        relax.op.matmul(x3, y3, out_dtype="float16"),
        relax.TensorStructInfo((6, 2, 3, 4, 7), "float16"),
    )
    _check_inference(
        bb,
        relax.op.matmul(x6, y3, out_dtype="float16"),
        relax.TensorStructInfo((6, 2, 3, 4, 7), "float16"),
    )


def test_matmul_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    k0 = tir.Var("k0", "int64")
    k1 = tir.Var("k1", "int64")
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    b1 = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((m, k0), "float32"))
    x1 = relax.Var("x", R.Tensor((k0,), "float32"))
    x2 = relax.Var("x", R.Tensor((a, b, m, k0), "float32"))
    x3 = relax.Var("x", R.Tensor((b, 1, m, k0), "float32"))
    x4 = relax.Var("x", R.Tensor((b, 1, m, k1), "float32"))
    y0 = relax.Var("y", R.Tensor((k0, n), "float32"))
    y1 = relax.Var("y", R.Tensor((k0,), "float32"))
    y2 = relax.Var("y", R.Tensor((a, b, k0, n), "float32"))
    y3 = relax.Var("y", R.Tensor((a, 1, c, k0, n), "float32"))
    y4 = relax.Var("y", R.Tensor((a, b1, c, k0, n), "float32"))

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorStructInfo((), "float32"))
    _check_inference(bb, relax.op.matmul(x1, y2), relax.TensorStructInfo((a, b, n), "float32"))
    _check_inference(bb, relax.op.matmul(x2, y1), relax.TensorStructInfo((a, b, m), "float32"))
    _check_inference(
        bb, relax.op.matmul(x3, y3), relax.TensorStructInfo((a, b, c, m, n), "float32")
    )
    _check_inference(
        bb, relax.op.matmul(x4, y3), relax.TensorStructInfo((a, b, c, m, n), "float32")
    )
    _check_inference(bb, relax.op.matmul(x3, y4), relax.TensorStructInfo(dtype="float32", ndim=5))


def test_matmul_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s1", relax.ShapeStructInfo(ndim=3))
    s2 = relax.Var("s3", relax.ShapeStructInfo(ndim=1))
    s3 = relax.Var("s4", relax.ShapeStructInfo(ndim=1))
    s5 = relax.Var("s5", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s5, "float32"))
    y0 = relax.Var("y", relax.TensorStructInfo(s1, "float32"))
    y1 = relax.Var("y", relax.TensorStructInfo(s2, "float32"))
    y2 = relax.Var("y", relax.TensorStructInfo(s3, "float32"))

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorStructInfo(dtype="float32", ndim=4))
    _check_inference(bb, relax.op.matmul(x1, y0), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.matmul(x2, y0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.matmul(x0, y1), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorStructInfo(dtype="float32", ndim=0))
    _check_inference(bb, relax.op.matmul(x2, y1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.matmul(x1, y2), relax.TensorStructInfo(dtype="float32", ndim=0))


def test_matmul_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float16"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((3, 4), "int8"))
    y1 = relax.Var("y", R.Tensor((4, 5), "int8"))
    x2 = relax.Var("x", R.Tensor((3, 4), "int64"))
    y2 = relax.Var("y", R.Tensor((4, 5), "int64"))

    _check_inference(bb, relax.op.matmul(x0, y0), relax.TensorStructInfo((3, 5), "float16"))
    _check_inference(bb, relax.op.matmul(x1, y1), relax.TensorStructInfo((3, 5), "int8"))
    _check_inference(bb, relax.op.matmul(x2, y2), relax.TensorStructInfo((3, 5), "int64"))


def test_matmul_infer_struct_info_mixed_precision():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float16"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((3, 4), "int8"))
    y1 = relax.Var("y", R.Tensor((4, 5), "int8"))
    x2 = relax.Var("x", R.Tensor((3, 4)))
    y2 = relax.Var("y", R.Tensor((4, 5)))

    _check_inference(
        bb,
        relax.op.matmul(x0, y0, out_dtype="float32"),
        relax.TensorStructInfo((3, 5), "float32"),
    )
    _check_inference(
        bb, relax.op.matmul(x1, y1, out_dtype="int32"), relax.TensorStructInfo((3, 5), "int32")
    )
    _check_inference(
        bb,
        relax.op.matmul(x2, y2, out_dtype="float32"),
        relax.TensorStructInfo((3, 5), "float32"),
    )


def test_matmul_infer_struct_info_zero_rank_input():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((), "float32"))
    y0 = relax.Var("y", R.Tensor((4, 5), "float32"))
    y1 = relax.Var("y", R.Tensor((), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.matmul(x0, y1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.matmul(x1, y0))


def test_matmul_infer_struct_info_not_broadcastable():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((2, 8, 3, 5, 6), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.matmul(x, y))


def test_matmul_infer_struct_info_unequal_reduction_length():
    bb = relax.BlockBuilder()
    k = tir.Var("k", "int64")
    x0 = relax.Var("x", R.Tensor((3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((3, k), "float32"))
    y0 = relax.Var("y", R.Tensor((6, 5), "float32"))
    y1 = relax.Var("y", R.Tensor((k + 1, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.matmul(x0, y0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.matmul(x1, y1))


def test_linear():
    # Since linear is only a sugar for transpose + matmul + add,
    # we only have brief tests here.
    bb = relax.BlockBuilder()
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x2 = relax.Var("x", R.Tensor("float32"))
    w1 = relax.Var("w", R.Tensor((5, 4), "float32"))
    w2 = relax.Var("w", R.Tensor((4,), "float32"))
    w3 = relax.Var("w", R.Tensor("float32"))
    b1 = relax.Var("b", R.Tensor((5,), "float32"))
    b2 = relax.Var("b", R.Tensor((), "float32"))

    # Need a scope to normalize non-leaf nodes
    with bb.function("func", [x1]):
        _check_inference(
            bb, relax.op.linear(x1, w1, b1), relax.TensorStructInfo((2, 3, 5), "float32")
        )
        _check_inference(
            bb, relax.op.linear(x1, w1, b2), relax.TensorStructInfo((2, 3, 5), "float32")
        )
        with pytest.raises(TVMError):
            bb.normalize(relax.op.linear(x1, w2, b1))  # error on Add with shape (2, 3, 5) and (4,)
        _check_inference(bb, relax.op.linear(x1, w2, b2), relax.TensorStructInfo((2, 3), "float32"))
        _check_inference(bb, relax.op.linear(x1, w3, b1), relax.TensorStructInfo(dtype="float32"))
        _check_inference(bb, relax.op.linear(x1, w3, b2), relax.TensorStructInfo(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w1, b1), relax.TensorStructInfo(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w1, b2), relax.TensorStructInfo(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w2, b1), relax.TensorStructInfo(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w2, b2), relax.TensorStructInfo(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w3, b1), relax.TensorStructInfo(dtype="float32"))
        _check_inference(bb, relax.op.linear(x2, w3, b2), relax.TensorStructInfo(dtype="float32"))

        # Fake output
        gv = bb.emit_func_output(relax.Tuple([]))


if __name__ == "__main__":
    tvm.testing.main()
