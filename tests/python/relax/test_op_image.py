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
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    assert relax.op.image.resize2d(x, (28, 28)).op == Op.get("relax.image.resize2d")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_resize2d_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor((2, 4, 32, 32, 16), "float32"))
    x3 = relax.Var("x", R.Tensor("float32", ndim=4))
    x4 = relax.Var("x", R.Tensor("float32", ndim=5))
    x5 = relax.Var("x", R.Tensor("float32"))
    x6 = relax.Var("x", R.Tensor(ndim=4))
    x7 = relax.Var("x", R.Tensor())
    x8 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32", vdev0))

    _check_inference(
        bb, relax.op.image.resize2d(x0, (28, 28)), relax.TensorStructInfo((2, 3, 28, 28), "float32")
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x8, (28, 28)),
        relax.TensorStructInfo((2, 3, 28, 28), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=28),
        relax.TensorStructInfo((2, 3, 28, 28), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=(28, 30)),
        relax.TensorStructInfo((2, 3, 28, 30), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x1, size=28, layout="NHWC"),
        relax.TensorStructInfo((2, 28, 28, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=28, out_dtype="float16"),
        relax.TensorStructInfo((2, 3, 28, 28), "float16"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x2, size=28, layout="NCHW16c"),
        relax.TensorStructInfo((2, 4, 28, 28, 16), "float32"),
    )
    _check_inference(
        bb, relax.op.image.resize2d(x3, size=28), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x4, size=28, layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb, relax.op.image.resize2d(x5, size=28), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb, relax.op.image.resize2d(x6, size=28), relax.TensorStructInfo(dtype="", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x6, size=28, out_dtype="float32"),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb, relax.op.image.resize2d(x7, size=28), relax.TensorStructInfo(dtype="", ndim=4)
    )


def test_resize2d_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c = tir.Var("c", "int64")
    ih = tir.Var("ih", "int64")
    iw = tir.Var("iw", "int64")
    oh = tir.Var("oh", "int64")
    ow = tir.Var("ow", "int64")
    x0 = relax.Var("x", R.Tensor((n, c, ih, iw), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c, ih, iw, 16), "float32"))

    _check_inference(
        bb, relax.op.image.resize2d(x0, size=oh), relax.TensorStructInfo((n, c, oh, oh), "float32")
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x0, size=(oh, ow)),
        relax.TensorStructInfo((n, c, oh, ow), "float32"),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x1, size=(oh, ow), layout="NCHW16c"),
        relax.TensorStructInfo((n, c, oh, ow, 16), "float32"),
    )


def test_resize2d_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.image.resize2d(x0, size=32), relax.TensorStructInfo(dtype="float32", ndim=4)
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x1, size=32, layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )
    _check_inference(
        bb,
        relax.op.image.resize2d(x2, size=32, layout="NCHW16c"),
        relax.TensorStructInfo(dtype="float32", ndim=5),
    )


def test_resize2d_infer_struct_info_pool_size_var():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    s0 = relax.Var("s", relax.ShapeStructInfo((30, 30)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))

    _check_inference(
        bb,
        relax.op.image.resize2d(x0, s0),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb, relax.op.image.resize2d(x0, s1), relax.TensorStructInfo(dtype="float32", ndim=4)
    )


def test_resize2d_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "int64"))
    _check_inference(
        bb, relax.op.image.resize2d(x0, size=28), relax.TensorStructInfo((2, 3, 28, 28), "float16")
    )
    _check_inference(
        bb, relax.op.image.resize2d(x1, size=28), relax.TensorStructInfo((2, 3, 28, 28), "int8")
    )
    _check_inference(
        bb, relax.op.image.resize2d(x2, size=28), relax.TensorStructInfo((2, 3, 28, 28), "int64")
    )


def test_resize2d_infer_struct_info_wrong_layout_string():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x, size=28, layout="OIHW"))


def test_resize2d_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32, 3), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, size=28, layout="NCHW16c"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x1, size=28, layout="NCHW"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x2, size=28))


def test_resize2d_wrong_pool_size_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float16"))
    s0 = relax.ShapeExpr((3,))
    s1 = relax.Var("s", relax.ShapeStructInfo((30, 30, 30)))
    s2 = relax.Var("s", relax.ShapeStructInfo(ndim=3))
    s3 = relax.Var("s", relax.ShapeStructInfo(ndim=1))
    s4 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    s5 = relax.Var("s", relax.ShapeStructInfo())

    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, (3, 3, 3)))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, s0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, s1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, s2))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, s3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, s4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, s5))


def test_resize2d_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 28, 28), "float32")))
    x2 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    s0 = relax.Var("s", R.Tensor((3, 3)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x0, size=32))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x1, size=32))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.image.resize2d(x2, s0))


if __name__ == "__main__":
    tvm.testing.main()
