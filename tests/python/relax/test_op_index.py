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
from tvm.script import ir as I, relax as R, tir as T


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    idx = relax.Var("idx", R.Tensor((2,), "float32"))
    assert relax.op.take(x, idx, axis=1).op == Op.get("relax.take")
    assert relax.op.strided_slice(x, axes=[0], begin=[0], end=[2]).op == Op.get(
        "relax.strided_slice"
    )
    assert relax.op.dynamic_strided_slice(x, x, x, x).op == Op.get("relax.dynamic_strided_slice")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_take_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((4, 10), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((4, 10)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((4, 10), "float32", vdev0))
    y0 = relax.Var("y", R.Tensor((10,), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=1))
    y2 = relax.Var("y", R.Tensor((10,)))
    y3 = relax.Var("y", R.Tensor(ndim=1))
    idx0 = relax.Var("idx", R.Tensor((6,), "int64"))
    idx1 = relax.Var("idx", R.Tensor("int64", ndim=1))
    idx2 = relax.Var("idx", R.Tensor((6,)))
    idx3 = relax.Var("idx", R.Tensor(ndim=1))
    idx4 = relax.Var("idx", R.Tensor((6, 4), "int64"))
    idx5 = relax.Var("idx", R.Tensor("int64", ndim=2))
    idx6 = relax.Var("idx", R.Tensor((6, 4)))
    idx7 = relax.Var("idx", R.Tensor(ndim=2))
    idx8 = relax.Var("idx", R.Tensor((6,), "int64", vdev0))

    _check_inference(bb, relax.op.take(x0, idx0, axis=1), relax.TensorStructInfo((4, 6), "float32"))
    _check_inference(
        bb, relax.op.take(x6, idx8, axis=1), relax.TensorStructInfo((4, 6), "float32", vdev0)
    )
    _check_inference(
        bb, relax.op.take(x0, idx0, axis=-1), relax.TensorStructInfo((4, 6), "float32")
    )
    _check_inference(
        bb, relax.op.take(x1, idx0, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.take(x2, idx0, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x3, idx0, axis=1), relax.TensorStructInfo((4, 6), dtype=""))
    _check_inference(bb, relax.op.take(x4, idx0, axis=1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(x5, idx0, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.take(x0, idx1, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x1, idx1, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.take(x2, idx1, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x3, idx1, axis=1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(x4, idx1, axis=1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(x5, idx1, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.take(x0, idx2, axis=1), relax.TensorStructInfo((4, 6), "float32"))
    _check_inference(
        bb, relax.op.take(x1, idx2, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.take(x2, idx2, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x3, idx2, axis=1), relax.TensorStructInfo((4, 6), dtype=""))
    _check_inference(bb, relax.op.take(x4, idx2, axis=1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(x5, idx2, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.take(x0, idx3, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x1, idx3, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.take(x2, idx3, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x3, idx3, axis=1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(x4, idx3, axis=1), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(x5, idx3, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.take(x0, idx4, axis=0), relax.TensorStructInfo((6, 4, 10), dtype="float32")
    )
    _check_inference(
        bb, relax.op.take(x0, idx4, axis=1), relax.TensorStructInfo((4, 6, 4), dtype="float32")
    )
    _check_inference(
        bb, relax.op.take(x1, idx4, axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.take(x2, idx4, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.take(x3, idx4, axis=1), relax.TensorStructInfo((4, 6, 4), dtype="")
    )
    _check_inference(bb, relax.op.take(x4, idx4, axis=1), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.take(x5, idx4, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.take(x0, idx5, axis=0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.take(x0, idx5, axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.take(x1, idx5, axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.take(x2, idx5, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x3, idx5, axis=1), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.take(x4, idx5, axis=1), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.take(x5, idx5, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.take(x0, idx6, axis=0), relax.TensorStructInfo((6, 4, 10), dtype="float32")
    )
    _check_inference(
        bb, relax.op.take(x0, idx6, axis=1), relax.TensorStructInfo((4, 6, 4), dtype="float32")
    )
    _check_inference(
        bb, relax.op.take(x1, idx6, axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.take(x2, idx6, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.take(x3, idx6, axis=1), relax.TensorStructInfo((4, 6, 4), dtype="")
    )
    _check_inference(bb, relax.op.take(x4, idx6, axis=1), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.take(x5, idx6, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.take(x0, idx7, axis=0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.take(x0, idx7, axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.take(x1, idx7, axis=1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.take(x2, idx7, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x3, idx7, axis=1), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.take(x4, idx7, axis=1), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, relax.op.take(x5, idx7, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.take(y0, idx0), relax.TensorStructInfo((6,), "float32"))
    _check_inference(bb, relax.op.take(y1, idx0), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.take(y2, idx0), relax.TensorStructInfo((6,), dtype=""))
    _check_inference(bb, relax.op.take(y3, idx0), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.take(y0, idx1), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.take(y1, idx1), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.take(y2, idx1), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.take(y3, idx1), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.take(y0, idx2), relax.TensorStructInfo((6,), "float32"))
    _check_inference(bb, relax.op.take(y1, idx2), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.take(y2, idx2), relax.TensorStructInfo((6,), dtype=""))
    _check_inference(bb, relax.op.take(y3, idx2), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.take(y0, idx3), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.take(y1, idx3), relax.TensorStructInfo(dtype="float32", ndim=1))
    _check_inference(bb, relax.op.take(y2, idx3), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.take(y3, idx3), relax.TensorStructInfo(dtype="", ndim=1))
    _check_inference(bb, relax.op.take(y0, idx4), relax.TensorStructInfo((6, 4), "float32"))
    _check_inference(bb, relax.op.take(y1, idx4), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.take(y2, idx4), relax.TensorStructInfo((6, 4), dtype=""))
    _check_inference(bb, relax.op.take(y3, idx4), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(y0, idx5), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.take(y1, idx5), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.take(y2, idx5), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(y3, idx5), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(y0, idx6), relax.TensorStructInfo((6, 4), "float32"))
    _check_inference(bb, relax.op.take(y1, idx6), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.take(y2, idx6), relax.TensorStructInfo((6, 4), dtype=""))
    _check_inference(bb, relax.op.take(y3, idx6), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(y0, idx7), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.take(y1, idx7), relax.TensorStructInfo(dtype="float32", ndim=2))
    _check_inference(bb, relax.op.take(y2, idx7), relax.TensorStructInfo(dtype="", ndim=2))
    _check_inference(bb, relax.op.take(y3, idx7), relax.TensorStructInfo(dtype="", ndim=2))


def test_take_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    i = tir.Var("i", "int64")
    j = tir.Var("j", "int64")
    k = tir.Var("k", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))
    y0 = relax.Var("y", R.Tensor((n,), "float32"))
    y1 = relax.Var("y", R.Tensor((n,)))
    idx0 = relax.Var("idx", R.Tensor((i,), "int64"))
    idx1 = relax.Var(
        "idx",
        R.Tensor(
            (i,),
        ),
    )
    idx2 = relax.Var(
        "idx",
        R.Tensor(
            (i, j, k),
        ),
    )

    _check_inference(bb, relax.op.take(x0, idx0, axis=1), relax.TensorStructInfo((m, i), "float32"))
    _check_inference(bb, relax.op.take(x1, idx0, axis=1), relax.TensorStructInfo((m, i), dtype=""))
    _check_inference(bb, relax.op.take(x0, idx1, axis=1), relax.TensorStructInfo((m, i), "float32"))
    _check_inference(bb, relax.op.take(x1, idx1, axis=1), relax.TensorStructInfo((m, i), dtype=""))
    _check_inference(
        bb, relax.op.take(x1, idx2, axis=1), relax.TensorStructInfo((m, i, j, k), dtype="")
    )
    _check_inference(
        bb, relax.op.take(x1, idx2, axis=1), relax.TensorStructInfo((m, i, j, k), dtype="")
    )
    _check_inference(bb, relax.op.take(y0, idx0), relax.TensorStructInfo((i,), "float32"))
    _check_inference(bb, relax.op.take(y1, idx0), relax.TensorStructInfo((i,), dtype=""))
    _check_inference(bb, relax.op.take(y0, idx1), relax.TensorStructInfo((i,), "float32"))
    _check_inference(bb, relax.op.take(y1, idx1), relax.TensorStructInfo((i,), dtype=""))
    _check_inference(bb, relax.op.take(y0, idx2), relax.TensorStructInfo((i, j, k), "float32"))
    _check_inference(bb, relax.op.take(y1, idx2), relax.TensorStructInfo((i, j, k), dtype=""))


def test_take_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    sx0 = relax.Var("sx", relax.ShapeStructInfo((4, 10)))
    sx1 = relax.Var("sx", relax.ShapeStructInfo(ndim=2))
    sx2 = relax.Var("sx", relax.ShapeStructInfo())
    sidx0 = relax.Var("sidx", relax.ShapeStructInfo((6,)))
    sidx1 = relax.Var("sidx", relax.ShapeStructInfo(ndim=1))
    x0 = relax.Var("x", relax.TensorStructInfo(sx0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(sx1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(sx2, "float32"))
    x3 = relax.Var("x", R.Tensor((4, 10), "float32"))
    idx0 = relax.Var("idx", relax.TensorStructInfo(sidx0, "int64"))
    idx1 = relax.Var("idx", relax.TensorStructInfo(sidx1, "int64"))
    idx2 = relax.Var("idx", R.Tensor((6,), "int64"))

    _check_inference(
        bb, relax.op.take(x0, idx0, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x0, idx1, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x0, idx2, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x1, idx0, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x1, idx1, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x1, idx2, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(bb, relax.op.take(x2, idx0, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x2, idx1, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.take(x2, idx2, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.take(x3, idx0, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb, relax.op.take(x3, idx1, axis=1), relax.TensorStructInfo(dtype="float32", ndim=2)
    )


def test_take_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((4, 10), "float16"))
    x1 = relax.Var("x", R.Tensor((4, 10), "int16"))
    x2 = relax.Var("x", R.Tensor((4, 10), "int32"))
    idx0 = relax.Var("idx", R.Tensor((6,), "int32"))
    idx1 = relax.Var("idx", R.Tensor((6,), "int8"))
    idx2 = relax.Var("idx", R.Tensor((6,), "uint32"))

    _check_inference(bb, relax.op.take(x0, idx0, axis=1), relax.TensorStructInfo((4, 6), "float16"))
    _check_inference(bb, relax.op.take(x1, idx0, axis=1), relax.TensorStructInfo((4, 6), "int16"))
    _check_inference(bb, relax.op.take(x2, idx0, axis=1), relax.TensorStructInfo((4, 6), "int32"))
    _check_inference(bb, relax.op.take(x0, idx1, axis=1), relax.TensorStructInfo((4, 6), "float16"))
    _check_inference(bb, relax.op.take(x1, idx1, axis=1), relax.TensorStructInfo((4, 6), "int16"))
    _check_inference(bb, relax.op.take(x2, idx1, axis=1), relax.TensorStructInfo((4, 6), "int32"))
    _check_inference(bb, relax.op.take(x0, idx2, axis=1), relax.TensorStructInfo((4, 6), "float16"))
    _check_inference(bb, relax.op.take(x1, idx2, axis=1), relax.TensorStructInfo((4, 6), "int16"))
    _check_inference(bb, relax.op.take(x2, idx2, axis=1), relax.TensorStructInfo((4, 6), "int32"))


def test_take_infer_struct_info_indices_not_integer_dtype():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((4, 10), "float32"))
    idx0 = relax.Var("idx", R.Tensor((6, 6), "float32"))
    idx1 = relax.Var("idx", R.Tensor((6, 6), "float64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx1, axis=1))


def test_take_infer_struct_info_multi_dimensional_without_axis():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((4, 10), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    idx0 = relax.Var("idx", R.Tensor((6,), "int64"))
    idx1 = relax.Var("idx", R.Tensor("int64", ndim=1))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x0, idx0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x1, idx0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x2, idx0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x0, idx1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x1, idx1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x2, idx1))


def test_take_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((4, 10), "float32"))
    idx = relax.Var("idx", R.Tensor((6,), "int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx, axis=-3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx, axis=2))


def test_take_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((4, 10)))
    x1 = relax.Var("x", R.Tensor((4, 10), "float32"))
    idx0 = relax.Var("idx", relax.ShapeStructInfo((6,)))
    idx1 = relax.Var("idx", R.Tensor((6,), "int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x0, idx1, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x1, idx0, axis=1))


def test_strided_slice_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((8, 9, 10, 10), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((8, 9, 10, 10)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((8, 9, 10, 10), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.strided_slice(
            x0, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
        ),
        relax.TensorStructInfo((4, 9, 10, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x6, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
        ),
        relax.TensorStructInfo((4, 9, 10, 3), "float32", vdev0),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x1, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
        ),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x2, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
        ),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x3, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
        ),
        relax.TensorStructInfo((4, 9, 10, 3), dtype=""),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x4, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
        ),
        relax.TensorStructInfo(dtype="", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x5, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
        ),
        relax.TensorStructInfo(dtype=""),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x0, axes=[-1, -3, -4], begin=[8, 0, 1], end=[0, 9, 8], strides=[-3, 1, 2]
        ),
        relax.TensorStructInfo((4, 9, 10, 3), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[1, 2], begin=[1, 0], end=[8, 9]),
        relax.TensorStructInfo((8, 7, 9, 10), "float32"),
    )


def test_strided_slice_infer_struct_info_shape_out_of_range():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((20, 10, 5), "float32"))
    _check_inference(
        bb,
        relax.op.strided_slice(
            x0, axes=[0, 1, 2], begin=[20, 10, 4], end=[0, 0, 1], strides=[-1, -3, -2]
        ),
        relax.TensorStructInfo((19, 3, 2), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x0, axes=[0, 1, 2], begin=[200, 10, 4], end=[0, 0, 1], strides=[-1, -3, -2]
        ),
        relax.TensorStructInfo((19, 3, 2), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x0, axes=[0, 1, 2], begin=[200, 10, 100], end=[0, 0, 1], strides=[-1, -3, -5]
        ),
        relax.TensorStructInfo((19, 3, 1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(
            x0, axes=[0, 1, 2], begin=[-21, -11, -6], end=[1, 1, 1], strides=[1000, 1000, 1000]
        ),
        relax.TensorStructInfo((1, 1, 1), "float32"),
    )


def test_strided_slice_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[0], begin=[1], end=[3]),
        relax.TensorStructInfo((tir.min(3, m) - tir.min(1, m), n), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[0], begin=[1], end=[8], strides=[3]),
        relax.TensorStructInfo(((tir.min(8, m) + 2 - tir.min(1, m)) // 3, n), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x1, axes=[0], begin=[1], end=[3]),
        relax.TensorStructInfo((tir.min(3, m) - tir.min(1, m), n), dtype=""),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x1, axes=[0], begin=[1], end=[8], strides=[3]),
        relax.TensorStructInfo(((tir.min(8, m) + 2 - tir.min(1, m)) // 3, n), dtype=""),
    )


def test_strided_slice_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((8, 10)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s0, dtype=""))
    x4 = relax.Var("x", relax.TensorStructInfo(s1, dtype=""))
    x5 = relax.Var("x", relax.TensorStructInfo(s2, dtype=""))

    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x1, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x2, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x3, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo(dtype="", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x4, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo(dtype="", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x5, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo(dtype=""),
    )


def test_strided_slice_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((8, 9), "float16"))
    x1 = relax.Var("x", R.Tensor((8, 9), "int32"))
    x2 = relax.Var("x", R.Tensor((8, 9), "int64"))

    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo((8, 9), "float16"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x1, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo((8, 9), "int32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x2, axes=[0], begin=[0], end=[8]),
        relax.TensorStructInfo((8, 9), "int64"),
    )


def test_strided_slice_infer_struct_info_symbolic_begin_end_strides():
    bb = relax.BlockBuilder()
    var = tir.Var("var", "int64")
    size_var = tir.SizeVar("size_var", "int64")
    x = relax.Var("x", R.Tensor((8, 9), "float32"))

    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[var], end=[8]),
        relax.TensorStructInfo(
            (tir.max(8 - tir.max(tir.if_then_else(var < 0, var + 8, var), 0), 0), 9),
            dtype="float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[size_var], end=[8]),
        relax.TensorStructInfo((tir.max(8 - size_var, 0), 9), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[var]),
        relax.TensorStructInfo(
            (tir.min(tir.max(tir.if_then_else(var < 0, var + 8, var), 0), 8), 9), dtype="float32"
        ),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[size_var]),
        relax.TensorStructInfo((tir.min(size_var, 8), 9), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[8], strides=[var]),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[8], strides=[size_var]),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )


def test_strided_slice_infer_struct_info_symbolic_begin_end_strides_inbound():
    bb = relax.BlockBuilder()
    var = tir.Var("var", "int64")
    size_var = tir.SizeVar("size_var", "int64")
    x = relax.Var("x", R.Tensor((8, 9), "float32"))

    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[var], end=[8], assume_inbound=True),
        relax.TensorStructInfo(
            (8 - tir.if_then_else(var < 0, var + 8, var), 9),
            dtype="float32",
        ),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[size_var], end=[8], assume_inbound=True),
        relax.TensorStructInfo((8 - size_var, 9), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[var], assume_inbound=True),
        relax.TensorStructInfo((tir.if_then_else(var < 0, var + 8, var), 9), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[size_var], assume_inbound=True),
        relax.TensorStructInfo((size_var, 9), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[8], strides=[var], assume_inbound=True),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[8], strides=[var], assume_inbound=True),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )


def test_strided_slice_infer_struct_info_no_axis():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    s0 = relax.Var("s", relax.ShapeStructInfo((m, n)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor(dtype="float32", ndim=2))
    x2 = relax.Var("x", R.Tensor(dtype="float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[], begin=[], end=[]),
        relax.TensorStructInfo((m, n), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x1, axes=[], begin=[], end=[]),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x2, axes=[], begin=[], end=[]),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x3, axes=[], begin=[], end=[]),
        relax.TensorStructInfo(s0, "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x4, axes=[], begin=[], end=[]),
        relax.TensorStructInfo(s1, "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x5, axes=[], begin=[], end=[]),
        relax.TensorStructInfo(s2, "float32"),
    )


def test_strided_slice_begin_end_strides_int64():
    x = relax.Var("x", R.Tensor((8, 9, 10, 10), "float32"))
    strided_slice = relax.op.strided_slice(
        x, axes=[0, 1, 3], begin=[1, 0, 8], end=[8, 9, 0], strides=[2, 1, -3]
    )

    assert strided_slice.attrs.begin[0].dtype == "int64"
    assert strided_slice.attrs.begin[1].dtype == "int64"
    assert strided_slice.attrs.begin[2].dtype == "int64"
    assert strided_slice.attrs.end[0].dtype == "int64"
    assert strided_slice.attrs.end[1].dtype == "int64"
    assert strided_slice.attrs.end[2].dtype == "int64"
    assert strided_slice.attrs.strides[0].dtype == "int64"
    assert strided_slice.attrs.strides[1].dtype == "int64"
    assert strided_slice.attrs.strides[2].dtype == "int64"


def test_strided_slice_inconsistent_axes_begin_end_strides_length():
    x = relax.Var("x", R.Tensor((8, 9), "float32"))

    with pytest.raises(TVMError):
        relax.op.strided_slice(x, axes=[1], begin=[], end=[9])
    with pytest.raises(TVMError):
        relax.op.strided_slice(x, axes=[1], begin=[0], end=[])
    with pytest.raises(TVMError):
        relax.op.strided_slice(x, axes=[1], begin=[0], end=[9], strides=[])


def test_strided_slice_infer_struct_info_repetitive_axes():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((8, 9), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x, axes=[0, 0], begin=[0, 0], end=[8, 8]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x, axes=[0, -2], begin=[0, 0], end=[8, 8]))


def test_strided_slice_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((8, 9), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x, axes=[2], begin=[0], end=[8]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x, axes=[-3], begin=[0], end=[8]))


def test_strided_slice_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((8, 9)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((8, 9), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x0, axes=[0], begin=[0], end=[8]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x1, axes=[0], begin=[0], end=[8]))


def test_dynamic_strided_slice_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((8, 9, 10, 10), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((8, 9, 10, 10)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())

    b0 = relax.Var("begin", R.Tensor((4,), "int64"))
    e0 = relax.Var("end", R.Tensor((4,), "int64"))
    s0 = relax.Var("strides", R.Tensor((4,), "int64"))
    b1 = relax.Var("begin", R.Tensor((4,)))
    e1 = relax.Var("end", R.Tensor((4,)))
    s1 = relax.Var("stride", R.Tensor((4,)))

    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x0, b0, e0, s0),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x1, b0, e0, s0),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x2, b0, e0, s0),
        R.Tensor("float32", ndim=-1),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x3, b0, e0, s0),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x4, b0, e0, s0),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x5, b0, e0, s0),
        R.Tensor(ndim=-1),
    )

    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x0, b1, e1, s1),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x1, b1, e1, s1),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x2, b1, e1, s1),
        R.Tensor("float32", ndim=-1),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x3, b1, e1, s1),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x4, b1, e1, s1),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x5, b1, e1, s1),
        R.Tensor(ndim=-1),
    )


def test_dynamic_strided_slice_infer_struct_info_symbolic():
    bb = relax.BlockBuilder()
    i = tir.Var("i", "int64")
    j = tir.Var("j", "int64")
    k = tir.Var("k", "int64")
    l = tir.Var("l", "int64")
    x0 = relax.Var("x", R.Tensor((i, j, k, l), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((i, j, k, l)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())

    b0 = relax.Var("begin", R.Tensor((4,), "int64"))
    e0 = relax.Var("end", R.Tensor((4,), "int64"))
    s0 = relax.Var("stride", R.Tensor((4,), "int64"))
    b1 = relax.Var("begin", R.Tensor((4,)))
    e1 = relax.Var("end", R.Tensor((4,)))
    s1 = relax.Var("stride", R.Tensor((4,)))

    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x0, b0, e0, s0),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x1, b0, e0, s0),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x2, b0, e0, s0),
        R.Tensor("float32", ndim=-1),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x3, b0, e0, s0),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x4, b0, e0, s0),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x5, b0, e0, s0),
        R.Tensor(ndim=-1),
    )

    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x0, b1, e1, s1),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x1, b1, e1, s1),
        R.Tensor("float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x2, b1, e1, s1),
        R.Tensor("float32", ndim=-1),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x3, b1, e1, s1),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x4, b1, e1, s1),
        R.Tensor(ndim=4),
    )
    _check_inference(
        bb,
        relax.op.dynamic_strided_slice(x5, b1, e1, s1),
        R.Tensor(ndim=-1),
    )


def test_dynamic_strided_slice_infer_struct_info_arg_wrong_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((8, 9, 10, 10), "float32"))
    b0 = relax.Var("begin", R.Tensor((4,), "float32"))
    e0 = relax.Var("end", R.Tensor((4,), "float32"))
    s0 = relax.Var("stride", R.Tensor((4,), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x0, b0, e0, s0))


def test_dynamic_strided_slice_infer_struct_info_arg_wrong_shape_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((8, 9, 10, 10), "float32"))
    m = tir.Var("m", "int64")
    # invalid arg
    b0 = relax.Var("begin", R.Tensor("int64", ndim=2))
    b1 = relax.Var("begin", R.Tensor((1,), "int64"))
    b2 = relax.Var("begin", R.Tensor((2, 2), "int64"))
    b3 = relax.Var("begin", R.Tensor((m,), "int64"))
    # valid args
    e0 = relax.Var("end", R.Tensor((4,), "int64"))
    s0 = relax.Var("stride", R.Tensor((4,), "int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x0, b0, e0, s0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x0, b1, e0, s0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x0, b2, e0, s0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.strided_slice(x0, b3, e0, s0))


def test_legalize_dynamic_begin_end():
    """relax.op.strided_slice FLegalize must support dynamic begin/end"""

    @I.ir_module
    class before:
        @R.function
        def main(A: R.Tensor((16, 16), "float32"), B: R.Shape(["index"])) -> R.Tensor((1, 16)):
            index = T.int64()
            return R.strided_slice(A, [0], [index], [index + 1], assume_inbound=True)

    @I.ir_module
    class expected:
        @R.function
        def main(A: R.Tensor((16, 16), "float32"), B: R.Shape(["index"])) -> R.Tensor((1, 16)):
            index = T.int64()
            return R.call_tir(
                expected.strided_slice,
                (A,),
                out_sinfo=R.Tensor((1, 16), "float32"),
                tir_vars=R.shape([index]),
            )

        @T.prim_func(private=True)
        def strided_slice(
            A: T.Buffer((T.int64(16), T.int64(16))),
            B: T.Buffer((T.int64(1), T.int64(16))),
            index: T.int64,
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            for iters in T.grid(*B.shape):
                with T.block("T_dynamic_strided_slice"):
                    i, j = T.axis.remap("SS", iters)
                    B[i, j] = A[i + index, j]

    after = tvm.relax.transform.LegalizeOps()(before)
    tvm.ir.assert_structural_equal(expected, after)


if __name__ == "__main__":
    tvm.testing.main()
