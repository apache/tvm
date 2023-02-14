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
    idx = relax.Var("idx", R.Tensor((2,), "float32"))
    assert relax.op.take(x, idx, axis=1).op == Op.get("relax.take")
    assert relax.op.strided_slice(x, axes=[0], begin=[0], end=[2]).op == Op.get(
        "relax.strided_slice"
    )


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_take_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((4, 10), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((4, 10)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())
    y0 = relax.Var("y", R.Tensor((10,), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=1))
    y2 = relax.Var("y", R.Tensor((10,)))
    y3 = relax.Var("y", R.Tensor(ndim=1))
    idx0 = relax.Var("idx", R.Tensor((6,), "int64"))
    idx1 = relax.Var("idx", R.Tensor("int64", ndim=1))
    idx2 = relax.Var("idx", R.Tensor((6,)))
    idx3 = relax.Var("idx", R.Tensor(ndim=1))

    _check_inference(bb, relax.op.take(x0, idx0, axis=1), relax.TensorStructInfo((4, 6), "float32"))
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


def test_take_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    i = tir.Var("i", "int64")
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

    _check_inference(bb, relax.op.take(x0, idx0, axis=1), relax.TensorStructInfo((m, i), "float32"))
    _check_inference(bb, relax.op.take(x1, idx0, axis=1), relax.TensorStructInfo((m, i), dtype=""))
    _check_inference(bb, relax.op.take(x0, idx1, axis=1), relax.TensorStructInfo((m, i), "float32"))
    _check_inference(bb, relax.op.take(x1, idx1, axis=1), relax.TensorStructInfo((m, i), dtype=""))
    _check_inference(bb, relax.op.take(y0, idx0), relax.TensorStructInfo((i,), "float32"))
    _check_inference(bb, relax.op.take(y1, idx0), relax.TensorStructInfo((i,), dtype=""))
    _check_inference(bb, relax.op.take(y0, idx1), relax.TensorStructInfo((i,), "float32"))
    _check_inference(bb, relax.op.take(y1, idx1), relax.TensorStructInfo((i,), dtype=""))


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


def test_take_infer_struct_info_indices_not_one_dimensional():
    bb = relax.BlockBuilder()
    sidx0 = relax.Var("sidx", relax.ShapeStructInfo((6, 6)))
    sidx1 = relax.Var("sidx", relax.ShapeStructInfo(()))
    sidx2 = relax.Var("sidx", relax.ShapeStructInfo(ndim=2))
    sidx3 = relax.Var("sidx", relax.ShapeStructInfo(ndim=0))
    sidx4 = relax.Var("sidx", relax.ShapeStructInfo())
    x = relax.Var("x", R.Tensor((4, 10), "float32"))
    idx0 = relax.Var("idx", R.Tensor((6, 6), "int64"))
    idx1 = relax.Var("idx", R.Tensor((), "int64"))
    idx2 = relax.Var("idx", R.Tensor("int64", ndim=2))
    idx3 = relax.Var("idx", R.Tensor("int64", ndim=0))
    idx4 = relax.Var("idx", R.Tensor("int64"))
    idx5 = relax.Var("idx", relax.TensorStructInfo(sidx0, "int64"))
    idx6 = relax.Var("idx", relax.TensorStructInfo(sidx1, "int64"))
    idx7 = relax.Var("idx", relax.TensorStructInfo(sidx2, "int64"))
    idx8 = relax.Var("idx", relax.TensorStructInfo(sidx3, "int64"))
    idx9 = relax.Var("idx", relax.TensorStructInfo(sidx4, "int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx1, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx2, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx3, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx4, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx5, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx6, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx7, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx8, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.take(x, idx9, axis=1))


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
    x0 = relax.Var("x", R.Tensor((8, 9, 10, 10), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((8, 9, 10, 10)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())

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


def test_strided_slice_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[0], begin=[1], end=[3]),
        relax.TensorStructInfo((2, n), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x0, axes=[0], begin=[1], end=[8], strides=[3]),
        relax.TensorStructInfo((3, n), "float32"),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x1, axes=[0], begin=[1], end=[3]),
        relax.TensorStructInfo((2, n), dtype=""),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x1, axes=[0], begin=[1], end=[8], strides=[3]),
        relax.TensorStructInfo((3, n), dtype=""),
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
    a = tir.Var("a", "int64")
    x = relax.Var("x", R.Tensor((8, 9), "float32"))

    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[a], end=[8]),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[a]),
        relax.TensorStructInfo(dtype="float32", ndim=2),
    )
    _check_inference(
        bb,
        relax.op.strided_slice(x, axes=[0], begin=[0], end=[8], strides=[a]),
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


if __name__ == "__main__":
    tvm.testing.main()
