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
from typing import Callable

import pytest
import tvm
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op, VDevice
from tvm.script import relax as R


def test_op_correctness():
    cond = relax.Var("cond", R.Tensor((2, 3), "bool"))
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("x", R.Tensor((2, 3), "float32"))
    assert relax.op.where(cond, x, y).op == Op.get("relax.where")
    assert relax.op.argmax(x).op == Op.get("relax.argmax")
    assert relax.op.argmin(x).op == Op.get("relax.argmin")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_where_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    cond0 = relax.Var("cond", R.Tensor((6, 5, 1, 3, 1), "bool"))
    cond1 = relax.Var("cond", R.Tensor("bool", ndim=5))
    cond2 = relax.Var("cond", R.Tensor("bool"))
    cond3 = relax.Var("cond", R.Tensor((6, 5, 1, 3, 1), "bool", vdev0))
    x0 = relax.Var("x", R.Tensor((5, 1, 3, 2), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((5, 1, 3, 2)))
    x4 = relax.Var("x", R.Tensor(ndim=4))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((5, 1, 3, 2), "float32", vdev0))
    y0 = relax.Var("y", R.Tensor((4, 3, 1), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=3))
    y2 = relax.Var("y", R.Tensor("float32"))
    y3 = relax.Var("y", R.Tensor((4, 3, 1)))
    y4 = relax.Var("y", R.Tensor(ndim=3))
    y5 = relax.Var("y", R.Tensor())
    y6 = relax.Var("y", R.Tensor((4, 3, 1), "float32", vdev0))

    _check_inference(
        bb, relax.op.where(cond0, x0, y0), relax.TensorStructInfo((6, 5, 4, 3, 2), "float32")
    )
    _check_inference(
        bb, relax.op.where(cond3, x6, y6), relax.TensorStructInfo((6, 5, 4, 3, 2), "float32", vdev0)
    )
    _check_inference(
        bb, relax.op.where(cond0, x1, y0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.where(cond0, x2, y0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.where(cond0, x3, y0), relax.TensorStructInfo((6, 5, 4, 3, 2), dtype="")
    )
    _check_inference(bb, relax.op.where(cond0, x4, y0), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.where(cond0, x5, y0), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.where(cond0, x1, y1), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.where(cond0, x2, y1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond0, x3, y1), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.where(cond0, x4, y1), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.where(cond0, x5, y1), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.where(cond0, x2, y2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond0, x3, y2), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.where(cond0, x4, y2), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.where(cond0, x5, y2), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.where(cond0, x3, y3), relax.TensorStructInfo((6, 5, 4, 3, 2), dtype="")
    )
    _check_inference(bb, relax.op.where(cond0, x4, y3), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.where(cond0, x5, y3), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.where(cond0, x4, y4), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(bb, relax.op.where(cond0, x5, y4), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.where(cond0, x5, y5), relax.TensorStructInfo(dtype=""))
    _check_inference(
        bb, relax.op.where(cond1, x0, y0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.where(cond1, x2, y0), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond2, x0, y0), relax.TensorStructInfo(dtype="float32"))


def test_where_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d0 = tir.Var("d", "int64")
    d1 = tir.Var("d", "int64")
    e = tir.Var("e", "int64")
    cond = relax.Var("cond", R.Tensor((a, b, 1, d0, 1), "bool"))
    x0 = relax.Var("x", R.Tensor((b, 1, d0, e), "float32"))
    x1 = relax.Var("x", R.Tensor((b, 1, d1, e), "float32"))
    x2 = relax.Var("x", R.Tensor((b, 1, d0, e)))
    y0 = relax.Var("y", R.Tensor((c, d0, 1), "float32"))
    y1 = relax.Var("y", R.Tensor((c, d0, 1)))

    _check_inference(
        bb, relax.op.where(cond, x0, y0), relax.TensorStructInfo((a, b, c, d0, e), "float32")
    )
    _check_inference(
        bb, relax.op.where(cond, x1, y0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.where(cond, x2, y0), relax.TensorStructInfo((a, b, c, d0, e), dtype="")
    )
    _check_inference(
        bb, relax.op.where(cond, x0, y1), relax.TensorStructInfo((a, b, c, d0, e), dtype="")
    )
    _check_inference(bb, relax.op.where(cond, x1, y1), relax.TensorStructInfo(dtype="", ndim=5))
    _check_inference(
        bb, relax.op.where(cond, x2, y1), relax.TensorStructInfo((a, b, c, d0, e), dtype="")
    )


def test_where_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    scond0 = relax.Var("scond", relax.ShapeStructInfo((6, 5, 1, 3, 1)))
    scond1 = relax.Var("scond", relax.ShapeStructInfo(ndim=5))
    scond2 = relax.Var("scond", relax.ShapeStructInfo())
    sx0 = relax.Var("sx", relax.ShapeStructInfo((5, 1, 3, 2)))
    sx1 = relax.Var("sx", relax.ShapeStructInfo(ndim=4))
    sx2 = relax.Var("sx", relax.ShapeStructInfo())
    sy0 = relax.Var("sy", relax.ShapeStructInfo((4, 3, 1)))
    sy1 = relax.Var("sy", relax.ShapeStructInfo(ndim=3))
    sy2 = relax.Var("sy", relax.ShapeStructInfo())
    s0 = relax.Var("s", relax.ShapeStructInfo((6, 5, 4, 3, 2)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=5))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    cond0 = relax.Var("cond", relax.TensorStructInfo(scond0, "bool"))
    cond1 = relax.Var("cond", relax.TensorStructInfo(scond1, "bool"))
    cond2 = relax.Var("cond", relax.TensorStructInfo(scond2, "bool"))
    cond3 = relax.Var("cond", relax.TensorStructInfo(s0, "bool"))
    cond4 = relax.Var("cond", relax.TensorStructInfo(s1, "bool"))
    cond5 = relax.Var("cond", relax.TensorStructInfo(s2, "bool"))
    x0 = relax.Var("x", relax.TensorStructInfo(sx0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(sx1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(sx2, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x4 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x5 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))
    y0 = relax.Var("y", relax.TensorStructInfo(sy0, "float32"))
    y1 = relax.Var("y", relax.TensorStructInfo(sy1, "float32"))
    y2 = relax.Var("y", relax.TensorStructInfo(sy2, "float32"))
    y3 = relax.Var("y", relax.TensorStructInfo(s0, "float32"))
    y4 = relax.Var("y", relax.TensorStructInfo(s1, "float32"))
    y5 = relax.Var("y", relax.TensorStructInfo(s2, "float32"))

    _check_inference(
        bb, relax.op.where(cond0, x0, y0), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.where(cond0, x0, y1), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.where(cond0, x0, y2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.where(cond0, x1, y1), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.where(cond0, x1, y2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond0, x2, y2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.where(cond1, x1, y1), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.where(cond1, x1, y2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond1, x2, y2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond2, x2, y2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond3, x3, y3), relax.TensorStructInfo(s0, "float32"))
    _check_inference(
        bb, relax.op.where(cond3, x3, y4), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.where(cond3, x4, y3), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(
        bb, relax.op.where(cond4, x3, y3), relax.TensorStructInfo(dtype="float32", ndim=5)
    )
    _check_inference(bb, relax.op.where(cond4, x4, y4), relax.TensorStructInfo(s1, "float32"))
    _check_inference(bb, relax.op.where(cond4, x4, y5), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond4, x5, y4), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond5, x4, y4), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.where(cond5, x5, y5), relax.TensorStructInfo(s2, "float32"))


def test_where_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    cond = relax.Var("cond", R.Tensor((6, 5, 1, 3, 1), "bool"))
    x0 = relax.Var("x", R.Tensor((5, 1, 3, 2), "float16"))
    y0 = relax.Var("y", R.Tensor((4, 3, 1), "float16"))
    x1 = relax.Var("x", R.Tensor((5, 1, 3, 2), "int8"))
    y1 = relax.Var("y", R.Tensor((4, 3, 1), "int8"))
    x2 = relax.Var("x", R.Tensor((5, 1, 3, 2), "int32"))
    y2 = relax.Var("y", R.Tensor((4, 3, 1), "int32"))

    _check_inference(
        bb, relax.op.where(cond, x0, y0), relax.TensorStructInfo((6, 5, 4, 3, 2), "float16")
    )
    _check_inference(
        bb, relax.op.where(cond, x1, y1), relax.TensorStructInfo((6, 5, 4, 3, 2), "int8")
    )
    _check_inference(
        bb, relax.op.where(cond, x2, y2), relax.TensorStructInfo((6, 5, 4, 3, 2), "int32")
    )


def test_where_infer_struct_info_cond_not_boolean():
    bb = relax.BlockBuilder()
    cond0 = relax.Var("cond", R.Tensor((2, 3), "float32"))
    cond1 = relax.Var("cond", R.Tensor((2, 3)))
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond0, x, y))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond1, x, y))


def test_where_infer_struct_info_shape_unequal_const_int():
    bb = relax.BlockBuilder()
    cond0 = relax.Var("cond", R.Tensor((6, 5, 1, 4, 1), "bool"))
    cond1 = relax.Var("cond", R.Tensor((6, 5, 1, 3, 1), "bool"))
    x0 = relax.Var("x", R.Tensor((5, 1, 4, 2), "float32"))
    x1 = relax.Var("x", R.Tensor((5, 1, 3, 2), "float32"))
    y0 = relax.Var("y", R.Tensor((4, 4, 1), "float32"))
    y1 = relax.Var("y", R.Tensor((4, 3, 1), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond0, x1, y1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond1, x0, y1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond1, x1, y0))


def test_where_infer_struct_info_dtype_mismatch():
    bb = relax.BlockBuilder()
    cond = relax.Var("cond", R.Tensor((2, 3), "bool"))
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    y1 = relax.Var("y", R.Tensor((2, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond, x0, y0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond, x1, y1))


def test_where_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    cond0 = relax.Var("cond", relax.ShapeStructInfo((2, 3)))
    cond1 = relax.Var("cond", R.Tensor((2, 3), "bool"))
    x0 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))
    x1 = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", relax.TupleStructInfo([R.Tensor((2, 3), "float32")]))
    y1 = relax.Var("y", R.Tensor((2, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond0, x1, y1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond1, x0, y1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.where(cond1, x1, y0))


(argmax_argmin_op,) = tvm.testing.parameters((relax.op.argmax,), (relax.op.argmin,))


def test_argmax_argmin_infer_struct_info(argmax_argmin_op: Callable):
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    x4 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32", vdev0))

    _check_inference(bb, argmax_argmin_op(x0, axis=1), relax.TensorStructInfo((2, 4, 5), "int64"))
    _check_inference(
        bb, argmax_argmin_op(x4, axis=1), relax.TensorStructInfo((2, 4, 5), "int64", vdev0)
    )
    _check_inference(
        bb,
        argmax_argmin_op(x0, axis=1, keepdims=True),
        relax.TensorStructInfo((2, 1, 4, 5), "int64"),
    )
    _check_inference(bb, argmax_argmin_op(x0, axis=None), relax.TensorStructInfo((), "int64"))
    _check_inference(
        bb,
        argmax_argmin_op(x0, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "int64"),
    )
    _check_inference(
        bb, argmax_argmin_op(x1, axis=1), relax.TensorStructInfo(dtype="int64", ndim=3)
    )
    _check_inference(
        bb,
        argmax_argmin_op(x1, axis=1, keepdims=True),
        relax.TensorStructInfo(dtype="int64", ndim=4),
    )
    _check_inference(bb, argmax_argmin_op(x1, axis=None), relax.TensorStructInfo((), "int64"))
    _check_inference(
        bb,
        argmax_argmin_op(x1, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "int64"),
    )
    _check_inference(bb, argmax_argmin_op(x2, axis=1), relax.TensorStructInfo(dtype="int64"))
    _check_inference(
        bb,
        argmax_argmin_op(x2, axis=1, keepdims=True),
        relax.TensorStructInfo(dtype="int64"),
    )
    _check_inference(bb, argmax_argmin_op(x2, axis=None), relax.TensorStructInfo((), "int64"))
    _check_inference(
        bb,
        argmax_argmin_op(x2, axis=None, keepdims=True),
        relax.TensorStructInfo(dtype="int64"),
    )
    _check_inference(
        bb, argmax_argmin_op(x3, axis=1), relax.TensorStructInfo((2, 4, 5), dtype="int64")
    )
    _check_inference(
        bb,
        argmax_argmin_op(x3, axis=1, keepdims=True),
        relax.TensorStructInfo((2, 1, 4, 5), dtype="int64"),
    )
    _check_inference(bb, argmax_argmin_op(x3, axis=None), relax.TensorStructInfo((), dtype="int64"))
    _check_inference(
        bb,
        argmax_argmin_op(x3, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), dtype="int64"),
    )
    _check_inference(
        bb,
        argmax_argmin_op(x0, axis=1, keepdims=True),
        relax.TensorStructInfo((2, 1, 4, 5), "int64"),
    )
    _check_inference(bb, argmax_argmin_op(x0, axis=-1), relax.TensorStructInfo((2, 3, 4), "int64"))


def test_argmax_argmin_infer_struct_info_shape_symbolic(argmax_argmin_op: Callable):
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "int64"))

    _check_inference(bb, argmax_argmin_op(x, axis=1), relax.TensorStructInfo((a, c, d), "int64"))
    _check_inference(
        bb,
        argmax_argmin_op(x, axis=1, keepdims=True),
        relax.TensorStructInfo((a, 1, c, d), "int64"),
    )
    _check_inference(bb, argmax_argmin_op(x, axis=None), relax.TensorStructInfo((), "int64"))
    _check_inference(
        bb,
        argmax_argmin_op(x, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "int64"),
    )


def test_argmax_argmin_infer_struct_info_shape_var(argmax_argmin_op: Callable):
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "int64"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "int64"))

    _check_inference(bb, argmax_argmin_op(x0), relax.TensorStructInfo((), dtype="int64"))
    _check_inference(
        bb, argmax_argmin_op(x0, keepdims=True), relax.TensorStructInfo((1, 1, 1, 1), dtype="int64")
    )
    _check_inference(
        bb, argmax_argmin_op(x0, axis=2), relax.TensorStructInfo(dtype="int64", ndim=3)
    )
    _check_inference(
        bb,
        argmax_argmin_op(x0, axis=2, keepdims=True),
        relax.TensorStructInfo(dtype="int64", ndim=4),
    )
    _check_inference(bb, argmax_argmin_op(x1), relax.TensorStructInfo((), dtype="int64"))
    _check_inference(bb, argmax_argmin_op(x1, keepdims=True), relax.TensorStructInfo(dtype="int64"))
    _check_inference(bb, argmax_argmin_op(x1, axis=2), relax.TensorStructInfo(dtype="int64"))
    _check_inference(
        bb, argmax_argmin_op(x1, axis=2, keepdims=True), relax.TensorStructInfo(dtype="int64")
    )


def test_argmax_argmin_infer_struct_info_more_input_dtype(argmax_argmin_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))

    _check_inference(bb, argmax_argmin_op(x0), relax.TensorStructInfo((), "int64"))
    _check_inference(bb, argmax_argmin_op(x1), relax.TensorStructInfo((), "int64"))


def test_argmax_argmin_infer_struct_info_axis_out_of_range(argmax_argmin_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int64"))
    x1 = relax.Var("x", R.Tensor("int64", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(argmax_argmin_op(x0, axis=4))
    with pytest.raises(TVMError):
        bb.normalize(argmax_argmin_op(x0, axis=-5))
    with pytest.raises(TVMError):
        bb.normalize(argmax_argmin_op(x1, axis=4))
    with pytest.raises(TVMError):
        bb.normalize(argmax_argmin_op(x1, axis=-5))


def test_argmax_argmin_infer_struct_info_wrong_input_type(argmax_argmin_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "int64")))

    with pytest.raises(TVMError):
        bb.normalize(argmax_argmin_op(x0))
    with pytest.raises(TVMError):
        bb.normalize(argmax_argmin_op(x1))


if __name__ == "__main__":
    tvm.testing.main()
