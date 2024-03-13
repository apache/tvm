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
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    assert relax.op.abs(x).op == Op.get("relax.abs")
    assert relax.op.acos(x).op == Op.get("relax.acos")
    assert relax.op.acosh(x).op == Op.get("relax.acosh")
    assert relax.op.asin(x).op == Op.get("relax.asin")
    assert relax.op.asinh(x).op == Op.get("relax.asinh")
    assert relax.op.atan(x).op == Op.get("relax.atan")
    assert relax.op.atanh(x).op == Op.get("relax.atanh")
    assert relax.op.ceil(x).op == Op.get("relax.ceil")
    assert relax.op.cos(x).op == Op.get("relax.cos")
    assert relax.op.cosh(x).op == Op.get("relax.cosh")
    assert relax.op.exp(x).op == Op.get("relax.exp")
    assert relax.op.floor(x).op == Op.get("relax.floor")
    assert relax.op.isfinite(x).op == Op.get("relax.isfinite")
    assert relax.op.isinf(x).op == Op.get("relax.isinf")
    assert relax.op.isnan(x).op == Op.get("relax.isnan")
    assert relax.op.log(x).op == Op.get("relax.log")
    assert relax.op.negative(x).op == Op.get("relax.negative")
    assert relax.op.round(x).op == Op.get("relax.round")
    assert relax.op.rsqrt(x).op == Op.get("relax.rsqrt")
    assert relax.op.sigmoid(x).op == Op.get("relax.sigmoid")
    assert relax.op.sin(x).op == Op.get("relax.sin")
    assert relax.op.sinh(x).op == Op.get("relax.sinh")
    assert relax.op.square(x).op == Op.get("relax.square")
    assert relax.op.sqrt(x).op == Op.get("relax.sqrt")
    assert relax.op.tan(x).op == Op.get("relax.tan")
    assert relax.op.tanh(x).op == Op.get("relax.tanh")
    assert relax.op.clip(x, 0, 6).op == Op.get("relax.clip")
    assert relax.op.erf(x).op == Op.get("relax.erf")

    x = relax.Var("x", R.Tensor((2, 3), "int32"))
    assert relax.op.bitwise_not(x).op == Op.get("relax.bitwise_not")

    x = relax.Var("x", R.Tensor((2, 3), "bool"))
    assert relax.op.logical_not(x).op == Op.get("relax.logical_not")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


unary_arith_op, require_float_dtype = tvm.testing.parameters(
    (relax.op.abs, False),
    (relax.op.acos, True),
    (relax.op.acosh, True),
    (relax.op.asin, True),
    (relax.op.asinh, True),
    (relax.op.atan, True),
    (relax.op.atanh, True),
    (relax.op.ceil, False),
    (relax.op.cos, True),
    (relax.op.cosh, True),
    (relax.op.exp, True),
    (relax.op.floor, False),
    (relax.op.log, True),
    (relax.op.negative, False),
    (relax.op.round, False),
    (relax.op.rsqrt, True),
    (relax.op.sigmoid, True),
    (relax.op.sign, False),
    (relax.op.sin, True),
    (relax.op.sinh, True),
    (relax.op.square, False),
    (relax.op.sqrt, True),
    (relax.op.tan, True),
    (relax.op.tanh, True),
)


def test_unary_arith_infer_struct_info(unary_arith_op: Callable):
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 3), "float32", vdev0))

    _check_inference(bb, unary_arith_op(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, unary_arith_op(x5), relax.TensorStructInfo((2, 3), "float32", vdev0))
    _check_inference(bb, unary_arith_op(x1), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(bb, unary_arith_op(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, unary_arith_op(x3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, unary_arith_op(x4), relax.TensorStructInfo(dtype=""))


def test_unary_arith_infer_struct_info_shape_symbolic(unary_arith_op: Callable):
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, unary_arith_op(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, unary_arith_op(x1), relax.TensorStructInfo((4, n), "float32"))


def test_unary_arith_infer_struct_info_shape_var(unary_arith_op: Callable):
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(bb, unary_arith_op(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, unary_arith_op(x1), relax.TensorStructInfo(s1, "float32"))


def test_unary_arith_infer_struct_info_more_input_dtype(
    unary_arith_op: Callable, require_float_dtype: bool
):
    if require_float_dtype:
        return

    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(bb, unary_arith_op(x0), relax.TensorStructInfo((2, 3), "float64"))
    _check_inference(bb, unary_arith_op(x1), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, unary_arith_op(x2), relax.TensorStructInfo((2, 3), "int64"))


def test_unary_arith_infer_struct_info_invalid_input_dtype(
    unary_arith_op: Callable, require_float_dtype: bool
):
    if not require_float_dtype:
        return

    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int64"))

    with pytest.raises(TVMError):
        bb.normalize(unary_arith_op(x0))
    with pytest.raises(TVMError):
        bb.normalize(unary_arith_op(x1))


def test_unary_arith_wrong_input_number(unary_arith_op: Callable):
    x = relax.Var("x", R.Tensor((2, 3), "float32"))

    with pytest.raises(TypeError):
        unary_arith_op(x, x)
    with pytest.raises(TypeError):
        unary_arith_op(x, x, x)


def test_unary_arith_infer_struct_info_wrong_input_type(unary_arith_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(unary_arith_op(x0))
    with pytest.raises(TVMError):
        bb.normalize(unary_arith_op(x1))


def test_clip_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((2, 3), "float32", vdev0))

    _check_inference(bb, relax.op.clip(x0, 0, 6), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.clip(x5, 0, 6), relax.TensorStructInfo((2, 3), "float32", vdev0))
    _check_inference(bb, relax.op.clip(x1, 0, 6), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.clip(x2, 0, 6), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.clip(x3, 0, 6), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.clip(x4, 0, 6), relax.TensorStructInfo(dtype=""))

    # Symbolic
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x5 = relax.Var("x", R.Tensor((m, n), "float32"))
    x6 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, relax.op.clip(x5, 0, 6), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.clip(x6, 0, 6), relax.TensorStructInfo((4, n), "float32"))


if __name__ == "__main__":
    tvm.testing.main()
