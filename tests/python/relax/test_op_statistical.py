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
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    assert relax.op.max(x).op == Op.get("relax.max")
    assert relax.op.mean(x).op == Op.get("relax.mean")
    assert relax.op.min(x).op == Op.get("relax.min")
    assert relax.op.prod(x).op == Op.get("relax.prod")
    assert relax.op.std(x).op == Op.get("relax.std")
    assert relax.op.sum(x).op == Op.get("relax.sum")
    assert relax.op.variance(x).op == Op.get("relax.variance")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_statistical_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    x4 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32", vdev0))

    _check_inference(bb, relax.op.sum(x0, axis=[1, 2]), relax.TensorStructInfo((2, 5), "float32"))
    _check_inference(
        bb, relax.op.sum(x4, axis=[1, 2]), relax.TensorStructInfo((2, 5), "float32", vdev0)
    )
    _check_inference(
        bb,
        relax.op.sum(x0, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((2, 1, 1, 5), "float32"),
    )
    _check_inference(bb, relax.op.sum(x0, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.sum(x0, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "float32"),
    )
    _check_inference(
        bb, relax.op.mean(x1, axis=[1, 2]), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb,
        relax.op.mean(x1, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.mean(x1, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.mean(x1, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "float32"),
    )
    _check_inference(
        bb, relax.op.variance(x2, axis=[1, 2]), relax.TensorStructInfo(dtype="float32")
    )
    _check_inference(
        bb,
        relax.op.variance(x2, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(bb, relax.op.variance(x2, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.variance(x2, axis=None, keepdims=True),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(bb, relax.op.max(x3, axis=[1, 2]), relax.TensorStructInfo((2, 5), dtype=""))
    _check_inference(
        bb,
        relax.op.max(x3, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((2, 1, 1, 5), dtype=""),
    )
    _check_inference(bb, relax.op.max(x3, axis=None), relax.TensorStructInfo((), dtype=""))
    _check_inference(
        bb,
        relax.op.max(x3, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), dtype=""),
    )
    _check_inference(bb, relax.op.prod(x0, axis=[1, 2]), relax.TensorStructInfo((2, 5), "float32"))
    _check_inference(
        bb,
        relax.op.prod(x0, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((2, 1, 1, 5), "float32"),
    )
    _check_inference(bb, relax.op.std(x0, axis=[1, 2]), relax.TensorStructInfo((2, 5), "float32"))
    _check_inference(
        bb,
        relax.op.std(x0, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((2, 1, 1, 5), "float32"),
    )
    _check_inference(bb, relax.op.sum(x0, axis=[-1, -4]), relax.TensorStructInfo((3, 4), "float32"))
    _check_inference(bb, relax.op.sum(x0, axis=[]), relax.TensorStructInfo((2, 3, 4, 5), "float32"))


def test_statistical_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    d = tir.Var("d", "int64")
    x = relax.Var("x", R.Tensor((a, b, c, d), "float32"))

    _check_inference(bb, relax.op.min(x, axis=[1, 2]), relax.TensorStructInfo((a, d), "float32"))
    _check_inference(
        bb,
        relax.op.min(x, axis=[1, 2], keepdims=True),
        relax.TensorStructInfo((a, 1, 1, d), "float32"),
    )
    _check_inference(bb, relax.op.min(x, axis=None), relax.TensorStructInfo((), "float32"))
    _check_inference(
        bb,
        relax.op.min(x, axis=None, keepdims=True),
        relax.TensorStructInfo((1, 1, 1, 1), "float32"),
    )


def test_statistical_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(bb, relax.op.max(x0), relax.TensorStructInfo((), dtype="float32"))
    _check_inference(
        bb, relax.op.max(x0, keepdims=True), relax.TensorStructInfo((1, 1, 1, 1), dtype="float32")
    )
    _check_inference(
        bb, relax.op.max(x0, axis=[2, 3]), relax.TensorStructInfo(dtype="float32", ndim=2)
    )
    _check_inference(
        bb,
        relax.op.max(x0, axis=[2, 3], keepdims=True),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(bb, relax.op.max(x1), relax.TensorStructInfo((), dtype="float32"))
    _check_inference(bb, relax.op.max(x1, keepdims=True), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.max(x1, axis=[2, 3]), relax.TensorStructInfo(dtype="float32"))
    _check_inference(
        bb, relax.op.max(x1, axis=[2, 3], keepdims=True), relax.TensorStructInfo(dtype="float32")
    )


def test_statistical_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))

    _check_inference(bb, relax.op.sum(x0), relax.TensorStructInfo((), "float16"))
    _check_inference(bb, relax.op.sum(x1), relax.TensorStructInfo((), "int8"))


def test_statistical_infer_struct_info_axis_out_of_range_repetitive():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x0, axis=[4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x1, axis=[3, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x0, axis=[-1, 3]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x1, axis=[-4, -4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.mean(x0, axis=[-5]))


def test_statistical_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.variance(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.variance(x1))


(scan_op,) = tvm.testing.parameters(
    (relax.op.cumprod,),
    (relax.op.cumsum,),
)


def test_scan_op_infer_struct_info(scan_op: Callable):
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 10, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 10, 4)))
    x4 = relax.Var("x", R.Tensor(ndim=3))
    x5 = relax.Var("x", R.Tensor())
    x6 = relax.Var("x", R.Tensor((2, 10, 4), "float32", vdev0))

    _check_inference(bb, scan_op(x0, axis=1), relax.TensorStructInfo((2, 10, 4), "float32"))
    _check_inference(bb, scan_op(x6, axis=1), relax.TensorStructInfo((2, 10, 4), "float32", vdev0))
    _check_inference(bb, scan_op(x1, axis=1), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(bb, scan_op(x2, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, scan_op(x3, axis=1), relax.TensorStructInfo((2, 10, 4), dtype=""))
    _check_inference(bb, scan_op(x4, axis=1), relax.TensorStructInfo(dtype="", ndim=3))
    _check_inference(bb, scan_op(x5, axis=1), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, scan_op(x0), relax.TensorStructInfo((80,), "float32"))
    _check_inference(
        bb, scan_op(x0, axis=1, dtype="int32"), relax.TensorStructInfo((2, 10, 4), "int32")
    )


def test_scan_op_infer_struct_info_shape_symbolic(scan_op: Callable):
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(bb, scan_op(x, axis=1), relax.TensorStructInfo((a, b, c), "float32"))
    _check_inference(bb, scan_op(x), relax.TensorStructInfo((a * b * c,), "float32"))


def test_scan_op_infer_struct_info_more_input_dtype(scan_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))

    _check_inference(bb, scan_op(x0, axis=1), relax.TensorStructInfo((2, 3, 4), "float16"))
    _check_inference(bb, scan_op(x1, axis=1), relax.TensorStructInfo((2, 3, 4), "int8"))


def test_scan_op_wrong_input_number(scan_op: Callable):
    x = relax.Var("x", R.Tensor((3, 4, 5), "float32"))
    y = relax.Var("y", R.Tensor((2, 3, 4), "float32"))

    with pytest.raises(TVMError):
        scan_op(x, y)


def test_scan_opinfer_struct_info_wrong_input_type(scan_op: Callable):
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4, 5), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(scan_op(x0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(scan_op(x1, axis=1))


if __name__ == "__main__":
    tvm.testing.main()
