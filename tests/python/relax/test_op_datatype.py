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
import numpy as np  # type: ignore


import pytest
import tvm
import tvm.testing
from tvm import relax, tir
from tvm import TVMError
from tvm.ir import Op
from tvm.script import relax as R


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    c = relax.Constant(tvm.nd.array(np.array([1, 2, 3], dtype="float16")))
    assert relax.op.astype(x, "float16").op == Op.get("relax.astype")
    assert relax.op.wrap_param(c, "float32").op == Op.get("relax.wrap_param")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_astype_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=2))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor(ndim=2))
    x5 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.astype(x0, "float16"), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(
        bb, relax.op.astype(x1, "float16"), relax.TensorStructInfo(dtype="float16", ndim=2)
    )
    _check_inference(bb, relax.op.astype(x2, "float16"), relax.TensorStructInfo(dtype="float16"))
    _check_inference(bb, relax.op.astype(x3, "float16"), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(
        bb, relax.op.astype(x4, "float16"), relax.TensorStructInfo(dtype="float16", ndim=2)
    )
    _check_inference(bb, relax.op.astype(x5, "float16"), relax.TensorStructInfo(dtype="float16"))


def test_astype_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m, n)))

    _check_inference(bb, relax.op.astype(x0, "float16"), relax.TensorStructInfo((m, n), "float16"))
    _check_inference(bb, relax.op.astype(x1, "float16"), relax.TensorStructInfo((m, n), "float16"))


def test_astype_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3)))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s2 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    x2 = relax.Var("x", relax.TensorStructInfo(s2, "float32"))

    _check_inference(bb, relax.op.astype(x0, "float16"), relax.TensorStructInfo(s0, "float16"))
    _check_inference(bb, relax.op.astype(x1, "float16"), relax.TensorStructInfo(s1, "float16"))
    _check_inference(bb, relax.op.astype(x2, "float16"), relax.TensorStructInfo(s2, "float16"))


def test_astype_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int32"))

    _check_inference(bb, relax.op.astype(x0, "float32"), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.astype(x1, "int32"), relax.TensorStructInfo((2, 3), "int32"))
    _check_inference(bb, relax.op.astype(x2, "int8"), relax.TensorStructInfo((2, 3), "int8"))


def test_astype_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.astype(x0, "float16"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.astype(x1, "float16"))


def test_wrap_param_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Constant(tvm.nd.array(np.zeros([1, 2, 3], dtype="float16")))
    x1 = relax.Constant(tvm.nd.array(np.zeros([1, 2, 3], dtype="int8")))
    _check_inference(
        bb, relax.op.wrap_param(x0, "float32"), relax.TensorStructInfo((1, 2, 3), "float32")
    )
    _check_inference(
        bb, relax.op.wrap_param(x1, "int32"), relax.TensorStructInfo((1, 2, 3), "int32")
    )


if __name__ == "__main__":
    tvm.testing.main()
