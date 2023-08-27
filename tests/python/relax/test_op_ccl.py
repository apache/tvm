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
    assert relax.op.ccl.allreduce(x).op == Op.get("relax.ccl.allreduce")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_allreduce_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())
    x5 = relax.Var("x", R.Tensor((3, 4)))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.ccl.allreduce(x1), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.ccl.allreduce(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.ccl.allreduce(x3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.ccl.allreduce(x4), relax.TensorStructInfo(dtype=""))
    _check_inference(bb, relax.op.ccl.allreduce(x5), relax.TensorStructInfo((3, 4), dtype=""))


def test_allreduce_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.ccl.allreduce(x1), relax.TensorStructInfo((4, n), "float32"))


def test_allreduce_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.ccl.allreduce(x1), relax.TensorStructInfo(s1, "float32"))


def test_allreduce_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(bb, relax.op.ccl.allreduce(x0), relax.TensorStructInfo((2, 3), "float64"))
    _check_inference(bb, relax.op.ccl.allreduce(x1), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.ccl.allreduce(x2), relax.TensorStructInfo((2, 3), "int64"))


if __name__ == "__main__":
    tvm.testing.main()
