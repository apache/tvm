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
import tvm
import tvm.testing
from tvm import relax, tir
from tvm.ir import Op
from tvm.script import relax as R


def test_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((2, 3), "uint8"))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "int8"))
    assert relax.op.quantize(x, s, zp, 1, "int8").op == Op.get("relax.quantize")
    assert relax.op.dequantize(dx, s, zp, 1, "float32").op == Op.get("relax.dequantize")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_qdq_op_infer_struct_info():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((2, 3), "uint8"))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "int8"))
    _check_inference(
        bb, relax.op.quantize(x, s, zp, 1, "int8"), relax.TensorStructInfo((2, 3), "int8")
    )
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 1, "float32"),
        relax.TensorStructInfo((2, 3), "float32"),
    )


def test_qdq_op_infer_struct_info_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor((n, 3), "float32"))
    dx = relax.Var("dx", R.Tensor((n, 3), "int8"))
    s = relax.Var("s", R.Tensor([3], "float32"))
    zp = relax.Var("zp", R.Tensor([3], "int8"))
    _check_inference(
        bb, relax.op.quantize(x, s, zp, 1, "int8"), relax.TensorStructInfo((n, 3), "int8")
    )
    _check_inference(
        bb,
        relax.op.dequantize(dx, s, zp, 1, "float32"),
        relax.TensorStructInfo((n, 3), "float32"),
    )


if __name__ == "__main__":
    tvm.testing.main()
