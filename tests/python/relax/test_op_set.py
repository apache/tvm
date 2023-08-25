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
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    assert relax.op.unique(x).op == Op.get("relax.unique")


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_unique_infer_struct_info():
    bb = relax.BlockBuilder()
    vdev0 = VDevice("llvm")
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4)))
    x4 = relax.Var("x", R.Tensor((2, 3, 4), "float32", vdev0))

    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x4, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="float32", ndim=1, vdevice=vdev0),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=True, return_counts=False, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=False, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=True, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=False, return_counts=False, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=False, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=False, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=False, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=False, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=-2),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, sorted=True, return_index=True, return_inverse=True, return_counts=True, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, sorted=True, return_index=True, return_inverse=True, return_counts=True, axis=1
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=True, return_counts=False, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=True, return_counts=False, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=False, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=False, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x2, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x2, return_index=True, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=False, return_counts=False, axis=1),
        relax.TupleStructInfo(
            [relax.TensorStructInfo(dtype="float32"), relax.TensorStructInfo(dtype="int64", ndim=1)]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=False, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=False, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x3, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorStructInfo(dtype="", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x3, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x3, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c = tir.Var("c", "int64")
    x = relax.Var("x", R.Tensor((a, b, c), "float32"))

    _check_inference(
        bb,
        relax.op.unique(
            x, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=False, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo((2, 3, 4)))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorStructInfo(dtype="float32", ndim=3),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x0, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo(dtype="float32", ndim=1),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=False, return_counts=False, axis=1),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x1, return_index=False, return_inverse=False, return_counts=True, axis=None
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=False, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [relax.TensorStructInfo(dtype="float32"), relax.TensorStructInfo(dtype="int64", ndim=1)]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=False, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3, 4), "int32"))

    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float16", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="int8", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x2, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="int32", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
                relax.TensorStructInfo(dtype="int64", ndim=1),
            ]
        ),
    )


def test_unique_infer_struct_info_input_zero_rank():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(()))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=0))
    x0 = relax.Var("x", R.Tensor((), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=0))
    x2 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x3 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(
        bb,
        relax.op.unique(x0, return_index=True, return_inverse=True, return_counts=True, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((1,), "float32"),
                relax.TensorStructInfo((1,), "int64"),
                relax.TensorStructInfo((1,), "int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(x1, return_index=True, return_inverse=True, return_counts=False, axis=None),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((1,), "float32"),
                relax.TensorStructInfo((1,), "int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x2, return_index=True, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TupleStructInfo(
            [relax.TensorStructInfo((1,), "float32"), relax.TensorStructInfo((1,), "int64")]
        ),
    )
    _check_inference(
        bb,
        relax.op.unique(
            x3, return_index=False, return_inverse=False, return_counts=False, axis=None
        ),
        relax.TensorStructInfo((1,), "float32"),
    )


def test_unique_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    x1 = relax.Var("x", R.Tensor((), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.unique(x0, axis=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.unique(x0, axis=-4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.unique(x1, axis=0))


def test_unique_infer_struct_info_wrong_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3, 4), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.unique(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.unique(x1))


if __name__ == "__main__":
    tvm.testing.main()
