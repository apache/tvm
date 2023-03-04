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
    assert relax.op.nn.relu(x).op == Op.get("relax.nn.relu")
    assert relax.op.nn.gelu(x).op == Op.get("relax.nn.gelu")
    assert relax.op.nn.silu(x).op == Op.get("relax.nn.silu")
    assert relax.op.nn.softmax(x).op == Op.get("relax.nn.softmax")
    assert relax.op.nn.log_softmax(x).op == Op.get("relax.nn.log_softmax")
    assert relax.op.nn.dropout(x).op == Op.get("relax.nn.dropout")

    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    gamma = relax.Var("gamma", R.Tensor((3,), "float32"))
    beta = relax.Var("beta", R.Tensor((3,), "float32"))
    moving_mean = relax.Var("moving_mean", R.Tensor((3,), "float32"))
    moving_var = relax.Var("moving_var", R.Tensor((3,), "float32"))
    assert relax.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1).op == Op.get(
        "relax.nn.batch_norm"
    )
    assert relax.op.nn.layer_norm(x, gamma, beta, axes=1).op == Op.get("relax.nn.layer_norm")

    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    assert relax.op.nn.cross_entropy_with_logits(x, y).op == Op.get(
        "relax.nn.cross_entropy_with_logits"
    )


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_linear_unit_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.nn.relu(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(bb, relax.op.nn.silu(x1), relax.TensorStructInfo(dtype="float32", ndim=3))
    _check_inference(bb, relax.op.nn.gelu(x2), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.nn.relu(x3), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.nn.gelu(x4), relax.TensorStructInfo(dtype=""))


def test_linear_unit_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, relax.op.nn.silu(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.nn.relu(x1), relax.TensorStructInfo((4, n), "float32"))


def test_linear_unit_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(bb, relax.op.nn.gelu(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.nn.relu(x1), relax.TensorStructInfo(s1, "float32"))


def test_linear_unit_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(bb, relax.op.nn.relu(x0), relax.TensorStructInfo((2, 3), "float64"))
    _check_inference(bb, relax.op.nn.relu(x1), relax.TensorStructInfo((2, 3), "int8"))
    _check_inference(bb, relax.op.nn.relu(x2), relax.TensorStructInfo((2, 3), "int64"))


def test_linear_unit_infer_struct_info_invalid_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.gelu(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.silu(x1))


def test_linear_unit_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.gelu(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.silu(x1))


def test_softmax_log_softmax_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())

    _check_inference(bb, relax.op.nn.softmax(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.nn.softmax(x1, axis=0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(bb, relax.op.nn.softmax(x2, axis=1), relax.TensorStructInfo(dtype="float32"))
    _check_inference(bb, relax.op.nn.softmax(x3, axis=-1), relax.TensorStructInfo((2, 3), dtype=""))
    _check_inference(bb, relax.op.nn.softmax(x4, axis=-2), relax.TensorStructInfo(dtype=""))

    _check_inference(bb, relax.op.nn.log_softmax(x0), relax.TensorStructInfo((2, 3), "float32"))
    _check_inference(
        bb, relax.op.nn.log_softmax(x1, axis=0), relax.TensorStructInfo(dtype="float32", ndim=3)
    )
    _check_inference(
        bb, relax.op.nn.log_softmax(x2, axis=1), relax.TensorStructInfo(dtype="float32")
    )
    _check_inference(
        bb, relax.op.nn.log_softmax(x3, axis=-1), relax.TensorStructInfo((2, 3), dtype="")
    )
    _check_inference(bb, relax.op.nn.log_softmax(x4, axis=-2), relax.TensorStructInfo(dtype=""))


def test_softmax_log_softmax_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m, n), "float32"))
    x1 = relax.Var("x", R.Tensor((4, n), "float32"))

    _check_inference(bb, relax.op.nn.softmax(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(bb, relax.op.nn.softmax(x1, axis=0), relax.TensorStructInfo((4, n), "float32"))

    _check_inference(bb, relax.op.nn.log_softmax(x0), relax.TensorStructInfo((m, n), "float32"))
    _check_inference(
        bb, relax.op.nn.log_softmax(x1, axis=0), relax.TensorStructInfo((4, n), "float32")
    )


def test_softmax_log_softmax_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(bb, relax.op.nn.softmax(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.nn.softmax(x1), relax.TensorStructInfo(s1, "float32"))

    _check_inference(bb, relax.op.nn.log_softmax(x0), relax.TensorStructInfo(s0, "float32"))
    _check_inference(bb, relax.op.nn.log_softmax(x1), relax.TensorStructInfo(s1, "float32"))


def test_softmax_log_softmax_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "float64"))

    _check_inference(bb, relax.op.nn.softmax(x0), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.nn.softmax(x1), relax.TensorStructInfo((2, 3), "float64"))

    _check_inference(bb, relax.op.nn.log_softmax(x0), relax.TensorStructInfo((2, 3), "float16"))
    _check_inference(bb, relax.op.nn.log_softmax(x1), relax.TensorStructInfo((2, 3), "float64"))


def test_softmax_log_softmax_infer_struct_info_invalid_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int64"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.softmax(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.softmax(x1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.log_softmax(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.log_softmax(x1))


def test_softmax_log_softmax_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.softmax(x, axis=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.softmax(x, axis=-4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.log_softmax(x, axis=3))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.log_softmax(x, axis=-4))


def test_softmax_log_softmax_wrong_with_multiple_axes():
    x = relax.Var("x", R.Tensor((2, 3, 4), "float32"))

    with pytest.raises(TVMError):
        relax.op.nn.softmax(x, axis=[1, 2])
    with pytest.raises(TVMError):
        relax.op.nn.softmax(x, axis=[-1, -2, -3])
    with pytest.raises(TVMError):
        relax.op.nn.log_softmax(x, axis=[1, 2])
    with pytest.raises(TVMError):
        relax.op.nn.log_softmax(x, axis=[-1, -2, -3])


def test_softmax_log_softmax_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.softmax(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.softmax(x1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.log_softmax(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.log_softmax(x1))


def test_batch_norm_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor(ndim=4))
    x4 = relax.Var("x", R.Tensor())
    gamma0 = relax.Var("gamma", R.Tensor((3,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor("float32", ndim=1))
    gamma2 = relax.Var("gamma", R.Tensor(ndim=1))
    beta0 = relax.Var("beta", R.Tensor((3,), "float32"))
    beta1 = relax.Var("beta", R.Tensor((3,)))
    moving_mean0 = relax.Var("moving_mean", R.Tensor((3,), "float32"))
    moving_mean1 = relax.Var("moving_mean", R.Tensor((3,)))
    moving_var0 = relax.Var("moving_var", R.Tensor((3,), "float32"))
    moving_var1 = relax.Var("moving_var", R.Tensor("float32", ndim=1))
    moving_var2 = relax.Var("moving_var", R.Tensor(ndim=1))

    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma0, beta0, moving_mean0, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 28, 28), "float32"),
                relax.TensorStructInfo((3,), "float32"),
                relax.TensorStructInfo((3,), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma0, beta0, moving_mean0, moving_var0, axis=-3),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 28, 28), "float32"),
                relax.TensorStructInfo((3,), "float32"),
                relax.TensorStructInfo((3,), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x1, gamma0, beta0, moving_mean0, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=4),
                relax.TensorStructInfo((3,), "float32"),
                relax.TensorStructInfo((3,), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma1, beta0, moving_mean0, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 28, 28), "float32"),
                relax.TensorStructInfo((3,), "float32"),
                relax.TensorStructInfo((3,), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma0, beta0, moving_mean0, moving_var1, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 28, 28), "float32"),
                relax.TensorStructInfo((3,), "float32"),
                relax.TensorStructInfo(dtype="float32", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x1, gamma1, beta0, moving_mean0, moving_var1, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=4),
                relax.TensorStructInfo((3,), "float32"),
                relax.TensorStructInfo(dtype="float32", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x2, gamma1, beta0, moving_mean0, moving_var1, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32"),
                relax.TensorStructInfo((3,), "float32"),
                relax.TensorStructInfo(dtype="float32", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x3, gamma2, beta1, moving_mean1, moving_var2, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(ndim=4, dtype=""),
                relax.TensorStructInfo((3,), dtype=""),
                relax.TensorStructInfo(dtype="", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x4, gamma2, beta1, moving_mean1, moving_var2, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype=""),
                relax.TensorStructInfo((3,), dtype=""),
                relax.TensorStructInfo(dtype="", ndim=1),
            ]
        ),
    )


def test_batch_norm_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    c0 = tir.Var("c", "int64")
    c1 = tir.Var("c", "int64")
    h = tir.Var("h", "int64")
    w = tir.Var("w", "int64")
    x0 = relax.Var("x", R.Tensor((n, c0, h, w), "float32"))
    x1 = relax.Var("x", R.Tensor((n, c1, h, w), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    gamma0 = relax.Var("gamma", R.Tensor((c0,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((c1,), "float32"))
    gamma2 = relax.Var("gamma", R.Tensor("float32", ndim=1))
    beta = relax.Var("beta", R.Tensor((c0,), "float32"))
    moving_mean = relax.Var("moving_mean", R.Tensor((c0,), "float32"))
    moving_var0 = relax.Var("moving_var", R.Tensor((c0,), "float32"))
    moving_var1 = relax.Var("moving_var", R.Tensor((c1,), "float32"))
    moving_var2 = relax.Var("moving_var", R.Tensor("float32", ndim=1))

    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma0, beta, moving_mean, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((n, c0, h, w), "float32"),
                relax.TensorStructInfo((c0,), "float32"),
                relax.TensorStructInfo((c0,), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x1, gamma0, beta, moving_mean, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=4),
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="float32", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x2, gamma0, beta, moving_mean, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=4),
                relax.TensorStructInfo((c0,), "float32"),
                relax.TensorStructInfo((c0,), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma1, beta, moving_mean, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=4),
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="float32", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma0, beta, moving_mean, moving_var1, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=4),
                relax.TensorStructInfo(dtype="float32", ndim=1),
                relax.TensorStructInfo(dtype="float32", ndim=1),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma2, beta, moving_mean, moving_var0, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((n, c0, h, w), "float32"),
                relax.TensorStructInfo((c0,), "float32"),
                relax.TensorStructInfo((c0,), "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma0, beta, moving_mean, moving_var2, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((n, c0, h, w), "float32"),
                relax.TensorStructInfo((c0,), "float32"),
                relax.TensorStructInfo(dtype="float32", ndim=1),
            ]
        ),
    )


def test_batch_norm_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s1", relax.ShapeStructInfo())
    s2 = relax.Var("s2", relax.ShapeStructInfo(ndim=1))
    s3 = relax.Var("s3", relax.ShapeStructInfo(ndim=1))
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    gamma = relax.Var("gamma", relax.TensorStructInfo(s2, "float32"))
    beta = relax.Var("beta", relax.TensorStructInfo(s3, "float32"))
    moving_mean = relax.Var("moving_mean", relax.TensorStructInfo(s2, "float32"))
    moving_var = relax.Var("moving_var", relax.TensorStructInfo(s3, "float32"))

    _check_inference(
        bb,
        relax.op.nn.batch_norm(x0, gamma, beta, moving_mean, moving_var, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(s0, "float32"),
                relax.TensorStructInfo(s2, "float32"),
                relax.TensorStructInfo(s3, "float32"),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.batch_norm(x1, gamma, beta, moving_mean, moving_var, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(s1, "float32"),
                relax.TensorStructInfo(s2, "float32"),
                relax.TensorStructInfo(s3, "float32"),
            ]
        ),
    )


def test_batch_norm_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float16"))
    gamma = relax.Var("gamma", R.Tensor((3,), "float16"))
    beta = relax.Var("beta", R.Tensor((3,), "float16"))
    moving_mean = relax.Var("moving_mean", R.Tensor((3,), "float16"))
    moving_var = relax.Var("moving_var", R.Tensor((3,), "float16"))

    _check_inference(
        bb,
        relax.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 3, 28, 28), "float16"),
                relax.TensorStructInfo((3,), "float16"),
                relax.TensorStructInfo((3,), "float16"),
            ]
        ),
    )


def test_batch_norm_infer_struct_info_invalid_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    gamma0 = relax.Var("gamma", R.Tensor((3,), "int8"))
    beta0 = relax.Var("beta", R.Tensor((3,), "int8"))
    moving_mean0 = relax.Var("moving_mean", R.Tensor((3,), "int8"))
    moving_var0 = relax.Var("moving_var", R.Tensor((3,), "int8"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int32"))
    gamma1 = relax.Var("gamma", R.Tensor((3,), "int32"))
    beta1 = relax.Var("beta", R.Tensor((3,), "int32"))
    moving_mean1 = relax.Var("moving_mean", R.Tensor((3,), "int32"))
    moving_var1 = relax.Var("moving_var", R.Tensor((3,), "int32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x0, gamma0, beta0, moving_mean0, moving_var0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x1, gamma1, beta1, moving_mean1, moving_var1, axis=1))


def test_batch_norm_infer_struct_info_axis_out_of_range():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    gamma = relax.Var("gamma", R.Tensor((3,), "float32"))
    beta = relax.Var("beta", R.Tensor((3,), "float32"))
    moving_mean = relax.Var("moving_mean", R.Tensor((3,), "float32"))
    moving_var = relax.Var("moving_var", R.Tensor((3,), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=4))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x, gamma, beta, moving_mean, moving_var, axis=-5))


def test_batch_norm_infer_struct_info_dtype_mismatch():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 28, 28), "int8"))
    gamma0 = relax.Var("gamma", R.Tensor((3,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((3,)))
    beta = relax.Var("beta", R.Tensor((3,), "float32"))
    moving_mean = relax.Var("moving_mean", R.Tensor((3,), "float32"))
    moving_var0 = relax.Var("moving_var", R.Tensor((3,), "float32"))
    moving_var1 = relax.Var("moving_var", R.Tensor((3,), "float16"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x1, gamma0, beta, moving_mean, moving_var0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x0, gamma1, beta, moving_mean, moving_var0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x0, gamma0, beta, moving_mean, moving_var1, axis=1))


def test_batch_norm_infer_struct_info_ndim_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((3,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((3, 1), "float32"))
    beta = relax.Var("beta", R.Tensor((3,), "float32"))
    moving_mean = relax.Var("moving_mean", R.Tensor((3,), "float32"))
    moving_var0 = relax.Var("moving_var", R.Tensor((3,), "float32"))
    moving_var1 = relax.Var("moving_var", R.Tensor((1, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x, gamma1, beta, moving_mean, moving_var0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x, gamma0, beta, moving_mean, moving_var1, axis=1))


def test_batch_norm_infer_struct_info_shape_mismatch():
    bb = relax.BlockBuilder()
    c = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", R.Tensor((2, c, 28, 28), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((3,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((4,), "float32"))
    gamma2 = relax.Var("gamma", R.Tensor((c + 2,), "float32"))
    beta0 = relax.Var("beta", R.Tensor((3,), "float32"))
    beta1 = relax.Var("beta", R.Tensor((c,), "float32"))
    moving_mean0 = relax.Var("moving_mean", R.Tensor((3,), "float32"))
    moving_mean1 = relax.Var("moving_mean", R.Tensor((c,), "float32"))
    moving_var0 = relax.Var("moving_var", R.Tensor((3,), "float32"))
    moving_var1 = relax.Var("moving_var", R.Tensor((4,), "float32"))
    moving_var2 = relax.Var("moving_var", R.Tensor((c,), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x0, gamma1, beta0, moving_mean0, moving_var0, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x0, gamma0, beta0, moving_mean0, moving_var1, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x1, gamma2, beta1, moving_mean1, moving_var2, axis=1))


def test_batch_norm_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 28, 28), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((2, 3, 28, 28)))
    gamma0 = relax.Var("gamma", R.Tensor((3,), "float32"))
    gamma1 = relax.Var("gamma", relax.FuncStructInfo([], R.Tensor((3,), "float32")))
    beta = relax.Var("beta", R.Tensor((3,), "float32"))
    moving_mean = relax.Var("moving_mean", R.Tensor((3,), "float32"))
    moving_var = relax.Var("moving_var", R.Tensor((3,), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x1, gamma0, beta, moving_mean, moving_var, axis=1))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.batch_norm(x0, gamma1, beta, moving_mean, moving_var, axis=1))


def test_layer_norm_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor("float32", ndim=2))
    gamma2 = relax.Var("gamma", R.Tensor((4, 5)))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "float32"))
    beta1 = relax.Var("beta", R.Tensor((4, 5)))

    _check_inference(
        bb,
        relax.op.nn.layer_norm(x0, gamma0, beta0, axes=[-2, -1]),
        relax.TensorStructInfo((2, 3, 4, 5), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x0, gamma0, beta0, axes=[-2, 3]),
        relax.TensorStructInfo((2, 3, 4, 5), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x1, gamma0, beta0, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x2, gamma0, beta0, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x0, gamma1, beta0, axes=[-2, -1]),
        relax.TensorStructInfo((2, 3, 4, 5), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x3, gamma2, beta1, axes=[-2, -1]),
        relax.TensorStructInfo((2, 3, 4, 5), dtype=""),
    )


def test_layer_norm_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c0 = tir.Var("c", "int64")
    c1 = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((n, a, b, c0), "float32"))
    x1 = relax.Var("x", R.Tensor((n, a, b, c1), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    gamma0 = relax.Var("gamma", R.Tensor((b, c0), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((b, c1), "float32"))
    beta = relax.Var("beta", R.Tensor((b, c0), "float32"))

    _check_inference(
        bb,
        relax.op.nn.layer_norm(x0, gamma0, beta, axes=[-2, -1]),
        relax.TensorStructInfo((n, a, b, c0), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x1, gamma0, beta, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x0, gamma1, beta, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x2, gamma0, beta, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x2, gamma1, beta, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )


def test_layer_norm_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s1", relax.ShapeStructInfo())
    s2 = relax.Var("s2", relax.ShapeStructInfo(ndim=2))
    s3 = relax.Var("s3", relax.ShapeStructInfo(ndim=2))
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    gamma = relax.Var("gamma", relax.TensorStructInfo(s2, "float32"))
    beta = relax.Var("beta", relax.TensorStructInfo(s3, "float32"))

    _check_inference(
        bb,
        relax.op.nn.layer_norm(x0, gamma, beta, axes=[2, 3]),
        relax.TensorStructInfo(s0, "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x1, gamma, beta, axes=[2, 3]),
        relax.TensorStructInfo(s1, "float32"),
    )


def test_layer_norm_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "float16"))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float64"))
    gamma1 = relax.Var("gamma", R.Tensor((4, 5), "float64"))
    beta1 = relax.Var("beta", R.Tensor((4, 5), "float64"))

    _check_inference(
        bb,
        relax.op.nn.layer_norm(x0, gamma0, beta0, axes=[-2, -1]),
        relax.TensorStructInfo((2, 3, 4, 5), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.layer_norm(x1, gamma1, beta1, axes=[-2, -1]),
        relax.TensorStructInfo((2, 3, 4, 5), "float64"),
    )


def test_layer_norm_infer_struct_info_invalid_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "int8"))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "int8"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int32"))
    gamma1 = relax.Var("gamma", R.Tensor((4, 5), "int32"))
    beta1 = relax.Var("beta", R.Tensor((4, 5), "int32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x0, gamma0, beta0, axes=[-2, -1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x1, gamma1, beta1, axes=[-2, -1]))


def test_layer_norm_infer_struct_info_axis_out_of_range_and_repetitive():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    gamma = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    beta = relax.Var("beta", R.Tensor((4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x, gamma, beta, axes=[3, 4]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x, gamma, beta, axes=[3, -1]))


def test_layer_norm_infer_struct_info_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((4, 5), "int8"))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "float32"))
    beta1 = relax.Var("beta", R.Tensor((4, 5)))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x, gamma1, beta0, axes=[-2, -1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x, gamma0, beta1, axes=[-2, -1]))


def test_layer_norm_infer_struct_info_ndim_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((4,), "float32"))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "float32"))
    beta1 = relax.Var("beta", R.Tensor((3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x, gamma1, beta0, axes=[-2, -1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x, gamma0, beta1, axes=[-2, -1]))


def test_layer_norm_infer_struct_info_shape_mismatch():
    bb = relax.BlockBuilder()
    c0 = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, c0), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((4, 6), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((4, c0), "float32"))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "float32"))
    beta1 = relax.Var("beta", R.Tensor((4, c0 - 2), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x0, gamma0, beta0, axes=[-2, -1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x1, gamma1, beta1, axes=[-2, -1]))


def test_layer_norm_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    gamma1 = relax.Var("gamma", relax.FuncStructInfo([], R.Tensor((4, 5), "float32")))
    beta = relax.Var("beta", R.Tensor((4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x1, gamma0, beta, axes=[-2, -1]))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.layer_norm(x0, gamma1, beta, axes=[-2, -1]))


def test_group_norm_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=4))
    x2 = relax.Var("x", R.Tensor("float32"))
    x3 = relax.Var("x", R.Tensor((2, 3, 4, 5)))
    gamma0 = relax.Var("gamma", R.Tensor((4,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor("float32", ndim=1))
    gamma2 = relax.Var("gamma", R.Tensor((4,)))
    beta0 = relax.Var("beta", R.Tensor((4,), "float32"))
    beta1 = relax.Var("beta", R.Tensor((4,)))

    _check_inference(
        bb,
        relax.op.nn.group_norm(x0, gamma0, beta0, num_groups=2, channel_axis=-2, axes=[-1]),
        relax.TensorStructInfo((2, 3, 4, 5), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x0, gamma0, beta0, num_groups=2, channel_axis=-2, axes=[-1]),
        relax.TensorStructInfo((2, 3, 4, 5), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x1, gamma0, beta0, num_groups=2, channel_axis=-2, axes=[-1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x2, gamma0, beta0, num_groups=2, channel_axis=-2, axes=[-1]),
        relax.TensorStructInfo(dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x0, gamma1, beta0, num_groups=2, channel_axis=-2, axes=[-1]),
        relax.TensorStructInfo((2, 3, 4, 5), dtype="float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x3, gamma2, beta1, num_groups=2, channel_axis=-2, axes=[-1]),
        relax.TensorStructInfo((2, 3, 4, 5), dtype=""),
    )


def test_group_norm_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    n = tir.Var("n", "int64")
    a = tir.Var("a", "int64")
    b = tir.Var("b", "int64")
    c0 = tir.Var("c", "int64")
    c1 = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((n, a, b, c0), "float32"))
    x1 = relax.Var("x", R.Tensor((n, a, b, c1), "float32"))
    x2 = relax.Var("x", R.Tensor("float32", ndim=4))
    gamma0 = relax.Var("gamma", R.Tensor((a,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((a,), "float32"))
    beta = relax.Var("beta", R.Tensor((a,), "float32"))

    _check_inference(
        bb,
        relax.op.nn.group_norm(x0, gamma0, beta, num_groups=2, channel_axis=-3, axes=[-2, -1]),
        relax.TensorStructInfo((n, a, b, c0), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x1, gamma0, beta, num_groups=2, channel_axis=-3, axes=[-2, -1]),
        relax.TensorStructInfo((n, a, b, c1), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x0, gamma1, beta, num_groups=2, channel_axis=-3, axes=[-2, -1]),
        relax.TensorStructInfo((n, a, b, c0), "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x2, gamma0, beta, num_groups=2, channel_axis=-3, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x2, gamma1, beta, num_groups=2, channel_axis=-3, axes=[-2, -1]),
        relax.TensorStructInfo(dtype="float32", ndim=4),
    )


def test_group_norm_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s0", relax.ShapeStructInfo(ndim=4))
    s1 = relax.Var("s1", relax.ShapeStructInfo())
    s2 = relax.Var("s2", relax.ShapeStructInfo(ndim=1))
    s3 = relax.Var("s3", relax.ShapeStructInfo(ndim=1))
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))
    gamma = relax.Var("gamma", relax.TensorStructInfo(s2, "float32"))
    beta = relax.Var("beta", relax.TensorStructInfo(s3, "float32"))

    _check_inference(
        bb,
        relax.op.nn.group_norm(x0, gamma, beta, num_groups=2, channel_axis=-2, axes=[1, 3]),
        relax.TensorStructInfo(s0, "float32"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x1, gamma, beta, num_groups=2, channel_axis=-2, axes=[1, 3]),
        relax.TensorStructInfo(s1, "float32"),
    )


def test_group_norm_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float16"))
    gamma0 = relax.Var("gamma", R.Tensor((3,), "float16"))
    beta0 = relax.Var("beta", R.Tensor((3,), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float64"))
    gamma1 = relax.Var("gamma", R.Tensor((3,), "float64"))
    beta1 = relax.Var("beta", R.Tensor((3,), "float64"))

    _check_inference(
        bb,
        relax.op.nn.group_norm(x0, gamma0, beta0, num_groups=3, channel_axis=1, axes=[-2, -1]),
        relax.TensorStructInfo((2, 3, 4, 5), "float16"),
    )
    _check_inference(
        bb,
        relax.op.nn.group_norm(x1, gamma1, beta1, num_groups=3, channel_axis=1, axes=[-2, -1]),
        relax.TensorStructInfo((2, 3, 4, 5), "float64"),
    )


def test_group_norm_infer_struct_info_invalid_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int8"))
    gamma0 = relax.Var("gamma", R.Tensor((4,), "int8"))
    beta0 = relax.Var("beta", R.Tensor((4,), "int8"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, 5), "int32"))
    gamma1 = relax.Var("gamma", R.Tensor((4,), "int32"))
    beta1 = relax.Var("beta", R.Tensor((4,), "int32"))

    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x0, gamma0, beta0, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )
    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x1, gamma1, beta1, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )


def test_group_norm_infer_struct_info_axis_out_of_range_and_repetitive():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    gamma = relax.Var("gamma", R.Tensor((4,), "float32"))
    beta = relax.Var("beta", R.Tensor((4,), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x, gamma, beta, num_groups=2, channel_axis=-2, axes=[3, 4])
        )
    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x, gamma, beta, num_groups=2, channel_axis=-2, axes=[3, -1])
        )


def test_group_norm_infer_struct_info_dtype_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((4,), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((4,), "int8"))
    beta0 = relax.Var("beta", R.Tensor((4,), "float32"))
    beta1 = relax.Var("beta", R.Tensor((4,)))

    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x, gamma1, beta0, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )
    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x, gamma0, beta1, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )


def test_group_norm_infer_struct_info_ndim_mismatch():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((4,), "float32"))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "float32"))
    beta1 = relax.Var("beta", R.Tensor((3, 4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x, gamma1, beta0, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )
    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x, gamma0, beta1, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )


def test_group_norm_infer_struct_info_shape_mismatch():
    bb = relax.BlockBuilder()
    c0 = tir.Var("c", "int64")
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4, c0), "float32"))
    gamma0 = relax.Var("gamma", R.Tensor((4, 6), "float32"))
    gamma1 = relax.Var("gamma", R.Tensor((4, c0), "float32"))
    beta0 = relax.Var("beta", R.Tensor((4, 5), "float32"))
    beta1 = relax.Var("beta", R.Tensor((4, c0 - 2), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x0, gamma0, beta0, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )
    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x1, gamma1, beta1, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )


def test_group_norm_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 4, 5), "float32"))
    x1 = relax.Var("x", relax.ShapeStructInfo((2, 3, 4, 5)))
    gamma0 = relax.Var("gamma", R.Tensor((4, 5), "float32"))
    gamma1 = relax.Var("gamma", relax.FuncStructInfo([], R.Tensor((4, 5), "float32")))
    beta = relax.Var("beta", R.Tensor((4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x1, gamma0, beta, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )
    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.nn.group_norm(x0, gamma1, beta, num_groups=2, channel_axis=-2, axes=[-2, -1])
        )


def test_dropout_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor("float32", ndim=3))
    x2 = relax.Var("x", R.Tensor("float32", ndim=-1))
    x3 = relax.Var("x", R.Tensor((2, 3)))
    x4 = relax.Var("x", R.Tensor())

    _check_inference(
        bb,
        relax.op.nn.dropout(x0),
        relax.TupleStructInfo(
            [relax.TensorStructInfo((2, 3), "float32"), relax.TensorStructInfo((2, 3), "float32")]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.dropout(x1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(dtype="float32", ndim=3),
                relax.TensorStructInfo(dtype="float32", ndim=3),
            ]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.dropout(x2),
        relax.TupleStructInfo(
            [relax.TensorStructInfo(dtype="float32"), relax.TensorStructInfo(dtype="float32")]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.dropout(x3),
        relax.TupleStructInfo(
            [relax.TensorStructInfo((2, 3), dtype=""), relax.TensorStructInfo((2, 3), dtype="")]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.dropout(x4),
        relax.TupleStructInfo([relax.TensorStructInfo(dtype=""), relax.TensorStructInfo(dtype="")]),
    )


def test_dropout_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x = relax.Var("x", R.Tensor((m, n), "float32"))

    _check_inference(
        bb,
        relax.op.nn.dropout(x),
        relax.TupleStructInfo(
            [relax.TensorStructInfo((m, n), "float32"), relax.TensorStructInfo((m, n), "float32")]
        ),
    )


def test_dropout_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s1 = relax.Var("s", relax.ShapeStructInfo())
    x0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    x1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(
        bb,
        relax.op.nn.dropout(x0),
        relax.TupleStructInfo(
            [relax.TensorStructInfo(s0, "float32"), relax.TensorStructInfo(s0, "float32")]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.dropout(x1),
        relax.TupleStructInfo(
            [relax.TensorStructInfo(s1, "float32"), relax.TensorStructInfo(s1, "float32")]
        ),
    )


def test_dropout_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float64"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int64"))

    _check_inference(
        bb,
        relax.op.nn.dropout(x0),
        relax.TupleStructInfo(
            [relax.TensorStructInfo((2, 3), "float64"), relax.TensorStructInfo((2, 3), "float64")]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.dropout(x1),
        relax.TupleStructInfo(
            [relax.TensorStructInfo((2, 3), "int8"), relax.TensorStructInfo((2, 3), "int8")]
        ),
    )
    _check_inference(
        bb,
        relax.op.nn.dropout(x2),
        relax.TupleStructInfo(
            [relax.TensorStructInfo((2, 3), "int64"), relax.TensorStructInfo((2, 3), "int64")]
        ),
    )


def test_dropout_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.dropout(x0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.dropout(x1))


def test_cross_entropy_infer_struct_info():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=2))
    y2 = relax.Var("y", R.Tensor((2, 3)))
    y3 = relax.Var("y", R.Tensor(ndim=2))

    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x, y0), relax.TensorStructInfo((), "float32")
    )
    _check_inference(
        bb,
        relax.op.nn.cross_entropy_with_logits(x, y1),
        relax.TensorStructInfo((), dtype="float32"),
    )
    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x, y2), relax.TensorStructInfo((), dtype="")
    )
    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x, y3), relax.TensorStructInfo((), dtype="")
    )


def test_cross_entropy_infer_struct_info_shape_symbolic():
    bb = relax.BlockBuilder()
    m0 = tir.Var("m", "int64")
    m1 = tir.Var("m", "int64")
    n = tir.Var("n", "int64")
    x0 = relax.Var("x", R.Tensor((m0, n), "float32"))
    x1 = relax.Var("x", R.Tensor((m1, n), "float32"))
    y = relax.Var("y", R.Tensor((m0, n), "float32"))

    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x0, y), relax.TensorStructInfo((), "float32")
    )
    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x1, y), relax.TensorStructInfo((), "float32")
    )


def test_cross_entropy_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    s0 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    s1 = relax.Var("s", relax.ShapeStructInfo(ndim=2))
    x = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    y0 = relax.Var("x", relax.TensorStructInfo(s0, "float32"))
    y1 = relax.Var("x", relax.TensorStructInfo(s1, "float32"))

    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x, y0), relax.TensorStructInfo((), "float32")
    )
    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x, y1), relax.TensorStructInfo((), "float32")
    )


def test_cross_entropy_infer_struct_info_more_input_dtype():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float16"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float16"))
    x1 = relax.Var("x", R.Tensor((2, 3), "int8"))
    y1 = relax.Var("y", R.Tensor((2, 3), "int8"))
    x2 = relax.Var("x", R.Tensor((2, 3), "int32"))
    y2 = relax.Var("y", R.Tensor((2, 3), "int32"))

    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x0, y0), relax.TensorStructInfo((), "float16")
    )
    _check_inference(
        bb, relax.op.nn.cross_entropy_with_logits(x1, y1), relax.TensorStructInfo((), "int8")
    )


def test_cross_entropy_infer_struct_info_wrong_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 4), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 3), "float32"))
    y1 = relax.Var("y", R.Tensor("float32", ndim=4))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.cross_entropy_with_logits(x1, y0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.cross_entropy_with_logits(x0, y1))


def test_cross_entropy_infer_struct_info_shape_mismatch():
    bb = relax.BlockBuilder()
    m = tir.Var("m", "int64")
    x0 = relax.Var("x", R.Tensor((2, 3), "float32"))
    y0 = relax.Var("y", R.Tensor((2, 4), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.cross_entropy_with_logits(x0, y0))


def test_cross_entropy_infer_struct_info_wrong_input_type():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", relax.ShapeStructInfo((2, 3)))
    x1 = relax.Var("x", relax.FuncStructInfo([], R.Tensor((2, 3), "float32")))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.cross_entropy_with_logits(x0, y))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.nn.cross_entropy_with_logits(x1, y))


if __name__ == "__main__":
    tvm.testing.main()
