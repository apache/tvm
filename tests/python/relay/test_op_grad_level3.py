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
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay.testing import check_grad, run_infer_type, run_opt_pass, _np_randn_from_type
from tvm.relay.transform import gradient
import tvm.testing

executor_kind = tvm.testing.parameter("debug")


@tvm.testing.uses_gpu
def test_clip(executor_kind):
    for dtype in ("float32", "float64"):
        ref = lambda x: np.where(
            x > 10.0, np.zeros_like(x), np.where(x < 1.0, np.zeros_like(x), np.ones_like(x))
        )
        x = relay.var("x", relay.TensorType((10, 4), dtype))
        y = tvm.relay.clip(x, 1.0, 10.0)

        data = np.random.rand(10, 4).astype(dtype) * 11.0
        ref_grad = ref(data)
        fwd_func = relay.Function([x], y)
        fwd_func = run_infer_type(fwd_func)
        bwd_func = run_infer_type(gradient(fwd_func))

        for target, dev in tvm.testing.enabled_targets():
            op_res, (op_grad,) = relay.create_executor(
                executor_kind, device=dev, target=target
            ).evaluate(bwd_func)(data)
            np.testing.assert_allclose(op_grad.numpy(), ref_grad, rtol=0.01)


def verify_transpose_grad(d_shape, axes=None, executor_kind="vm"):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    fwd_func = relay.Function([data], relay.transpose(data, axes=axes))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_transpose_grad(executor_kind):
    verify_transpose_grad((1, 2, 3, 4), executor_kind=executor_kind)
    verify_transpose_grad((1, 2, 3, 4), axes=(0, 2, 3, 1), executor_kind=executor_kind)


def test_negative_grad(executor_kind):
    data = relay.var("data", relay.TensorType((10, 4), "float32"))
    fwd_func = relay.Function([data], relay.negative(data))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_cast_grad(executor_kind):
    data = relay.var("data", relay.TensorType((10, 4), "float32"))
    fwd_func = relay.Function([data], relay.cast(data, "float64"))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_cast_like_grad(executor_kind):
    data = relay.var("data", shape=(10, 4), dtype="float32")
    like = relay.var("like", shape=(1,), dtype="float64")
    fwd_func = relay.Function([data, like], relay.cast_like(data, like))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_copy_grad(executor_kind):
    data = relay.var("data", relay.TensorType((10, 4), "float64"))
    fwd_func = relay.Function([data], relay.copy(data))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_take_grad(executor_kind):
    data_dtype = relay.TensorType((3, 4, 5), "float64")
    data = relay.var("data", data_dtype)
    indices = relay.var("indices", relay.TensorType((relay.Any(),), "int32"))
    inputs = [_np_randn_from_type(data_dtype, scale=1e-5), np.array([1, 2], dtype="int32")]
    test_inputs = [inputs[0]]

    # take on axis
    fwd_func = relay.Function([data, indices], relay.take(data, indices, axis=1))
    check_grad(fwd_func, inputs=inputs, test_inputs=test_inputs, executor_kind=executor_kind)

    # take on flattened
    fwd_func = relay.Function([data, indices], relay.take(data, indices, axis=None))
    check_grad(fwd_func, inputs=inputs, test_inputs=test_inputs, executor_kind=executor_kind)


def test_stack_grad(executor_kind):
    args = [relay.var(c, shape=(2, 3, 4), dtype="float64") for c in "xyz"]
    fwd_func = relay.Function(args, relay.stack(args, axis=0))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_squeeze_grad(executor_kind):
    data = relay.var("data", shape=(2, 1, 1, 3, 4, 1), dtype="float64")
    fwd_func = relay.Function([data], relay.squeeze(data))
    fwd_func_subset = relay.Function([data], relay.squeeze(data, axis=[1, -1]))
    check_grad(fwd_func, executor_kind=executor_kind)
    check_grad(fwd_func_subset, executor_kind=executor_kind)


def test_arange_grad(executor_kind):
    # TODO: testing arange numerically is strange because two-sided approx can
    #       produce different output shapes
    dtype = "float64"
    start = relay.var("start", relay.TensorType((), dtype))
    stop = relay.var("stop", relay.TensorType((), dtype))
    step = relay.var("step", relay.TensorType((), dtype))
    values = [np.array(v, dtype=dtype) for v in [2.5, 9.5, 1.8]]
    fwd_func = relay.Function([start, stop, step], relay.arange(start, stop, step, dtype))
    check_grad(fwd_func, inputs=values, executor_kind=executor_kind)


def test_gather_nd_grad(executor_kind):
    data = relay.var("data", relay.TensorType((2, 3), "float64"))
    indices = relay.var("indices", relay.TensorType((2, 4), "int64"))
    fwd = relay.Function([data, indices], relay.gather_nd(data, indices))
    data_np = np.random.rand(2, 3).astype("float64")
    indices_np = np.array([[0, 1, 1, 0], [0, 1, 0, 0]], dtype="int64")
    check_grad(
        fwd, inputs=[data_np, indices_np], test_inputs=[data_np], executor_kind=executor_kind
    )


def test_reshape_like_grad(executor_kind):
    data = relay.var("data", shape=(2, 3, 4), dtype="float32")
    shape_like = relay.var("shape_like", shape=(6, 2, 2), dtype="float32")
    fwd_func = relay.Function([data, shape_like], relay.reshape_like(data, shape_like))
    check_grad(fwd_func, executor_kind=executor_kind)


def test_zeros_ones_grad_const_ints():
    # when shape is static (i.e. not an input), there is no gradient at all
    static_ty = relay.TensorType([2, 3, 4], dtype="float32")
    expected_ty = relay.TupleType([static_ty, relay.TupleType([])])

    for op in [relay.zeros, relay.ones]:
        fwd_func = relay.Function([], op(static_ty.concrete_shape, static_ty.dtype))
        bwd_func = run_infer_type(gradient(run_infer_type(fwd_func)))
        tvm.ir.assert_structural_equal(bwd_func.ret_type, expected_ty)


def test_zeros_ones_grad_const_expr():
    # when shape is static (i.e. not an input), there is no gradient at all
    shape_const = relay.const(np.array([2, 3, 4]), dtype="int32") * relay.const(1, dtype="int32")
    static_ty = relay.TensorType([2, 3, 4], dtype="float32")
    dyn_ty = relay.TensorType([relay.Any(), relay.Any(), relay.Any()], dtype="float32")
    expected_ty_static = relay.TupleType([static_ty, relay.TupleType([])])
    expected_ty_dyn = relay.TupleType([dyn_ty, relay.TupleType([])])

    for op in [relay.zeros, relay.ones]:
        # with DynamicToStatic, the shape should be concretized
        fwd_func = relay.Function([], op(shape_const, static_ty.dtype))
        fwd_func = run_opt_pass(fwd_func, relay.transform.DynamicToStatic())
        bwd_func = run_infer_type(gradient(run_infer_type(fwd_func)))
        tvm.ir.assert_structural_equal(bwd_func.ret_type, expected_ty_static)

        fwd_func = relay.Function([], op(shape_const, static_ty.dtype))
        bwd_func = run_infer_type(gradient(run_infer_type(fwd_func)))
        tvm.ir.assert_structural_equal(bwd_func.ret_type, expected_ty_dyn)


def test_zeros_ones_grad_dynamic(executor_kind):
    rank = np.random.randint(low=1, high=5, dtype="int32")
    dyn_shape = np.random.randint(low=1, high=4, size=(rank,), dtype="int32")
    shape_data = relay.var("shape_data", shape=(rank,), dtype="int32")

    for op, op_ref in [(relay.zeros, np.zeros), (relay.ones, np.ones)]:
        fwd_func = relay.Function([shape_data], op(shape_data, dtype="float32"))
        bwd_func = run_infer_type(gradient(run_infer_type(fwd_func)))

        for target, dev in tvm.testing.enabled_targets():
            res, (grad,) = relay.create_executor(executor_kind, device=dev, target=target).evaluate(
                bwd_func
            )(dyn_shape)
            tvm.testing.assert_allclose(res.numpy(), op_ref(dyn_shape, dtype="float32"))
            tvm.testing.assert_allclose(grad.numpy(), np.zeros((rank,), dtype="int32"))


if __name__ == "__main__":
    tvm.testing.main()
