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

import tvm
from tvm import relay
from tvm.relay import create_executor, transform
from tvm.relay.testing import rand, run_infer_type
import tvm.testing
from tvm.testing import assert_allclose


def test_tc():
    """Simple testcase, check that transformation typechecks."""
    mod = tvm.IRModule()

    shape = (20, 20)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x1 = relay.var("x1", t)
    x2 = relay.var("x2", t)
    # f(x1,x2) = (x1-x2)*x2
    y = relay.Function([x1, x2], (x1 - x2) * x2)

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)

    # function input/output types should remain the same
    assert mod["main"].checked_type == relay.FuncType([t, t], t)


def test_add():
    """Simple add testcase. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    # f(x) = x+x
    y = relay.Function([x], x + x)

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    x = rand(dtype, *shape)
    y = create_executor(mod=mod).evaluate(y)(x)
    assert_allclose(y.numpy(), x.numpy() + x.numpy())


def test_add_tuple():
    """Add elements of tuple. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    tensor_type = relay.TensorType(shape, dtype)
    t = relay.TupleType([tensor_type, tensor_type])

    x = relay.var("x", t)
    # f((x1,x2)) = x1 + x2
    y = relay.Function([x], relay.TupleGetItem(x, 0) + relay.TupleGetItem(x, 1))

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    mod = tvm.transform.PrintIR(show_meta_data=True)(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], tensor_type)

    x = (rand(dtype, *shape), rand(dtype, *shape))
    y = create_executor(mod=mod).evaluate(y)(x)
    assert_allclose(y.numpy(), x[0].numpy() + x[1].numpy())


def test_mult():
    """Simple multiplication testcase. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (15, 15)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    # f(x) = x*x
    y = relay.Function([x], x * x)

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    x = rand(dtype, *shape)
    y = create_executor(mod=mod).evaluate(y)(x)
    assert_allclose(y.numpy(), x.numpy() * x.numpy())


def test_ret_tuple():
    """Test tuple return type. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    # f(x) = (x,x)
    func = relay.Function([x], relay.Tuple([x, x * relay.const(2.0)]))
    func = run_infer_type(func)

    mod["main"] = func
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], relay.TupleType([t, t]))

    x = rand(dtype, *shape)
    y = create_executor(mod=mod).evaluate(func)(x)
    assert_allclose(y[0].numpy(), x.numpy())
    assert_allclose(y[1].numpy(), x.numpy() * 2.0)


def test_add_broadcast():
    """Test adding matrices of different size. Check types and semantic equivalence."""
    mod = tvm.IRModule()

    shape1 = (3, 4, 1)
    shape2 = (1, 5)
    dtype = "float32"
    t1 = relay.TensorType(shape1, dtype)
    t2 = relay.TensorType(shape2, dtype)

    x1 = relay.var("x1", t1)
    x2 = relay.var("x2", t2)
    func = relay.Function([x1, x2], x1 + x2)
    func = run_infer_type(func)

    mod["main"] = func
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    func = mod["main"]

    x1_np = rand(dtype, *shape1).numpy()
    x2_np = rand(dtype, *shape2).numpy()
    expected_forward = x1_np + x2_np

    expected_forward_type = relay.TensorType(expected_forward.shape, dtype)
    assert mod["main"].checked_type == relay.FuncType([t1, t2], expected_forward_type)

    forward = create_executor(mod=mod).evaluate(func)(x1_np, x2_np)

    assert_allclose(forward.numpy(), expected_forward)


def test_reverse_ad_identity():
    """Simple test with reverse mode ad."""
    # of f(x) = x
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)

    func = relay.Function([x], x)
    func = run_infer_type(func)
    back_func = transform.gradient(func)
    back_func = run_infer_type(back_func)

    mod["main"] = back_func
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    back_func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType(
        [t], relay.TupleType([t, relay.TupleType([t])])
    )

    x = rand(dtype, *shape)
    (forward), (grad,) = create_executor(mod=mod).evaluate(back_func)(x)
    assert_allclose(forward.numpy(), x.numpy())
    assert_allclose(grad.numpy(), np.ones_like(x.numpy()))


def test_multivar_reverse_ad():
    """Simple test with multivariate reverse mode ad."""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.var("y", t)

    func = relay.Function([x, y], (x * y) * relay.const(np.ones(shape, dtype)))
    func = run_infer_type(func)
    back_func = transform.gradient(func)
    back_func = run_infer_type(back_func)

    mod["main"] = back_func
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    back_func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType(
        [t, t], relay.TupleType([t, relay.TupleType([t, t])])
    )

    x = rand(dtype, *shape)
    y = rand(dtype, *shape)
    (forward), (grad_x, grad_y,) = create_executor(mod=mod).evaluate(
        back_func
    )(x, y)
    assert_allclose(forward.numpy(), x.numpy() * y.numpy())
    assert_allclose(grad_x.numpy(), y.numpy())
    assert_allclose(grad_y.numpy(), x.numpy())


def test_partial_eval():
    """Test transformation following reverse mode ad and PartialEval"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    func = relay.Function([], relay.const(np.ones(shape, dtype)))
    func = run_infer_type(func)
    back_func = transform.gradient(func)
    back_func = run_infer_type(back_func)

    mod["main"] = back_func
    mod = transform.InferType()(mod)
    back_func = mod["main"]

    transform.PartialEvaluate()(mod)


def test_after_partial_eval():
    """Test transformation following reverse mode ad and PartialEval"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.var("y", t)

    func = relay.Function([x, y], (x * y) * relay.const(np.ones(shape, dtype)))
    func = run_infer_type(func)
    back_func = transform.gradient(func)
    back_func = run_infer_type(back_func)

    mod["main"] = back_func
    back_func = mod["main"]

    seq = tvm.transform.Sequential(
        [
            transform.PartialEvaluate(),
            transform.InferType(),
            transform.LazyGradientInit(),
            transform.InferType(),
            transform.DeadCodeElimination(),
            transform.InferType(),
        ]
    )

    mod = seq(mod)

    assert mod["main"].checked_type == relay.FuncType(
        [t, t], relay.TupleType([t, relay.TupleType([t, t])])
    )

    x = rand(dtype, *shape)
    y = rand(dtype, *shape)
    (forward), (grad_x, grad_y,) = create_executor(mod=mod).evaluate(
        back_func
    )(x, y)
    assert_allclose(forward.numpy(), x.numpy() * y.numpy())
    assert_allclose(grad_x.numpy(), y.numpy())
    assert_allclose(grad_y.numpy(), x.numpy())


def test_before_partial_eval():
    """Test transformation before PartialEval"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.var("y", t)

    func = relay.Function([x, y], x * y)
    func = run_infer_type(func)
    back_func = transform.gradient(func)
    back_func = run_infer_type(back_func)

    mod["main"] = back_func
    seq = tvm.transform.Sequential(
        [
            transform.LazyGradientInit(),
            transform.PartialEvaluate(),
            transform.InferType(),
            transform.DeadCodeElimination(),
            transform.InferType(),
        ]
    )
    mod = seq(mod)
    back_func = mod["main"]

    assert mod["main"].checked_type == relay.FuncType(
        [t, t], relay.TupleType([t, relay.TupleType([t, t])])
    )

    x = rand(dtype, *shape)
    y = rand(dtype, *shape)
    (forward), (grad_x, grad_y,) = create_executor(mod=mod).evaluate(
        back_func
    )(x, y)
    assert_allclose(forward.numpy(), x.numpy() * y.numpy())
    assert_allclose(grad_x.numpy(), y.numpy())
    assert_allclose(grad_y.numpy(), x.numpy())


def test_zeros():
    """Simple test using "zeros" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.zeros(shape, dtype))

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    x = rand(dtype, *shape)
    y = create_executor(mod=mod).evaluate(y)(x)
    assert_allclose(y.numpy(), x.numpy())


def test_ones():
    """Simple test using "ones" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.ones(shape, dtype))

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    x = rand(dtype, *shape)
    y = create_executor(mod=mod).evaluate(y)(x)
    assert_allclose(y.numpy(), x.numpy() + np.ones_like(x.numpy()))


def test_zeros_like():
    """Simple test using "zeros_like" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.zeros_like(x))

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    x = rand(dtype, *shape)
    y = create_executor(mod=mod).evaluate(y)(x)
    assert_allclose(y.numpy(), x.numpy())


def test_ones_like():
    """Simple test using "ones_like" op"""
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.ones_like(x))

    mod["main"] = y
    mod = transform.InferType()(mod)
    mod = transform.LazyGradientInit()(mod)
    y = mod["main"]

    assert mod["main"].checked_type == relay.FuncType([t], t)

    x = rand(dtype, *shape)
    y = create_executor(mod=mod).evaluate(y)(x)
    assert_allclose(y.numpy(), x.numpy() + np.ones_like(x.numpy()))


if __name__ == "__main__":
    tvm.testing.main()
