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
from tvm.testing import assert_allclose
import pytest

def test_tc():
  # test typechecks
  mod = tvm.IRModule()

  shape = (20, 20)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x1 = relay.var("x1", t)
  x2 = relay.var("x2", t)
  # f(x1,x2) = (x1-x2)*x2
  y = relay.Function([x1, x2], (x1 - x2) * x2)

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  # function input/output types should remain the same
  assert mod["main"].checked_type == relay.FuncType([t, t], t)

def test_add():
  # test simple add
  mod = tvm.IRModule()

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  # f(x) = x+x
  y = relay.Function([x], x+x)

  mod["main"] = y
  mod = transform.GradientCell()(mod)
  y = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], t)

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy() + x.asnumpy())

def test_add_tuple():
  # test input tuple and add items
  mod = tvm.IRModule()

  shape = (10, 10)
  dtype = 'float32'
  tensor_type = relay.TensorType(shape, dtype)
  t = relay.TupleType([tensor_type, tensor_type])

  x = relay.var("x", t)
  # f((x1,x2)) = x1 + x2
  y = relay.Function([x], relay.TupleGetItem(x, 0) + relay.TupleGetItem(x, 1))

  mod["main"] = y
  mod = transform.GradientCell()(mod)
  mod = transform.PrintIR(show_meta_data=True)(mod)
  y = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], tensor_type)

  ex = create_executor(mod=mod)
  x = (rand(dtype, *shape), rand(dtype, *shape))
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x[0].asnumpy() + x[1].asnumpy())

def test_mult():
  # test simple add
  mod = tvm.IRModule()

  shape = (15, 15)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  # f(x) = x*x
  y = relay.Function([x], x * x)

  mod["main"] = y
  mod = transform.GradientCell()(mod)
  y = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], t)

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy() * x.asnumpy())

def test_ret_tuple():
  # test return tuple
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  # f(x) = (x,x)
  func = relay.Function([x], relay.Tuple([x,x * relay.const(2.0)]))
  func = run_infer_type(func)

  mod["main"] = func
  mod = transform.GradientCell()(mod)
  func = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], relay.TupleType([t, t]))

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = ex.evaluate(func)(x)
  assert_allclose(y[0].asnumpy(), x.asnumpy())
  assert_allclose(y[1].asnumpy(), x.asnumpy() * 2.0)

def test_broadcast():
  # test broadcast add
  mod = tvm.IRModule()
  
  shape1 = (3, 4, 1)
  shape2 = (1, 5)
  dtype = 'float32'
  t1 = relay.TensorType(shape1, dtype)
  t2 = relay.TensorType(shape2, dtype)

  x1 = relay.var("x1", t1)
  x2 = relay.var("x2", t2)
  func = relay.Function([x1,x2], x1 + x2)
  func = run_infer_type(func)
  back_func = transform.gradient(func)
  back_func = run_infer_type(back_func)

  mod["main"] = back_func
  mod = transform.GradientCell()(mod)
  back_func = mod["main"]

  x1_np = rand(dtype, *shape1).asnumpy()
  x2_np = rand(dtype, *shape2).asnumpy()
  expected_forward = x1_np + x2_np

  expected_forward_type = relay.TensorType(expected_forward.shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([t1, t2],
                                                    relay.TupleType([expected_forward_type, relay.TupleType([t1, t2])]))

  ex = create_executor(mod=mod)
  (forward), (grad_x1, grad_x2, ) = ex.evaluate(back_func)(x1_np, x2_np)

  assert_allclose(forward.asnumpy(), expected_forward)
  assert_allclose(grad_x1.asnumpy(), np.ones_like(expected_forward).sum(axis=2, keepdims=True))
  assert_allclose(grad_x2.asnumpy(), np.ones_like(expected_forward).sum(axis=(0,1), keepdims=True).squeeze(axis=0))

def test_reverse_ad_identity():
  # test correctness after reverse mode ad
  # of f(x) = x
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)

  func = relay.Function([x], x)
  func = run_infer_type(func)
  back_func = transform.gradient(func)
  back_func = run_infer_type(back_func)

  mod["main"] = back_func
  mod = transform.GradientCell()(mod)
  back_func = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t],
                                                    relay.TupleType([t, relay.TupleType([t])]))

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  (forward), (grad,) = ex.evaluate(back_func)(x)
  assert_allclose(forward.asnumpy(), x.asnumpy())
  assert_allclose(grad.asnumpy(), np.ones_like(x.asnumpy()))

def test_multivar_reverse_ad():
  # test correctness after reverse mode ad
  # of multivariate function
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.var("y", t)

  func = relay.Function([x, y],  (x * y) * relay.const(np.ones(shape, dtype)))
  func = run_infer_type(func)
  back_func = transform.gradient(func)
  back_func = run_infer_type(back_func)

  mod["main"] = back_func
  mod = transform.GradientCell()(mod)
  back_func = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t, t],
                                                    relay.TupleType([t, relay.TupleType([t, t])]))

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = rand(dtype, *shape)
  (forward), (grad_x, grad_y, ) = ex.evaluate(back_func)(x, y)
  assert_allclose(forward.asnumpy(), x.asnumpy() * y.asnumpy())
  assert_allclose(grad_x.asnumpy(), y.asnumpy())
  assert_allclose(grad_y.asnumpy(), x.asnumpy())

def test_after_partial_eval():
  # test GradientCell transformation after PartialEval
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.var("y", t)

  func = relay.Function([x, y], (x * y) * relay.const(np.ones(shape, dtype)))
  func = run_infer_type(func)
  back_func = transform.gradient(func)
  back_func = run_infer_type(back_func)

  mod["main"] = back_func
  back_func = mod["main"]

  seq = transform.Sequential([
    transform.PartialEvaluate(),
    transform.GradientCell(),
    transform.DeadCodeElimination()
  ])

  mod = seq(mod)

  assert mod["main"].checked_type == relay.FuncType([t, t],
                                                    relay.TupleType([t, relay.TupleType([t, t])]))

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = rand(dtype, *shape)
  (forward), (grad_x, grad_y,) = ex.evaluate(back_func)(x, y)
  assert_allclose(forward.asnumpy(), x.asnumpy() * y.asnumpy())
  assert_allclose(grad_x.asnumpy(), y.asnumpy())
  assert_allclose(grad_y.asnumpy(), x.asnumpy())

def test_before_partial_eval():
  # test GradientCell transformation before PartialEval
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.var("y", t)

  func = relay.Function([x, y], x * y)
  func = run_infer_type(func)
  back_func = transform.gradient(func)
  back_func = run_infer_type(back_func)

  mod["main"] = back_func
  seq = transform.Sequential([
    transform.GradientCell(),
    transform.PartialEvaluate(),
    transform.DeadCodeElimination()
  ])
  mod = seq(mod)
  back_func = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t, t],
                                                    relay.TupleType([t, relay.TupleType([t, t])]))

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = rand(dtype, *shape)
  (forward), (grad_x, grad_y,) = ex.evaluate(back_func)(x, y)
  assert_allclose(forward.asnumpy(), x.asnumpy() * y.asnumpy())
  assert_allclose(grad_x.asnumpy(), y.asnumpy())
  assert_allclose(grad_y.asnumpy(), x.asnumpy())

def test_zeros():
  # test with zeros operator
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.zeros(shape, dtype))

  mod["main"] = y
  mod = transform.GradientCell()(mod)
  y = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], t)

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy())

def test_ones():
  # test with ones operator
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.ones(shape, dtype))

  mod["main"] = y
  mod = transform.GradientCell()(mod)
  y = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], t)

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy() + np.ones_like(x.asnumpy()))

def test_zeros_like():
  # test with zeros_like operator
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.zeros_like(x))

  mod["main"] = y
  mod = transform.GradientCell()(mod)
  y = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], t)

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy())

def test_ones_like():
  # test with ones_like operator
  mod = tvm.IRModule()
  
  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.ones_like(x))

  mod["main"] = y
  mod = transform.GradientCell()(mod)
  y = mod["main"]

  assert mod["main"].checked_type == relay.FuncType([t], t)

  ex = create_executor(mod=mod)
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy() + np.ones_like(x.asnumpy()))

if __name__ == "__main__":
  pytest.main([__file__])
