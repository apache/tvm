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
from tvm import te
from tvm import relay
from tvm.relay import create_executor, transform
from tvm.relay.testing import rand, run_infer_type, check_grad
from tvm.relay.analysis import assert_alpha_equal
from tvm.relay.op import add, multiply
from tvm.relay.prelude import Prelude, TensorArrayOps
from tvm.testing import assert_allclose
import pytest

def grad_cell_type(mod, shape, dtype):
  grad_type = mod.get_global_type_var("GradCell")
  type_arg = relay.TensorType(shape, dtype)
  return grad_type(type_arg)

def test_add():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x+x)

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type], new_type)

def test_add_tuple():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x1 = relay.var("x1", t)
  x2 = relay.var("x2", t)
  t1 = relay.Tuple([x1, x2])
  y = relay.Function([x1, x2], relay.TupleGetItem(t1,0) + relay.TupleGetItem(t1,1))

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type, new_type], new_type)

def test_mult():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (15, 15)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x * x)

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type], new_type)

def test_tc():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (20, 20)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x1 = relay.var("x1", t)
  x2 = relay.var("x2", t)

  y = relay.Function([x1, x2], (x1 - x2) * x2)

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type, new_type], new_type)

def test_ret_tuple():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.RefCreate(x)
  func = relay.Function([x], relay.Tuple([x,y]))
  func = run_infer_type(func)

  mod["main"] = func
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type], relay.TupleType([new_type, relay.RefType(new_type)]))

def test_broadcast():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

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

  x1_np = rand(dtype, *shape1).asnumpy()
  x2_np = rand(dtype, *shape2).asnumpy()
  expected_forward =  x1_np + x2_np
  x1_type = grad_cell_type(mod, shape1, dtype)
  x2_type = grad_cell_type(mod, shape2, dtype)
  expected_forward_type = grad_cell_type(mod, expected_forward.shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([x1_type, x2_type],
                                                    relay.TupleType([expected_forward_type, relay.TupleType([x1_type, x2_type])]))

  ex = create_executor()
  (forward), (grad_x1, grad_x2, ) = ex.evaluate(back_func)(x1_np, x2_np)

  assert_allclose(forward.asnumpy(), expected_forward)
  assert_allclose(grad_x1.asnumpy(), np.ones_like(expected_forward).sum(axis=2, keepdims=True))
  assert_allclose(grad_x2.asnumpy(), np.ones_like(expected_forward).sum(axis=(0,1), keepdims=True).squeeze(axis=0))

def test_reverse_ad_identity():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

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

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type],
                                                    relay.TupleType([new_type, relay.TupleType([new_type])]))

  ex = create_executor()
  x = rand(dtype, *shape)
  (forward), (grad,) = ex.evaluate(back_func)(x)
  assert_allclose(forward.asnumpy(), x.asnumpy())
  assert_allclose(grad.asnumpy(), np.ones_like(x.asnumpy()))

def test_multivar_reverse_ad():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

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

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type, new_type],
                                                    relay.TupleType([new_type, relay.TupleType([new_type, new_type])]))

  ex = create_executor()
  x = rand(dtype, *shape)
  y = rand(dtype, *shape)
  (forward), (grad_x, grad_y, ) = ex.evaluate(back_func)(x, y)
  assert_allclose(forward.asnumpy(), x.asnumpy() * y.asnumpy())
  assert_allclose(grad_x.asnumpy(), y.asnumpy())
  assert_allclose(grad_y.asnumpy(), x.asnumpy())

def test_partial_eval_before():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

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

  seq = transform.Sequential([
    transform.PartialEvaluate(),
    transform.GradientCell(),
    transform.DeadCodeElimination()
  ])

  mod = seq(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type, new_type],
                                                    relay.TupleType([new_type, relay.TupleType([new_type, new_type])]))

  ex = create_executor()
  x = rand(dtype, *shape)
  y = rand(dtype, *shape)
  (forward), (grad_x, grad_y,) = ex.evaluate(back_func)(x, y)
  assert_allclose(forward.asnumpy(), x.asnumpy() * y.asnumpy())
  assert_allclose(grad_x.asnumpy(), y.asnumpy())
  assert_allclose(grad_y.asnumpy(), x.asnumpy())

def test_partial_eval_after_multivar():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

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

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type, new_type],
                                                    relay.TupleType([new_type, relay.TupleType([new_type, new_type])]))

  ex = create_executor()
  x = rand(dtype, *shape)
  y = rand(dtype, *shape)
  (forward), (grad_x, grad_y,) = ex.evaluate(back_func)(x, y)
  assert_allclose(forward.asnumpy(), x.asnumpy() * y.asnumpy())
  assert_allclose(grad_x.asnumpy(), y.asnumpy())
  assert_allclose(grad_y.asnumpy(), x.asnumpy())

def test_zeros():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.zeros(shape, dtype))

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type], new_type)

  ex = create_executor()
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy())

def test_ones():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.ones(shape, dtype))

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type], new_type)

  ex = create_executor()
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy() + np.ones_like(x.asnumpy()))

def test_zeros():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.zeros_like(x))

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type], new_type)

  ex = create_executor()
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy())

def test_ones_like():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'
  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)
  y = relay.Function([x], x + relay.ones_like(x))

  mod["main"] = y
  mod = transform.GradientCell()(mod)

  new_type = grad_cell_type(mod, shape, dtype)
  assert mod["main"].checked_type == relay.FuncType([new_type], new_type)

  ex = create_executor()
  x = rand(dtype, *shape)
  y = ex.evaluate(y)(x)
  assert_allclose(y.asnumpy(), x.asnumpy() + np.ones_like(x.asnumpy()))

if __name__ == "__main__":
  pytest.main([__file__])
