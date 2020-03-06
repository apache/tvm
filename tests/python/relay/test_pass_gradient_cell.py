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
from tvm.relay.analysis import free_vars, free_type_vars, assert_alpha_equal
from tvm.relay import create_executor, transform
from tvm.relay.testing import rand, run_infer_type
from tvm.relay.op import add, multiply
from tvm.relay.prelude import Prelude, TensorArrayOps
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

def test_reverse_ad():
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

  # new_type = grad_cell_type(mod, shape, dtype)
  # assert mod["main"].checked_type == relay.FuncType([new_type],)


if __name__ == "__main__":
  pytest.main([__file__])