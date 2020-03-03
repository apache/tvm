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
from tvm.relay.build_module import optimize
from tvm.relay.transform import GradientCell
from tvm.relay.testing import rand, run_infer_type
from tvm.relay.op import add, multiply
from tvm.relay.prelude import Prelude, TensorArrayOps

def test_zero_tensor():
  mod = tvm.IRModule()
  mod.import_from_std("gradient.rly")

  shape = (10, 10)
  dtype = 'float32'

  t = relay.TensorType(shape, dtype)

  x = relay.var("x", t)

  y = relay.Function([x], multiply(x,x))
  mod["main"] = y

  mod = transform.GradientCell()(mod)


  # mod = transform.PrintIR(True)(mod)

  print("---------------------------")

  # gradcell = mod.get_global_type_var("GradCell")(t)
  # x_np = np.zeros(shape, dtype)

  # gradcell = mod.get_global_type_var("GradCell")
  # y = tvm.relay.TypeCall(gradcell, [t])
  #
  # addcell = mod.get_global_var("AddGradCell")
  # fromcell = mod.get_global_var("FromGradCell")

  # mod, params = optimize(mod, target="llvm", params={"x": x_nd})

  #ex = create_executor(mod=mod)
  #a = ex.evaluate(addFunc)(x_nd)


  # mod = transform.InferType()(mod)
  print("hi")

if __name__ == "__main__":
  test_zero_tensor()

