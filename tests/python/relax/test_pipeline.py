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
from tvm import relax
from tvm.script import relax as R


def test_pipeline_compile():
    pipeline = relax.get_pipeline()

    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")):
            lv0 = R.add(x, y)
            return lv0

    mod = Mod
    mod = pipeline(mod)
    target = tvm.target.Target("llvm", host="llvm")

    ex = relax.build(mod, target)
    x_np = np.random.rand(3, 4).astype(np.float32)
    y_np = np.random.rand(3, 4).astype(np.float32)
    x = tvm.nd.array(x_np)
    y = tvm.nd.array(y_np)

    vm = relax.VirtualMachine(ex, tvm.cpu())
    z = vm["main"](x, y)
    tvm.testing.assert_allclose(z.numpy(), x_np + y_np, rtol=1e-7, atol=1e-7)
