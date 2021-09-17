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
import scipy
from scipy import special
import tvm
import tvm.testing
import tvm.relay as relay
from tvm import topi
from tvm import te
from tvm.contrib import graph_executor
from tvm.topi import testing


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_fastmath(target, dev):
    def test_apply(relay_op, name, f_numpy, low, high, step, dtype="float32"):
        a_np = np.arange(low, high, step).astype(dtype).reshape((1, -1))
        b_np = f_numpy(a_np)

        x = relay.var("x", shape=a_np.shape, dtype="float32")
        y = relay_op(x)
        func = relay.Function([x], y)
        mod = tvm.IRModule.from_expr(func)

        with tvm.transform.PassContext(opt_level=3, required_pass=["FastMath"]):
            graph, lib, params = relay.build(mod, target=target, params=None)

        # Check that the op related to fast math have been convered to function in lib
        func_name = "tvmgen_default_fused_" + name
        # When there're multiple targets in tvm.testing.parametrize_targets, the function
        # built will have a "_1" in function name
        assert func_name in graph

        m = graph_executor.create(graph, lib, dev)
        # Set inputs
        m.set_input("x", tvm.nd.array(a_np, dev))
        m.set_input(**params)
        # Execute
        m.run()
        # Get outputs
        tvm_output = m.get_output(0)
        tvm.testing.assert_allclose(tvm_output.numpy(), b_np, rtol=1e-5, atol=1e-5)

    test_apply(relay.exp, "fast_exp", np.exp, low=-88, high=88, step=0.01)
    test_apply(relay.erf, "fast_erf", scipy.special.erf, low=-10, high=10, step=0.01)
    test_apply(relay.tanh, "fast_tanh", np.tanh, low=-10, high=10, step=0.01)
    test_apply(
        relay.nn.fast_softmax,
        "nn_fast_softmax",
        tvm.topi.testing.softmax_python,
        low=-10,
        high=10,
        step=0.01,
    )


if __name__ == "__main__":
    test_fastmath()
