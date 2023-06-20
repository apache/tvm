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
from tvm.contrib import graph_executor
from tvm.relay.frontend.common import infer_shape
import tvm.testing


@tvm.testing.requires_llvm
def test_reshape_0_copy():
    shape0 = (56, 224)
    shape1 = (112, 112)
    in_name0 = "infeats0"
    in_name1 = "infeats1"
    x0 = relay.var(in_name0, shape=shape0, dtype="float32")
    x0 = relay.reshape(x0, shape1)

    x1 = relay.var(in_name1, shape=shape1, dtype="float32")
    mat = relay.nn.matmul(x0, x1)
    _y = relay.reshape(mat, (-1))
    func = relay.Function(relay.analysis.free_vars(_y), _y)
    mod = tvm.IRModule.from_expr(func)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target="llvm")
    m = graph_executor.GraphModule(lib["default"](tvm.cpu(0)))

    data_ndarray0 = tvm.nd.array(
        np.random.random(shape0).astype(np.float32), device=tvm.device("llvm", 0)
    )
    data_ndarray1 = tvm.nd.array(
        np.random.random(shape1).astype(np.float32), device=tvm.device("llvm", 0)
    )

    def expected():
        m.set_input(in_name0, data_ndarray0)
        m.set_input(in_name1, data_ndarray1)
        m.run()
        return m.get_output(0).numpy()

    def zero_copy():
        outshape = infer_shape(_y)
        output_view = tvm.nd.empty(outshape, device=tvm.device("llvm", 0))
        m.set_input_zero_copy(in_name0, data_ndarray0)
        m.set_input_zero_copy(in_name1, data_ndarray1)
        m.set_output_zero_copy(0, output_view)
        m.run()
        return output_view.numpy()

    golden_out = expected()
    out = zero_copy()
    np.testing.assert_equal(golden_out, out)


if __name__ == "__main__":
    test_reshape_0_copy()
