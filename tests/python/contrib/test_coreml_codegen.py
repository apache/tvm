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
import numpy as np

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.op.annotation import compiler_begin, compiler_end

def check_result(mod, map_inputs, out_shape, result, tol=1e-3, target="llvm",
                 ctx=tvm.cpu(), params=None):
    def check_graph_runtime_result():
        with relay.build_config(opt_level=3):
            json, lib, param = relay.build(mod, target=target, params=params)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, ctx=ctx)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    check_graph_runtime_result()

@pytest.mark.skip('skip because coremltools is not available in CI')
def test_coreml_codegen():
    shape = (1,)
    x = relay.var('x', shape=shape)
    y = relay.var('y', shape=shape)
    _x = relay.annotation.compiler_begin(x, "coremlcompiler")
    _y = relay.annotation.compiler_begin(y, "coremlcompiler")
    z = _x + _x
    p = _y * _y
    p = relay.annotation.compiler_end(p, "coremlcompiler")
    z = relay.annotation.compiler_end(z, "coremlcompiler")
    f = relay.Function([x, y], p - z)
    x_data = np.random.rand(*shape).astype('float32')
    y_data = np.random.rand(*shape).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = f
    mod = transform.PartitionGraph()(mod)

    check_result(mod, {"x": x_data, "y": y_data}, shape, (y_data * y_data) - (x_data + x_data))

if __name__ == "__main__":
    test_coreml_codegen()
