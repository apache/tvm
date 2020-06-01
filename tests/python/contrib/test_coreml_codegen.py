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
from tvm.relay import transform
from tvm.contrib.target import coreml as _coreml

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

def test_coreml_codegen():
    try:
        import coremltools
    except ImportError:
        print("skip because coremltools is not available")
        return
    shape = (1,)

    def create_graph():
        x = relay.var('x', shape=shape)
        y = relay.var('y', shape=shape)
        z = x + x
        p = y * y
        return relay.Function([x, y], p - z)

    def expected():
        target = "coremlcompiler"
        mod = tvm.IRModule()

        # function 0
        f0_i0 = relay.var(target + "_0_i0", shape=shape)
        func0 = relay.Function([f0_i0], f0_i0 * f0_i0)

        func0 = func0.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func0 = func0.with_attr("Compiler", target)
        func0 = func0.with_attr("global_symbol", target + "_0")
        gv0 = relay.GlobalVar(target + "_0")
        mod[gv0] = func0

        # function 2
        f2_i0 = relay.var(target + "_2_i0", shape=shape)
        func2 = relay.Function([f2_i0], f2_i0 + f2_i0)

        func2 = func2.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
        func2 = func2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        func2 = func2.with_attr("Compiler", target)
        func2 = func2.with_attr("global_symbol", target + "_2")
        gv2 = relay.GlobalVar(target + "_2")
        mod[gv2] = func2

        # body
        x = relay.var('x', shape=shape)
        y = relay.var('y', shape=shape)
        func = relay.Function([x, y], gv0(y) - gv2(x))
        mod["main"] = func
        return mod

    x_data = np.random.rand(*shape).astype('float32')
    y_data = np.random.rand(*shape).astype('float32')
    mod = tvm.IRModule()
    mod["main"] = create_graph()
    mod = transform.AnnotateTarget("coremlcompiler")(mod)
    mod = transform.PartitionGraph()(mod)

    assert tvm.ir.structural_equal(mod, expected(), map_free_vars=True)

    check_result(mod, {"x": x_data, "y": y_data}, shape, (y_data * y_data) - (x_data + x_data))

if __name__ == "__main__":
    test_coreml_codegen()
