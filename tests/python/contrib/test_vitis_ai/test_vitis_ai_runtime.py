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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name, W0611, C0413

""" Vitis-AI runtime test """

import sys
import numpy as np

import pytest
pytest.importorskip('pyxir')
import pyxir.contrib.target.DPUCADX8G

import tvm
import tvm.relay.testing
from tvm import relay
from tvm import runtime
from tvm.relay import transform
from tvm.contrib import util
from tvm.relay.backend import compile_engine
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.vitis_ai import annotation
from tvm.contrib.target import vitis_ai



def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm",
                 ctx=tvm.cpu(), params=None):
    """ To check the result between reference and byoc vitis-ai flow"""

    def update_lib(lib):
        tmp_path = util.tempdir()
        lib_name = 'lib.so'
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path)
        lib = runtime.load_module(lib_path)
        return lib

    def check_graph_runtime_result():
        compile_engine.get().clear()
        with tvm.transform.PassContext(opt_level=3,
                                       config={'relay.ext.vitis_ai.options.target' : 'DPUCADX8G'}):
            json, lib, param = relay.build(mod, target=target, params=params)
        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()

        out_shapes = out_shape if isinstance(out_shape, list) else [out_shape]
        results = result if isinstance(result, list) else [result]

        for idx, shape in enumerate(out_shapes):
            out = tvm.nd.empty(shape, ctx=ctx)
            out = rt_mod.get_output(idx, out)

            tvm.testing.assert_allclose(out.asnumpy(), results[idx], rtol=tol, atol=tol)

    check_graph_runtime_result()


def test_extern_vai_resnet18():
    """Test resnet18 model using Vitis-AI byoc flow"""
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    if not tvm.get_global_func("relay.ext.vitis_ai", True):
        print("skip because VITIS-AI codegen is not available")
        return

    dtype = 'float32'
    ishape = (1, 3, 224, 224)

    mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1)
    mod["main"] = bind_params_by_name(mod["main"], params)
    mod = annotation(mod, params, "DPUCADX8G")
    mod = transform.MergeCompilerRegions()(mod)
    mod = transform.PartitionGraph()(mod)

    ref_mod, params = relay.testing.resnet.get_workload(num_layers=18, batch_size=1)
    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu(0))
    i_data = np.random.uniform(0, 1, ishape).astype(dtype)

    ref_res = ref_ex.evaluate()(i_data, **params)

    check_result(mod, {"data": i_data},
                 (1, 1000), ref_res.asnumpy(), tol=1e-5, params=params)
if __name__ == "__main__":
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        sys.exit(0)
    test_extern_vai_resnet18()
