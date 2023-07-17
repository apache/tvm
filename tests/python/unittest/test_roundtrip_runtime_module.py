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
""" Test roundtrip of runtime modules """
# pylint: disable=missing-docstring

import pytest
import tvm
import tvm.testing
from tvm import TVMError
from tvm import relay
import numpy as np
from tvm.contrib.graph_executor import GraphModule


def test_csource_module():
    mod = tvm.runtime._ffi_api.CSourceModuleCreate("", "cc", [], None)
    # source module that is not binary serializable.
    # Thus, it would raise an error.
    assert not mod.is_binary_serializable
    with pytest.raises(TVMError):
        tvm.ir.load_json(tvm.ir.save_json(mod))


def test_aot_module():
    mod = tvm.get_global_func("relay.build_module._AOTExecutorCodegen")()
    # aot module that is not binary serializable.
    # Thus, it would raise an error.
    assert not mod.is_binary_serializable
    with pytest.raises(TVMError):
        tvm.ir.load_json(tvm.ir.save_json(mod))


@tvm.testing.requires_gpu
def test_recursive_imports():
    x = relay.var("x", shape=(1, 10), dtype="float32")
    y = relay.var("y", shape=(1, 10), dtype="float32")
    z = relay.add(x, y)
    func = relay.Function([x, y], z)
    mod = relay.build_module._build_module_no_factory(func, target="cuda")

    assert mod.is_binary_serializable
    # GraphExecutorFactory Module contains LLVM Module and LLVM Module contains cuda Module.
    assert mod.type_key == "GraphExecutorFactory"
    assert mod.imported_modules[0].type_key == "llvm"
    assert mod.imported_modules[0].imported_modules[0].type_key == "cuda"

    new_mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    assert new_mod.is_binary_serializable
    # GraphExecutorFactory Module contains LLVM Module and LLVM Module contains cuda Module.
    assert new_mod.type_key == "GraphExecutorFactory"
    # This type key is now `library` rather than llvm.
    assert new_mod.imported_modules[0].type_key == "library"
    assert new_mod.imported_modules[0].imported_modules[0].type_key == "cuda"

    dev = tvm.cuda()
    data_x = tvm.nd.array(np.random.rand(1, 10).astype("float32"), dev)
    data_y = tvm.nd.array(np.random.rand(1, 10).astype("float32"), dev)

    graph_mod = GraphModule(mod["default"](dev))
    graph_mod.set_input("x", data_x)
    graph_mod.set_input("y", data_y)
    graph_mod.run()
    expected = graph_mod.get_output(0)

    graph_mod = GraphModule(new_mod["default"](dev))
    graph_mod.set_input("x", data_x)
    graph_mod.set_input("y", data_y)
    graph_mod.run()
    output = graph_mod.get_output(0)
    tvm.testing.assert_allclose(output.numpy(), expected.numpy(), atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
