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
from tvm import relay
from tvm.meta_schedule.testing.custom_builder_runner import (
    build_relay_with_tensorrt,
)
from tvm.relay import testing
from tvm.relay.op.contrib import tensorrt
from tvm import TVMError

has_tensorrt_codegen = pytest.mark.skipif(
    not tensorrt.is_tensorrt_compiler_enabled(),
    reason="TensorRT codegen not available",
)
has_tensorrt_runtime = pytest.mark.skipif(
    not tensorrt.is_tensorrt_runtime_enabled(),
    reason="TensorRT runtime not available",
)

@has_tensorrt_codegen
@has_tensorrt_runtime
def test_tensorrt():
    data_shape = (1, 1280, 14, 14)
    dtype = "float32"

    data = relay.var("data", relay.TensorType(data_shape, dtype))
    net = relay.nn.relu(data)
    inputs = relay.analysis.free_vars(net)
    f = relay.Function(inputs, net)
    mod, params = testing.create_workload(f)
    mod = build_relay_with_tensorrt(mod, "cuda", params)
    
    # json runtime is binary serializable. so roundtrip works.
    assert mod.is_binary_serializable
    tvm.ir.load_json(tvm.ir.save_json(mod))    
    

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

if __name__ == "__main__":
    tvm.testing.main()