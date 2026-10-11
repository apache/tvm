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
"""Device-selection C entry points in generated host modules."""

import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.parametrize("target,system_lib", [("c", False), ("llvm", False), ("llvm", True)])
@pytest.mark.parametrize("operation", ["set", "invalid_set", "get"])
def test_backend_device_api_module(tmp_path, target, system_lib, operation):
    if target == "llvm" and not env.has_llvm():
        pytest.skip("LLVM is not available")

    if operation == "get":

        @T.function
        def main(
            self: T.handle, args: T.handle, num_args: T.int32, result: T.handle("int32")
        ) -> T.int32:
            T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main"})
            status = T.call_extern("TVMBackendGetDeviceIndex", 1, result, ty="int32")
            if status != 0:
                T.evaluate(T.throw_last_error())
            return 0
    else:
        device_type = -1 if operation == "invalid_set" else 1

        @T.function
        def main(self: T.handle, args: T.handle, num_args: T.int32, result: T.handle) -> T.int32:
            T.func_attr({"calling_conv": 1, "global_symbol": "__tvm_ffi_main"})
            status = T.call_extern("TVMBackendSetDevice", device_type, 0, ty="int32")
            if status != 0:
                T.evaluate(T.throw_last_error())
            return 0

    ir_module = tvm.IRModule.from_expr(main)
    if system_lib:
        ir_module = ir_module.with_attr("system_lib_prefix", "device_api_")
    mod = tvm.tirx.build(ir_module, target=target)
    source = mod.inspect_source()
    symbol = "TVMBackendGetDeviceIndex" if operation == "get" else "TVMBackendSetDevice"
    assert symbol in source
    if target == "llvm":
        assert ("@" if system_lib else "@__") + symbol in source
    path = str(tmp_path / "device_api.so")
    mod.export_library(path)
    loaded = tvm.runtime.load_module(path)
    for module in [mod, loaded] if target == "llvm" else [loaded]:
        if operation == "get":
            with pytest.raises(RuntimeError, match="GetDeviceIndex"):
                module["main"]()
        elif operation == "invalid_set":
            with pytest.raises(ValueError, match="local device type"):
                module["main"]()
        else:
            module["main"]()


def test_cuda_host_device_selection():
    @T.function
    def main(A: T.Tensor((1,), "float32")):
        T.func_attr({"target": T.target("cuda", host="cuda_host")})
        T.evaluate(0)

    lowered = tvm.tirx.transform.MakePackedAPI()(tvm.IRModule.from_expr(main))
    source = tvm.get_global_func("target.build.cuda_host")(
        lowered, tvm.target.Target("cuda_host")
    ).inspect_source()
    assert "CUDADeviceGuard" in source
    assert "cuda_host requires a CUDA device" in source
    assert "TVMBackendSetDevice" not in source
    assert "__tvm_set_device" not in source


if __name__ == "__main__":
    tvm.testing.main()
