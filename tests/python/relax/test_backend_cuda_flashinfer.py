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
"""Host checks for FlashInfer's generated TVM bindings."""

import pytest
import tvm_ffi.cpp

from tvm.relax.backend.cuda.flashinfer import (
    _rename_exported_func_names,
    _wrap_prefill_plan,
)


@pytest.mark.parametrize("num_plan_args", [19, 21])
def test_prefill_plan_interface(tmp_path, num_plan_args):
    parameters = ", ".join(f"int64_t arg{i}" for i in range(num_plan_args))
    returned = ", ".join(f"arg{i}" for i in range(num_plan_args))
    source = tmp_path / "batch_prefill_jit_binding.cu"
    source.write_text(
        "#include <tvm/ffi/function.h>\n"
        "#include <tvm/ffi/container/array.h>\n"
        "using tvm::ffi::Array;\n"
        "using TensorView = int64_t;\n"
        f"Array<int64_t> BatchPrefillWithKVCachePlan({parameters}) {{\n"
        f"  return {{{returned}}};\n"
        "}\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchPrefillWithKVCachePlan);\n"
    )
    _wrap_prefill_plan([source])
    _rename_exported_func_names([source], "batch_prefill")
    cpp_source = tmp_path / "binding.cpp"
    cpp_source.write_text(source.read_text())
    module = tvm_ffi.cpp.load(name=f"flashinfer_plan_{num_plan_args}", cpp_files=[str(cpp_source)])
    args = list(range(1, 20))
    args[11], args[14], args[17] = True, False, True
    expected = [int(arg) for arg in args]
    if num_plan_args == 21:
        expected += [0, 0]
    assert list(module["batch_prefill_plan"](*args)) == expected
