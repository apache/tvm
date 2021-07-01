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
"""BYOC External Compiler Implementation for Android NNAPI target."""
import tvm
from .relayir_to_nnapi_converter import Converter as _Converter

def _get_c_type(tipe): 
    """Get matching C type for Relay types."""
    dtype = str(tipe.dtype)
    if dtype == "float32": 
        return "float"
    if dtype == "float16": 
        return "uint16_t"
    if dtype == "int32": 
        return "int32_t"
    assert dtype == "int64", f"{dtype} is unsupported"
    return "int64_t"


@tvm.register_func("relay.ext.android_nnapi")
def _codegen(func): 
    """Codegen Relay IR to Android NNAPI.

    Parameters
    ----------
    func: tvm.relay.Function
        The Relay IR function to be codegened.

    Returns
    -------
    mod: runtime.CSourceModule
        The resulting Android NNAPI in C++ source code.

    Notes
    -----
    Certain function attributes should be configured:

    * func.attrs.NnapiTargetVersion: (int) The targeting API level of Android.
    """
    assert isinstance(func, tvm.relay.Function), "Only Function can be codegened to Android NNAPI"
    code = """#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <android/NeuralNetworks.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <dlpack/dlpack.h>

namespace {
"""

    sid = str(func.attrs.global_symbol);
    class_name = sid + "_class";
    options = {
        "class": {
            "self": {
                "name": class_name,
            },
        },
        "target": {
            "api_level": int(func.attrs.NnapiTargetVersion),
        },
    }
    code += _Converter(options).convert(func)
    code += "\n"

    instance_name = sid + "_model"
    code += f"  {class_name} {instance_name};\n"

    sid_impl_name = sid + "_";
    code += f"  void {sid_impl_name}(::tvm::runtime::TVMArgs args, ::tvm::runtime::TVMRetValue *rv) {{\n"
    code += f"    CHECK_EQ(args.num_args, {len(func.params) + 1}) << \"num_args is expected to be {len(func.params) + 1}\";\n"
    code += f"    {instance_name}.execute(\n"
    for i, p in enumerate(func.params): 
        assert isinstance(p.checked_type, tvm.relay.TensorType), "Function parameter is expected to be a tensor"
        code += f"      reinterpret_cast< {_get_c_type(p.checked_type)}* >(args[{i}].operator DLTensor*()->data), \n"
    assert isinstance(func.body.checked_type, tvm.relay.TensorType), "Function output is expected to be a tensor"
    code += f"      reinterpret_cast< {_get_c_type(func.body.checked_type)}* >(args[{len(func.params)}].operator DLTensor*()->data)\n"
    code += f"    );\n"
    code += "    *rv = 0;\n"
    code += f"  }} // {sid_impl_name}\n"
    code += "} // anonymous namespace\n"
    code += f"TVM_DLL_EXPORT_PACKED_FUNC({sid}, {sid_impl_name});\n"

    return tvm.get_global_func("runtime.CSourceModuleCreate")(code, "c", [sid], [])
