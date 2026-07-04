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
import ctypes

import numpy as np

import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.support import cc, utils


def _lookup_module():
    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def ramp(A: T.handle):
            T.func_attr({"global_symbol": "ramp"})
            T.call_packed("test_codegen_lookup", A)

    return Module


def test_static_init():
    @tvm.register_global_func("test_static_callback")
    def test_cb(sh, A):
        assert isinstance(sh, ctypes.c_void_p)
        return sh

    @I.ir_module
    class Module:
        @T.prim_func(s_tir=True)
        def ramp(A: T.handle):
            T.func_attr({"global_symbol": "ramp"})
            n = T.int64()
            Ab = T.match_buffer(A, (n,), "int64")
            T.call_packed(
                "test_static_callback",
                T.call_intrin("handle", "tirx.tvm_static_handle"),
                Ab.data,
            )

    mod = Module
    f = tvm.driver.build(mod, target="llvm")
    a = tvm.runtime.tensor(np.zeros(10, dtype="int64"))
    f(a)


def test_generated_lookup_uses_tvm_ffi_symbol():
    mod = _lookup_module()

    c_source = tvm.compile(mod, target="c").mod.inspect_source()
    assert "TVMFFIEnvModLookupFromImports(" in c_source
    assert "TVMBackendGetFuncFromEnv(" not in c_source

    llvm_source = tvm.compile(mod, target="llvm").mod.inspect_source("ll")
    assert "@__TVMFFIEnvModLookupFromImports" in llvm_source
    assert "@__TVMBackendGetFuncFromEnv" not in llvm_source


def test_legacy_lookup_context_slot_is_still_populated():
    temp = utils.tempdir()
    source_path = temp.relpath("legacy_lookup_context.cc")
    library_path = temp.relpath("legacy_lookup_context.so")
    with open(source_path, "w", encoding="utf-8") as source_file:
        source_file.write(
            r"""
#if defined(_WIN32)
#define TVM_TEST_EXPORT __declspec(dllexport)
#else
#define TVM_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {
TVM_TEST_EXPORT void* __TVMBackendGetFuncFromEnv = nullptr;
TVM_TEST_EXPORT void* __TVMFFIEnvModLookupFromImports = nullptr;
TVM_TEST_EXPORT void* __tvm_ffi__library_ctx = nullptr;
}
"""
        )

    cc.create_shared(library_path, [source_path])
    loaded_module = tvm.runtime.load_module(library_path)
    loaded_library = ctypes.CDLL(library_path)
    assert loaded_module is not None
    assert ctypes.c_void_p.in_dll(loaded_library, "__TVMBackendGetFuncFromEnv").value
    assert ctypes.c_void_p.in_dll(loaded_library, "__TVMFFIEnvModLookupFromImports").value


if __name__ == "__main__":
    test_static_init()
