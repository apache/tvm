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
"""Small native modules shared by runtime tests."""

import pytest
import tvm_ffi

import tvm
from tvm.support import cc


@pytest.fixture(scope="module")
def native_source():
    return r"""
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
void add_one(tvm::ffi::TensorView a, tvm::ffi::TensorView b) {
  for (int64_t i = 0; i < a.size(0); ++i) {
    static_cast<float*>(b.data_ptr())[i] = static_cast<float*>(a.data_ptr())[i] + 1;
  }
}
int64_t answer() { return 42; }
int64_t with_arg(int64_t) { return 42; }
void callback(tvm::ffi::Function func) { func(); }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(main, add_one);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(answer, answer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(with_arg, with_arg);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(callback, callback);
"""


@pytest.fixture(scope="module")
def native_compile_options():
    return [
        "-std=c++17",
        "-I" + tvm_ffi.libinfo.find_include_path(),
        "-I" + tvm_ffi.libinfo.find_dlpack_include_path(),
    ]


@pytest.fixture
def source_module(native_source):
    return tvm.runtime._ffi_api.CSourceModuleCreate(
        native_source, "cc", ["main", "answer", "with_arg", "callback"], None
    )


@pytest.fixture(scope="module")
def native_object(tmp_path_factory, native_source, native_compile_options):
    directory = tmp_path_factory.mktemp("runtime_module")
    source = directory / "module.cc"
    source.write_text(native_source)
    path = directory / "module.o"
    cc.create_shared(str(path), [str(source)], options=["-c", *native_compile_options])
    return path


@pytest.fixture(scope="module")
def native_library(native_object):
    path = native_object.with_suffix(".so")
    cc.create_shared(str(path), [str(native_object)])
    return path


@pytest.fixture(scope="module")
def system_library_source(native_source):
    return (
        native_source
        + r"""
#include <tvm/ffi/extra/c_env_api.h>
struct RegisterSystemLibrary {
  RegisterSystemLibrary() {
    TVMFFIEnvModRegisterSystemLibSymbol("__tvm_ffi_main", (void*)__tvm_ffi_main);
    TVMFFIEnvModRegisterSystemLibSymbol("__tvm_ffi_answer", (void*)__tvm_ffi_answer);
  }
};
static RegisterSystemLibrary register_system_library;
"""
    )
