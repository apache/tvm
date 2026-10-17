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
"""Shared native module fixtures for runtime and code generation coverage."""

import pytest


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
void callback(tvm::ffi::Function func) { func(); }
TVM_FFI_DLL_EXPORT_TYPED_FUNC(main, add_one);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(answer, answer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(callback, callback);
"""


@pytest.fixture(scope="module")
def native_compile_options():
    import tvm_ffi

    return [
        "-std=c++17",
        "-I" + tvm_ffi.libinfo.find_include_path(),
        "-I" + tvm_ffi.libinfo.find_dlpack_include_path(),
    ]


@pytest.fixture
def source_module(native_source):
    import tvm_tirx

    return tvm_tirx.runtime._ffi_api.CSourceModuleCreate(
        native_source, "cc", ["main", "answer", "callback"], None
    )


@pytest.fixture(scope="module")
def native_object(tmp_path_factory, native_source, native_compile_options):
    from tvm_tirx.support import cc

    directory = tmp_path_factory.mktemp("runtime_module")
    source = directory / "module.cc"
    source.write_text(native_source)
    path = directory / "module.o"
    cc.create_shared(str(path), [str(source)], options=["-c", *native_compile_options])
    return path


@pytest.fixture(scope="module")
def native_library(native_object):
    from tvm_tirx.support import cc

    path = native_object.with_suffix(".so")
    cc.create_shared(str(path), [str(native_object)])
    return path
