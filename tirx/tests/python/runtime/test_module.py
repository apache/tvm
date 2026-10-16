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
"""Runtime and FFI module loading and properties."""

import os
import subprocess
import sys

import numpy as np
import pytest
import tvm_ffi

import tvm_tirx
import tvm_tirx.testing
from tvm_tirx.support import cc


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


def test_module_properties(source_module, native_library):
    assert tvm_tirx.runtime.Module is tvm_ffi.Module
    assert tvm_tirx.runtime.load_module is tvm_ffi.load_module
    assert tvm_tirx.runtime.system_lib is tvm_ffi.system_lib
    assert not hasattr(source_module, "export_library")
    assert not hasattr(source_module, "time_evaluator")
    assert source_module.is_binary_serializable()
    assert not source_module.is_runnable()
    assert source_module.is_compilation_exportable()
    loaded = tvm_tirx.runtime.load_module(str(native_library))
    assert loaded.kind == "library"
    assert loaded.is_runnable()
    assert not loaded.is_compilation_exportable()
    assert loaded.implements_function("answer")
    assert loaded.get_function("answer")() == 42


def test_dso_module_load(native_library):
    module = tvm_tirx.runtime.load_module(str(native_library))
    a = tvm_tirx.runtime.tensor(np.arange(10, dtype="float32"))
    b = tvm_tirx.runtime.empty((10,), "float32")
    module(a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    # Verify loading and execution in a fresh process using only the runtime library.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import numpy as np
import tvm_tirx
assert tvm_tirx.base._RUNTIME_ONLY
module = tvm_tirx.runtime.load_module(sys.argv[1])
a = tvm_tirx.runtime.tensor(np.arange(10, dtype="float32"))
b = tvm_tirx.runtime.empty((10,), "float32")
module(a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)
""",
            str(native_library),
        ],
        env={**os.environ, "TVM_USE_RUNTIME_LIB": "1"},
        check=True,
    )


@pytest.mark.skipif(sys.platform == "win32", reason="system library registration on Unix")
def test_system_library(system_library_source, native_compile_options, tmp_path):
    source = tmp_path / "system.cc"
    source.write_text(system_library_source)
    path = tmp_path / "system.so"
    cc.create_shared(str(path), [str(source)], options=native_compile_options)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import ctypes
import sys
import numpy as np
import tvm_tirx
handle = ctypes.CDLL(sys.argv[1])
module = tvm_tirx.runtime.system_lib()
assert module["answer"]() == 42
a = tvm_tirx.runtime.tensor(np.arange(10, dtype="float32"))
b = tvm_tirx.runtime.empty((10,), "float32")
module["main"](a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)
""",
            str(path),
        ],
        check=True,
    )


if __name__ == "__main__":
    tvm_tirx.testing.main()
