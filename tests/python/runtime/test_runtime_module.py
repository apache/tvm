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
"""Runtime module loading, properties, exports, and timing."""

import os
import subprocess
import sys
import time

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.runtime.module import BenchmarkResult
from tvm.support import cc


def test_module_properties(source_module, native_library):
    assert source_module.is_binary_serializable()
    assert not source_module.is_runnable()
    assert source_module.is_compilation_exportable()
    loaded = tvm.runtime.load_module(str(native_library))
    assert loaded.kind == "library"
    assert loaded.is_runnable()
    assert not loaded.is_compilation_exportable()
    assert loaded.implements_function("answer")
    assert loaded.get_function("answer")() == 42


def test_dso_module_load(native_library):
    module = tvm.runtime.load_module(str(native_library))
    a = tvm.runtime.tensor(np.arange(10, dtype="float32"))
    b = tvm.runtime.empty((10,), "float32")
    module(a, b)
    np.testing.assert_equal(b.numpy(), a.numpy() + 1)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import numpy as np
import tvm
assert tvm.base._RUNTIME_ONLY
module = tvm.runtime.load_module(sys.argv[1])
a = tvm.runtime.tensor(np.arange(10, dtype="float32"))
b = tvm.runtime.empty((10,), "float32")
module(a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)
""",
            str(native_library),
        ],
        env={**os.environ, "TVM_USE_RUNTIME_LIB": "1"},
        check=True,
    )


def test_import_static_library(source_module, native_compile_options, tmp_path):
    source = tmp_path / "other.cc"
    source.write_text(
        "#include <tvm/ffi/function.h>\n"
        "int other() { return 17; }\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(other, other);\n"
    )
    obj = tmp_path / "other.o"
    cc.create_shared(str(obj), [str(source)], options=["-c", *native_compile_options])
    imported = tvm.runtime.load_static_library(str(obj), ["other"])
    assert imported.implements_function("other")
    assert imported.is_compilation_exportable()
    source_module.import_module(imported)
    path = tmp_path / "combined.so"
    source_module.export_library(str(path), options=native_compile_options)
    loaded = tvm.runtime.load_module(str(path))
    assert loaded.kind == "library"
    assert len(loaded.imports) == 0
    for name, expected in (("answer", 42), ("other", 17)):
        assert loaded.implements_function(name)
        assert loaded.get_function(name)() == expected
    assert not loaded.is_compilation_exportable()


def test_combine_module(native_object, native_compile_options, tmp_path):
    source = tmp_path / "other.cc"
    source.write_text(
        "#include <tvm/ffi/function.h>\n"
        "int other() { return 17; }\n"
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(other, other);\n"
    )
    obj = tmp_path / "other.o"
    cc.create_shared(str(obj), [str(source)], options=["-c", *native_compile_options])
    path = tmp_path / "combined.so"
    cc.create_shared(str(path), [str(native_object), str(obj)])
    module = tvm.runtime.load_module(str(path))
    assert module["answer"]() == 42
    assert module["other"]() == 17


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
import tvm
handle = ctypes.CDLL(sys.argv[1])
module = tvm.runtime.system_lib()
assert module["answer"]() == 42
a = tvm.runtime.tensor(np.arange(10, dtype="float32"))
b = tvm.runtime.empty((10,), "float32")
module["main"](a, b)
np.testing.assert_equal(b.numpy(), a.numpy() + 1)
""",
            str(path),
        ],
        check=True,
    )


def test_min_repeat_ms(native_library):
    calls = []

    def callback():
        time.sleep(0.1)
        calls.append(1)

    module = tvm.runtime.load_module(str(native_library))
    module.time_evaluator("callback", tvm.cpu(), number=1, repeat=1)(callback)
    assert len(calls) == 2
    module.time_evaluator("callback", tvm.cpu(), number=1, repeat=1, min_repeat_ms=1000)(callback)
    assert len(calls) > 12


def test_benchmark_result():
    result = BenchmarkResult([1, 2, 2, 5])
    assert result.mean == 2.5
    assert result.median == 2.0
    assert result.min == 1
    assert result.max == 5
    assert result.std == 1.5


if __name__ == "__main__":
    tvm.testing.main()
