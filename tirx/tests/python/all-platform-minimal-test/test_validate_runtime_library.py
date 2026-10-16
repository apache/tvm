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
"""Post-install runtime and AOT checks.

These live in ``tests/python/all-platform-minimal-test`` so the standard suite and
the cibuildwheel ``test-command`` run a single pytest invocation. The assertions
for wheel packaging check things the rest of the suite cannot check -- that LLVM is
enabled (the other LLVM test merely *skips* when LLVM is absent) and that the CUDA
runtime library got bundled -- so each is gated behind a ``TVM_WHEEL_EXPECT_*`` env
var and SKIPS unless that var is set. cibuildwheel sets the vars (see
``CIBW_TEST_ENVIRONMENT`` in ``.github/actions/build-wheel-for-publish``); ordinary
source-build CI (e.g. ``main.yml``) leaves them unset, so these tests skip there and
never fail a non-wheel / non-LLVM / non-CUDA build. The CPU AOT case also runs
on Linux/macOS source builds with LLVM enabled.
"""

import glob
import os
import subprocess
import sys
from pathlib import Path

import pytest

import tvm_tirx


def test_llvm_enabled():
    """Every published TVM wheel ships with LLVM enabled. Only assert this when
    validating a wheel (``TVM_WHEEL_EXPECT_LLVM=1``); skip otherwise so source
    builds with LLVM off do not fail."""
    if os.environ.get("TVM_WHEEL_EXPECT_LLVM") != "1":
        pytest.skip("LLVM enablement only asserted during wheel validation")
    assert tvm_tirx.runtime.enabled("llvm"), "wheel was not built with LLVM enabled"


def test_cuda_runtime_present():
    """The bundled CUDA runtime library must be present in tvm/lib."""
    if os.environ.get("TVM_WHEEL_EXPECT_CUDA_RUNTIME") != "1":
        pytest.skip("CUDA runtime not expected in this wheel")
    libdir = Path(tvm_tirx.__file__).resolve().parent / "lib"
    present = glob.glob(str(libdir / "libtvm_runtime_cuda.*")) or glob.glob(
        str(libdir / "tvm_runtime_cuda.*")
    )
    assert present, "CUDA runtime expected but not bundled in tvm/lib"


def test_cpu_aot(tmp_path):
    if sys.platform not in ("linux", "darwin") or not tvm_tirx.runtime.enabled("llvm"):
        pytest.skip("CPU AOT requires Linux/macOS and LLVM")
    from tvm_tirx.script import tirx as T

    @T.function
    def add_one(A: T.Tensor((16,), "int32", align=4), B: T.Tensor((16,), "int32", align=4)):
        for i in range(16):
            B[i] = A[i] + 1

    module = tvm_tirx.tirx.compile(add_one, target="llvm")
    path = tmp_path / "model.so"
    module.export_library(path)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import numpy as np
import tvm_ffi
module = tvm_ffi.load_module(sys.argv[1])
a = np.arange(16, dtype="int32")
b = np.empty_like(a)
module["add_one"](tvm_ffi.from_dlpack(a), tvm_ffi.from_dlpack(b))
np.testing.assert_array_equal(b, a + 1)
assert "tvm" not in sys.modules and "tvm_tirx" not in sys.modules
""",
            str(path),
        ],
        check=True,
    )
