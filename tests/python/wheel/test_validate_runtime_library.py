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
"""Post-install checks for a built TVM wheel.

Run by cibuildwheel against the installed wheel (``test-command`` in
``[tool.cibuildwheel]``). These assert the two wheel-specific things the standard
``tests/python/all-platform-minimal-test`` suite cannot: that LLVM is enabled (its
LLVM test merely *skips* when LLVM is absent), and that the CUDA runtime library
got bundled (when ``TVM_WHEEL_EXPECT_CUDA_RUNTIME=1``). The functional LLVM
compile / ndarray ops are covered by that all-platform suite.
"""

import glob
import os
from pathlib import Path

import pytest

import tvm


def test_llvm_enabled():
    """Every TVM wheel ships with LLVM enabled. The all-platform suite only skips
    (does not fail) when LLVM is absent, so assert presence here."""
    assert tvm.runtime.enabled("llvm"), "wheel was not built with LLVM enabled"


def test_cuda_runtime_present():
    """The bundled CUDA runtime library must be present in tvm/lib."""
    if os.environ.get("TVM_WHEEL_EXPECT_CUDA_RUNTIME") != "1":
        pytest.skip("CUDA runtime not expected in this wheel")
    libdir = Path(tvm.__file__).resolve().parent / "lib"
    present = glob.glob(str(libdir / "libtvm_runtime_cuda.*")) or glob.glob(
        str(libdir / "tvm_runtime_cuda.*")
    )
    assert present, "CUDA runtime expected but not bundled in tvm/lib"
