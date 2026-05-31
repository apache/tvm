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
``[tool.cibuildwheel]``). Each check is opt-in via an environment variable so
the same tests apply across the wheel matrix: when the variable is unset the
check is skipped.
"""

import glob
import os
from pathlib import Path

import pytest

import tvm


def _expect(name):
    """Return True/False for an expectation env var, or None when unset."""
    value = os.environ.get(name, "")
    if value == "":
        return None
    return value.strip().lower() in ("1", "true", "yes", "on")


def _libdir():
    return Path(tvm.__file__).resolve().parent / "lib"


def test_llvm_compile():
    """import tvm and run a minimal LLVM compile+execute."""
    if _expect("TVM_EXPECT_LLVM_ENABLED") is False:
        pytest.skip("LLVM not expected in this wheel")
    if not tvm.runtime.enabled("llvm"):
        if _expect("TVM_EXPECT_LLVM_ENABLED"):
            pytest.fail("llvm runtime expected but not enabled")
        pytest.skip("llvm runtime not enabled")

    import numpy as np

    from tvm import te

    n = 8
    a = te.placeholder((n,), name="a", dtype="float32")
    b = te.placeholder((n,), name="b", dtype="float32")
    c = te.compute((n,), lambda i: a[i] + b[i], name="c")
    exe = tvm.compile(te.create_prim_func([a, b, c]), target="llvm")

    dev = tvm.cpu()
    a_nd = tvm.runtime.tensor(np.arange(n, dtype="float32"), dev)
    b_nd = tvm.runtime.tensor(np.arange(n, dtype="float32") * 2, dev)
    c_nd = tvm.runtime.tensor(np.zeros(n, dtype="float32"), dev)
    exe(a_nd, b_nd, c_nd)
    np.testing.assert_allclose(c_nd.numpy(), a_nd.numpy() + b_nd.numpy(), rtol=1e-6)


def test_no_dynamic_llvm():
    """When static LLVM is expected, the wheel must not ship a dynamic libLLVM."""
    if not _expect("TVM_EXPECT_STATIC_LLVM"):
        pytest.skip("static LLVM not required for this wheel")
    dynamic = glob.glob(str(_libdir() / "libLLVM*"))
    assert not dynamic, f"wheel ships dynamic LLVM libraries: {dynamic}"


def test_cuda_runtime_present():
    """When a CUDA runtime is expected, libtvm_runtime_cuda must be bundled."""
    expected = _expect("TVM_EXPECT_CUDA_RUNTIME")
    if expected is None:
        pytest.skip("CUDA runtime expectation not set")
    libdir = _libdir()
    present = bool(
        glob.glob(str(libdir / "libtvm_runtime_cuda.*"))
        or glob.glob(str(libdir / "tvm_runtime_cuda.*"))
    )
    assert present == expected, f"cuda runtime present={present}, expected={expected}"
