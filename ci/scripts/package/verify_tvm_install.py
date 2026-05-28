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
"""Verify an installed TVM wheel imports and ships the expected runtime DSO."""

from __future__ import annotations

import os
from pathlib import Path
import sys


def expect_bool(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean value, got {value!r}")


def _clear_external_library_overrides() -> None:
    for name in ("TVM_LIBRARY_PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        if name in os.environ:
            print(f"clearing {name} before importing tvm")
            os.environ.pop(name, None)


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _assert_loaded_runtime_from_wheel(libdir: Path, runtime_candidates: list[Path]) -> None:
    import tvm.base as tvm_base  # pylint: disable=import-outside-toplevel

    loaded_runtime = Path(tvm_base._LIB_RUNTIME._name).resolve()  # pylint: disable=protected-access
    expected_runtime_paths = {candidate.resolve() for candidate in runtime_candidates}
    print("loaded runtime library:", loaded_runtime)
    if loaded_runtime not in expected_runtime_paths:
        expected = ", ".join(str(path) for path in sorted(expected_runtime_paths))
        raise RuntimeError(
            f"loaded runtime library is not from the installed wheel: "
            f"got {loaded_runtime}, expected one of {expected}"
        )


def _dynamic_llvm_libs(libdir: Path) -> list[Path]:
    if sys.platform == "darwin":
        patterns = ["libLLVM*.dylib"]
    elif sys.platform == "win32":
        patterns = ["LLVM*.dll", "libLLVM*.dll"]
    else:
        patterns = ["libLLVM*.so", "libLLVM*.so.*"]
    found: set[Path] = set()
    for pattern in patterns:
        found.update(libdir.glob(pattern))
    return sorted(found)


def main() -> int:
    _clear_external_library_overrides()

    import tvm  # pylint: disable=import-outside-toplevel

    root = Path(tvm.__file__).resolve().parent
    libdir = root / "lib"
    if sys.platform == "darwin":
        runtime_candidates = [libdir / "libtvm_runtime.dylib"]
        cuda_runtime_candidates = [libdir / "libtvm_runtime_cuda.dylib"]
    elif sys.platform == "win32":
        runtime_candidates = [libdir / "tvm_runtime.dll", libdir / "libtvm_runtime.dll"]
        cuda_runtime_candidates = [
            libdir / "tvm_runtime_cuda.dll",
            libdir / "libtvm_runtime_cuda.dll",
        ]
    else:
        runtime_candidates = [libdir / "libtvm_runtime.so"]
        cuda_runtime_candidates = [libdir / "libtvm_runtime_cuda.so"]

    print("tvm version:", tvm.__version__)
    print("tvm package:", root)
    llvm_enabled = bool(tvm.runtime.enabled("llvm"))
    cuda_enabled = bool(tvm.runtime.enabled("cuda"))
    runtime_lib = _first_existing(runtime_candidates)
    cuda_runtime = _first_existing(cuda_runtime_candidates)
    runtime_present = any(candidate.exists() for candidate in runtime_candidates)
    cuda_runtime_present = any(candidate.exists() for candidate in cuda_runtime_candidates)
    dynamic_llvm_libs = _dynamic_llvm_libs(libdir)

    print("llvm enabled:", llvm_enabled)
    print("cuda runtime enabled:", cuda_enabled)
    print("runtime library:", runtime_lib)
    if not runtime_present:
        raise RuntimeError(
            "runtime library is missing; checked "
            + ", ".join(str(candidate) for candidate in runtime_candidates)
        )
    _assert_loaded_runtime_from_wheel(libdir, runtime_candidates)
    print("cuda runtime present:", cuda_runtime_present)
    if cuda_runtime_present:
        print("cuda runtime library:", cuda_runtime)
    print("dynamic LLVM libraries:", [str(path) for path in dynamic_llvm_libs])

    expected_llvm = expect_bool("TVM_EXPECT_LLVM_ENABLED")
    if expected_llvm is not None and llvm_enabled != expected_llvm:
        raise RuntimeError(f"llvm enabled: expected {expected_llvm}, got {llvm_enabled}")
    expected_static_llvm = expect_bool("TVM_EXPECT_STATIC_LLVM")
    if expected_static_llvm and dynamic_llvm_libs:
        raise RuntimeError(
            "expected LLVM to be linked statically, but dynamic LLVM libraries are present: "
            + ", ".join(str(path) for path in dynamic_llvm_libs)
        )
    expected_cuda_runtime = expect_bool("TVM_EXPECT_CUDA_RUNTIME")
    if expected_cuda_runtime is not None and cuda_runtime_present != expected_cuda_runtime:
        raise RuntimeError(
            f"cuda runtime present: expected {expected_cuda_runtime}, got {cuda_runtime_present}"
        )
    expected_cuda = expect_bool("TVM_EXPECT_CUDA_ENABLED")
    if expected_cuda is not None and cuda_enabled != expected_cuda:
        raise RuntimeError(f"cuda runtime enabled: expected {expected_cuda}, got {cuda_enabled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
