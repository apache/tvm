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

import faulthandler
import os
from pathlib import Path
import signal
import sys

faulthandler.enable(all_threads=True)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import numpy as np


def log(*args: object) -> None:
    print(*args, flush=True)


def _enable_python_fault_handler() -> None:
    """Install Python's signal handler after native libraries may install theirs."""
    faulthandler.enable(all_threads=True)
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except (AttributeError, RuntimeError, ValueError):
        pass


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
            log(f"clearing {name} before importing tvm")
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
    log("loaded runtime library:", loaded_runtime)
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


def _log_tvm_ffi_details() -> None:
    import tvm_ffi  # pylint: disable=import-outside-toplevel

    ffi_lib = getattr(tvm_ffi, "LIB", None)
    ffi_lib_path = getattr(ffi_lib, "_name", None)
    log("tvm_ffi version:", getattr(tvm_ffi, "__version__", "<unknown>"))
    log("tvm_ffi package:", Path(tvm_ffi.__file__).resolve().parent)
    if ffi_lib_path:
        log("tvm_ffi library:", Path(ffi_lib_path).resolve())


def _log_loaded_native_libraries() -> None:
    if sys.platform != "linux":
        return
    maps_path = Path("/proc/self/maps")
    if not maps_path.exists():
        return
    interesting_names = (
        "libtvm",
        "libLLVM",
        "libstdc++",
        "libgcc_s",
        "libxml2",
        "libzstd",
        "liblzma",
    )
    loaded: set[str] = set()
    for line in maps_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        path = line.rsplit(maxsplit=1)[-1]
        if "/" not in path:
            continue
        if any(name in path for name in interesting_names):
            loaded.add(path)
    log("loaded native libraries:")
    for path in sorted(loaded):
        log("  ", path)


def _verify_llvm_tirx_compile() -> None:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import te  # pylint: disable=import-outside-toplevel

    log("llvm tirx compile smoke: starting")
    extent = 8
    log("llvm tirx compile smoke: create numpy inputs")
    lhs_np = np.arange(extent, dtype="float32")
    rhs_np = np.arange(extent, dtype="float32") * np.float32(2)
    out_np = np.zeros(extent, dtype="float32")

    log("llvm tirx compile smoke: create placeholders")
    lhs = te.placeholder((extent,), name="lhs", dtype="float32")
    rhs = te.placeholder((extent,), name="rhs", dtype="float32")
    log("llvm tirx compile smoke: create compute")
    out = te.compute((extent,), lambda i: lhs[i] + rhs[i], name="out")
    log("llvm tirx compile smoke: create prim func")
    prim_func = te.create_prim_func([lhs, rhs, out])
    log("llvm tirx compile smoke: compile")
    executable = tvm.compile(prim_func, target="llvm")

    log("llvm tirx compile smoke: create tensors")
    dev = tvm.cpu()
    lhs_t = tvm.runtime.tensor(lhs_np, dev)
    rhs_t = tvm.runtime.tensor(rhs_np, dev)
    out_t = tvm.runtime.tensor(out_np, dev)
    log("llvm tirx compile smoke: execute")
    executable(lhs_t, rhs_t, out_t)
    log("llvm tirx compile smoke: check output")
    np.testing.assert_allclose(out_t.numpy(), lhs_np + rhs_np, rtol=1e-6)
    log("llvm tirx compile smoke: passed")


def _verify_relax_compile() -> None:
    import tvm  # pylint: disable=import-outside-toplevel
    from tvm import relax  # pylint: disable=import-outside-toplevel

    log("llvm relax compile smoke: starting")
    log("llvm relax compile smoke: create numpy inputs")
    lhs_np = np.arange(8, dtype="float32")
    rhs_np = np.arange(8, dtype="float32") * np.float32(3)
    dev = tvm.cpu()

    log("llvm relax compile smoke: create vars")
    lhs = relax.Var("lhs", relax.TensorStructInfo((8,), "float32"))
    rhs = relax.Var("rhs", relax.TensorStructInfo((8,), "float32"))
    log("llvm relax compile smoke: create module")
    builder = relax.BlockBuilder()
    with builder.function("main", [lhs, rhs]):
        out = builder.emit(relax.op.add(lhs, rhs))
        builder.emit_func_output(out)

    log("llvm relax compile smoke: compile")
    executable = tvm.compile(builder.get(), target="llvm")
    log("llvm relax compile smoke: create vm")
    vm = relax.VirtualMachine(executable, dev)
    log("llvm relax compile smoke: execute")
    out = vm["main"](tvm.runtime.tensor(lhs_np, dev), tvm.runtime.tensor(rhs_np, dev))
    log("llvm relax compile smoke: check output")
    np.testing.assert_allclose(out.numpy(), lhs_np + rhs_np, rtol=1e-6)
    log("llvm relax compile smoke: passed")


def main() -> int:
    _clear_external_library_overrides()

    log("import tvm: starting")
    import tvm  # pylint: disable=import-outside-toplevel
    log("import tvm: passed")
    _enable_python_fault_handler()
    _log_tvm_ffi_details()

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

    log("tvm version:", tvm.__version__)
    log("tvm package:", root)
    llvm_enabled = bool(tvm.runtime.enabled("llvm"))
    cuda_enabled = bool(tvm.runtime.enabled("cuda"))
    runtime_lib = _first_existing(runtime_candidates)
    cuda_runtime = _first_existing(cuda_runtime_candidates)
    runtime_present = any(candidate.exists() for candidate in runtime_candidates)
    cuda_runtime_present = any(candidate.exists() for candidate in cuda_runtime_candidates)
    dynamic_llvm_libs = _dynamic_llvm_libs(libdir)

    log("llvm enabled:", llvm_enabled)
    log("cuda runtime enabled:", cuda_enabled)
    log("runtime library:", runtime_lib)
    if not runtime_present:
        raise RuntimeError(
            "runtime library is missing; checked "
            + ", ".join(str(candidate) for candidate in runtime_candidates)
        )
    _assert_loaded_runtime_from_wheel(libdir, runtime_candidates)
    log("cuda runtime present:", cuda_runtime_present)
    if cuda_runtime_present:
        log("cuda runtime library:", cuda_runtime)
    log("dynamic LLVM libraries:", [str(path) for path in dynamic_llvm_libs])
    _log_loaded_native_libraries()

    expected_llvm = expect_bool("TVM_EXPECT_LLVM_ENABLED")
    if expected_llvm is not None and llvm_enabled != expected_llvm:
        raise RuntimeError(f"llvm enabled: expected {expected_llvm}, got {llvm_enabled}")
    if llvm_enabled:
        _verify_llvm_tirx_compile()
        _verify_relax_compile()
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
    log("verify tvm install: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
