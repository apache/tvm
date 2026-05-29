#!/usr/bin/env bash
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

set -euo pipefail

VERIFY_SCRIPT="${1:?usage: verify_tvm_linux.sh /path/to/verify_tvm_install.py}"

echo "Linux wheel verify diagnostics"
echo "TVM_WHEEL_DEBUG_SYMBOLS=${TVM_WHEEL_DEBUG_SYMBOLS:-}"
echo "SKBUILD_CMAKE_BUILD_TYPE=${SKBUILD_CMAKE_BUILD_TYPE:-}"
echo "SKBUILD_INSTALL_STRIP=${SKBUILD_INSTALL_STRIP:-}"
echo "CFLAGS=${CFLAGS:-}"
echo "CXXFLAGS=${CXXFLAGS:-}"

for name in TVM_LIBRARY_PATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH; do
  if [[ -n "${!name:-}" ]]; then
    echo "clearing ${name} before importing tvm"
    unset "${name}"
  fi
done

python - <<'PY'
from __future__ import annotations

from pathlib import Path
import shlex
import subprocess

import tvm


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(shlex.quote(part) for part in cmd), flush=True)
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=capture,
            text=True,
        )
    except FileNotFoundError:
        print(f"{cmd[0]} is not available", flush=True)
        return subprocess.CompletedProcess(cmd, 127, "", "")


def print_section_markers(path: Path) -> None:
    result = run(["readelf", "-S", str(path)], capture=True)
    text = result.stdout or ""
    markers = [".symtab", ".debug_info", ".debug_line", ".gnu_debuglink"]
    for marker in markers:
        print(f"{path.name} contains {marker}: {marker in text}", flush=True)


def print_symbol_sample(path: Path) -> None:
    result = run(["nm", "-C", str(path)], capture=True)
    if result.returncode != 0:
        print(f"nm failed for {path.name} with code {result.returncode}", flush=True)
        if result.stderr:
            print(result.stderr, flush=True)
        return
    patterns = (
        "CodeGenLLVM",
        "LLVMModuleNode",
        "LLVMTargetInfo",
        "target.build.llvm",
        "BuildLLVM",
    )
    matches = [line for line in result.stdout.splitlines() if any(p in line for p in patterns)]
    print(f"{path.name} symbol sample count: {len(matches)}", flush=True)
    for line in matches[:40]:
        print("  ", line, flush=True)


root = Path(tvm.__file__).resolve().parent
libdir = root / "lib"
print("diagnostic tvm package:", root, flush=True)
print("diagnostic libdir:", libdir, flush=True)
for libname in (
    "libtvm_compiler.so",
    "libtvm_runtime.so",
    "libtvm_runtime_extra.so",
    "libtvm_runtime_cuda.so",
):
    path = libdir / libname
    print(f"{libname}: exists={path.exists()}", flush=True)
    if not path.exists():
        continue
    print(f"{libname}: size={path.stat().st_size} bytes", flush=True)
    run(["file", str(path)])
    run(["readelf", "-n", str(path)])
    print_section_markers(path)
    if libname == "libtvm_compiler.so":
        print_symbol_sample(path)
PY

if command -v gdb >/dev/null 2>&1; then
  gdb_cmd=(
    gdb -q -batch -return-child-result
    -ex "set pagination off"
    -ex "set print frame-arguments all"
    -ex "set print elements 0"
    -ex "set backtrace limit 200"
    -ex "run"
    -ex 'info symbol $pc'
    -ex "info registers"
    -ex 'x/16i $pc-32'
    -ex "thread apply all bt full"
    -ex "info sharedlibrary"
    --args python -u -X faulthandler "${VERIFY_SCRIPT}"
  )
  printf 'Running under gdb:'
  printf ' %q' "${gdb_cmd[@]}"
  printf '\n'
  set +e
  "${gdb_cmd[@]}"
  status=$?
  set -e
  echo "gdb verify exited with status ${status}"
  exit "${status}"
fi

exec python -u -X faulthandler "${VERIFY_SCRIPT}"
