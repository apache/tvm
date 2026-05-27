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
"""Validate ELF linkage inside a repaired TVM wheel."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

TVM_EXTERNAL_LIBS = {
    "libtvm_ffi.so",
}


def _run(command: list[str], env: dict[str, str] | None = None) -> str:
    try:
        proc = subprocess.run(
            command,
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as err:
        output = err.stdout or ""
        raise RuntimeError(f"{' '.join(command)} failed:\n{output}") from err
    return proc.stdout


def _dynamic_entries(path: Path) -> tuple[list[str], list[str]]:
    output = _run(["readelf", "-d", str(path)])
    needed = re.findall(r"Shared library: \[(.*?)\]", output)
    rpaths = re.findall(r"Library (?:rpath|runpath): \[(.*?)\]", output)
    return needed, [entry for rpath in rpaths for entry in rpath.split(":") if entry]


def _ldd(path: Path) -> dict[str, str]:
    output = _run(["ldd", str(path)], env={**os.environ, "LD_LIBRARY_PATH": ""})
    resolved: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if "=>" not in line:
            continue
        name, target = line.split("=>", 1)
        target = target.strip()
        resolved[name.strip()] = target.split(" ", 1)[0]
    return resolved


def validate(wheel: Path) -> None:
    if sys.platform != "linux":
        print("ELF wheel validation skipped on non-Linux platform")
        return
    for command in ("readelf", "ldd"):
        if shutil.which(command) is None:
            raise RuntimeError(f"{command} is required for ELF wheel validation")

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with zipfile.ZipFile(wheel) as zf:
            zf.extractall(root)

        libdir = root / "tvm" / "lib"
        if not libdir.is_dir():
            raise RuntimeError(f"wheel does not contain {libdir.relative_to(root)}")

        bundled_tvm_ffi = sorted(
            str(path.relative_to(root)) for path in root.rglob("libtvm_ffi*.so*") if path.is_file()
        )
        if bundled_tvm_ffi:
            raise RuntimeError(
                "TVM wheel must depend on tvm_ffi instead of bundling libtvm_ffi: "
                + ", ".join(bundled_tvm_ffi)
            )

        libs = {path.name: path for path in sorted(libdir.glob("*.so*")) if path.is_file()}
        if "libtvm_runtime.so" not in libs:
            raise RuntimeError("wheel does not contain tvm/lib/libtvm_runtime.so")
        bundled_llvm = sorted(
            str(path.relative_to(root)) for path in root.rglob("libLLVM*.so*") if path.is_file()
        )
        if bundled_llvm:
            raise RuntimeError(
                "TVM wheel must link LLVM statically instead of bundling libLLVM: "
                + ", ".join(bundled_llvm)
            )

        errors: list[str] = []
        for lib in libs.values():
            needed, rpaths = _dynamic_entries(lib)
            llvm_needed = sorted(name for name in needed if name.startswith("libLLVM"))
            if llvm_needed:
                errors.append(
                    f"{lib.relative_to(root)} links dynamic LLVM libraries: {llvm_needed}"
                )

            internal_needed = sorted(name for name in needed if name in libs)
            if internal_needed and "$ORIGIN" not in rpaths:
                errors.append(
                    f"{lib.relative_to(root)} needs {internal_needed} but RUNPATH/RPATH is {rpaths}"
                )

            resolved = _ldd(lib)
            for name in internal_needed:
                target = resolved.get(name)
                if target is None:
                    errors.append(f"{lib.relative_to(root)}: ldd did not report {name}")
                    continue
                if target == "not":
                    errors.append(f"{lib.relative_to(root)}: {name} is not found")
                    continue
                if Path(target).resolve() != libs[name].resolve():
                    errors.append(
                        f"{lib.relative_to(root)}: {name} resolved to {target}, "
                        f"expected {libs[name].relative_to(root)}"
                    )

            unexpected_tvm_deps = sorted(
                name
                for name in needed
                if name.startswith("libtvm_") and name not in libs and name not in TVM_EXTERNAL_LIBS
            )
            if unexpected_tvm_deps:
                errors.append(
                    f"{lib.relative_to(root)} has unresolved TVM deps: {unexpected_tvm_deps}"
                )

        if errors:
            raise RuntimeError("ELF wheel validation failed:\n" + "\n".join(errors))

        print(f"ELF wheel validation passed for {wheel.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    try:
        validate(args.wheel)
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
