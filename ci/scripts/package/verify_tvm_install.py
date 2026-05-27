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

from pathlib import Path
import sys

import tvm


def main() -> int:
    root = Path(tvm.__file__).resolve().parent
    libdir = root / "lib"
    if sys.platform == "darwin":
        runtime_lib = libdir / "libtvm_runtime.dylib"
        cuda_sidecar = libdir / "libtvm_runtime_cuda.dylib"
    elif sys.platform == "win32":
        runtime_lib = libdir / "tvm_runtime.dll"
        cuda_sidecar = libdir / "tvm_runtime_cuda.dll"
    else:
        runtime_lib = libdir / "libtvm_runtime.so"
        cuda_sidecar = libdir / "libtvm_runtime_cuda.so"

    print("tvm version:", tvm.__version__)
    print("tvm package:", root)
    print("llvm enabled:", tvm.runtime.enabled("llvm"))
    print("cuda runtime enabled:", tvm.runtime.enabled("cuda"))
    print("runtime library:", runtime_lib)
    if not runtime_lib.exists():
        raise RuntimeError(f"runtime library is missing: {runtime_lib}")
    print("cuda sidecar present:", cuda_sidecar.exists())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
