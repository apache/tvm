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
"""The config command must work without importing native runtime dependencies."""

import os
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

import tvm_tirx


def test_config_without_native_imports(tmp_path):
    # A startup hook rejects native loads and dependencies even when they are installed.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\n"
        "class BlockNative:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname.split('.')[0] in ('tvm_ffi', 'torch', 'numpy') or "
        "fullname in ('tvm.base', 'tvm.runtime', 'tvm.backend', 'tvm.driver', 'tvm_tirx.base'):\n"
        "            raise AssertionError(fullname)\n"
        "sys.meta_path.insert(0, BlockNative())\n"
        "def audit(event, args):\n"
        "    if event == 'ctypes.dlopen' and args[0] is not None: raise AssertionError(args)\n"
        "sys.addaudithook(audit)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        (
            str(tmp_path),
            str(Path(tvm_tirx.__file__).parent.parent),
        )
    )
    for entry in ("module", "console"):
        command = [sys.executable, "-m", "tvm_tirx.config"]
        if entry == "console":
            executable = Path(sysconfig.get_path("scripts")) / "tvm-tirx-config"
            if not executable.is_file():
                continue
            command = [str(executable)]
        include = subprocess.check_output([*command, "--includedir"], env=env, text=True).strip()
        assert (Path(include) / "tvm" / "runtime").is_dir()
        libraries = subprocess.check_output(
            [*command, "--libfiles", "runtime"], env=env, text=True
        ).strip()
        libraries = shlex.split(libraries)
        assert len(libraries) == 2
        assert all(Path(library).is_file() for library in libraries)
        cmake_dir = subprocess.check_output([*command, "--cmakedir"], env=env, text=True).strip()
        assert (Path(cmake_dir) / "tvm_tirx-config.cmake").is_file()
    if shutil.which("c++") and sys.platform != "win32":
        _check_config_public_headers_compile()


def _check_config_public_headers_compile():
    flags = shlex.split(
        subprocess.check_output(
            [sys.executable, "-m", "tvm_tirx.config", "--cxxflags", "runtime"], text=True
        )
    )
    subprocess.run(
        ["c++", *flags, "-x", "c++", "-fsyntax-only", "-"],
        input="#include <tvm/runtime/base.h>\n#include <tvm/ffi/container/tensor.h>\n",
        text=True,
        check=True,
    )


if __name__ == "__main__":
    import tvm_tirx.testing

    tvm_tirx.testing.main()
