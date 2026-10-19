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
"""Configuration entry points must not require native runtime dependencies."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("entrypoint", ["module", "tvm-config", "tvm-config.exe"])
def test_config_entrypoint_without_native_imports(tmp_path, entrypoint):
    package = Path(importlib.util.find_spec("tvm").origin).parent
    env = dict(os.environ, PYTHONPATH=str(package.parent))
    if entrypoint == "module":
        command = [sys.executable, "-S", "-m", "tvm.config"]
    else:
        # Console launchers import the target before normalizing sys.argv[0].
        script = tmp_path / entrypoint
        script.write_text("from tvm.config import __main__\n__main__()\n")
        command = [sys.executable, "-S", str(script)]
    result = subprocess.run(
        [*command, "--includedir"], env=env, check=True, capture_output=True, text=True
    )
    include = Path(result.stdout.strip())
    assert (include / "tvm" / "runtime").is_dir()
