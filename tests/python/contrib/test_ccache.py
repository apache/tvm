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
"""Test contrib.cc with ccache"""
import os
import pytest
import shutil
import tempfile
import tvm
from tvm.contrib.cc import create_shared, create_executable, _is_linux_like, _is_windows_like


def _src_gen(text):
    return """
#include <iostream>

int main() {
    std::cout << "text";
    return 0;
}""".replace(
        "text", text
    )


def _compile(f_create, text, output):
    with tempfile.TemporaryDirectory() as temp_dir:
        src_path = os.path.join(temp_dir, "src.cpp")
        with open(src_path, "w", encoding="utf-8") as file:
            file.write(_src_gen(text))
        log_path = os.path.join(temp_dir, "log.txt")
        ccache_env = {
            "CCACHE_COMPILERCHECK": "content",
            "CCACHE_LOGFILE": log_path,
        }
        f_create(output, ["src.cpp"], ["-c"], cwd=temp_dir, ccache_env=ccache_env)
        with open(log_path, "r", encoding="utf-8") as file:
            log = file.read()
        return log


@pytest.mark.skipif(shutil.which("ccache") is None, reason="ccache not installed")
def test_shared():
    if _is_linux_like():
        _ = _compile(create_shared, "shared", "main.o")
        log = _compile(create_shared, "shared", "main.o")
        assert "Succeeded getting cached result" in log
    elif _is_windows_like():
        _ = _compile(create_shared, "shared", "main.obj")
        log = _compile(create_shared, "shared", "main.obj")
        assert "Succeeded getting cached result" in log


@pytest.mark.skipif(shutil.which("ccache") is None, reason="ccache not installed")
def test_executable():
    if _is_linux_like():
        _ = _compile(create_executable, "executable", "main")
        log = _compile(create_executable, "executable", "main")
        assert "Succeeded getting cached result" in log
    elif _is_windows_like():
        _ = _compile(create_executable, "executable", "main.exe")
        log = _compile(create_executable, "executable", "main.exe")
        assert "Succeeded getting cached result" in log


if __name__ == "__main__":
    tvm.testing.main()
