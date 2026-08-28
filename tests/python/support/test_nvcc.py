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
"""Tests for functions in tvm/python/tvm/support/nvcc.py."""

import os

import pytest

import tvm.testing
from tvm.support import nvcc


def _make_cuda_root(root, triples):
    """Create a fake CUDA toolkit exposing the given ``targets/<triple>`` dirs."""
    for triple in triples:
        os.makedirs(os.path.join(root, "targets", triple, "include"))
    return str(root)


@pytest.mark.parametrize(
    "machine,available,expected",
    [
        # ARM64 server toolkits ship the headers under "sbsa-linux".
        ("aarch64", ["sbsa-linux", "aarch64-linux"], "sbsa-linux"),
        # Embedded/L4T toolkits only provide "aarch64-linux".
        ("aarch64", ["aarch64-linux"], "aarch64-linux"),
        ("arm64", ["sbsa-linux"], "sbsa-linux"),
        ("x86_64", ["x86_64-linux"], "x86_64-linux"),
    ],
)
def test_find_cuda_target_include(tmp_path, monkeypatch, machine, available, expected):
    """The architecture-specific include dir matches the installed toolkit layout."""
    monkeypatch.setattr(nvcc.platform, "machine", lambda: machine)
    monkeypatch.setattr(nvcc.platform, "system", lambda: "Linux")
    cuda_path = _make_cuda_root(tmp_path, available)
    assert nvcc._find_cuda_target_include(cuda_path) == os.path.join(
        cuda_path, "targets", expected, "include"
    )


def test_find_cuda_target_include_absent(tmp_path, monkeypatch):
    """Toolkits without a ``targets/`` layout report no architecture-specific dir."""
    monkeypatch.setattr(nvcc.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(nvcc.platform, "system", lambda: "Linux")
    assert nvcc._find_cuda_target_include(str(tmp_path)) is None


if __name__ == "__main__":
    tvm.testing.main()
