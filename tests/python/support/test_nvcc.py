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
from tvm.backend.cuda.target import arch_from_compute_version, compute_version_from_arch
from tvm.support import nvcc
from tvm.target import Target

_ARCH_COMPUTE_VERSION_CASES = [
    ("sm_89", "8.9"),
    ("sm_90a", "9.0.a"),
    ("sm_103a", "10.3.a"),
    ("sm_110a", "11.0.a"),
]


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


@pytest.mark.parametrize(
    "compute_version,expected",
    [("8.9", "sm_89"), ("9.0", "sm_90a"), ("10.3", "sm_103a"), ("11.0", "sm_110a")],
)
def test_arch_from_compute_version(compute_version, expected):
    assert arch_from_compute_version(compute_version) == expected


@pytest.mark.parametrize("compute_version", ["110", "11", "11.00", "x.0"])
def test_arch_from_invalid_compute_version(compute_version):
    with pytest.raises(ValueError, match="Invalid CUDA compute capability"):
        arch_from_compute_version(compute_version)


@pytest.mark.parametrize(
    "arch,expected",
    _ARCH_COMPUTE_VERSION_CASES,
)
def test_compute_version_from_arch(arch, expected):
    assert compute_version_from_arch(arch) == expected


@pytest.mark.parametrize("arch", ["compute_110", "sm_", "sm_xx", "sm_110_a"])
def test_compute_version_from_invalid_arch(arch):
    with pytest.raises(ValueError, match="Expected a CUDA architecture"):
        compute_version_from_arch(arch)


@pytest.mark.parametrize(
    "arch,expected",
    _ARCH_COMPUTE_VERSION_CASES,
)
def test_get_target_compute_version_from_target(arch, expected):
    target = Target({"kind": "cuda", "arch": arch})
    assert nvcc.get_target_compute_version(target) == expected


if __name__ == "__main__":
    tvm.testing.main()
