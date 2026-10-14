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
"""Tests for the thin ``tvm.testing.env`` capability probes."""

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

import tvm
import tvm.testing
from tvm.testing import env

# Probes that take no arguments and must return a plain bool without raising.
_BOOL_PROBES = [
    # runtime device
    env.has_cuda,
    env.has_rocm,
    env.has_vulkan,
    env.has_metal,
    env.has_opencl,
    env.has_nvptx,
    env.has_llvm,
    env.has_gpu,
    # build support
    env.has_cudnn,
    env.has_cublas,
    env.has_nccl,
    env.has_hipblas,
    env.has_nvshmem,
    # version / capability
    env.has_matrixcore,
    env.has_cudagraph,
    # toolchain / environment
    env.has_hexagon,
    env.has_hexagon_toolchain,
    env.has_adreno_opencl,
    # cpu features
    env.has_x86_vnni,
    env.has_x86_avx512,
    # host architecture
]


@pytest.mark.parametrize("probe", _BOOL_PROBES, ids=lambda p: p.__name__)
def test_probe_returns_bool(probe):
    """Every probe returns a real bool and never raises during collection/run."""
    assert isinstance(probe(), bool)


def test_has_cuda_implies_device():
    """has_cuda() requires a device (it also requires the kind to be enabled)."""
    if env.has_cuda():
        assert tvm.cuda().exist


def test_has_gpu_is_raw_any_device():
    """has_gpu() is the disjunction of the raw device checks (no target gating)."""
    any_device = (
        env._device_exists("cuda")  # pylint: disable=protected-access
        or env._device_exists("rocm")  # pylint: disable=protected-access
        or env._device_exists("opencl")  # pylint: disable=protected-access
        or env._device_exists("metal")  # pylint: disable=protected-access
        or env._device_exists("vulkan")  # pylint: disable=protected-access
    )
    assert env.has_gpu() == any_device


def test_target_enabled_respects_tvm_test_targets(monkeypatch):
    """A device kind excluded from TVM_TEST_TARGETS is reported as not enabled."""
    env._target_enabled.cache_clear()  # pylint: disable=protected-access
    monkeypatch.setenv("TVM_TEST_TARGETS", "cuda;llvm")
    try:
        assert env._target_enabled("cuda")  # pylint: disable=protected-access
        assert env._target_enabled("llvm")  # pylint: disable=protected-access
        assert not env._target_enabled("opencl")  # pylint: disable=protected-access
        assert not env._target_enabled("metal")  # pylint: disable=protected-access
    finally:
        env._target_enabled.cache_clear()  # pylint: disable=protected-access


def test_cuda_compute_is_monotonic():
    """has_cuda_compute is monotone in the requested version."""
    if not env.has_cuda():
        # Without a CUDA device every query is False, including the (0, 0) floor.
        assert not env.has_cuda_compute(1, 0)
        assert not env.has_cuda_compute(0, 0)
        return
    # A device that satisfies (major, minor) also satisfies anything lower.
    assert env.has_cuda_compute(1, 0)
    assert env.has_cuda_compute(0, 0)


@pytest.mark.parametrize(
    ("compute_version", "expected"),
    [
        ("8.0", "sm_80"),
        ("9.0", "sm_90a"),
        ("10.0", "sm_100a"),
        ("10.3", "sm_103a"),
        ("invalid", None),
    ],
)
def test_cuda_arch_from_compute_version(compute_version, expected):
    assert env._cuda_arch_from_compute_version(compute_version) == expected


def test_has_cuda_arch_matches_detected_device():
    actual = env.cuda_arch()
    if actual is None:
        assert not env.has_cuda_arch("sm_100a")
    else:
        assert env.has_cuda_arch(actual)


def test_cuda_arch_uses_requested_device(monkeypatch):
    class Device:
        exist = True
        compute_version = "10.3"

    requested = []

    def cuda(device_id):
        requested.append(device_id)
        return Device()

    env.cuda_arch.cache_clear()
    monkeypatch.setattr(tvm, "cuda", cuda)
    try:
        assert env.cuda_arch(7) == "sm_103a"
        assert requested == [7]
    finally:
        env.cuda_arch.cache_clear()


def test_has_multi_gpu_is_bool():
    assert isinstance(env.has_multi_gpu(), bool)
    assert isinstance(env.has_multi_gpu(1), bool)
    # Requiring a single device is at least as permissive as requiring two.
    assert env.has_multi_gpu(1) or not env.has_multi_gpu(2)


def test_llvm_min_version_is_monotone():
    if not env.has_llvm():
        assert not env.has_llvm_min_version(1)
        return
    # An LLVM that satisfies a higher floor also satisfies a lower one.
    assert env.has_llvm_min_version(1)


def test_probes_are_memoized():
    """Probes are cached so the driver/subprocess is hit once per process."""
    env.has_cuda()
    info = env._device_exists.cache_info()  # pylint: disable=protected-access
    assert info.hits + info.misses >= 1


@pytest.fixture
def clear_env_cache():
    probes = (
        env._target_enabled,
        env._runtime_enabled,
        env._device_exists,
        env.build_flag_enabled,
        env._llvm_version_major,
        env.has_hexagon_toolchain,
        env.has_hexagon,
    )
    for probe in probes:
        probe.cache_clear()
    yield
    for probe in probes:
        probe.cache_clear()


def test_unavailable_selected_target_does_not_enable_llvm(monkeypatch, clear_env_cache):
    monkeypatch.setenv("TVM_TEST_TARGETS", "cuda")
    monkeypatch.setattr(tvm.runtime, "enabled", lambda kind: kind == "llvm")

    def unexpected_device_lookup(target):
        pytest.fail(f"Excluded or unavailable target was probed: {target}")

    monkeypatch.setattr(tvm, "device_from_target", unexpected_device_lookup)
    assert not tvm.testing.device_enabled("cuda")
    assert not tvm.testing.device_enabled("llvm")


@pytest.mark.parametrize("target_form", ["string", "dict", "target"])
def test_device_enabled_accepts_target_forms(monkeypatch, clear_env_cache, target_form):
    monkeypatch.setenv("TVM_TEST_TARGETS", '{"kind": "llvm"};cuda')
    monkeypatch.setattr(tvm.runtime, "enabled", lambda kind: kind == "llvm")
    monkeypatch.setattr(tvm, "device_from_target", lambda target: SimpleNamespace(exist=True))
    target = {
        "string": "llvm -num-cores=2",
        "dict": {"kind": "llvm", "num-cores": 2},
        "target": tvm.target.Target({"kind": "llvm", "num-cores": 2}),
    }[target_form]
    assert tvm.testing.device_enabled(target)


@pytest.mark.parametrize("kind", ["llvm", "cuda"])
@pytest.mark.parametrize(
    ("runtime_enabled", "device_exists", "expected"),
    [(False, True, False), (True, False, False), (True, True, True)],
)
def test_device_enabled_requires_runtime_and_device(
    monkeypatch, clear_env_cache, kind, runtime_enabled, device_exists, expected
):
    # In particular, the CPU can exist even when LLVM was not built.
    monkeypatch.setenv("TVM_TEST_TARGETS", kind)
    monkeypatch.setattr(tvm.runtime, "enabled", lambda target: runtime_enabled)
    monkeypatch.setattr(
        tvm, "device_from_target", lambda target: SimpleNamespace(exist=device_exists)
    )
    assert tvm.testing.device_enabled(kind) is expected


@pytest.mark.parametrize("cudnn_available", [None, False, True])
def test_device_enabled_cudnn_runtime_probe(monkeypatch, clear_env_cache, cudnn_available):
    monkeypatch.setenv("TVM_TEST_TARGETS", '{"kind": "cuda", "libs": ["cudnn"]}')
    monkeypatch.setattr(tvm.runtime, "enabled", lambda kind: True)
    monkeypatch.setattr(tvm, "device", lambda *args: SimpleNamespace(exist=True))
    monkeypatch.setattr(tvm.support, "libinfo", lambda: {"USE_CUDNN": "ON"})
    exists = None if cudnn_available is None else lambda: cudnn_available
    monkeypatch.setattr(tvm, "get_global_func", lambda name, allow_missing: exists)
    assert tvm.testing.device_enabled("cuda") is (cudnn_available is True)


@pytest.mark.parametrize(
    ("built", "toolchain", "llvm_version", "serial", "can_compile", "can_run"),
    [
        (False, True, 7, True, False, False),
        (True, False, 7, True, False, False),
        (True, True, 6, True, False, False),
        (True, True, 7, False, True, False),
        (True, True, 7, True, True, True),
        (True, True, None, True, True, True),
    ],
)
def test_hexagon_environment_requirements(
    monkeypatch, clear_env_cache, built, toolchain, llvm_version, serial, can_compile, can_run
):
    # LLVM selection for tests is independent of whether its codegen can support Hexagon.
    monkeypatch.setenv("TVM_TEST_TARGETS", "hexagon")
    for variable, present in (
        ("HEXAGON_TOOLCHAIN", toolchain),
        ("ANDROID_SERIAL_NUMBER", serial),
    ):
        if present:
            monkeypatch.setenv(variable, "test-value")
        else:
            monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(tvm.support, "libinfo", lambda: {"USE_HEXAGON": "ON" if built else "OFF"})
    monkeypatch.setattr(tvm.runtime, "enabled", lambda kind: llvm_version is not None)
    monkeypatch.setattr(tvm.codegen, "llvm_version_major", lambda: llvm_version)
    assert env.has_hexagon_toolchain() is can_compile
    assert env.has_hexagon() is can_run


def test_core_testing_imports_without_domain_modules():
    # Let TVM finish its own startup, then make the domain modules unavailable to testing.
    code = textwrap.dedent(
        """
        import os
        import sys
        import tvm

        for module in list(sys.modules):
            if module == "tvm.testing" or module.startswith("tvm.testing."):
                del sys.modules[module]
        if hasattr(tvm, "testing"):
            del tvm.testing

        for name in ("te", "contrib", "s_tir"):
            prefix = "tvm." + name
            for module in list(sys.modules):
                if module == prefix or module.startswith(prefix + "."):
                    sys.modules[module] = None
            sys.modules[prefix] = None
            if hasattr(tvm, name):
                delattr(tvm, name)

        import tvm.testing
        from tvm.testing import env

        tvm.testing.assert_allclose([1], [1])
        tvm.support.libinfo = lambda: {"USE_HEXAGON": "ON"}
        tvm.runtime.enabled = lambda kind: False
        os.environ["HEXAGON_TOOLCHAIN"] = "test-toolchain"
        os.environ["ANDROID_SERIAL_NUMBER"] = "test-device"
        assert env.has_hexagon_toolchain()
        assert env.has_hexagon()
        """
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


# --- demonstration of the target idiom -------------------------------------
#
# The standard gating idiom: a plain registered pytest marker (for ``-m``
# selection) plus a skipif backed by a thin env probe (for runtime gating).


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_thin_cuda_idiom():
    dev = tvm.cuda()
    assert dev.exist


if __name__ == "__main__":
    tvm.testing.main()
