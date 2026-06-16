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
"""Thin capability probes for test gating.

This module exposes small ``has_*`` predicates that report whether the
current environment can run a given feature.  They are meant to be used
with plain pytest markers and ``skipif``::

    import pytest
    import tvm.testing

    @pytest.mark.gpu
    @pytest.mark.skipif(not tvm.testing.env.has_cuda(), reason="need cuda")
    def test_my_cuda_kernel():
        ...

Every probe is memoized with :func:`functools.cache`, so the
underlying device query / ``nvcc`` subprocess runs at most once per
process even though ``skipif`` evaluates the predicate at import time for
every decorated test.  Probes never raise: when support is absent they
return ``False`` (or a zero version tuple) rather than propagating an
error out of collection.

Three kinds of probe live here:

* **runtime device** probes (``has_cuda``, ``has_gpu`` …) ask whether a
  usable device of a given kind is present;
* **build-support** probes (``has_cutlass``, ``has_cudnn`` …) ask whether
  an optional library was compiled into the runtime;
* **version / capability** probes (``has_cuda_compute``,
  ``has_tensorcore`` …) ask about a finer capability of a present device
  or toolchain.
"""

import functools
import os
import platform

import tvm

__all__ = [
    "has_aarch64_sme",
    "has_aarch64_sve",
    "has_adreno_opencl",
    "has_aprofile_aem_fvp",
    "has_arm_dot",
    "has_arm_fp16",
    # cpu features
    "has_cpu_feature",
    "has_cublas",
    # runtime device
    "has_cuda",
    # version / capability
    "has_cuda_compute",
    "has_cudagraph",
    # build support
    "has_cudnn",
    "has_cutlass",
    "has_gpu",
    # toolchain / environment
    "has_hexagon",
    "has_hexagon_toolchain",
    "has_hipblas",
    "has_llvm",
    "has_llvm_min_version",
    "has_matrixcore",
    "has_metal",
    "has_mrvl",
    "has_multi_gpu",
    "has_nccl",
    "has_nnapi",
    "has_nvcc_version",
    "has_nvptx",
    "has_nvshmem",
    "has_opencl",
    "has_openclml",
    "has_rocm",
    "has_rpc",
    "has_tensorcore",
    "has_vulkan",
    "has_x86_amx",
    "has_x86_avx512",
    "has_x86_vnni",
    "is_aarch64",
    # host architecture
    "is_x86",
]


@functools.cache
def _device_exists(kind: str, index: int = 0) -> bool:
    """Return whether ``tvm.device(kind, index)`` is present and usable."""
    try:
        return bool(tvm.device(kind, index).exist)
    except Exception:  # pylint: disable=broad-except
        # A missing backend / driver must skip the test, not crash collection.
        return False


@functools.cache
def _build_flag_enabled(flag: str) -> bool:
    """Return whether an optional build flag (e.g. ``USE_CUTLASS``) is on.

    A flag counts as enabled unless it is explicitly disabled, so library
    flags carrying a path (rather than a boolean) still register as present.
    Callers gate on this via ``@pytest.mark.skipif(not has_cutlass(), ...)``.
    """
    try:
        value = tvm.support.libinfo().get(flag, "OFF")
        return str(value).lower() not in ("off", "false", "0")
    except Exception:  # pylint: disable=broad-except
        return False


@functools.cache
def _target_enabled(kind: str) -> bool:
    """True if ``kind`` is selected by ``TVM_TEST_TARGETS`` (or the default set).

    Honors the ``TVM_TEST_TARGETS`` opt-out, so CI can exclude a flaky
    backend (e.g. opencl) via ``TVM_TEST_TARGETS`` and have its tests skip
    even when a device is physically present.
    """
    try:
        from tvm.testing.utils import _tvm_test_targets  # pylint: disable=import-outside-toplevel

        for target in _tvm_test_targets():
            k = target["kind"] if isinstance(target, dict) else str(target).split()[0]
            if k == kind:
                return True
        return False
    except Exception:  # pylint: disable=broad-except
        return True  # fail open: the device check still gates


@functools.cache
def _runtime_enabled(kind: str) -> bool:
    """True if the runtime was built with support for target ``kind``.

    Used for kinds whose device existence does not imply the backend was
    compiled in -- notably ``llvm``, which maps to the always-present CPU
    device, so ``tvm.device("llvm").exist`` is True even on ``USE_LLVM=OFF``.
    """
    try:
        return bool(tvm.runtime.enabled(kind))
    except Exception:  # pylint: disable=broad-except
        return False


def _device_usable(kind: str) -> bool:
    """True if ``kind`` is enabled for this run and a ``kind`` device exists.

    The TVM_TEST_TARGETS opt-out is checked first so that an excluded backend
    never probes a (possibly crashy) device.
    """
    return _target_enabled(kind) and _device_exists(kind)


# --- runtime device probes -------------------------------------------------


def has_cuda() -> bool:
    """True if a CUDA device is present and enabled in TVM_TEST_TARGETS."""
    return _device_usable("cuda")


def has_rocm() -> bool:
    """True if a ROCm device is present and enabled in TVM_TEST_TARGETS."""
    return _device_usable("rocm")


def has_vulkan() -> bool:
    """True if a Vulkan device is present and enabled in TVM_TEST_TARGETS."""
    return _device_usable("vulkan")


def has_metal() -> bool:
    """True if a Metal device is present and enabled in TVM_TEST_TARGETS."""
    return _device_usable("metal")


def has_opencl() -> bool:
    """True if an OpenCL device is present and enabled in TVM_TEST_TARGETS."""
    return _device_usable("opencl")


def has_nvptx() -> bool:
    """True if NVPTX is usable: a (CUDA) device, plus the LLVM backend it needs."""
    return _device_usable("nvptx") and has_llvm()


def has_llvm() -> bool:
    """True if the LLVM backend was built in and enabled in TVM_TEST_TARGETS.

    Uses ``tvm.runtime.enabled`` rather than device existence: ``llvm`` maps to
    the CPU device, which exists even on a ``USE_LLVM=OFF`` build.
    """
    return _target_enabled("llvm") and _runtime_enabled("llvm")


def has_gpu() -> bool:
    """True if any GPU backend (cuda/rocm/opencl/metal/vulkan) is present."""
    return (
        _device_exists("cuda")
        or _device_exists("rocm")
        or _device_exists("opencl")
        or _device_exists("metal")
        or _device_exists("vulkan")
    )


@functools.cache
def has_multi_gpu(count: int = 2) -> bool:
    """True if at least ``count`` devices of a single GPU backend exist."""
    for kind in ("cuda", "rocm", "opencl", "metal", "vulkan"):
        if all(_device_exists(kind, index) for index in range(count)):
            return True
    return False


# --- build-support probes --------------------------------------------------
#
# These wrap the optional-library build flags.  Features that extend CUDA /
# ROCm additionally require the parent device to be present.


def has_cudnn() -> bool:
    """True if cuDNN was built in and a CUDA device is present."""
    return has_cuda() and _build_flag_enabled("USE_CUDNN")


def has_cublas() -> bool:
    """True if cuBLAS was built in and a CUDA device is present."""
    return has_cuda() and _build_flag_enabled("USE_CUBLAS")


def has_nccl() -> bool:
    """True if NCCL was built in and a CUDA device is present."""
    return has_cuda() and _build_flag_enabled("USE_NCCL")


def has_hipblas() -> bool:
    """True if hipBLAS was built in and a ROCm device is present."""
    return has_rocm() and _build_flag_enabled("USE_HIPBLAS")


def has_cutlass() -> bool:
    """True if CUTLASS support was built into the runtime."""
    return _build_flag_enabled("USE_CUTLASS")


def has_rpc() -> bool:
    """True if RPC support was built into the runtime."""
    return _build_flag_enabled("USE_RPC")


def has_nnapi() -> bool:
    """True if NNAPI codegen support was built into the runtime."""
    return _build_flag_enabled("USE_NNAPI_CODEGEN")


def has_openclml() -> bool:
    """True if OpenCLML (CLML) support was built into the runtime."""
    return _build_flag_enabled("USE_CLML")


def has_mrvl() -> bool:
    """True if the Marvell (MRVL) backend was built into the runtime."""
    return _build_flag_enabled("USE_MRVL")


@functools.cache
def has_nvshmem() -> bool:
    """True if the disco NVSHMEM runtime is available (requires CUDA).

    Probes the runtime global function rather than the ``USE_NVSHMEM`` build
    flag, since the flag can be set in builds that do not ship the runtime.
    """
    try:
        return has_cuda() and (
            tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid", allow_missing=True)
            is not None
        )
    except Exception:  # pylint: disable=broad-except
        return False


# --- version / capability probes -------------------------------------------


@functools.cache
def _cuda_compute_version() -> tuple:
    """Return the (major, minor) CUDA compute version, or (0, 0) if unknown."""
    try:
        from tvm.support import nvcc  # pylint: disable=import-outside-toplevel

        arch = nvcc.get_target_compute_version()
        return nvcc.parse_compute_version(arch)
    except Exception:  # pylint: disable=broad-except
        return (0, 0)


def has_cuda_compute(major: int, minor: int = 0, exact: bool = False) -> bool:
    """True if the CUDA compute capability satisfies ``(major, minor)``.

    When ``exact`` is False (default) the check is ``compute >= (major,
    minor)``; when True it requires an exact match.  Returns False when no
    CUDA device is present, so it implies :func:`has_cuda`.
    """
    if not has_cuda():
        return False
    compute = _cuda_compute_version()
    want = (major, minor)
    if exact:
        return compute == want
    return compute >= want


@functools.cache
def _nvcc_version() -> tuple:
    """Return the (major, minor, release) nvcc version, or (0, 0, 0)."""
    try:
        from tvm.support import nvcc  # pylint: disable=import-outside-toplevel

        return nvcc.get_cuda_version()
    except Exception:  # pylint: disable=broad-except
        return (0, 0, 0)


def has_nvcc_version(major: int, minor: int = 0, release: int = 0) -> bool:
    """True if a CUDA device is present and nvcc is at least ``(major, minor, release)``.

    Returns False when no CUDA device is present, so it implies :func:`has_cuda`.
    Gate a test with ``@pytest.mark.skipif(not env.has_nvcc_version(11, 4),
    reason="need nvcc >= 11.4")`` (add ``@pytest.mark.gpu`` for GPU selection).
    """
    return has_cuda() and _nvcc_version() >= (major, minor, release)


@functools.cache
def _llvm_version_major() -> int:
    """Return the major LLVM version, or 0 if LLVM is unavailable."""
    try:
        return int(tvm.target.codegen.llvm_version_major())
    except Exception:  # pylint: disable=broad-except
        return 0


def has_llvm_min_version(major: int) -> bool:
    """True if LLVM is available and its major version is at least ``major``."""
    return has_llvm() and _llvm_version_major() >= major


@functools.cache
def has_tensorcore() -> bool:
    """True if a CUDA device with Tensor Core support (compute >= 7) exists."""
    try:
        from tvm.support import nvcc  # pylint: disable=import-outside-toplevel

        return has_cuda() and bool(nvcc.have_tensorcore(tvm.cuda().compute_version))
    except Exception:  # pylint: disable=broad-except
        return False


@functools.cache
def has_matrixcore() -> bool:
    """True if a ROCm device with Matrix Core support (compute >= 8) exists."""
    try:
        from tvm.support import rocm  # pylint: disable=import-outside-toplevel

        return has_rocm() and bool(rocm.have_matrixcore(tvm.rocm().compute_version))
    except Exception:  # pylint: disable=broad-except
        return False


@functools.cache
def has_cudagraph() -> bool:
    """True if a CUDA device is present and the toolkit supports CUDA Graphs.

    Implies :func:`has_cuda`: ``nvcc.have_cudagraph()`` only checks the
    toolkit version, so the device guard must be explicit.  Gate a test with
    ``@pytest.mark.skipif(not tvm.testing.env.has_cudagraph(), reason=...)``
    (add ``@pytest.mark.gpu`` for CI selection).
    """
    try:
        from tvm.support import nvcc  # pylint: disable=import-outside-toplevel

        return has_cuda() and bool(nvcc.have_cudagraph())
    except Exception:  # pylint: disable=broad-except
        return False


# --- toolchain / environment probes ----------------------------------------


@functools.cache
def has_hexagon_toolchain() -> bool:
    """True if the Hexagon toolchain is available for compilation."""
    try:
        from tvm.contrib.hexagon import (  # pylint: disable=import-outside-toplevel
            _ci_env_check,
        )

        return _build_flag_enabled("USE_HEXAGON") and _ci_env_check._compile_time_check() is True
    except Exception:  # pylint: disable=broad-except
        return False


@functools.cache
def has_hexagon() -> bool:
    """True if Hexagon can both compile and run (toolchain + attached device)."""
    try:
        from tvm.contrib.hexagon import (  # pylint: disable=import-outside-toplevel
            _ci_env_check,
        )

        return has_hexagon_toolchain() and _ci_env_check._run_time_check() is True
    except Exception:  # pylint: disable=broad-except
        return False


@functools.cache
def has_adreno_opencl() -> bool:
    """True if remote Adreno OpenCL testing is configured (RPC_TARGET set)."""
    return _build_flag_enabled("USE_OPENCL") and os.environ.get("RPC_TARGET") is not None


@functools.cache
def has_aprofile_aem_fvp() -> bool:
    """True if the AProfile AEM FVP simulator is on PATH."""
    try:
        import shutil  # pylint: disable=import-outside-toplevel

        return shutil.which("FVP_Base_RevC-2xAEMvA") is not None
    except Exception:  # pylint: disable=broad-except
        return False


# --- cpu feature probes ----------------------------------------------------


@functools.cache
def _has_cpu_feature(features) -> bool:
    """True if the host CPU advertises the given LLVM target ``features``."""
    try:
        codegen = tvm.target.codegen
        cpu = codegen.llvm_get_system_cpu()
        triple = codegen.llvm_get_system_triple()
        target = tvm.target.Target({"kind": "llvm", "mtriple": triple, "mcpu": cpu})
        return bool(codegen.target_has_features(features, target))
    except Exception:  # pylint: disable=broad-except
        return False


def has_cpu_feature(features) -> bool:
    """True if the host CPU supports ``features`` (a name or list of names)."""
    if isinstance(features, list):
        features = tuple(features)
    return _has_cpu_feature(features)


def has_arm_dot() -> bool:
    """True if the host CPU supports the ARM dot-product instructions."""
    return has_cpu_feature("dotprod")


def has_arm_fp16() -> bool:
    """True if the host CPU supports ARM Neon FP16 instructions."""
    return has_cpu_feature("fullfp16")


def has_aarch64_sve() -> bool:
    """True if the host CPU supports AArch64 SVE."""
    return has_cpu_feature("sve")


def has_aarch64_sme() -> bool:
    """True if the host CPU supports AArch64 SME."""
    return has_cpu_feature("sme")


def has_x86_vnni() -> bool:
    """True if the host CPU supports x86 VNNI (AVX512-VNNI or AVX-VNNI)."""
    return has_cpu_feature("avx512vnni") or has_cpu_feature("avxvnni")


def has_x86_avx512() -> bool:
    """True if the host CPU supports the x86 AVX512 extensions."""
    return has_cpu_feature(["avx512bw", "avx512cd", "avx512dq", "avx512vl", "avx512f"])


def has_x86_amx() -> bool:
    """True if the host CPU supports the x86 AMX (int8) extensions."""
    return has_cpu_feature("amx-int8")


# --- host architecture probes ----------------------------------------------


def is_x86() -> bool:
    """True if running on an x86_64 host."""
    return platform.machine() == "x86_64"


def is_aarch64() -> bool:
    """True if running on an aarch64 host."""
    return platform.machine() == "aarch64"
