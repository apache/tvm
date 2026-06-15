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
    env.has_cutlass,
    env.has_rpc,
    env.has_nnapi,
    env.has_openclml,
    env.has_mrvl,
    env.has_nvshmem,
    # version / capability
    env.has_tensorcore,
    env.has_matrixcore,
    env.has_cudagraph,
    # toolchain / environment
    env.has_hexagon,
    env.has_hexagon_toolchain,
    env.has_adreno_opencl,
    env.has_aprofile_aem_fvp,
    # cpu features
    env.has_arm_dot,
    env.has_arm_fp16,
    env.has_aarch64_sve,
    env.has_aarch64_sme,
    env.has_x86_vnni,
    env.has_x86_avx512,
    env.has_x86_amx,
    # host architecture
    env.is_x86,
    env.is_aarch64,
]


@pytest.mark.parametrize("probe", _BOOL_PROBES, ids=lambda p: p.__name__)
def test_probe_returns_bool(probe):
    """Every probe returns a real bool and never raises during collection/run."""
    assert isinstance(probe(), bool)


def test_has_cuda_matches_device():
    """has_cuda() agrees with the underlying device query."""
    assert env.has_cuda() == bool(tvm.cuda().exist)


def test_has_gpu_implied_by_backends():
    """has_gpu() is the disjunction of the individual GPU backends."""
    any_backend = (
        env.has_cuda() or env.has_rocm() or env.has_opencl() or env.has_metal() or env.has_vulkan()
    )
    assert env.has_gpu() == any_backend


def test_tensorcore_implies_cuda():
    """Tensor Core support cannot be reported without a CUDA device."""
    if env.has_tensorcore():
        assert env.has_cuda()


def test_cudagraph_implies_cuda():
    """CUDA Graph support cannot be reported without a CUDA device."""
    if env.has_cudagraph():
        assert env.has_cuda()


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


def test_has_multi_gpu_is_bool():
    assert isinstance(env.has_multi_gpu(), bool)
    assert isinstance(env.has_multi_gpu(1), bool)
    # Requiring a single device is at least as permissive as requiring two.
    assert env.has_multi_gpu(1) or not env.has_multi_gpu(2)


@pytest.mark.parametrize(
    "probe,flag",
    [
        (env.has_cutlass, "USE_CUTLASS"),
        (env.has_rpc, "USE_RPC"),
        (env.has_nnapi, "USE_NNAPI_CODEGEN"),
        (env.has_openclml, "USE_CLML"),
        (env.has_mrvl, "USE_MRVL"),
    ],
    ids=lambda v: getattr(v, "__name__", v),
)
def test_build_flag_probe_matches_libinfo(probe, flag):
    """Pure build-flag probes agree with the build-info flag they wrap."""
    assert probe() == env._build_flag_enabled(flag)  # pylint: disable=protected-access


@pytest.mark.parametrize(
    "probe,parent",
    [
        (env.has_cudnn, env.has_cuda),
        (env.has_cublas, env.has_cuda),
        (env.has_nccl, env.has_cuda),
        (env.has_hipblas, env.has_rocm),
    ],
    ids=lambda v: v.__name__,
)
def test_library_probe_implies_parent_device(probe, parent):
    """A CUDA/ROCm library cannot be reported without its parent device."""
    if probe():
        assert parent()


def test_llvm_min_version_is_monotone():
    if not env.has_llvm():
        assert not env.has_llvm_min_version(1)
        return
    # An LLVM that satisfies a higher floor also satisfies a lower one.
    assert env.has_llvm_min_version(1)


def test_hexagon_run_implies_toolchain():
    """Full Hexagon support implies the compile-time toolchain is present."""
    if env.has_hexagon():
        assert env.has_hexagon_toolchain()


def test_probes_are_memoized():
    """Probes are cached so the driver/subprocess is hit once per process."""
    env.has_cuda()
    info = env._device_exists.cache_info()  # pylint: disable=protected-access
    assert info.hits + info.misses >= 1


# --- demonstration of the target idiom -------------------------------------
#
# The standard gating idiom: a plain registered pytest marker (for ``-m``
# selection) plus a skipif backed by a thin env probe (for runtime gating).


@pytest.mark.cuda
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_thin_cuda_idiom():
    dev = tvm.cuda()
    assert dev.exist


if __name__ == "__main__":
    tvm.testing.main()
