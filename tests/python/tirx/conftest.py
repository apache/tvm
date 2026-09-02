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
"""Shared CUDA device and exact-architecture handling for TIRx tests."""

import os
import re

import pytest

from tvm.testing import env

_CUDA_ARCH_PATTERN = re.compile(r"sm_[1-9][0-9]*(?:a|f)?\Z")
_XDIST_CUDA_DEVICE = None


def _visible_cuda_archs():
    try:
        import torch
    except ImportError:
        return ()

    if not torch.cuda.is_available():
        return ()
    return tuple(env.cuda_arch(device_id) for device_id in range(torch.cuda.device_count()))


def _require_homogeneous_cuda_archs():
    arches = _visible_cuda_archs()
    if not arches:
        return None
    if any(arch is None for arch in arches):
        raise pytest.UsageError(f"could not determine every visible CUDA architecture: {arches}")
    if len(set(arches)) != 1:
        profile = ", ".join(f"cuda:{device_id}={arch}" for device_id, arch in enumerate(arches))
        raise pytest.UsageError(
            "mixed-architecture CUDA pools are unsupported by the TIRx test suite: " + profile
        )
    return arches[0]


def _set_cuda_device_for_xdist_worker():
    global _XDIST_CUDA_DEVICE

    try:
        import torch
    except ImportError:
        return None

    if not torch.cuda.is_available():
        return None
    if _XDIST_CUDA_DEVICE is None:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
        worker_index = int(worker[2:]) if worker.startswith("gw") and worker[2:].isdigit() else 0
        _XDIST_CUDA_DEVICE = worker_index % torch.cuda.device_count()
    torch.cuda.set_device(_XDIST_CUDA_DEVICE)
    return _XDIST_CUDA_DEVICE


def pytest_configure(config):
    del config
    _require_homogeneous_cuda_archs()
    _set_cuda_device_for_xdist_worker()


def pytest_runtest_setup(item):
    current_device = _set_cuda_device_for_xdist_worker()
    marker = item.get_closest_marker("cuda_arch")
    if marker is None:
        return

    unknown_kwargs = set(marker.kwargs) - {"device"}
    if unknown_kwargs:
        raise pytest.UsageError(
            f"cuda_arch marker has unsupported keyword(s): {sorted(unknown_kwargs)}"
        )
    arches = tuple(marker.args)
    if not arches or any(
        not isinstance(arch, str) or _CUDA_ARCH_PATTERN.fullmatch(arch) is None for arch in arches
    ):
        raise pytest.UsageError(
            "cuda_arch marker requires one or more canonical architectures such as sm_100a"
        )

    device_source = marker.kwargs.get("device", "cuda0")
    if device_source == "cuda0":
        device_id = 0
    elif device_source == "current":
        device_id = current_device if current_device is not None else 0
    else:
        raise pytest.UsageError("cuda_arch marker device must be 'cuda0' or 'current'")

    actual = env.cuda_arch(device_id)
    if actual not in arches:
        required = ", ".join(arches)
        pytest.skip(
            f"requires CUDA architecture {required} on {device_source}; "
            f"actual architecture is {actual or 'unavailable'}"
        )
