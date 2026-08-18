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
"""Correctness coverage for workloads maintained by tirx-kernels."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

_WORKSPACE_TIRX_KERNELS = Path(__file__).resolve().parents[4] / "tirx-kernels"
if _WORKSPACE_TIRX_KERNELS.exists():
    sys.path.insert(0, str(_WORKSPACE_TIRX_KERNELS))

kernel_registry = pytest.importorskip("tirx_kernels.registry")
kernel_runner = pytest.importorskip("tirx_kernels.runner")
bench_suite_run = pytest.importorskip("tirx_kernels.bench_suite.run")


def _load_workloads():
    """The pinned bench sweep, across both tirx-kernels layouts.

    Newer checkouts keep one config file per kernel and assemble the sweep with
    ``load_config_dir``; older ones list it in a single ``workloads.yaml``.
    """
    if hasattr(bench_suite_run, "load_config_dir"):
        return bench_suite_run.load_config_dir()
    return bench_suite_run.load_workloads(bench_suite_run.DEFAULT_WORKLOADS)


_WORKLOADS = _load_workloads()
_KERNELS = {
    kernel_name: kernel_registry.load_kernel(kernel_name, strict=True)
    for kernel_name in sorted({workload["kernel"] for workload in _WORKLOADS})
}
_DISTRIBUTED_KERNELS = frozenset(
    # Both MegaMoE names are listed so the test works across tirx-kernels
    # checkouts from before and after the sm100_fp8_fp4_mega_moe rename.
    {
        "allgather_gemm",
        "deepgemm_fp8_fp4_mega_moe",
        "gemm_reduce_scatter",
        "sm100_fp8_fp4_mega_moe",
    }
)
_MEGA_MOE_KERNELS = frozenset({"deepgemm_fp8_fp4_mega_moe", "sm100_fp8_fp4_mega_moe"})
_XDIST_CUDA_DEVICE = None


def _manifest_kernel_config_cases():
    cases = []
    for workload in _WORKLOADS:
        kernel_name = workload["kernel"]
        label = workload["config"]
        mod = _KERNELS[kernel_name]
        configs = getattr(mod, "BENCH_CONFIGS", getattr(mod, "CONFIGS", []))
        matches = [config for config in configs if config.get("label") == label]
        if len(matches) != 1:
            raise ValueError(
                f"{kernel_name}::{label} must resolve to exactly one benchmark config, "
                f"found {len(matches)}"
            )
        config = matches[0]
        required_devices = int(config.get("num_processes", config.get("world_size", 1)))
        if workload["num_gpus"] != required_devices:
            raise ValueError(
                f"{kernel_name}::{label} declares num_gpus={workload['num_gpus']}, "
                f"but its config requires {required_devices}"
            )
        marks = (
            pytest.mark.xdist_group(name="distributed_device_zero")
            if kernel_name in _DISTRIBUTED_KERNELS
            else ()
        )
        cases.append(pytest.param(kernel_name, config, id=f"{kernel_name}::{label}", marks=marks))
    return cases


def _set_cuda_device_for_xdist_worker():
    global _XDIST_CUDA_DEVICE

    try:
        import torch
    except ImportError:
        return False

    if not torch.cuda.is_available():
        return False

    if _XDIST_CUDA_DEVICE is None:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
        worker_index = int(worker[2:]) if worker.startswith("gw") and worker[2:].isdigit() else 0
        _XDIST_CUDA_DEVICE = worker_index % torch.cuda.device_count()
    torch.cuda.set_device(_XDIST_CUDA_DEVICE)
    return True


def _visible_cuda_device_count():
    try:
        import torch
    except ImportError:
        return 0

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _required_cuda_device_count(config):
    return int(config.get("num_processes", config.get("world_size", 1)))


@contextmanager
def _registry_gpu_lock(kernel_name, config):
    try:
        import fcntl

        import torch
    except ImportError:
        yield
        return

    if not torch.cuda.is_available():
        yield
        return

    if kernel_name in _DISTRIBUTED_KERNELS:
        device_ids = range(_required_cuda_device_count(config))
    else:
        device_ids = (torch.cuda.current_device(),)

    lock_files = []
    try:
        for device_id in device_ids:
            lock_path = Path("/tmp") / f"tirx-kernels-registry-correctness-device{device_id}.lock"
            lock_file = lock_path.open("w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            lock_files.append(lock_file)

        torch.cuda.empty_cache()
        min_free_gb = float(os.environ.get("TIRX_KERNEL_TEST_MIN_FREE_GB", "32"))
        min_free_bytes = int(min_free_gb * 1024**3)
        for device_id in device_ids:
            free_bytes, _ = torch.cuda.mem_get_info(device_id)
            if free_bytes < min_free_bytes:
                pytest.skip(
                    f"CUDA device {device_id} has less than {min_free_gb:g} GiB free memory"
                )
        yield
    finally:
        try:
            torch.cuda.empty_cache()
        finally:
            for lock_file in reversed(lock_files):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()


@pytest.mark.parametrize(("kernel_name", "config"), _manifest_kernel_config_cases())
def test_manifest_tirx_kernel_correctness(kernel_name, config):
    _set_cuda_device_for_xdist_worker()
    required_devices = _required_cuda_device_count(config)
    visible_devices = _visible_cuda_device_count()
    if required_devices > visible_devices:
        pytest.skip(
            f"requires {required_devices} CUDA devices, but only {visible_devices} are visible"
        )
    if kernel_name in _MEGA_MOE_KERNELS:
        pytest.skip(
            "MegaMoE requires its dedicated multi-process scheduler; this suite's "
            "processes own CUDA contexts that its physical-device assignment rejects"
        )
    with _registry_gpu_lock(kernel_name, config):
        kernel_runner.run_kernel_test(kernel_name, config, registry=_KERNELS)
