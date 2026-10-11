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
"""CUDA launch configuration, index calls, and both native launch backends."""

import os
import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.backend.cuda.launch import (
    AccessPolicyWindow,
    KernelAttributes,
    LaunchConfig,
    ProgrammaticEvent,
)
from tvm.backend.cuda.launch._impl import pack_kernel_attrs, pack_launch
from tvm.script import tirx as T
from tvm.testing import env


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid": 0},
        {"block": -1},
        {"block": (1024, 2)},
        {"block": (2**32 - 1,) * 3},
        {"grid": ()},
        {"block": (1, 2, 3, 4)},
        {"block": 1.5},
        {"grid": True},
        {"grid": 2**32},
        {"cluster": 3},
        {"preferred_cluster": 2},
        {"cooperative": 2},
        {"dynamic_smem_bytes": -1},
        {"priority": 2**31},
        {"shared_memory_mode": "unknown"},
        {"preferred_shared_memory_carveout": 101},
    ],
)
def test_invalid_launch_config(kwargs):
    with pytest.raises((ValueError, TypeError)):
        LaunchConfig(**({"grid": 8, "block": 128} | kwargs))


def test_configuration_is_immutable_and_presence_is_preserved():
    config = LaunchConfig(grid=8, block=(32, 2))
    assert config.grid == (8,)
    with pytest.raises(FrozenInstanceError):
        config.block = (128,)
    ordinary, _ = pack_launch(config)
    explicit, _ = pack_launch(config.replace(cluster=1, preferred_cluster=1))
    assert "cluster.x" not in ordinary
    assert "cluster.x" in explicit and "preferred_cluster.x" in explicit
    assert config.cluster is None


def test_dynamic_values_are_operands_with_their_original_types():
    grid = T.dynamic("grid", "int32")
    stream = T.dynamic("stream", "handle")
    ratio = T.dynamic("ratio", "float32")
    cooperative = T.dynamic("cooperative", "bool")
    fields, args = pack_launch(
        LaunchConfig(
            grid=grid,
            block=32,
            stream=stream,
            cooperative=cooperative,
            access_policy_window=AccessPolicyWindow(base_ptr=stream, num_bytes=64, hit_ratio=ratio),
            programmatic_event=ProgrammaticEvent(event=stream),
        )
    )
    values = dict(zip(fields, args))
    for name, value in [
        ("grid.x", grid),
        ("stream", stream),
        ("access_policy_window.hit_ratio", ratio),
        ("cooperative", cooperative),
    ]:
        assert values[name].same_as(value)
    assert all(isinstance(field, str) for field in fields)
    with pytest.raises(TypeError, match="compile-time"):
        KernelAttributes(min_blocks_per_sm=grid)
    with pytest.raises(ValueError, match="static"):
        pack_kernel_attrs(
            KernelAttributes(required_block_size=True), LaunchConfig(grid=1, block=grid)
        )
    with pytest.raises(ValueError, match="static"):
        pack_kernel_attrs(KernelAttributes(min_blocks_per_sm=1), LaunchConfig(grid=1, block=grid))
    with pytest.raises(TypeError):
        AccessPolicyWindow(base_ptr=stream)
    with pytest.raises(ValueError, match="hit_ratio"):
        AccessPolicyWindow(base_ptr=stream, num_bytes=64, hit_ratio=1.5)


def test_launch_region_roundtrip_and_host_only_operands():
    @T.function
    def kernel(A: T.Tensor((128,), "int32"), n: T.int32, stream: T.handle):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=n, block=32, stream=stream))
        bx, tx = T.cuda.block_idx("x"), T.cuda.thread_idx("x")
        A[bx * 32 + tx] = bx

    parsed = tvm.script.from_source(kernel.script(), extra_vars={"T": T})
    tvm.ir.assert_structural_equal(kernel, parsed)
    target = tvm.target.Target("cuda", host="llvm")
    with target:
        mod = tvm.tirx.transform.LowerTIRx()(
            tvm.IRModule({"main": kernel.with_attr("target", target)})
        )
        mod = tvm.tirx.transform.SplitHostDevice()(mod)
    devices = [f for f in mod.functions.values() if f.attrs and f.attrs.get("calling_conv") == 2]
    assert len(devices) == 1
    assert len(devices[0].params) == 1
    assert "cuda.launch_fields" in devices[0].attrs
    calls = []
    tvm_ffi.structural_walk(
        mod["main"],
        lambda node: (
            calls.append(node)
            if isinstance(node, tvm.ir.Call)
            and getattr(node.op, "name", "") == "tirx.call_ffi_kernel"
            else None
        ),
    )
    assert len(calls) == 1
    assert calls[0].attrs.num_kernel_args == 1
    assert "stream" in calls[0].attrs.launch_fields


def _compile(kernel, host, tmp_path):
    from tvm.backend.cuda import export_cuda_host

    arch = env.cuda_arch(0)
    target = tvm.target.Target({"kind": "cuda", "arch": arch}, host=host)
    built = tvm.compile(kernel, target=target).mod
    if host == "llvm":
        return built, None
    import tvm_ffi.cpp

    source = export_cuda_host(built)
    assert "#include <tvm/runtime/" not in source
    assert "#include <tvm/tirx/" not in source
    library = tvm_ffi.cpp.build_inline(
        name="launch_" + kernel.attrs["global_symbol"],
        cuda_sources=source,
        extra_cuda_cflags=[f"-arch={arch}"],
        extra_ldflags=["-lcuda"],
        build_directory=str(tmp_path),
        backend="cuda",
    )
    return tvm_ffi.load_module(library), library


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("host", ["llvm", "cuda_host"])
def test_multidimensional_indices_and_ffi_only_export(host, tmp_path):
    @T.function
    def coordinates(A: T.Tensor((8, 64, 4), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2, 2, 2), block=(16, 2, 2)))
        bx, by, bz = T.cuda.block_idx("x"), T.cuda.block_idx("y"), T.cuda.block_idx("z")
        tid = T.cuda.linear_thread_id()
        lane = T.cuda.lane_id()
        warp = T.cuda.warp_id()
        A[bx + by * 2 + bz * 4, tid, 0] = tid
        A[bx + by * 2 + bz * 4, tid, 1] = lane
        A[bx + by * 2 + bz * 4, tid, 2] = warp
        A[bx + by * 2 + bz * 4, tid, 3] = T.cuda.block_dim("y") + T.cuda.grid_dim("z")

    module, library = _compile(coordinates, host, tmp_path)

    def check():
        out = tvm.runtime.empty((8, 64, 4), "int32", tvm.cuda(0))
        module["coordinates"](out)
        values = np.arange(64, dtype="int32")
        expected = np.stack((values, values % 32, values // 32, np.full(64, 4)), axis=-1)
        np.testing.assert_array_equal(out.numpy(), np.broadcast_to(expected, (8, 64, 4)))
        if library:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
import torch
import tvm_ffi
assert "tvm" not in sys.modules
m = tvm_ffi.load_module(sys.argv[1])
a = torch.empty((8, 64, 4), dtype=torch.int32, device="cuda")
m["coordinates"](a)
v = torch.arange(64, dtype=torch.int32, device="cuda")
expected = torch.stack((v, v % 32, v // 32, torch.full_like(v, 4)), -1)
assert torch.equal(a, expected.expand(8, 64, 4))
assert "tvm" not in sys.modules
""",
                    str(library),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )

    tvm.testing.run_with_gpu_lock(check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="requires CUDA compute >= 9.0")
@pytest.mark.parametrize("host", ["llvm", "cuda_host"])
@pytest.mark.parametrize("required", [False, True])
def test_cluster_launch_and_required_block_dimensions(host, required, tmp_path):
    if required and not env.has_nvcc_version(13):
        pytest.skip("required block dimensions need CUDA 13 or newer")

    @T.function
    def clusters(A: T.Tensor((8, 3), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(4, 2), block=32, cluster=(2, 1)),
            kernel_attrs=T.cuda.KernelAttributes(required_block_size=required),
        )
        bx, by = T.cuda.block_idx("x"), T.cuda.block_idx("y")
        if T.cuda.thread_idx("x") == 0:
            A[bx + by * 4, 0] = T.cuda.cluster_id("x")
            A[bx + by * 4, 1] = T.cuda.cluster_cta_id("x")
            A[bx + by * 4, 2] = T.cuda.cluster_dim("x")

    module, _ = _compile(clusters, host, tmp_path)

    def check():
        out = tvm.runtime.empty((8, 3), "int32", tvm.cuda(0))
        module["clusters"](out)
        x = np.arange(8) % 4
        np.testing.assert_array_equal(out.numpy(), np.stack((x // 2, x % 2, np.full(8, 2)), -1))

    tvm.testing.run_with_gpu_lock(check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="requires CUDA compute >= 10.0")
@pytest.mark.skipif(not env.has_nvcc_version(12, 8), reason="requires CUDA 12.8 or newer")
@pytest.mark.parametrize("host", ["llvm", "cuda_host"])
def test_preferred_cluster_keeps_actual_hardware_coordinates(host, tmp_path):
    @T.function
    def preferred(A: T.Tensor((16, 5), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(4, 4), block=32, cluster=1, preferred_cluster=(2, 2))
        )
        bx, by = T.cuda.block_idx("x"), T.cuda.block_idx("y")
        if T.cuda.thread_idx("x") == 0:
            A[bx + by * 4, 0] = T.cuda.cluster_cta_id("x")
            A[bx + by * 4, 1] = T.cuda.cluster_cta_id("y")
            A[bx + by * 4, 2] = T.cuda.cluster_dim("x")
            A[bx + by * 4, 3] = T.cuda.cluster_dim("y")
            A[bx + by * 4, 4] = T.cuda.cta_pair_id()

    module, _ = _compile(preferred, host, tmp_path)

    def check():
        out = tvm.runtime.empty((16, 5), "int32", tvm.cuda(0))
        module["preferred"](out)
        values = out.numpy()
        for index, (cx, cy, sx, sy, pair) in enumerate(values):
            assert (sx, sy) in ((1, 1), (2, 2))
            assert cx == index % 4 % sx
            assert cy == index // 4 % sy
            assert pair == (cx + sx * cy) % 2

    tvm.testing.run_with_gpu_lock(check)


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("host", ["llvm", "cuda_host"])
def test_increasing_dynamic_shared_memory(host, tmp_path):
    import torch

    @T.function
    def shared(A: T.Tensor((1,), "int32"), n: T.int32, bytes_: T.int32):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=bytes_))
        scratch = T.alloc_tensor((n,), "int32", scope="shared.dyn")
        if T.cuda.thread_idx("x") == 0:
            scratch[n - 1] = n
            A[0] = scratch[n - 1]

    module, _ = _compile(shared, host, tmp_path)

    def check():
        out = tvm.runtime.empty((1,), "int32", tvm.cuda(0))
        limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        max_n = min(32768, limit // 4)
        for n in (256, max_n // 2, max_n, 256):
            module["shared"](out, n, n * 4)
            assert out.numpy()[0] == n
        with pytest.raises(Exception, match="smaller than"):
            module["shared"](out, 32, 64)

    tvm.testing.run_with_gpu_lock(check)


def test_static_shared_memory_override_cannot_shrink_allocation():
    @T.function
    def kernel():
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32, dynamic_smem_bytes=64))
        scratch = T.alloc_tensor((128,), "int32", scope="shared.dyn")
        scratch[0] = T.cuda.thread_idx("x")

    with pytest.raises(ValueError, match="smaller than"):
        tvm.compile(kernel, backend_config={"cuda": {"arch": "sm_80"}})
