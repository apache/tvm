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
"""Explicit CUDA compile settings, independent device groups and artifact replay."""

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.backend.cuda import CompileConfig
from tvm.backend.cuda.transforms import BindCompileConfig
from tvm.script import tirx as T
from tvm.testing import env


def _require_compiler(compiler):
    if compiler == "nvcc":
        if shutil.which("nvcc") is None:
            pytest.skip("NVCC is not installed")
        from tvm.support.nvcc import get_cuda_version

        version = get_cuda_version()
    else:
        nvrtc = pytest.importorskip("cuda.bindings.nvrtc")
        status, major, minor = nvrtc.nvrtcVersion()
        assert status == nvrtc.nvrtcResult.NVRTC_SUCCESS
        version = (major, minor)
    if version < (12, 8):
        pytest.skip("sm_100a compile tests require CUDA 12.8 or newer")


def test_presence_validation_and_roundtrip():
    defaults = CompileConfig(arch="sm_100a", fast_math=True, defines=["OLD=1"], ptxas_opt_level=3)
    local = CompileConfig(fast_math=False, defines=[], ptxas_opt_level=0)
    result = defaults.overlay(local)
    assert result.fast_math is False
    assert result.defines == ()
    assert result.ptxas_opt_level == 0
    assert result.arch == "sm_100a"
    assert CompileConfig.from_json(result.to_json()) == result
    assert CompileConfig().to_dict() == {}
    with pytest.raises(FrozenInstanceError):
        result.fast_math = True
    with pytest.raises(TypeError):
        CompileConfig(ftz=T.int32(1))
    with pytest.raises(ValueError, match="fatbin"):
        CompileConfig(compiler="nvrtc", target_format="fatbin")
    with pytest.raises(ValueError, match="between"):
        CompileConfig(ptxas_reg_usage_level=11)
    with pytest.raises(TypeError):
        CompileConfig(nvrtc_options="--ftz=false")


@pytest.mark.parametrize(
    "name,option",
    [
        ("nvrtc_options", "--ftz=false"),
        ("nvcc_options", "-arch=sm_90"),
        ("nvcc_options", "--gpu-code=sm_90"),
        ("nvcc_options", "--options-file=flags.txt"),
        ("nvcc_options", "-I/include"),
        ("ptxas_options", "-O1"),
        ("ptxas_options", "--register-usage-level=2"),
    ],
)
def test_raw_flags_cannot_override_fields(name, option):
    with pytest.raises(ValueError, match="structured"):
        CompileConfig(**{name: [option]})


def test_math_overrides_and_disabled_preset():
    config = CompileConfig(arch="sm_100a", ftz=False).resolved()
    flags = config.compiler_options()
    assert flags.index("--use_fast_math") < flags.index("--ftz=false")
    assert "--use_fast_math" not in config.with_overrides(fast_math=False).compiler_options()
    assert not any(
        "line-info" in flag for flag in config.with_overrides(dump_dir="/tmp").compiler_options()
    )


def test_entry_ir_roundtrip_and_resolution():
    @T.function
    def kernel(A: T.Tensor((32,), "float32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            compile_config=T.cuda.CompileConfig(arch="sm_90a", ftz=False),
        )
        A[T.cuda.thread_idx("x")] = T.float32(1)

    tvm.ir.assert_structural_equal(
        kernel, tvm.script.from_source(kernel.script(), extra_vars={"T": T})
    )
    tvm.ir.assert_structural_equal(kernel, tvm.ir.load_json(tvm.ir.save_json(kernel)))
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="llvm")
    mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"main": kernel}))
    mod = BindCompileConfig(CompileConfig(arch="sm_100a", lineinfo=True))(mod)
    entries = []
    tvm_ffi.structural_walk(
        mod,
        lambda node: (
            entries.append(node)
            if isinstance(node, tvm.ir.RegionStmt) and node.op.name == "tirx.device_entry"
            else None
        ),
    )
    assert len(entries) == 1
    entry = entries[0]
    assert entry.attrs["target"].arch == "sm_90a"
    config = CompileConfig.from_json(entry.attrs["cuda.compile_config"])
    assert config.ftz is False and config.lineinfo is True


def test_target_arch_conflict():
    from tvm.backend.cuda.compile_config import prepare_target

    with pytest.raises(ValueError, match="conflicts"):
        prepare_target({"kind": "cuda", "arch": "sm_90"}, CompileConfig(arch="sm_100a"))
    assert prepare_target(None, CompileConfig(arch="sm_100a")).arch == "sm_100a"


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("compiler", ["nvrtc", "nvcc"])
def test_compile_config_controls_actual_ftz(compiler):
    _require_compiler(compiler)

    @T.function
    def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        x = T.cuda.thread_idx("x")
        B[x] = A[x] * T.float32(0.5)

    arch = env.cuda_arch(0)
    inp = np.full(32, np.finfo(np.float32).tiny, dtype="float32")
    a = tvm.runtime.tensor(inp, tvm.cuda())
    outputs = []
    for ftz in (False, True):
        module = tvm.compile(
            kernel, compile_config=CompileConfig(arch=arch, compiler=compiler, ftz=ftz)
        )
        b = tvm.runtime.empty((32,), "float32", tvm.cuda())
        module.mod["kernel"](a, b)
        outputs.append(b.numpy())
        metadata = CompileConfig.from_json(
            module.mod.imports[0].inspect_source("cuda.compile_config")
        )
        assert metadata.ftz == ftz and metadata.compiler == compiler
    np.testing.assert_array_equal(outputs[0], inp * np.float32(0.5))
    np.testing.assert_array_equal(outputs[1], np.zeros_like(inp))


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
def test_two_entries_keep_compilers_math_and_order():
    _require_compiler("nvrtc")
    _require_compiler("nvcc")

    @T.function
    def kernel(
        A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32"), C: T.Tensor((32,), "float32")
    ):
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            compile_config=T.cuda.CompileConfig(compiler="nvrtc", ftz=False),
        ):
            x = T.cuda.thread_idx("x")
            B[x] = A[x] * T.float32(0.5)
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            compile_config=T.cuda.CompileConfig(compiler="nvcc", ftz=True),
        ):
            y = T.cuda.thread_idx("x")
            C[y] = B[y] * T.float32(0.5)

    arch = env.cuda_arch(0)
    module = tvm.compile(kernel, compile_config=CompileConfig(arch=arch))
    assert len(module.mod.imports) == 2
    configs = [
        CompileConfig.from_json(m.inspect_source("cuda.compile_config")) for m in module.mod.imports
    ]
    assert {c.compiler for c in configs} == {"nvcc", "nvrtc"}
    inp = np.full(32, np.finfo(np.float32).tiny, dtype="float32")
    a = tvm.runtime.tensor(inp, tvm.cuda())
    b = tvm.runtime.empty((32,), "float32", tvm.cuda())
    c = tvm.runtime.empty((32,), "float32", tvm.cuda())
    module.mod["kernel"](a, b, c)
    np.testing.assert_array_equal(b.numpy(), inp * np.float32(0.5))
    np.testing.assert_array_equal(c.numpy(), np.zeros_like(inp))


@pytest.mark.parametrize("compiler", ["nvrtc", "nvcc"])
def test_concurrent_source_compilation_and_diagnostics(tmp_path, compiler):
    _require_compiler(compiler)
    from tvm.backend.cuda.compiler import compile_source

    source = 'extern "C" __global__ void half_value(float* x) { x[0] = x[0] * 0.5f; }'
    base = CompileConfig(arch="sm_100a", compiler=compiler, target_format="ptx")
    configs = [base.with_overrides(ftz=ftz) for ftz in (False, True, False, True)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda cfg: compile_source(source, cfg), configs))
    for cfg, result in zip(configs, results):
        assert result.config.ftz == cfg.ftz
        assert (b"mul.ftz.f32" in result.binary) == cfg.ftz
    assert results[0].cache_key == results[2].cache_key
    assert results[0].cache_key != results[1].cache_key
    dumped = compile_source(source, configs[0].with_overrides(dump_dir=str(tmp_path)))
    assert dumped.cache_key == results[0].cache_key
    assert dumped.binary == results[0].binary
    assert (tmp_path / (dumped.cache_key + ".ptx")).read_bytes() == dumped.binary


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("fallback", [False, True])
def test_serialized_config_replays_in_fresh_process(tmp_path, monkeypatch, fallback):
    _require_compiler("nvrtc")

    @T.function
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        x = T.cuda.thread_idx("x")
        B[x] = A[x] * T.float32(0.5)

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1" if fallback else "0")
    config = CompileConfig(arch=env.cuda_arch(0), ftz=False, lineinfo=True)
    module = tvm.compile(main, compile_config=config)
    library = str(tmp_path / "replay.so")
    module.export_library(library)
    script = """
import sys
import numpy as np
import tvm
from tvm.backend.cuda import CompileConfig
module = tvm.runtime.load_module(sys.argv[1])
config = CompileConfig.from_json(module.imports[0].inspect_source("cuda.compile_config"))
assert config.ftz is False and config.lineinfo is True
x = np.full(32, np.finfo(np.float32).tiny, dtype="float32")
a = tvm.runtime.tensor(x, tvm.cuda())
b = tvm.runtime.empty((32,), "float32", tvm.cuda())
module["main"](a, b)
np.testing.assert_array_equal(b.numpy(), x * np.float32(0.5))
"""
    environment = dict(os.environ, TVM_COMPILE_FORCE_FALLBACK="0")
    result = subprocess.run(
        [sys.executable, "-c", script, library], env=environment, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_device_helpers_are_cloned_per_configuration():
    _require_compiler("nvrtc")
    from tvm.backend.cuda.transforms import SpecializeDeviceHelpers
    from tvm.script import ir as I

    @I.ir_module
    class Module:
        @T.function(private=True)
        def leaf(x: T.float32) -> T.float32:
            return x * T.float32(0.5)

        @T.function(private=True)
        def helper(x: T.float32) -> T.float32:
            return Module.leaf(x)

        @T.function
        def kernel(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
            with T.device_entry(
                launch=T.cuda.LaunchConfig(grid=1, block=32),
                compile_config=T.cuda.CompileConfig(arch="sm_90a", ftz=False),
            ):
                x = T.cuda.thread_idx("x")
                A[x] = Module.helper(A[x])
            with T.device_entry(
                launch=T.cuda.LaunchConfig(grid=1, block=32),
                compile_config=T.cuda.CompileConfig(arch="sm_100a", ftz=True),
            ):
                y = T.cuda.thread_idx("x")
                B[y] = Module.helper(B[y])

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="llvm")
    mod = tvm.tirx.transform.BindTarget(target)(Module)
    mod = BindCompileConfig()(mod)
    mod = tvm.tirx.transform.LowerTIRx()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)
    mod = SpecializeDeviceHelpers()(mod)
    groups = {}
    for gv, fn in mod.functions.items():
        if fn.attrs["target"].host is not None:
            continue
        config = CompileConfig.from_json(fn.attrs["cuda.compile_config"])
        groups.setdefault((config.arch, config.ftz), []).append(gv.name_hint)
    assert set(groups) == {("sm_90a", False), ("sm_100a", True)}
    assert all(len(names) == 3 for names in groups.values()), mod.script()
    assert set(groups[("sm_90a", False)]).isdisjoint(groups[("sm_100a", True)])
    dispatched = set()

    @tvm.transform.pass_instrument
    class CheckHelperTargets:
        def run_before_pass(self, lowered, info):
            if info.name == "tirx.TileDispatch":
                for gv, fn in lowered.functions.items():
                    if gv.name_hint.startswith(("helper", "leaf")):
                        dispatched.add(fn.attrs["target"].arch)
                        assert gv.name_hint not in ("helper", "leaf")

    with tvm.transform.PassContext(instruments=[CheckHelperTargets()]):
        built = tvm.compile(Module)
    assert dispatched == {"sm_90a", "sm_100a"}
    assert len(built.mod.imports) == 2
    assert {
        CompileConfig.from_json(m.inspect_source("cuda.compile_config")).arch
        for m in built.mod.imports
    } == {"sm_90a", "sm_100a"}


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
def test_cuda_host_embeds_independent_binaries_in_plain_cpp(tmp_path):
    _require_compiler("nvrtc")
    _require_compiler("nvcc")
    import tvm_ffi.cpp

    from tvm.backend.cuda.host import export_cuda_host
    from tvm.support.nvcc import find_cuda_path

    @T.function
    def main(
        A: T.Tensor((32,), "float32"),
        B: T.Tensor((32,), "float32"),
        C: T.Tensor((32,), "float32"),
        factor: T.float16,
        other: T.bfloat16,
    ):
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            compile_config=T.cuda.CompileConfig(compiler="nvrtc", ftz=False),
        ):
            x = T.cuda.thread_idx("x")
            B[x] = A[x] * T.Cast("float32", factor)
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            compile_config=T.cuda.CompileConfig(compiler="nvcc", ftz=True),
        ):
            y = T.cuda.thread_idx("x")
            C[y] = B[y] * T.Cast("float32", other)

    target = tvm.target.Target({"kind": "cuda", "arch": env.cuda_arch(0)}, host="cuda_host")
    module = tvm.compile(main, target=target)
    source = export_cuda_host(module.mod)
    assert "__global__" not in source
    cuda = find_cuda_path()
    library = tvm_ffi.cpp.build_inline(
        name="independent_cuda_entries",
        cpp_sources=source,
        backend="cpu",
        extra_include_paths=[cuda + "/include"],
        extra_ldflags=["-L" + cuda + "/lib64", "-lcuda", "-lcudart"],
        build_directory=str(tmp_path),
    )
    script = """
import sys
import torch
import tvm_ffi
assert 'tvm' not in sys.modules
module = tvm_ffi.load_module(sys.argv[1])
a = torch.full((32,), torch.finfo(torch.float32).tiny, device='cuda')
a[1::2] = 2.0
b, c = torch.empty_like(a), torch.empty_like(a)
module['main'](a, b, c, 0.5, 0.5)
assert torch.equal(b, a * 0.5)
expected = torch.zeros_like(c)
expected[1::2] = 0.5
assert torch.equal(c, expected)
assert 'tvm' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script, library], text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_payloads():
    import struct

    load = tvm.get_global_func("ffi.Module.load_from_bytes.cuda", allow_missing=True)
    if load is None:
        pytest.skip("CUDA runtime module loader is not linked")

    def string(value):
        return struct.pack("<Q", len(value)) + value

    code = b".version 8.0\n.target sm_80\n.address_size 64\n"
    old_binary = string(b"ptx") + struct.pack("<Q", 0) + string(code)
    module = load(old_binary)
    assert module["__tvm_cuda_binary"]()[0] == code
    old_source = string(b"cuda") + struct.pack("<Q", 0) + string(b"// legacy source")
    with pytest.raises(ValueError, match="regenerate"):
        load(old_source)


def test_removed_compile_environment(monkeypatch):
    from tvm.backend.cuda.compile_config import reject_legacy_compile_environment

    monkeypatch.setenv("TVM_CUDA_COMPILE_MODE", "nvcc")
    with pytest.raises(ValueError, match="CompileConfig.*compiler"):
        reject_legacy_compile_environment()


def test_generic_cuda_target_without_gpu(tmp_path):
    script = """
import tvm
import tvm.testing
from tvm.backend.cuda import CompileConfig
from tvm.backend.cuda.compile_config import prepare_target
from tvm.backend.cuda.transforms import BindCompileConfig
from tvm.script import tirx as T

target = tvm.target.Target("cuda", host="llvm")
assert "arch" not in target.attrs
assert not tvm.testing.device_enabled("cuda")
resolved = prepare_target(target, CompileConfig(arch="sm_100a"))
assert resolved.arch == "sm_100a"
assert resolved.host.kind.name == "llvm"

@T.function
def main(A: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    A[T.cuda.thread_idx("x")] = T.float32(1)

mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"main": main}))
try:
    BindCompileConfig()(mod)
except ValueError as error:
    assert "explicit" in str(error)
else:
    raise AssertionError("offline compilation must not guess an architecture")
"""
    path = tmp_path / "generic_cuda.py"
    path.write_text(script)
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", TVM_TEST_TARGETS="llvm;cuda")
    result = subprocess.run(
        [sys.executable, str(path)], env=environment, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("target", [None, "cuda", {"kind": "cuda"}, "target_object"])
def test_entry_architectures_compile_without_gpu_detection(tmp_path, target):
    _require_compiler("nvcc")
    script = """
import tvm
from tvm.script import tirx as T
from tvm.backend.cuda import CompileConfig
@T.function
def main(A: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32),
                   compile_config=T.cuda.CompileConfig(arch="sm_100a", compiler="nvcc"))
    A[T.cuda.thread_idx("x")] = T.float32(1)
target = TARGET
if target == "target_object":
    target = tvm.target.Target("cuda", host="llvm")
module = tvm.compile(main, target=target)
metadata = module.mod.imports[0].inspect_source("cuda.compile_config")
assert CompileConfig.from_json(metadata).arch == "sm_100a"
assert "arch" not in tvm.target.Target("cuda").attrs
"""
    # Use a source file because the TIRx construction decorator reads its definition.
    path = tmp_path / "offline.py"
    path.write_text(script.replace("TARGET", repr(target)))
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        [sys.executable, str(path)], env=environment, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
