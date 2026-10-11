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

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.backend.config import (
    backend_config_json,
    copy_backend_config,
    merge_backend_configs,
    parse_backend_config,
)
from tvm.backend.cuda import BackendConfig
from tvm.backend.cuda.backend_config import compiler_options, resolve_backend_config
from tvm.backend.cuda.transforms import BindBackendConfig
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


def test_defaults_overrides_snapshot_and_roundtrip():
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    defaults = resolve_backend_config(target=target)
    assert defaults["compiler"] == "nvrtc"
    assert defaults["nvcc"] == defaults["nvrtc"] == ["--use_fast_math"]
    assert defaults["ptxas"][-1] == "--register-usage-level=10"
    assert resolve_backend_config({}, target) == defaults
    local = {"cuda": BackendConfig(nvrtc=[], ptxas=["-O0"])}
    result = merge_backend_configs({"cuda": defaults}, local)
    assert result["cuda"]["nvrtc"] == []
    assert result["cuda"]["ptxas"] == ["-O0"]
    assert result["cuda"]["arch"] == "sm_100a"
    local["cuda"]["nvrtc"].append("--ftz=true")
    defaults["ptxas"].clear()
    assert result["cuda"]["nvrtc"] == []
    assert result["cuda"]["ptxas"] == ["-O0"]
    assert parse_backend_config(backend_config_json(result)) == result
    assert copy_backend_config(None) == copy_backend_config({}) == {}
    assert backend_config_json({"cuda": {"arch": "sm_100a", "nvrtc": []}}) == (
        backend_config_json({"cuda": {"nvrtc": [], "arch": "sm_100a"}})
    )
    # JSON sorts mapping keys, but never changes the order of compiler arguments.
    options = {"cuda": {"nvrtc": ["--ftz=true", "--ftz=false"]}}
    assert parse_backend_config(backend_config_json(options)) == options


@pytest.mark.parametrize(
    "config,exception",
    [
        ({"cuda": {"unknown": []}}, ValueError),
        ({"cuda": {"compiler": "clang"}}, ValueError),
        ({"cuda": {"arch": "compute_100"}}, ValueError),
        ({"cuda": {"arch": T.int32(100)}}, TypeError),
        ({"cuda": {"nvrtc": "--ftz=false"}}, TypeError),
        ({"cuda": {"nvcc": [1]}}, TypeError),
        ({"cuda": {"compiler": "nvrtc", "target_format": "fatbin"}}, ValueError),
        ({"rocm": {}}, ValueError),
        ({"compiler": "nvcc"}, ValueError),
    ],
)
def test_fixed_keys_and_types(config, exception):
    with pytest.raises(exception):
        copy_backend_config(config)


@pytest.mark.parametrize(
    "name,option",
    [
        ("nvrtc", "--gpu-architecture=sm_90"),
        ("nvcc", "-arch"),
        ("nvcc", "--gpu-code=sm_90"),
        ("nvcc", "--options-file=flags.txt"),
        ("nvcc", "--cubin"),
        ("nvrtc", "--ptxas-options=-O1"),
        ("ptxas", "--output-file=kernel.cubin"),
    ],
)
def test_managed_routing_flags(name, option):
    with pytest.raises(ValueError, match="managed"):
        copy_backend_config({"cuda": {name: [option]}})


@pytest.mark.parametrize("compiler,format", [("nvrtc", "cubin"), ("nvcc", "fatbin")])
def test_native_flags_and_default_format(monkeypatch, compiler, format):
    from tvm.backend.cuda.compiler import compile_source
    from tvm.support import nvcc

    calls = []

    def driver(source, output_format, arch, options, **kwargs):
        calls.append((output_format, arch, options))
        return b"artifact", "compiler output"

    monkeypatch.setattr(nvcc, f"_compile_cuda_{compiler}", driver)
    config = {
        "cuda": {
            "arch": "sm_100a",
            "compiler": compiler,
            compiler: ["--ftz=false", "-I/include", "--future-toolchain-option=1"],
            "ptxas": ["-O1", "--register-usage-level=2"],
        }
    }
    result = compile_source("// source", config)
    assert result.target_format == format
    assert calls[0][:2] == (format, "sm_100a")
    assert calls[0][2][:3] == config["cuda"][compiler]
    assert any("--register-usage-level=2" in flag for flag in calls[0][2])
    disabled = resolve_backend_config({"arch": "sm_100a", "nvrtc": [], "ptxas": []})
    assert compiler_options(disabled) == []
    linked = compile_source("#include <nvshmem.h>", config)
    assert linked.target_format == "cubin"


@pytest.mark.parametrize("tag_target", ["name", "dict", "export"])
def test_target_tag_defaults_and_precedence(tag_target):
    name = "test/backend-config"
    tvm.target.tag.register_tag(
        name,
        {
            "kind": "cuda",
            "arch": "sm_100a",
            "backend_config": {"cuda": {"compiler": "nvcc", "nvcc": ["-O1"]}},
        },
        override=True,
    )
    target = tvm.target.Target(name if tag_target == "name" else {"tag": name})
    if tag_target == "export":
        target = tvm.target.Target(target.export())
    resolved = resolve_backend_config(target=target)
    assert resolved["compiler"] == "nvcc" and resolved["nvcc"] == ["-O1"]
    assert resolved["ptxas"][-1] == "--register-usage-level=10"
    override = resolve_backend_config({"compiler": "nvrtc", "nvcc": []}, target)
    assert override["compiler"] == "nvrtc" and override["nvcc"] == []
    assert resolve_backend_config(target=target) == resolved


def test_entry_ir_roundtrip_and_resolution():
    @T.function
    def kernel(A: T.Tensor((32,), "float32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            backend_config={
                "cuda": {
                    "arch": "sm_90a",
                    "nvcc": ["--use_fast_math", "--ftz=false"],
                    "nvrtc": ["--use_fast_math", "--ftz=false"],
                }
            },
        )
        A[T.cuda.thread_idx("x")] = T.float32(1)

    tvm.ir.assert_structural_equal(
        kernel, tvm.script.from_source(kernel.script(), extra_vars={"T": T})
    )
    tvm.ir.assert_structural_equal(kernel, tvm.ir.load_json(tvm.ir.save_json(kernel)))
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="llvm")
    mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"main": kernel}))
    mod = BindBackendConfig(
        {
            "cuda": {
                "arch": "sm_100a",
                "nvcc": ["--use_fast_math", "--generate-line-info"],
                "nvrtc": ["--use_fast_math", "--generate-line-info"],
            }
        }
    )(mod)
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
    config = parse_backend_config(entry.attrs["backend_config"])
    assert "--ftz=false" in config["cuda"]["nvrtc"]
    assert "--generate-line-info" not in config["cuda"]["nvrtc"]


def test_target_arch_conflict():
    from tvm.backend.config import prepare_target

    with pytest.raises(ValueError, match="conflicts"):
        prepare_target({"kind": "cuda", "arch": "sm_90"}, {"cuda": {"arch": "sm_100a"}})
    assert prepare_target(None, {"cuda": {"arch": "sm_100a"}}).arch == "sm_100a"


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("compiler", ["nvrtc", "nvcc"])
def test_backend_config_controls_actual_ftz(compiler):
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
            kernel,
            backend_config={
                "cuda": {
                    "arch": arch,
                    "compiler": compiler,
                    "nvcc": ["--use_fast_math", "--ftz" + "=" + str(ftz).lower()],
                    "nvrtc": ["--use_fast_math", "--ftz" + "=" + str(ftz).lower()],
                }
            },
        )
        b = tvm.runtime.empty((32,), "float32", tvm.cuda())
        module.mod["kernel"](a, b)
        outputs.append(b.numpy())
        metadata = parse_backend_config(module.mod.imports[0].inspect_source("backend_config"))
        assert f"--ftz={str(ftz).lower()}" in metadata["cuda"][compiler]
        assert metadata["cuda"]["compiler"] == compiler
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
            backend_config={
                "cuda": {
                    "compiler": "nvrtc",
                    "nvcc": ["--use_fast_math", "--ftz=false"],
                    "nvrtc": ["--use_fast_math", "--ftz=false"],
                }
            },
        ):
            x = T.cuda.thread_idx("x")
            B[x] = A[x] * T.float32(0.5)
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            backend_config={
                "cuda": {
                    "compiler": "nvcc",
                    "nvcc": ["--use_fast_math", "--ftz=true"],
                    "nvrtc": ["--use_fast_math", "--ftz=true"],
                }
            },
        ):
            y = T.cuda.thread_idx("x")
            C[y] = B[y] * T.float32(0.5)

    arch = env.cuda_arch(0)
    module = tvm.compile(kernel, backend_config={"cuda": {"arch": arch}})
    assert len(module.mod.imports) == 2
    configs = [parse_backend_config(m.inspect_source("backend_config")) for m in module.mod.imports]
    assert {c["cuda"]["compiler"] for c in configs} == {"nvcc", "nvrtc"}
    inp = np.full(32, np.finfo(np.float32).tiny, dtype="float32")
    a = tvm.runtime.tensor(inp, tvm.cuda())
    b = tvm.runtime.empty((32,), "float32", tvm.cuda())
    c = tvm.runtime.empty((32,), "float32", tvm.cuda())
    module.mod["kernel"](a, b, c)
    np.testing.assert_array_equal(b.numpy(), inp * np.float32(0.5))
    np.testing.assert_array_equal(c.numpy(), np.zeros_like(inp))


@pytest.mark.parametrize("compiler", ["nvrtc", "nvcc"])
def test_concurrent_source_compilation(compiler):
    _require_compiler(compiler)
    from tvm.backend.cuda.compiler import compile_source

    source = 'extern "C" __global__ void half_value(float* x) { x[0] = x[0] * 0.5f; }'
    base = {"cuda": {"arch": "sm_100a", "compiler": compiler, "target_format": "ptx"}}
    configs = [
        merge_backend_configs(
            base,
            {
                "cuda": {
                    "nvcc": ["--use_fast_math", "--ftz" + "=" + str(ftz).lower()],
                    "nvrtc": ["--use_fast_math", "--ftz" + "=" + str(ftz).lower()],
                }
            },
        )
        for ftz in (False, True, False, True)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda cfg: compile_source(source, cfg), configs))
    for ftz, cfg, result in zip((False, True, False, True), configs, results):
        assert result.config["cuda"][compiler] == cfg["cuda"][compiler]
        assert (b"mul.ftz.f32" in result.binary) == ftz
    assert results[0].config == results[2].config
    assert results[0].binary == results[2].binary
    assert results[0].binary != results[1].binary


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("load_target", ["llvm", "cuda"])
def test_serialized_config_replays_in_fresh_process(tmp_path, monkeypatch, fallback, load_target):
    _require_compiler("nvrtc")

    @T.function
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        x = T.cuda.thread_idx("x")
        B[x] = A[x] * T.float32(0.5)

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1" if fallback else "0")
    config = {
        "cuda": {
            "arch": env.cuda_arch(0),
            "nvcc": ["--use_fast_math", "--ftz=false", "--generate-line-info"],
            "nvrtc": ["--use_fast_math", "--ftz=false", "--generate-line-info"],
        }
    }
    module = tvm.compile(main, backend_config=config)
    library = str(tmp_path / "replay.so")
    module.export_library(library)
    script = """
import sys
import numpy as np
import tvm
# The loader's Target may describe unrelated work or provide incompatible
# defaults. Source replay must use the configuration saved in the artifact.
with tvm.target.Target(LOAD_TARGET):
    module = tvm.runtime.load_module(sys.argv[1])
from tvm.backend.config import parse_backend_config
config = parse_backend_config(module.imports[0].inspect_source("backend_config"))["cuda"]
assert "--ftz=false" in config["nvrtc"] and "--generate-line-info" in config["nvrtc"]
x = np.full(32, np.finfo(np.float32).tiny, dtype="float32")
a = tvm.runtime.tensor(x, tvm.cuda())
b = tvm.runtime.empty((32,), "float32", tvm.cuda())
module["main"](a, b)
np.testing.assert_array_equal(b.numpy(), x * np.float32(0.5))
"""
    load_context = (
        "llvm"
        if load_target == "llvm"
        else {
            "kind": "cuda",
            "arch": "sm_90a",
            "backend_config": {"cuda": {"target_format": "fatbin"}},
        }
    )
    script = script.replace("LOAD_TARGET", repr(load_context))
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
                backend_config={
                    "cuda": {
                        "arch": "sm_90a",
                        "nvcc": ["--use_fast_math", "--ftz=false"],
                        "nvrtc": ["--use_fast_math", "--ftz=false"],
                    }
                },
            ):
                x = T.cuda.thread_idx("x")
                A[x] = Module.helper(A[x])
            with T.device_entry(
                launch=T.cuda.LaunchConfig(grid=1, block=32),
                backend_config={
                    "cuda": {
                        "arch": "sm_100a",
                        "nvcc": ["--use_fast_math", "--ftz=true"],
                        "nvrtc": ["--use_fast_math", "--ftz=true"],
                    }
                },
            ):
                y = T.cuda.thread_idx("x")
                B[y] = Module.helper(B[y])

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}, host="llvm")
    mod = tvm.tirx.transform.BindTarget(target)(Module)
    mod = BindBackendConfig()(mod)
    mod = tvm.tirx.transform.LowerTIRx()(mod)
    mod = tvm.tirx.transform.SplitHostDevice()(mod)
    mod = SpecializeDeviceHelpers()(mod)
    groups = {}
    for gv, fn in mod.functions.items():
        if fn.attrs["target"].host is not None:
            continue
        config = parse_backend_config(fn.attrs["backend_config"])
        groups.setdefault(
            (config["cuda"]["arch"], "--ftz=true" in config["cuda"]["nvrtc"]), []
        ).append(gv.name_hint)
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
        parse_backend_config(m.inspect_source("backend_config"))["cuda"]["arch"]
        for m in built.mod.imports
    } == {"sm_90a", "sm_100a"}


@pytest.mark.gpu
@pytest.mark.skipif(not tvm.cuda().exist, reason="requires CUDA")
@pytest.mark.parametrize("fallback", [False, True])
def test_cuda_host_embeds_independent_binaries_in_plain_cpp(tmp_path, monkeypatch, fallback):
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1" if fallback else "0")
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
            backend_config={
                "cuda": {
                    "compiler": "nvrtc",
                    "nvcc": ["--use_fast_math", "--ftz=false"],
                    "nvrtc": ["--use_fast_math", "--ftz=false"],
                }
            },
        ):
            x = T.cuda.thread_idx("x")
            B[x] = A[x] * T.Cast("float32", factor)
        with T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=32),
            backend_config={
                "cuda": {
                    "compiler": "nvcc",
                    "nvcc": ["--use_fast_math", "--ftz=true"],
                    "nvrtc": ["--use_fast_math", "--ftz=true"],
                }
            },
        ):
            y = T.cuda.thread_idx("x")
            C[y] = B[y] * T.Cast("float32", other)

    target = tvm.target.Target({"kind": "cuda", "arch": env.cuda_arch(0)}, host="cuda_host")
    module = tvm.compile(main, target=target)
    with tvm.target.Target("llvm"):
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
    from tvm.support.nvcc import tvm_callback_cuda_compile

    seen = []
    try:

        @tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)
        def mock_compile(source, config):
            seen.append(parse_backend_config(config))
            return [bytearray(code), "ptx"]

        assert load(old_source)["__tvm_cuda_binary"]()[0] == code
        assert seen == [{}]
    finally:
        tvm_ffi.register_global_func(
            "tvm_callback_cuda_compile", tvm_callback_cuda_compile, override=True
        )
    snapshot = backend_config_json({"cuda": {"arch": "sm_100a", "nvrtc": []}})
    current = old_binary + string(snapshot.encode())
    assert load(current).inspect_source("backend_config") == snapshot
    for malformed in (old_binary + b"x", current[:-1], current + b"x"):
        with pytest.raises(tvm.error.InternalError):
            load(malformed)


def test_removed_compile_environment(monkeypatch):
    from tvm.backend.cuda.backend_config import reject_legacy_compile_environment

    monkeypatch.setenv("TVM_CUDA_COMPILE_MODE", "nvcc")
    with pytest.raises(ValueError, match="backend_config.*compiler"):
        reject_legacy_compile_environment()


def test_generic_cuda_target_without_gpu(tmp_path):
    script = """
import tvm
import tvm.testing
from tvm.backend.cuda import BackendConfig
from tvm.backend.config import prepare_target
from tvm.backend.cuda.transforms import BindBackendConfig
from tvm.script import tirx as T

target = tvm.target.Target("cuda", host="llvm")
assert "arch" not in target.attrs
assert not tvm.testing.device_enabled("cuda")
resolved = prepare_target(target, {"cuda": BackendConfig(arch="sm_100a")})
assert resolved.arch == "sm_100a"
assert resolved.host.kind.name == "llvm"

@T.function
def main(A: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
    A[T.cuda.thread_idx("x")] = T.float32(1)

mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule({"main": main}))
try:
    BindBackendConfig()(mod)
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
from tvm.backend.cuda import BackendConfig
@T.function
def main(A: T.Tensor((32,), "float32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32),
                   backend_config={"cuda": T.cuda.BackendConfig(arch="sm_100a", compiler="nvcc")})
    A[T.cuda.thread_idx("x")] = T.float32(1)
target = TARGET
if target == "target_object":
    target = tvm.target.Target("cuda", host="llvm")
module = tvm.compile(main, target=target)
metadata = module.mod.imports[0].inspect_source("backend_config")
from tvm.backend.config import parse_backend_config
assert parse_backend_config(metadata)["cuda"]["arch"] == "sm_100a"
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


def test_source_fallback_does_not_invoke_compiler(monkeypatch):
    from tvm.support import nvcc

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")

    def unavailable(*args, **kwargs):
        raise AssertionError("source fallback must not require a CUDA compiler")

    monkeypatch.setattr(nvcc, "_compile_cuda_nvcc", unavailable)
    monkeypatch.setattr(nvcc, "_compile_cuda_nvrtc", unavailable)

    @T.function
    def main(A: T.Tensor((32,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        A[T.cuda.thread_idx("x")] = T.float32(1)

    module = tvm.compile(main, backend_config={"cuda": {"arch": "sm_100a"}})
    assert "__global__" in module.mod.imports[0].inspect_source("cuda")
    assert (
        parse_backend_config(module.mod.imports[0].inspect_source("backend_config"))["cuda"]["arch"]
        == "sm_100a"
    )


def test_identical_entries_share_one_module(monkeypatch):
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    config = {"cuda": {"arch": "sm_100a", "nvrtc": ["--ftz=false"]}}

    @T.function
    def main(A: T.Tensor((32,), "float32")):
        with T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32), backend_config=config):
            A[T.cuda.thread_idx("x")] = T.float32(1)
        with T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32), backend_config=config):
            A[T.cuda.thread_idx("x")] = T.float32(2)

    config["cuda"]["nvrtc"].clear()
    module = tvm.compile(main, backend_config={"cuda": {"arch": "sm_100a"}})
    assert len(module.mod.imports) == 1
    resolved = parse_backend_config(module.mod.imports[0].inspect_source("backend_config"))
    assert resolved["cuda"]["nvrtc"] == ["--ftz=false"]


def test_build_snapshots_config_before_passes(monkeypatch):
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    config = {"cuda": {"arch": "sm_100a", "nvrtc": ["--ftz=false"]}}

    @T.function
    def main(A: T.Tensor((32,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        A[T.cuda.thread_idx("x")] = T.float32(1)

    @tvm.transform.pass_instrument
    class ChangeCallerConfig:
        def run_before_pass(self, mod, info):
            config["cuda"]["nvrtc"] = ["--ftz=true"]

    with tvm.transform.PassContext(instruments=[ChangeCallerConfig()]):
        module = tvm.compile(main, backend_config=config)
    resolved = parse_backend_config(module.mod.imports[0].inspect_source("backend_config"))
    assert config["cuda"]["nvrtc"] == ["--ftz=true"]
    assert resolved["cuda"]["nvrtc"] == ["--ftz=false"]
