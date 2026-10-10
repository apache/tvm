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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for IKET lowering, metadata, installation checks, and trace contracts."""

import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import tvm
from tvm.backend.cuda import transforms as cuda_transforms
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.cuda.iket import IketProfiler

TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
ORACLE_PATH = Path(__file__).parent / "oracle" / "iket_official_cutlass_4_6_0_oracle.json"


@T.function
def serial_a(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("a")
    out[tx] = tx + 1


@T.function
def serial_b(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("b")
    out[tx] = tx + 2


@T.function
def plain_entry(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    tx = T.cuda.thread_idx("x")
    out[tx] = tx + 7


@T.function
def push_pop_kernel(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.range_push("outer")
    iket.range_push("inner")
    iket.mark("point")
    iket.range_pop()
    iket.range_pop()
    out[tx] = tx


@T.function
def token_loop(n: T.int32, out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    token: T.uint32 = iket.sentinel_token("sentinel")
    for i in T.serial(n, unroll=False):
        iket.range_end(token)
        if i % 2 == 0:
            token = iket.range_start("even")
        else:
            token = iket.range_start("odd")
    iket.range_end(token)
    out[tx] = tx + n


@T.function
def payload_kernel(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("payload", tx)
    out[tx] = tx


@T.function
def payload_types(n: T.int64, out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("bool", tx == 0)
    iket.mark("i8", T.int8(-8))
    iket.mark("u8", T.uint8(8))
    iket.mark("i16", T.int16(-16))
    iket.mark("u16", T.uint16(16))
    iket.mark("i32", T.int32(-32))
    iket.mark("u32", T.uint32(32))
    iket.mark("i64", n)
    iket.mark("u64", T.uint64(64))
    iket.mark("f32", T.float32(-3.25))
    iket.mark("f64", T.float64(6.5))
    token: T.uint32 = iket.range_start("token_payload", T.int32(-7))
    iket.range_end(token, T.int32(9))
    iket.range_push("stack_payload", T.float32(1.5))
    iket.range_pop()
    out[tx] = tx


@T.function
def payload_presence_mismatch(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    token: T.uint32 = iket.range_start("mismatch", tx)
    iket.range_end(token)
    out[tx] = tx


@T.function
def payload_type_mismatch(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    token: T.uint32 = iket.range_start("mismatch", tx)
    iket.range_end(token, T.uint32(tx))
    out[tx] = tx


@T.function
def sentinel_only_payload(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    token: T.uint32 = iket.sentinel_token("not-a-declaration")
    iket.range_end(token, out[tx])
    out[tx] = tx


@T.function
def payload_float16(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("bad", T.float16(1))
    out[tx] = tx


@T.function
def payload_bfloat16(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("bad", T.bfloat16(1))
    out[tx] = tx


@T.function
def payload_pointer(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    tx = T.cuda.thread_idx("x")
    T.evaluate(tvm.tirx.call_intrin("", "tirx.cuda.iket_mark", "bad", out.data))
    out[tx] = tx


@T.function
def payload_vector(out: T.Tensor((1,), "int32x4")):
    T.device_entry()
    T.evaluate(tvm.tirx.call_intrin("", "tirx.cuda.iket_mark", "bad", out[0]))


@T.function
def schema_i32(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("shared-schema", T.int32(tx))
    out[tx] = tx


@T.function
def schema_u32(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("shared-schema", T.uint32(tx))
    out[tx] = tx


@T.function
def schema_no_payload(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("shared-schema")
    out[tx] = tx


@T.function
def marks_30(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("e00")
    iket.mark("e01")
    iket.mark("e02")
    iket.mark("e03")
    iket.mark("e04")
    iket.mark("e05")
    iket.mark("e06")
    iket.mark("e07")
    iket.mark("e08")
    iket.mark("e09")
    iket.mark("e10")
    iket.mark("e11")
    iket.mark("e12")
    iket.mark("e13")
    iket.mark("e14")
    iket.mark("e15")
    iket.mark("e16")
    iket.mark("e17")
    iket.mark("e18")
    iket.mark("e19")
    iket.mark("e20")
    iket.mark("e21")
    iket.mark("e22")
    iket.mark("e23")
    iket.mark("e24")
    iket.mark("e25")
    iket.mark("e26")
    iket.mark("e27")
    iket.mark("e28")
    iket.mark("e29")
    out[tx] = tx


@T.function
def marks_31(out: T.Tensor((32,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
    iket.mark("e00")
    iket.mark("e01")
    iket.mark("e02")
    iket.mark("e03")
    iket.mark("e04")
    iket.mark("e05")
    iket.mark("e06")
    iket.mark("e07")
    iket.mark("e08")
    iket.mark("e09")
    iket.mark("e10")
    iket.mark("e11")
    iket.mark("e12")
    iket.mark("e13")
    iket.mark("e14")
    iket.mark("e15")
    iket.mark("e16")
    iket.mark("e17")
    iket.mark("e18")
    iket.mark("e19")
    iket.mark("e20")
    iket.mark("e21")
    iket.mark("e22")
    iket.mark("e23")
    iket.mark("e24")
    iket.mark("e25")
    iket.mark("e26")
    iket.mark("e27")
    iket.mark("e28")
    iket.mark("e29")
    iket.mark("e30")
    out[tx] = tx


def _compile(func, *, target=TARGET):
    return IketProfiler().compile(func, target=target, tir_pipeline="tirx")


def _cuda_source(executable):
    modules = executable.mod._collect_from_import_tree(  # pylint: disable=protected-access
        lambda module: module.kind == "cuda"
    )
    assert len(modules) == 1
    return modules[0].inspect_source("cuda")


def _official_global_bytes(source, symbol):
    match = re.search(rf"unsigned char {re.escape(symbol)}\[(\d+)\] = \{{([^}}]*)\}};", source)
    assert match is not None, symbol
    size = int(match.group(1))
    values = [int(value) for value in match.group(2).split(",") if value]
    assert len(values) == size
    return bytes(values)


def _event_bytes(source, name):
    match = re.search(rf"__iket_evt_decl_{re.escape(name)}_(\d+)_attrs", source)
    assert match is not None, name
    event_id = int(match.group(1))
    return event_id, _official_global_bytes(source, f"__iket_evt_decl_{name}_{event_id}_attrs")


def _many_marks(count):
    marks = "\n".join(f'    iket.mark("e{index:04d}")' for index in range(count))
    source = f"""@T.function
def main(out: T.Tensor((1,), "int32")):
    T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=1))
    iket = IketProfiler()
    tx = T.cuda.thread_idx("x")
{marks}
    out[tx] = 1
"""
    return tvm.script.from_source(source, {"T": T, "IketProfiler": IketProfiler})


def _packaged_nvdisasm():
    distribution = metadata.distribution("nvidia-cuda-nvdisasm")
    return Path(distribution.locate_file("nvidia/cu13/bin/nvdisasm"))


def _has_packaged_nvdisasm():
    try:
        return _packaged_nvdisasm().exists()
    except metadata.PackageNotFoundError:
        return False


def _nvrtc_disassemble(source, tmp_path):
    from tvm.support.nvcc import compile_cuda

    tmp_path.mkdir(parents=True, exist_ok=True)
    cubin = compile_cuda(
        source,
        compile_config=T.cuda.CompileConfig(
            arch="sm_100a", target_format="cubin", compiler="nvrtc"
        ),
    )
    cubin_path = tmp_path / "kernel.cubin"
    cubin_path.write_bytes(cubin)
    return subprocess.run(
        [_packaged_nvdisasm(), "-c", cubin_path],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_public_interface_is_official_only():
    signature = inspect.signature(IketProfiler.compile)
    assert "backend" not in signature.parameters
    profiler = IketProfiler()
    assert not hasattr(profiler, "capture")
    assert not hasattr(profiler, "export")
    assert not hasattr(tvm.tirx.transform, "LowerIket")
    assert callable(cuda_transforms.LowerIket)

    script = serial_a.script()
    assert 'T.cuda.iket_mark("a")' in script
    assert "T.tirx.iket" not in script
    assert (
        tvm.script.from_source(
            script, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}
        ).script()
        == script
    )

    payload_script = payload_types.script()
    assert 'T.cuda.iket_mark("i8", T.int8(-8))' in payload_script
    assert 'T.cuda.iket_range_start("token_payload", -7)' in payload_script
    assert "T.cuda.iket_range_end(token, 9)" in payload_script
    assert (
        tvm.script.from_source(
            payload_script, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx}
        ).script()
        == payload_script
    )


@pytest.mark.parametrize(
    "name",
    ("mark", "range_start", "range_end", "range_push", "range_pop", "sentinel_token"),
)
def test_annotation_ops_are_cuda_owned(name):
    op = tvm.ir.Op.get(f"tirx.cuda.iket_{name}")
    assert op.get_attr("TIRxOpCategory") == "device_intrin"
    assert op.get_attr("TDeviceIntrinsicNamespace") == "cuda"


def test_regular_lowering_strips_annotations_and_tokens():
    stripped = cuda_transforms.LowerIket()(tvm.IRModule({"main": token_loop}))
    script = stripped.script()
    assert "cuda.iket" not in script
    assert "sentinel" not in script
    assert "token:" not in script

    def make_kernel(with_annotation):
        @T.function
        def main(out: T.Tensor((32,), "int32")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
            iket = IketProfiler()
            tx = T.cuda.thread_idx("x")
            if with_annotation and tx % 2 == 0:
                iket.mark("strip-me")
            out[tx] = tx + 1

        return main

    sources = []
    for with_annotation in (False, True):
        executable = tvm.compile(
            tvm.IRModule({"main": make_kernel(with_annotation)}),
            target=TARGET,
            tir_pipeline="tirx",
        )
        sources.append(executable.mod.imports[0].inspect_source("cuda"))
    assert sources[0] == sources[1]
    assert "iket" not in sources[1].lower()


@pytest.mark.parametrize("config_env", ["IKET_INJECTION_CONFIG", "SMODEL_INJECTION_CONFIG"])
def test_verified_injected_child_automatically_enables_plain_jit(monkeypatch, config_env):
    monkeypatch.delenv("IKET_INJECTION_CONFIG", raising=False)
    monkeypatch.delenv("SMODEL_INJECTION_CONFIG", raising=False)
    monkeypatch.setenv("TVM_IKET_INJECTED_CHILD_ENABLE", "1")
    monkeypatch.delenv("TVM_IKET_OFFICIAL_PROFILE", raising=False)
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/verified/libsmodel_injection.so")
    monkeypatch.setenv(config_env, "/verified/config.json")
    executable = tvm.compile(tvm.IRModule({"main": serial_a}), target=TARGET, tir_pipeline="tirx")
    source = _cuda_source(executable)
    assert "__iket_evt_decl_a_1_attrs" in source


@pytest.mark.parametrize(
    "missing_env",
    ("CUDA_INJECTION64_PATH", "config", "TVM_IKET_INJECTED_CHILD_ENABLE"),
)
@pytest.mark.parametrize("config_env", ["IKET_INJECTION_CONFIG", "SMODEL_INJECTION_CONFIG"])
def test_injected_child_auto_enable_remains_fail_closed(monkeypatch, missing_env, config_env):
    monkeypatch.delenv("IKET_INJECTION_CONFIG", raising=False)
    monkeypatch.delenv("SMODEL_INJECTION_CONFIG", raising=False)
    values = {
        "TVM_IKET_INJECTED_CHILD_ENABLE": "1",
        "CUDA_INJECTION64_PATH": "/verified/libsmodel_injection.so",
        config_env: "/verified/config.json",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(config_env if missing_env == "config" else missing_env)
    executable = tvm.compile(tvm.IRModule({"main": serial_a}), target=TARGET, tir_pipeline="tirx")
    assert "iket" not in _cuda_source(executable).lower()


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("tvm.tirx.compilation_pipeline", "default_tir_pipeline"),
        ("tvm.tirx.compilation_pipeline", "tirx_pipeline"),
    ],
)
def test_tirx_pipelines_immediately_lower_iket(module_name, factory_name):
    factory = getattr(importlib.import_module(module_name), factory_name)
    visited = []

    @tvm.transform.pass_instrument
    class RecordPasses:
        def run_before_pass(self, mod, info):
            visited.append(("before", info.name))

        def run_after_pass(self, mod, info):
            visited.append(("after", info.name))

    pipeline, _, _ = factory()
    with tvm.transform.PassContext(instruments=[RecordPasses()]):
        pipeline(tvm.IRModule())

    split = visited.index(("after", tvm.tirx.transform.SplitHostDevice().info.name))
    assert visited[split + 1] == ("before", cuda_transforms.LowerIket().info.name)


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("tvm.backend.trn.pipeline", "trn_pipeline"),
    ],
)
def test_non_tirx_pipelines_do_not_lower_iket(module_name, factory_name):
    factory = getattr(importlib.import_module(module_name), factory_name)
    assert "LowerIket" not in inspect.getsource(factory)


def test_lowering_emits_native_dump_metadata_without_control_abi():
    source = _cuda_source(_compile(push_pop_kernel))

    meta = _official_global_bytes(source, "__iket_meta_info")
    assert [int.from_bytes(meta[offset : offset + 4], "little") for offset in range(0, 36, 4)] == [
        48,
        0,
        5,
        31,
        32,
        60,
        0xBABEF19D,
        0,
        3,
    ]
    point = _official_global_bytes(source, "__iket_evt_decl_point_3_attrs")
    assert [int.from_bytes(point[offset : offset + 4], "little") for offset in range(0, 28, 4)] == [
        60,
        3,
        3,
        0,
        0,
        0,
        5,
    ]
    assert point[28:33] == b"point"
    outer = _official_global_bytes(source, "__iket_range_decl_outer_2718680436_attrs")
    assert [int.from_bytes(outer[offset : offset + 4], "little") for offset in range(0, 24, 4)] == [
        72,
        0,
        2718680436,
        0xFFFFFFFF,
        2,
        0,
    ]
    assert outer[36:41] == b"outer"

    assert "mov.u32 %%t, %%globaltimer_lo" in source
    assert "st.weak.shared.u32 [%%r], %%t" in source
    assert "pmevent.mask %0" in source
    assert source.count("tvm_builtin_iket_official_event((uint)31)") == 2
    for removed_symbol in (
        "__tvm_iket_get_metadata",
        "__tvm_iket_set_context",
        "__tvm_iket_clear_context",
        "tvm_builtin_iket_prologue",
        "tvm_builtin_iket_finalize",
    ):
        assert removed_symbol not in source


def test_token_sentinel_and_dynamic_alternation_lowering():
    source = _cuda_source(_compile(token_loop))

    assert "token_ptr[0] = (uint)0" in source
    assert source.count("tvm_builtin_iket_official_event(token_ptr[0])") == 2
    assert "case 1:" in source
    assert "case 2:" in source
    assert "case 31:" in source
    assert "__iket_evt_decl_sentinel" not in source
    assert "__iket_range_decl_sentinel" not in source
    for name, event_id in (("even", 1), ("odd", 2)):
        event = _official_global_bytes(source, f"__iket_evt_decl_{name}_{event_id}_attrs")
        assert int.from_bytes(event[4:8], "little") == event_id
        assert int.from_bytes(event[16:20], "little") == 4


def test_payload_metadata_and_native_record_layout():
    source = _cuda_source(_compile(payload_types))
    expected_payload_types = {
        "bool": 2,
        "i8": 1,
        "u8": 2,
        "i16": 3,
        "u16": 4,
        "i32": 5,
        "u32": 6,
        "i64": 7,
        "u64": 16,
        "f32": 13,
        "f64": 14,
        "token_payload": 5,
        "stack_payload": 13,
    }
    for name, payload_type in expected_payload_types.items():
        _event_id, event = _event_bytes(source, name)
        assert int.from_bytes(event[8:12], "little") == 3
        assert int.from_bytes(event[12:16], "little") == payload_type

    assert int.from_bytes(_event_bytes(source, "token_payload")[1][16:20], "little") == 4
    assert int.from_bytes(_event_bytes(source, "stack_payload")[1][16:20], "little") == 1
    assert "activemask.b32 %%mask" in source
    assert "elect.sync _|%%p, %%mask" in source
    assert "@%%p st.weak.shared.b32 [%%r+4], %%payload32" in source
    assert "@%%p st.weak.shared.b64 [%%r+8], %%payload64" in source
    helper_start = source.index("template <unsigned int EventId>")
    helper_end = source.index('extern "C" __global__', helper_start)
    assert "__shfl" not in source[helper_start:helper_end]


def test_no_payload_native_helper_is_unchanged():
    source = _cuda_source(_compile(push_pop_kernel))
    helper = source[source.index("template <unsigned int EventId>") :]
    assert "activemask" not in helper
    assert "elect.sync" not in helper
    assert "st.weak.shared.u32 [%%r], %%t" in helper
    assert "st.weak.shared.u64" not in helper


def test_sentinel_only_has_no_declaration_and_guards_payload_evaluation():
    source = _cuda_source(_compile(sentinel_only_payload))
    assert "__iket_evt_decl" not in source
    assert "__iket_range_decl" not in source
    kernel = source[source.index("sentinel_only_payload_kernel") :]
    guard = kernel.index("if (token_ptr[0] != (uint)0)")
    payload_load = kernel.index("out_ptr[((int)threadIdx.x)]", guard)
    event = kernel.index("tvm_builtin_iket_official_event", guard)
    assert guard < event < payload_load


@pytest.mark.parametrize(
    ("kernel_func", "message"),
    [
        pytest.param(payload_float16, "supports only", id="float16"),
        pytest.param(payload_bfloat16, "supports only", id="bfloat16"),
        pytest.param(payload_pointer, "scalar numeric", id="pointer"),
        pytest.param(payload_vector, "scalar value", id="vector"),
    ],
)
def test_rejects_invalid_payload_types(kernel_func, message):
    with pytest.raises(TypeError, match=message):
        _compile(kernel_func)


@pytest.mark.parametrize(
    ("kernel_func", "message"),
    [
        pytest.param(payload_presence_mismatch, "both range_start and range_end", id="presence"),
        pytest.param(payload_type_mismatch, "changes payload type", id="dtype"),
    ],
)
def test_rejects_token_payload_schema_mismatch(kernel_func, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _compile(kernel_func)


def test_rejects_cross_kernel_payload_schema_conflicts():
    with pytest.raises(TypeError, match="changes payload type.*across kernels"):
        IketProfiler().compile(
            tvm.IRModule({"i32": schema_i32, "u32": schema_u32}),
            target=TARGET,
            tir_pipeline="tirx",
        )
    with pytest.raises(ValueError, match="changes payload presence.*across kernels"):
        IketProfiler().compile(
            tvm.IRModule({"i32": schema_i32, "none": schema_no_payload}),
            target=TARGET,
            tir_pipeline="tirx",
        )


def test_native_extended_module_and_architecture_boundaries(capfd):
    source = _cuda_source(_compile(marks_30))
    assert len(re.findall(r"__iket_evt_decl_e\d\d_\d+_attrs", source)) == 30
    assert int.from_bytes(_official_global_bytes(source, "__iket_meta_info")[12:16], "little") == 31

    source = _cuda_source(_compile(marks_31))
    warning = capfd.readouterr().err
    assert "ExtendedNativeDump" in warning
    assert len(re.findall(r"__iket_evt_decl_e\d\d_\d+_attrs", source)) == 31
    assert "__iket_evt_decl_e00_64_attrs" in source
    assert "__iket_evt_decl_e30_94_attrs" in source
    meta = _official_global_bytes(source, "__iket_meta_info")
    assert int.from_bytes(meta[12:16], "little") == 4095

    source = _cuda_source(
        IketProfiler().compile(
            tvm.IRModule({"marks_30": marks_30, "serial_a": serial_a}),
            target=TARGET,
            tir_pipeline="tirx",
        )
    )
    assert (
        int.from_bytes(_official_global_bytes(source, "__iket_meta_info")[12:16], "little") == 4095
    )
    with pytest.raises(ValueError, match="requires SM90 or newer"):
        _compile(serial_a, target=tvm.target.Target({"kind": "cuda", "arch": "sm_80"}))


def test_extended_declaration_limit(capfd):
    source = _cuda_source(_compile(_many_marks(4032)))
    capfd.readouterr()
    assert "__iket_evt_decl_e0000_64_attrs" in source
    assert "__iket_evt_decl_e4031_4095_attrs" in source
    with pytest.raises(ValueError, match="at most 4032.*got 4033"):
        _compile(_many_marks(4033))


@pytest.mark.parametrize("arch", ("sm_90a", "sm_103a", "sm_110a", "sm_120a"))
def test_extended_payload_compile_only_architectures(arch, capfd):
    executable = IketProfiler().compile(
        tvm.IRModule({"marks": marks_31, "payload": payload_types}),
        target=tvm.target.Target({"kind": "cuda", "arch": arch}),
        tir_pipeline="tirx",
    )
    capfd.readouterr()
    source = _cuda_source(executable)
    meta = _official_global_bytes(source, "__iket_meta_info")
    assert int.from_bytes(meta[12:16], "little") == 4095
    assert "elect.sync _|%%p, %%mask" in source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.skipif(
    not _has_packaged_nvdisasm(), reason="need the pinned nvidia-cuda-nvdisasm distribution"
)
def test_native_and_extended_payload_placeholder_sass(monkeypatch, tmp_path):
    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    native_no_payload_source = _cuda_source(_compile(push_pop_kernel))
    native_no_payload_sass = _nvrtc_disassemble(
        native_no_payload_source, tmp_path / "native-no-payload"
    )
    assert "ELECT" not in native_no_payload_sass

    native_payload_source = _cuda_source(_compile(payload_types))
    native_sass = _nvrtc_disassemble(native_payload_source, tmp_path / "native")
    assert "ELECT" in native_sass
    assert "STS" in native_sass
    assert native_sass.count("SHFL") == native_no_payload_sass.count("SHFL")

    extended_source = _cuda_source(_compile(marks_31))
    extended_sass = _nvrtc_disassemble(extended_source, tmp_path / "extended")
    assert "ELECT" not in extended_sass
    assert "PMTRIG" in extended_sass
    assert "STS.64" in extended_sass or "STS" in extended_sass

    extended_payload_source = _cuda_source(
        IketProfiler().compile(
            tvm.IRModule({"marks": marks_31, "payload": payload_types}),
            target=TARGET,
            tir_pipeline="tirx",
        )
    )
    _event_id, event = _event_bytes(extended_payload_source, "i64")
    assert int.from_bytes(event[8:12], "little") == 5
    assert "@%%p st.weak.shared.b32 [%%r+8], %%payload32" in extended_payload_source
    assert "@%%p st.weak.shared.b64 [%%r+8], %%payload64" in extended_payload_source
    extended_payload_sass = _nvrtc_disassemble(
        extended_payload_source, tmp_path / "extended-payload"
    )
    assert "ELECT" in extended_payload_sass


@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_multi_kernel_module_has_no_tvm_control_plane():
    executable = IketProfiler().compile(
        tvm.IRModule(
            {
                "plain_entry": plain_entry,
                "serial_a": serial_a,
                "serial_b": serial_b,
            }
        ),
        target=TARGET,
        tir_pipeline="tirx",
    )
    source = _cuda_source(executable)
    assert "__iket_evt_decl_a_1_attrs" in source
    assert "__iket_evt_decl_b_2_attrs" in source
    assert "plain_entry_kernel" in source
    for module in executable.mod._collect_from_import_tree(lambda _module: True):
        for control_name in (
            "__tvm_iket_get_metadata",
            "__tvm_iket_set_context",
            "__tvm_iket_clear_context",
        ):
            assert not module.implements_function(control_name, query_imports=False)


def test_proxy_fails_closed_and_forbids_export(tmp_path, monkeypatch):
    from tvm.tirx.cuda import iket as _iket_official

    executable = _compile(serial_a)
    with pytest.raises(RuntimeError, match="cannot be exported"):
        executable.export_library(tmp_path / "official.so")

    monkeypatch.setattr(_iket_official, "_validate_official_installation", lambda: None)
    monkeypatch.delenv("CUDA_INJECTION64_PATH", raising=False)
    with pytest.raises(RuntimeError, match="CUDA_INJECTION64_PATH was not supplied"):
        executable.jit()
    assert executable._executable._jitted_mod is None  # pylint: disable=protected-access


def test_environment_validation_is_not_process_cached(tmp_path, monkeypatch):
    from tvm.tirx.cuda import iket as _iket_official

    monkeypatch.setattr(_iket_official, "_validate_run_iket_entrypoint", lambda: None)
    monkeypatch.setattr(_iket_official, "_validate_nvrtc_available", lambda: None)
    injection = tmp_path / "libsmodel_injection.so"
    injection.write_bytes(b"injection")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"toolName": "tracker"}))
    monkeypatch.setenv("CUDA_INJECTION64_PATH", str(injection))
    monkeypatch.setenv("SMODEL_INJECTION_CONFIG", str(config))
    monkeypatch.delenv("TVM_IKET_OFFICIAL_PROFILE", raising=False)

    _iket_official.validate_official_environment()
    monkeypatch.delenv("CUDA_INJECTION64_PATH")
    with pytest.raises(RuntimeError, match="CUDA_INJECTION64_PATH was not supplied"):
        _iket_official.validate_official_environment()


@pytest.mark.parametrize("version", ("4.5.0", "4.6.0", "4.6.2", "5.0.0.dev1"))
def test_installation_ignores_package_versions(tmp_path, monkeypatch, version):
    from tvm.tirx.cuda import iket as _iket_official

    executable = tmp_path / "run-iket"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    entry_point = metadata.EntryPoint(
        name="run-iket", value="iket.cli.main:entrypoint", group="console_scripts"
    )

    def distribution(name):
        if name == "nvidia-cutlass-dsl-libs-base":
            return SimpleNamespace(version=version, entry_points=(entry_point,))
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_iket_official.metadata, "distribution", distribution)
    monkeypatch.setattr(_iket_official.shutil, "which", lambda _name: str(executable))
    monkeypatch.setattr(_iket_official, "_validate_nvrtc_available", lambda: None)
    assert _iket_official._validate_official_installation() == str(executable)  # pylint: disable=protected-access


@pytest.mark.parametrize("version", ((12, 9), (13, 2), (13, 4), (14, 0)))
def test_nvrtc_version_unrestricted(monkeypatch, version):
    from tvm.tirx.cuda import iket as _iket_official

    nvrtc = SimpleNamespace(nvrtcVersion=lambda: (0, *version))
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(nvrtc=nvrtc))
    _iket_official._validate_nvrtc_available()  # pylint: disable=protected-access


def test_nvrtc_error_status(monkeypatch):
    from tvm.tirx.cuda import iket as _iket_official

    nvrtc = SimpleNamespace(nvrtcVersion=lambda: (1, 0, 0))
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(nvrtc=nvrtc))
    with pytest.raises(_iket_official.IketProfileError, match="CUDA NVRTC is unavailable"):
        _iket_official._validate_nvrtc_available()  # pylint: disable=protected-access


@pytest.mark.parametrize("error_type", (ImportError, OSError, RuntimeError))
def test_nvrtc_load_error(monkeypatch, error_type):
    from tvm.tirx.cuda import iket as _iket_official

    def unavailable_nvrtc():
        raise error_type("NVRTC could not be loaded")

    nvrtc = SimpleNamespace(nvrtcVersion=unavailable_nvrtc)
    monkeypatch.setitem(sys.modules, "cuda.bindings", SimpleNamespace(nvrtc=nvrtc))
    with pytest.raises(_iket_official.IketProfileError, match="CUDA NVRTC is unavailable"):
        _iket_official._validate_nvrtc_available()  # pylint: disable=protected-access


@pytest.mark.parametrize("config_env", ["IKET_INJECTION_CONFIG", "SMODEL_INJECTION_CONFIG"])
def test_injection_environment_accepts_run_iket_two_passes(tmp_path, monkeypatch, config_env):
    from tvm.tirx.cuda import iket as _iket_official

    monkeypatch.delenv("IKET_INJECTION_CONFIG", raising=False)
    monkeypatch.delenv("SMODEL_INJECTION_CONFIG", raising=False)
    injection = tmp_path / "libiket_injection.so"
    injection.write_bytes(b"injection")
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("CUDA_INJECTION64_PATH", str(injection))
    monkeypatch.setenv(config_env, str(config_path))

    config_path.write_text(json.dumps({"toolName": "tracker", "toolConfig": {}}))
    _iket_official._validate_injection_environment()  # pylint: disable=protected-access

    instrument = tmp_path / "instrument.config.json"
    instrument.write_text("{}", encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "toolName": "iket",
                "toolConfig": {"appInstrument": str(instrument)},
            }
        ),
        encoding="utf-8",
    )
    _iket_official._validate_injection_environment()  # pylint: disable=protected-access
    config_path.write_text(json.dumps({"toolName": "other"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not generated by run-iket profile"):
        _iket_official._validate_injection_environment()  # pylint: disable=protected-access


def test_injection_environment_prefers_current_config(tmp_path, monkeypatch):
    from tvm.tirx.cuda import iket as _iket_official

    injection = tmp_path / "libiket_injection.so"
    injection.touch()
    current = tmp_path / "current.json"
    legacy = tmp_path / "legacy.json"
    current.write_text(json.dumps({"toolName": "tracker"}), encoding="utf-8")
    legacy.write_text("invalid", encoding="utf-8")
    monkeypatch.setenv("CUDA_INJECTION64_PATH", str(injection))
    monkeypatch.setenv("IKET_INJECTION_CONFIG", str(current))
    monkeypatch.setenv("SMODEL_INJECTION_CONFIG", str(legacy))
    _iket_official._validate_injection_environment()
    current.write_text("invalid", encoding="utf-8")
    legacy.write_text(json.dumps({"toolName": "tracker"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="IKET_INJECTION_CONFIG is not valid JSON"):
        _iket_official._validate_injection_environment()
    monkeypatch.setenv("IKET_INJECTION_CONFIG", "")
    _iket_official._validate_injection_environment()


def test_cutlass_4_6_0_oracle_manifest_integrity():
    oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    assert oracle["schema_version"] == 3
    assert oracle["profile"]["cutlass_dsl"] == "4.6.0"
    assert oracle["profile"]["nvdisasm_distribution"] == "13.3.73"
    assert "V13.3.73" in oracle["profile"]["nvdisasm"]
    assert "--dump-dir=<output>/<case>" in oracle["profile"]["compiler_flags"]
    assert set(oracle["cases"]) == {
        "native_no_payload",
        "native_payload",
        "extended_no_payload",
        "extended_payload",
    }
    for name, case in oracle["cases"].items():
        metadata_bytes = json.dumps(
            case["metadata"], sort_keys=True, separators=(",", ":")
        ).encode()
        assert hashlib.sha256(metadata_bytes).hexdigest() == case["metadata_sha256"]
        expected_method = 3 if name.startswith("native_") else 5
        assert case["instrument_methods"] == [expected_method]
        assert case["metadata"]["info"]["max_event_id"] == (31 if expected_method == 3 else 4095)
        user_events = [
            event for event in case["metadata"]["events"] if event["event_id"] not in (0, 31)
        ]
        assert len(user_events) == (30 if expected_method == 3 else 31)

    native_payloads = {
        event["event_name"]: event["payload_type"]
        for event in oracle["cases"]["native_payload"]["metadata"]["events"]
    }
    assert (
        native_payloads
        | {
            "bool": 2,
            "i8": 1,
            "u8": 2,
            "i16": 3,
            "u16": 4,
            "i32": 5,
            "u32": 6,
            "i64": 7,
            "u64": 16,
            "f32": 13,
            "f64": 14,
        }
        == native_payloads
    )

    abi = oracle["abi"]
    assert abi["sentinel_event_id"] == 0
    assert abi["range_pop_event_id"] == 31
    assert abi["native_user_event_ids"] == [1, 30]
    assert abi["extended_user_event_ids"] == [64, 4095]
    assert abi["max_user_declarations"] == 4032
    assert (
        abi["meta_info_bytes"],
        abi["event_attributes_bytes"],
        abi["range_attributes_bytes"],
    ) == (48, 60, 72)
    assert abi["record_layouts"]["native_payload_32"]["payload"] == [4, 4]
    assert abi["record_layouts"]["native_payload_64"]["payload"] == [8, 8]
    assert abi["record_layouts"]["extended_payload_32"]["payload"] == [8, 4]
    assert abi["record_layouts"]["extended_payload_64"]["payload"] == [8, 8]
    all_hashes = [
        *oracle["wheels"].values(),
        *(
            digest
            for case in oracle["cases"].values()
            for digest in (*case["artifact_sha256"].values(), case["metadata_sha256"])
        ),
    ]
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in all_hashes)


def test_external_trace_contract():
    trace_path = os.environ.get("TVM_IKET_OFFICIAL_TRACE_JSON")
    if trace_path is None:
        pytest.skip("set TVM_IKET_OFFICIAL_TRACE_JSON after a run-iket workload")
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    assert len(trace["launches"]) == 3
    launches = {launch["kernelName"]: launch for launch in trace["launches"]}
    strings = trace["stringTable"]

    launch = launches["canonical_iket_workload_kernel"]
    assert [strings[marker["markerNameIdx"]] for marker in launch["markers"]] == [
        "checkpoint",
        "inside_stack",
    ]
    ranges = {strings[item["rangeNameIdx"]]: item for item in launch["ranges"]}
    assert set(ranges) == {"token", "stack"}
    assert ranges["token"]["rangeType"] == 1
    assert [item["eventId"] for item in ranges["token"]["internalEvents"]] == [1, 1]
    assert ranges["stack"]["rangeType"] == 2
    assert [item["eventId"] for item in ranges["stack"]["internalEvents"]] == [2, 31]
    for item in ranges.values():
        assert item["startTs"] <= item["endTs"]
        assert [event["timestamp"] for event in item["internalEvents"]] == [
            item["startTs"],
            item["endTs"],
        ]

    native = launches["native_payload_workload_kernel"]
    markers = {strings[item["markerNameIdx"]]: item for item in native["markers"]}
    assert (markers["lane_payload"]["payloadType"], markers["lane_payload"]["payloadVal"]) == (
        5,
        100,
    )
    assert (
        markers["first_active_lane"]["payloadType"],
        markers["first_active_lane"]["payloadVal"],
    ) == (5, 5)
    assert (markers["wide_payload"]["payloadType"], markers["wide_payload"]["payloadVal"]) == (
        7,
        0x100000000,
    )
    assert (
        markers["negative_payload"]["payloadType"],
        markers["negative_payload"]["payloadVal"],
    ) == (5, 0xFFFFFFE0)
    assert (
        markers["bool_true_payload"]["payloadType"],
        markers["bool_true_payload"]["payloadVal"],
    ) == (2, 1)
    assert (
        markers["bool_false_payload"]["payloadType"],
        markers["bool_false_payload"]["payloadVal"],
    ) == (2, 0)
    assert (
        markers["float32_payload"]["payloadType"],
        markers["float32_payload"]["payloadVal"],
    ) == (13, 0xC0500000)
    assert (
        markers["float64_payload"]["payloadType"],
        markers["float64_payload"]["payloadVal"],
    ) == (14, 0x401A000000000000)
    ranges = {strings[item["rangeNameIdx"]]: item for item in native["ranges"]}
    assert [
        (event["payloadType"], event["payloadVal"])
        for event in ranges["token_payload"]["internalEvents"]
    ] == [(5, 200), (5, 300)]
    assert (
        ranges["stack_payload"]["internalEvents"][0]["payloadType"],
        ranges["stack_payload"]["internalEvents"][0]["payloadVal"],
    ) == (5, 400)

    extended = launches["extended_payload_workload_kernel"]
    markers = {strings[item["markerNameIdx"]]: item for item in extended["markers"]}
    assert len(markers) == 31
    assert (
        markers["extended_lane_payload"]["payloadType"],
        markers["extended_lane_payload"]["payloadVal"],
    ) == (5, 500)


def test_external_patch_contract(tmp_path):
    run_dir = os.environ.get("TVM_IKET_OFFICIAL_PATCH_RUN_DIR")
    if run_dir is None:
        pytest.skip("set TVM_IKET_OFFICIAL_PATCH_RUN_DIR after a trace-level run-iket workload")
    nvdisasm = os.environ.get("TVM_IKET_OFFICIAL_NVDISASM") or shutil.which("nvdisasm")
    if nvdisasm is None:
        pytest.skip("nvdisasm is required for official patch verification")

    verifier = Path(__file__).with_name("verify_iket_official_patch.py")
    subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--run-dir",
            run_dir,
            "--kernel",
            "canonical_iket_workload_kernel",
            "--nvdisasm",
            nvdisasm,
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    report = json.loads((tmp_path / "verification.json").read_text(encoding="utf-8"))
    oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["site_count"] > 0
    assert report["normalized_signature"] == oracle["abi"]["native_patched_hot_path"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in report["sha256"].values())
