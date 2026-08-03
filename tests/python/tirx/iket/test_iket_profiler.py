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

"""Tests for IKET lowering, metadata, installation locks, and trace contracts."""

import hashlib
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import tvm
from tvm.backend.cuda import transforms as cuda_transforms
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.cuda.iket import IketProfiler

TARGET = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
ORACLE_PATH = Path(__file__).parent / "oracle" / "iket_official_cutlass_4_6_0_oracle.json"


@T.prim_func
def serial_a(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    iket.mark("a")
    out[tx] = tx + 1


@T.prim_func
def serial_b(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    iket.mark("b")
    out[tx] = tx + 2


@T.prim_func
def plain_entry(out: T.Buffer((32,), "int32")):
    T.device_entry()
    tx = T.thread_id([32])
    out[tx] = tx + 7


@T.prim_func
def explicit_cuda_shuffle_guard(out: T.Buffer((64,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    bx = T.cta_id([2])
    tx = T.thread_id([64])
    warp = T.cuda.__shfl_sync(T.uint32(0xFFFFFFFF), tx // 32, 0, 32)
    if (bx == 0) & (warp == 0):
        iket.mark("warp-zero")
    out[tx] = tx


@T.prim_func
def push_pop_kernel(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    iket.range_push("outer")
    iket.range_push("inner")
    iket.mark("point")
    iket.range_pop()
    iket.range_pop()
    out[tx] = tx


@T.prim_func
def token_loop(n: T.int32, out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    token = iket.sentinel_token("sentinel")
    for i in T.serial(n, unroll=False):
        iket.range_end(token)
        if i % 2 == 0:
            token = iket.range_start("even")
        else:
            token = iket.range_start("odd")
    iket.range_end(token)
    out[tx] = tx + n


@T.prim_func
def overlapping_ranges(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    outer = iket.range_start("overlap")
    inner = iket.range_start("overlap")
    iket.range_end(inner)
    iket.range_end(outer)
    out[tx] = tx


@T.prim_func
def repeated_range_end(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    token = iket.range_start("twice")
    iket.range_end(token)
    iket.range_end(token)
    out[tx] = tx


@T.prim_func
def unbalanced_stack(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    iket.range_pop()
    out[tx] = tx


@T.prim_func
def payload_kernel(out: T.Buffer((32,), "int32")):
    T.device_entry()
    tx = T.thread_id([32])
    T.evaluate(tvm.tirx.call_intrin("", "tirx.cuda.iket_mark", "payload", tx))
    out[tx] = tx


@T.prim_func
def loop_carried_divergent_token(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    guard = T.alloc_local((1,), "int32")
    guard[0] = 0
    token = iket.sentinel_token("loop")
    for _i in T.serial(2, unroll=False):
        token = iket.sentinel_token("loop")
        if guard[0] == 0:
            token = iket.range_start("loop")
        iket.range_end(token)
        guard[0] = tx
    out[tx] = guard[0]


@T.prim_func
def while_carried_divergent_token(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    guard = T.alloc_local((1,), "int32")
    iteration = T.alloc_local((1,), "int32")
    guard[0] = 0
    iteration[0] = 0
    token = iket.sentinel_token("while-loop")
    while iteration[0] < 2:
        token = iket.sentinel_token("while-loop")
        if guard[0] == 0:
            token = iket.range_start("while-loop")
        iket.range_end(token)
        guard[0] = tx
        iteration[0] = iteration[0] + 1
    out[tx] = guard[0]


@T.prim_func
def annotated_outer_loop_with_nested_break(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    value = T.alloc_local((1,), "int32")
    value[0] = 0
    for _outer in T.serial(2, unroll=False):
        iket.range_push("outer")
        for inner in T.serial(2, unroll=False):
            if inner == 1:
                break
            value[0] = value[0] + 1
        iket.range_pop()
    out[tx] = value[0]


@T.prim_func
def marks_30(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
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


@T.prim_func
def marks_31(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
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


def test_public_interface_is_official_only():
    signature = inspect.signature(IketProfiler.compile)
    assert "backend" not in signature.parameters
    profiler = IketProfiler()
    assert not hasattr(profiler, "capture")
    assert not hasattr(profiler, "export")
    assert not hasattr(tvm.tirx.transform, "LowerIket")
    assert callable(cuda_transforms.LowerIket)

    script = serial_a.script()
    assert 'T.cuda.iket.mark("a")' in script
    assert "T.tirx.iket" not in script
    assert tvm.script.from_source(script).script() == script


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
        @T.prim_func
        def main(out: T.Buffer((32,), "int32")):
            T.device_entry()
            iket = IketProfiler()
            tx = T.thread_id([32])
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


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("tvm.tirx.compilation_pipeline", "default_tir_pipeline"),
        ("tvm.tirx.compilation_pipeline", "tirx_pipeline"),
    ],
)
def test_tirx_pipelines_immediately_lower_iket(module_name, factory_name):
    factory = getattr(importlib.import_module(module_name), factory_name)
    source = inspect.getsource(factory)
    assert re.search(
        r"tirx\.transform\.SplitHostDevice\(\),\s+cuda_transforms\.LowerIket\(\)", source
    )


@pytest.mark.parametrize(
    ("module_name", "factory_name"),
    [
        ("tvm.s_tir.pipeline", "default_s_tir_pipeline"),
        ("tvm.s_tir.backend.adreno.pipeline", "default_tir_pipeline"),
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
    assert "case 3:" in source
    assert "case 31:" in source
    for name, event_id in (("even", 1), ("odd", 2), ("sentinel", 3)):
        event = _official_global_bytes(source, f"__iket_evt_decl_{name}_{event_id}_attrs")
        assert int.from_bytes(event[4:8], "little") == event_id
        assert int.from_bytes(event[16:20], "little") == 4


def test_explicit_cuda_shuffle_broadcast_is_warp_uniform():
    source = _cuda_source(_compile(explicit_cuda_shuffle_guard))
    assert "__iket_evt_decl_warp_zero_1_attrs" in source


def test_nested_loop_control_does_not_reject_annotated_outer_loop():
    source = _cuda_source(_compile(annotated_outer_loop_with_nested_break))
    assert "__iket_evt_decl_outer_1_attrs" in source


@pytest.mark.parametrize(
    "kernel_func", (loop_carried_divergent_token, while_carried_divergent_token)
)
def test_unproven_warp_convergence_warns_and_compiles(kernel_func, capfd):
    source = _cuda_source(_compile(kernel_func))
    warning = capfd.readouterr().err

    assert "IKET warp convergence could not be proven for event site" in warning
    assert "continuing because convergence diagnostics are advisory" in warning
    assert "__iket_evt_decl" in source


@pytest.mark.parametrize(
    ("kernel_func", "message"),
    [
        pytest.param(payload_kernel, "does not support payloads", id="payload"),
        pytest.param(overlapping_ranges, "strictly alternating", id="overlap"),
        pytest.param(repeated_range_end, "strictly alternating", id="repeated-end"),
        pytest.param(unbalanced_stack, "balanced range_push/range_pop", id="unbalanced-stack"),
    ],
)
def test_rejects_unsupported_semantics(kernel_func, message):
    with pytest.raises(ValueError, match=message):
        _compile(kernel_func)


def test_declaration_module_and_architecture_boundaries():
    source = _cuda_source(_compile(marks_30))
    assert len(re.findall(r"__iket_evt_decl_e\d\d_\d+_attrs", source)) == 30

    with pytest.raises(ValueError, match="at most 30 declarations per kernel"):
        _compile(marks_31)
    with pytest.raises(ValueError, match="at most 30 distinct declarations"):
        IketProfiler().compile(
            tvm.IRModule({"marks_30": marks_30, "serial_a": serial_a}),
            target=TARGET,
            tir_pipeline="tirx",
        )
    with pytest.raises(ValueError, match="requires SM90 or newer"):
        _compile(serial_a, target=tvm.target.Target({"kind": "cuda", "arch": "sm_80"}))


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
    executable = _compile(serial_a)
    with pytest.raises(RuntimeError, match="cannot be exported"):
        executable.export_library(tmp_path / "official.so")

    monkeypatch.delenv("TVM_IKET_OFFICIAL_PROFILE", raising=False)
    with pytest.raises(RuntimeError, match="TVM_IKET_OFFICIAL_PROFILE must be set"):
        executable.jit()
    assert executable._executable._jitted_mod is None  # pylint: disable=protected-access


def test_environment_validation_is_not_process_cached(tmp_path, monkeypatch):
    from tvm.tirx.cuda import iket as _iket_official

    injection_path = tmp_path / "libsmodel_injection.so"
    injection_path.write_bytes(b"locked")
    injection_relative = "nvidia_cutlass_dsl/dsl_packages/iket/profiler/libsmodel_injection.so"
    profile = {
        "nvrtc_version": (13, 2),
        "versions": {"nvidia-cutlass-dsl-libs-base": "4.6.0"},
        "files": {
            "nvidia-cutlass-dsl-libs-base": {
                injection_relative: hashlib.sha256(b"locked").hexdigest()
            }
        },
    }

    class FakeDistribution:
        version = "4.6.0"

        @staticmethod
        def locate_file(_relative_path):
            return injection_path

    monkeypatch.setitem(_iket_official._OFFICIAL_PROFILES, "cutlass-4.6.0", profile)
    monkeypatch.setattr(_iket_official.metadata, "distribution", lambda _name: FakeDistribution())
    monkeypatch.setattr(_iket_official, "_validate_run_iket_entrypoint", lambda: None)
    monkeypatch.setattr(
        _iket_official, "_validate_injection_environment", lambda _expected_digest: None
    )
    monkeypatch.setattr(_iket_official, "_validate_nvrtc_version", lambda _version: None)
    monkeypatch.setenv("TVM_IKET_OFFICIAL_PROFILE", "cutlass-4.6.0")

    _iket_official.validate_official_environment()
    monkeypatch.delenv("TVM_IKET_OFFICIAL_PROFILE")
    with pytest.raises(RuntimeError, match="TVM_IKET_OFFICIAL_PROFILE must be set"):
        _iket_official.validate_official_environment()


def test_injection_environment_accepts_run_iket_two_passes(tmp_path, monkeypatch):
    from tvm.tirx.cuda import iket as _iket_official

    injection = tmp_path / "libsmodel_injection.so"
    injection.write_bytes(b"locked-injection")
    expected_digest = hashlib.sha256(injection.read_bytes()).hexdigest()
    config_path = tmp_path / "config.json"
    monkeypatch.setenv("CUDA_INJECTION64_PATH", str(injection))
    monkeypatch.setenv("SMODEL_INJECTION_CONFIG", str(config_path))

    config_path.write_text(json.dumps({"toolName": "tracker", "toolConfig": {}}))
    _iket_official._validate_injection_environment(expected_digest)  # pylint: disable=protected-access

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
    _iket_official._validate_injection_environment(expected_digest)  # pylint: disable=protected-access

    with pytest.raises(RuntimeError, match="locked run-iket binary"):
        _iket_official._validate_injection_environment("0" * 64)  # pylint: disable=protected-access
    config_path.write_text(json.dumps({"toolName": "other"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not generated by run-iket profile"):
        _iket_official._validate_injection_environment(  # pylint: disable=protected-access
            expected_digest
        )


def test_cutlass_4_6_0_oracle_manifest_integrity():
    oracle = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    assert oracle["schema_version"] == 2
    assert oracle["profile"]["cutlass_dsl"] == "4.6.0"
    assert oracle["profile"]["instrument_method"] == "NativeDump"
    assert "--dump-dir=<output>" in oracle["profile"]["compiler_flags"]
    metadata_bytes = json.dumps(oracle["metadata"], sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(metadata_bytes).hexdigest() == oracle["metadata_sha256"]
    abi = oracle["native_dump_abi"]
    assert abi["sentinel_event_id"] == 0
    assert abi["range_pop_event_id"] == 31
    assert (
        abi["meta_info_bytes"],
        abi["event_attributes_bytes"],
        abi["range_attributes_bytes"],
    ) == (48, 60, 72)
    assert abi["patched_hot_path"] == [
        "GLOBALTIMERLO",
        "ENCODE_EVENT_ID",
        "STORE_GLOBAL_32",
        "ADD_WRITE_PTR_64_4",
    ]
    all_hashes = [
        *oracle["artifact_sha256"].values(),
        *oracle["patch_artifact_sha256"].values(),
        *oracle["wheels"].values(),
        oracle["metadata_sha256"],
    ]
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in all_hashes)


def test_external_trace_contract():
    trace_path = os.environ.get("TVM_IKET_OFFICIAL_TRACE_JSON")
    if trace_path is None:
        pytest.skip("set TVM_IKET_OFFICIAL_TRACE_JSON after the locked run-iket workload")
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    assert len(trace["launches"]) == 1
    launch = trace["launches"][0]
    assert launch["kernelName"] == "canonical_iket_workload_kernel"
    strings = trace["stringTable"]
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
    assert report["normalized_signature"] == oracle["native_dump_abi"]["patched_hot_path"]
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in report["sha256"].values())
