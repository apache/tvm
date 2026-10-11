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

"""Compilation through the native TIRx pipeline."""

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.script import ir as I
from tvm.script import s_tir as Ts
from tvm.script import tirx as T
from tvm.tirx import compilation_pipeline


@pytest.mark.skipif(not tvm.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize("pipeline", [None, "default", "tirx"])
def test_default_pipeline_allocations(pipeline):
    @T.function
    def add_one(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32")):
        temp = T.alloc_tensor(ty_args=[T.Tensor((16,), "float32")])
        for i in range(16):
            temp[i] = A[i] + T.float32(1)
        for i in range(16):
            B[i] = temp[i]

    module = tvm.tirx.build(add_one, target="llvm", pipeline=pipeline)
    source = np.arange(16, dtype="float32")
    a = tvm.runtime.tensor(source)
    b = tvm.runtime.tensor(np.zeros(16, dtype="float32"))
    module(a, b)
    np.testing.assert_equal(b.numpy(), source + 1)


@pytest.mark.skipif(not tvm.runtime.enabled("llvm"), reason="LLVM is not enabled")
@pytest.mark.parametrize("pipeline", ["default", "tirx"])
def test_native_pipeline_preserves_stepped_iteration(pipeline):
    @T.function
    def kernel(A: T.Tensor((8,), "int32"), n: T.int32):
        for i in T.serial(0, n, step=2):
            if T.likely(i < 3):
                A[i] = 1
            else:
                A[i] = 2

    lowering, _, _ = tvm.tirx.get_tir_pipeline(
        pipeline, target=tvm.target.Target("llvm", host="llvm")
    )
    lowered = lowering(tvm.IRModule({"main": kernel}))
    loops = []
    tvm_ffi.structural_walk(lowered, (tvm.ir.For, loops.append))
    assert len(loops) == 1
    assert loops[0].min.value == 0
    assert loops[0].step.value == 2
    assert loops[0].extent.name == "n"

    # LLVM does not consume the likely hint; execute its ordinary condition too.
    @T.function
    def plain(A: T.Tensor((8,), "int32"), n: T.int32):
        for i in T.serial(0, n, step=2):
            if i < 3:
                A[i] = 1
            else:
                A[i] = 2

    module = tvm.tirx.build(plain, target="llvm", pipeline=pipeline)
    output = tvm.runtime.tensor(np.full(8, -1, dtype="int32"))
    module(output, 8)
    np.testing.assert_equal(output.numpy(), [1, -1, 1, -1, 2, -1, 2, -1])


def test_lower_thread_binding():
    @T.function
    def before(A: T.Tensor((32,), "int32")):
        for bx in T.parallel(1, annotations={"thread_binding": "blockIdx.x"}):
            for tx in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                A[tx] = tx
            for tx in T.parallel(32, annotations={"thread_binding": "threadIdx.x"}):
                A[tx] = A[tx] + 1

    @T.function
    def expected(A: T.Tensor((32,), "int32")):
        with T.launch_thread("blockIdx.x", 1) as bx:
            with T.launch_thread("threadIdx.x", 32) as tx:
                A[tx] = tx
            with T.launch_thread("threadIdx.x", 32) as tx:
                A[tx] = A[tx] + 1

    actual = tvm.s_tir.transform.LowerThreadBinding()(tvm.IRModule({"main": before}))
    tvm.ir.assert_structural_equal(actual["main"], expected.with_attr("global_symbol", "before"))


def _thread_launches(mod):
    launches = []

    def check_loop(loop):
        assert "thread_binding" not in loop.annotations

    def collect_region(region):
        if region.op.name == "tirx.launch_thread":
            launches.append(region)

    tvm_ffi.structural_walk(mod, [(tvm.ir.For, check_loop), (tvm.ir.RegionStmt, collect_region)])
    return launches


@pytest.mark.parametrize("pipeline_name", ["default", "tirx"])
def test_native_pipeline_lowers_annotated_parallel(pipeline_name):
    @T.function
    def kernel(A: T.Tensor((8,), "uint32")):
        for tx in T.parallel(
            T.uint32(8), step=T.uint32(1), annotations={"thread_binding": "threadIdx.x"}
        ):
            A[tx] = tx

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"}, host="llvm")
    pipeline, _, _ = tvm.tirx.get_tir_pipeline(pipeline_name, target=target)
    lowered = pipeline(tvm.IRModule({"main": kernel}))
    launches = _thread_launches(lowered)
    assert len(launches) == 1
    assert launches[0].args[0].value == "threadIdx.x"
    assert launches[0].args[1].value == 8
    assert launches[0].body_params[0].ty.dtype == "uint32"


def test_native_pipeline_preserves_symbolic_thread_extent():
    @T.function
    def kernel(A: T.Tensor((8,), "int32"), n: T.int32):
        for tx in T.parallel(n, annotations={"thread_binding": "threadIdx.x"}):
            if tx < 8:
                A[tx] = tx

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"}, host="llvm")
    pipeline, _, _ = tvm.tirx.get_tir_pipeline("tirx", target=target, prepare_only=True)
    lowered = pipeline(tvm.IRModule({"main": kernel}))
    launches = _thread_launches(lowered)
    assert len(launches) == 1
    tvm.ir.assert_structural_equal(launches[0].args[1], lowered["main"].params[1])


@pytest.mark.parametrize("pipeline_name", ["default", "s_tir", "adreno"])
@pytest.mark.parametrize("scheduled_helper", [False, True])
def test_mixed_pipeline_classifies_and_lowers_cloned_helpers(pipeline_name, scheduled_helper):
    @I.ir_module
    class Module:
        @T.function(private=True)
        def helper(value: T.int32) -> T.int32:
            return value + 1

        @Ts.function
        def scheduled(A: Ts.Tensor((8,), "int32")):
            Ts.evaluate(Module.helper(0))
            for tx in Ts.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("compute"):
                    vi = Ts.axis.spatial(8, tx)
                    A[vi] = Module.helper(vi)

        @T.function
        def native(A: T.Tensor((8,), "int32")):
            with T.launch_thread("threadIdx.x", 8) as tx:
                A[tx] = Module.helper(tx)

    mod = Module
    if scheduled_helper:
        mod = tvm.IRModule(dict(Module.functions))
        mod.update_func(mod.get_global_var("helper"), mod["helper"].with_attr("s_tir", True))
    target_options = (
        {"kind": "opencl", "keys": ["adreno", "opencl", "gpu"]}
        if pipeline_name == "adreno"
        else {"kind": "cuda", "arch": "sm_90"}
    )
    target = tvm.target.Target(target_options, host="llvm")
    if pipeline_name == "default":
        pipeline, _, _ = tvm.tirx.get_default_tir_pipeline(target, mod, target_to_bind=target)
    else:
        pipeline, _, _ = tvm.tirx.get_tir_pipeline(pipeline_name, target=target)
    lowered = pipeline(mod)
    assert lowered["helper"].attrs["target"].kind.name == target.kind.name
    assert lowered["helper_host"].attrs["target"].kind.name == "llvm"
    assert lowered["helper"].is_tirx == (not scheduled_helper)
    assert lowered["helper_host"].is_tirx == (not scheduled_helper)
    assert len(_thread_launches(lowered)) == 2

    def no_sblock(block):
        raise AssertionError(f"A cloned helper or entry escaped S-TIR lowering: {block}")

    tvm_ffi.structural_walk(lowered, (tvm.s_tir.SBlock, no_sblock))


def _register_legacy_pipeline(monkeypatch, name, inputs):
    @tvm.transform.module_pass(opt_level=0)
    def record(mod, _ctx):
        inputs.append(mod)
        return mod

    def finalize():
        return tvm.transform.Sequential([])

    def factory():
        return record, finalize, finalize

    monkeypatch.setitem(compilation_pipeline.PIPELINE_MAP, name, factory)
    return finalize


@pytest.mark.parametrize("default_selector", [False, True])
def test_legacy_pipeline_preserves_scheduled_input(monkeypatch, default_selector):
    @Ts.function
    def scheduled(A: Ts.Tensor((8,), "int32")):
        for tx in Ts.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
            with Ts.sblock("compute"):
                vi = Ts.axis.spatial(8, tx)
                A[vi] = vi

    inputs = []
    finalize = _register_legacy_pipeline(monkeypatch, "test_legacy", inputs)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"}, host="llvm")
    mod = tvm.IRModule({"main": scheduled})
    if default_selector:
        monkeypatch.setattr(
            compilation_pipeline,
            "_DEFAULT_PIPELINE_SELECTORS",
            [lambda _mod, _target: tvm.tirx.get_tir_pipeline("test_legacy")],
        )
        pipeline, finalize_host, finalize_device = tvm.tirx.get_default_tir_pipeline(
            target, mod, target_to_bind=target
        )
    else:
        pipeline, finalize_host, finalize_device = tvm.tirx.get_tir_pipeline(
            "test_legacy", target=target
        )
    assert finalize_host is finalize_device is finalize
    lowered = pipeline(mod)
    assert len(inputs) == 1
    assert lowered["main"].attrs["target"].kind.name == "cuda"
    tvm.ir.assert_structural_equal(lowered["main"].body, scheduled.body)
    tvm.s_tir.Schedule(lowered)


@pytest.mark.parametrize("explicit_region", [False, True])
@pytest.mark.parametrize("default_selector", [False, True])
def test_legacy_pipeline_handles_bound_helper_calls(monkeypatch, explicit_region, default_selector):
    @I.ir_module
    class Module:
        @T.function(private=True)
        def helper(value: T.int32) -> T.int32:
            return value + 1

        @Ts.function
        def scheduled(A: Ts.Tensor((8,), "int32")):
            for tx in Ts.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("compute"):
                    vi = Ts.axis.spatial(8, tx)
                    A[vi] = Module.helper(vi)

    mod = Module
    if explicit_region:
        scheduled = mod["scheduled"]
        entry = tvm.ir.RegionStmt("tirx.device_entry", [], [], {}, scheduled.body)
        mod = tvm.IRModule(dict(mod.functions))
        mod.update_func(mod.get_global_var("scheduled"), scheduled.with_body(entry))
    inputs = []
    _register_legacy_pipeline(monkeypatch, "test_legacy", inputs)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"}, host="llvm")
    if default_selector:
        monkeypatch.setattr(
            compilation_pipeline,
            "_DEFAULT_PIPELINE_SELECTORS",
            [lambda _mod, _target: tvm.tirx.get_tir_pipeline("test_legacy")],
        )

    def lower():
        if default_selector:
            pipeline, _, _ = tvm.tirx.get_default_tir_pipeline(target, mod, target_to_bind=target)
        else:
            pipeline, _, _ = tvm.tirx.get_tir_pipeline("test_legacy", target=target)
        return pipeline(mod)

    if explicit_region:
        lowered = lower()
        assert lowered["helper"].attrs["target"].kind.name == "cuda"
        assert len(inputs) == 1
    else:
        with pytest.raises(ValueError, match="must accept an explicit target keyword"):
            lower()
        assert not inputs


def test_target_aware_custom_factory_keeps_mixed_module(monkeypatch):
    @I.ir_module
    class Module:
        @T.function(private=True)
        def helper(value: T.int32) -> T.int32:
            return value + 1

        @Ts.function
        def scheduled(A: Ts.Tensor((8,), "int32")):
            for tx in Ts.parallel(8, annotations={"thread_binding": "threadIdx.x"}):
                with Ts.sblock("compute"):
                    vi = Ts.axis.spatial(8, tx)
                    A[vi] = Module.helper(vi)

    def factory(*, target):
        return tvm.tirx.get_tir_pipeline("s_tir", target=target)

    monkeypatch.setitem(compilation_pipeline.PIPELINE_MAP, "test_target_aware", factory)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"}, host="llvm")
    pipeline, _, _ = tvm.tirx.get_tir_pipeline("test_target_aware", target=target)
    lowered = pipeline(Module)
    assert lowered["helper"].attrs["target"].kind.name == "cuda"
    assert len(_thread_launches(lowered)) == 1


@pytest.mark.parametrize("selection", ["legacy", "fallback", "modern"])
def test_legacy_default_selector_receives_bound_module(monkeypatch, selection):
    @I.ir_module
    class Module:
        @T.function(private=True)
        def helper(value: T.int32) -> T.int32:
            return value + 1

        @T.function
        def kernel(A: T.Tensor((8,), "int32")):
            T.evaluate(Module.helper(0))
            with T.launch_thread("threadIdx.x", 8) as tx:
                A[tx] = Module.helper(tx)

    selected_modules = []
    bindings = []

    @tvm.transform.pass_instrument
    class RecordBindings:
        def run_after_pass(self, mod, info):
            if info.name == "tirx.BindTarget":
                bindings.append(mod)

    def legacy_selector(mod, target):
        selected_modules.append(mod)
        assert mod["kernel"].attrs["target"].kind.name == "cuda"
        assert "backend_config" in mod["kernel"].attrs
        assert mod["helper"].attrs["target"].kind.name == "cuda"
        assert mod["helper_host"].attrs["target"].kind.name == "llvm"
        return tvm.tirx.get_tir_pipeline("tirx") if selection == "legacy" else None

    def modern_selector(mod, target, *, target_to_bind=None):
        assert mod.same_as(selected_modules[0])
        assert target_to_bind is None
        return tvm.tirx.get_tir_pipeline("tirx", target=target_to_bind)

    selectors = [legacy_selector]
    if selection == "modern":
        selectors.append(modern_selector)
    monkeypatch.setattr(compilation_pipeline, "_DEFAULT_PIPELINE_SELECTORS", selectors)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"}, host="llvm")
    with tvm.transform.PassContext(instruments=[RecordBindings()]):
        pipeline, _, _ = tvm.tirx.get_default_tir_pipeline(target, Module, target_to_bind=target)
        assert len(bindings) == 1
        lowered = pipeline(Module)
        assert len(bindings) == 1
        assert lowered["helper_host"].attrs["target"].kind.name == "llvm"
        # A selected pipeline may be reused for another module with the same dialect.
        fresh = tvm.IRModule(dict(Module.functions))
        lowered_again = pipeline(fresh)
        assert len(bindings) == 2
        assert lowered_again["helper_host"].attrs["target"].kind.name == "llvm"
        # IRModule also supports in-place function-map updates between invocations.
        Module.update_func(
            Module.get_global_var("helper"), Module["helper"].with_attr("test_marker", 1)
        )
        updated = pipeline(Module)
        assert len(bindings) == 3
        assert updated["helper"].attrs["test_marker"] == 1
    assert len(selected_modules) == 1
    assert "target" not in Module["kernel"].attrs


def test_legacy_selector_pipeline_reuse_with_inplace_module_pass(monkeypatch):
    @T.function
    def kernel(A: T.Tensor((8,), "int32")):
        for i in range(8):
            A[i] = i

    mod = tvm.IRModule({"main": kernel})

    @tvm.transform.module_pass(opt_level=0)
    def update_inplace(mod, _ctx):
        gv = mod.get_global_var("main")
        func = mod[gv]
        count = func.attrs.get("test_count", 0)
        mod.update_func(gv, func.with_attr("test_count", count + 1))
        return mod

    def selector(mod, target):
        return (
            update_inplace,
            lambda: tvm.transform.Sequential([]),
            lambda: tvm.transform.Sequential([]),
        )

    monkeypatch.setattr(compilation_pipeline, "_DEFAULT_PIPELINE_SELECTORS", [selector])
    target = tvm.target.Target("llvm")
    pipeline, _, _ = tvm.tirx.get_default_tir_pipeline(target, mod, target_to_bind=target)
    assert pipeline(mod)["main"].attrs["test_count"] == 1
    assert pipeline(mod)["main"].attrs["test_count"] == 1
    assert "test_count" not in mod["main"].attrs


if __name__ == "__main__":
    tvm.testing.main()
