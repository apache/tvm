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
# ruff: noqa: F841

import numpy as np
import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import relax, tirx, topi
from tvm.contrib.thrust import can_use_thrust
from tvm.ir.base import assert_structural_equal
from tvm.relax.backend import DispatchSortScan
from tvm.s_tir import dlight
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tirx as T


def test_dispatch_scanop():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "llvm")):
            with R.dataflow():
                lv0 = R.cumsum(x, axis=1, dtype="float64", exclusive=False)
                lv1 = R.cumprod(lv0, axis=1, dtype="float64", exclusive=False)
                gv = lv1
                R.output(gv)
            return gv

    mod = DispatchSortScan()(Before)

    vdevices = [I.vdevice("llvm", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            lv0 = bb.emit_te(topi.cumsum, x, axis=1, dtype="float64", exclusive=False)
            out = bb.emit_te(topi.cumprod, lv0, axis=1, dtype="float64", exclusive=False)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_scanop_cuda():
    """R.cumsum and R.cumprod may be lowered with TOPI for GPU

    For the purpose of testing, this test case intentionally uses the
    `exclusive=True` argument to prevent the `R.cumsum` from being
    lowered to the packed func `"gpu_2d_continuous_cumsum"`.
    """

    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0)]})

        @R.function
        def main(x: R.Tensor(("m", 3), "float32", "cuda")):
            with R.dataflow():
                lv0 = R.cumsum(x, axis=1, exclusive=True)
                lv1 = R.cumprod(lv0, axis=1)
                gv = lv1
                R.output(gv)
            return gv

    target = tvm.target.Target("cuda", host="llvm")

    vdevices = [I.vdevice("cuda", 0)]
    m = tirx.Var("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("main", (x,), {"global_symbol": "main"}):
            with bb.dataflow():
                lv = bb.emit_te(
                    topi.gpu.cumsum,
                    x,
                    axis=1,
                    exclusive=True,
                )
                out = bb.emit_te(
                    topi.gpu.cumprod,
                    lv,
                    axis=1,
                )
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    with target:
        mod = DispatchSortScan()(Before)
        expected_mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback())(expected_mod)

    assert_structural_equal(mod, expected_mod, map_free_vars=True)


def test_dispatch_sort():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor(("m", 3), "float32", "llvm")):
            m = T.int64()
            with R.dataflow():
                lv = R.sort(x, axis=1, descending=False)
                gv = lv
                R.output(gv)
            return gv

    vdevices = [I.vdevice("llvm", 0)]
    m = tirx.Var("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(topi.sort, x, axis=1, is_ascend=True)
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, expected_mod)


@pytest.mark.xfail(reason="skipping broken tests")
def test_dispatch_sort_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                lv = R.sort(x, axis=1, descending=False)
                gv = lv
                R.output(gv)
            return gv

        @R.function
        def foo2(y: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                lv = R.sort(y, axis=0, descending=True)
                gv = lv
                R.output(gv)
            return gv

    target = tvm.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")

    vdevices = [I.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("foo", (x,), {"global_symbol": "foo"}):
            with bb.dataflow():
                out = bb.emit_te(
                    topi.gpu.sort,
                    x,
                    axis=1,
                )
                out = bb.emit_output(out)
            bb.emit_func_output(out)
        with bb.function("foo2", (y,), {"global_symbol": "foo2"}):
            with bb.dataflow():
                if can_use_thrust(target, "tvm.contrib.thrust.sort"):
                    workspace = bb.emit(
                        relax.op.builtin.alloc_tensor(
                            relax.ShapeExpr([4194568]), "uint8", runtime_device_index=0
                        )
                    )
                    out = bb.emit_te(
                        topi.gpu.sort_thrust,
                        y,
                        axis=0,
                        is_ascend=False,
                        workspace=workspace,
                    )
                else:
                    out = bb.emit_te(topi.gpu.sort, y, axis=0, is_ascend=False)
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    with target:
        mod = DispatchSortScan()(Before)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_argsort():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor(("m", 3), "float32", "llvm")):
            m = T.int64()
            with R.dataflow():
                lv = R.argsort(x, axis=1, descending=False, dtype="int32")
                gv = lv
                R.output(gv)
            return gv

    vdevices = [I.vdevice("llvm", 0)]
    m = tirx.Var("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(topi.argsort, x, axis=1, is_ascend=True, dtype="int32")
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, expected_mod)


def test_dispatch_argsort_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                lv = R.argsort(x, axis=1, descending=False)
                gv = lv
                R.output(gv)
            return gv

        @R.function
        def foo2(y: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                lv = R.argsort(y, axis=0, descending=True, dtype="int64")
                gv = lv
                R.output(gv)
            return gv

    target = tvm.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")

    vdevices = [I.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("foo", (x,), {"global_symbol": "foo"}):
            with bb.dataflow():
                out = bb.emit_te(topi.gpu.argsort, x, axis=1, is_ascend=True, dtype="int32")
                out = bb.emit_output(out)
            bb.emit_func_output(out)
        with bb.function("foo2", (y,), {"global_symbol": "foo2"}):
            with bb.dataflow():
                if can_use_thrust(target, "tvm.contrib.thrust.sort"):
                    workspace = bb.emit(
                        relax.op.builtin.alloc_tensor(
                            R.shape([8388872]), R.dtype("uint8"), R.prim_value(0), R.str("global")
                        )
                    )
                    out = bb.emit_te(
                        topi.gpu.argsort_thrust,
                        y,
                        axis=0,
                        is_ascend=False,
                        dtype="int64",
                        workspace=workspace,
                    )
                else:
                    out = bb.emit_te(topi.gpu.argsort, y, axis=0, is_ascend=False, dtype="int64")
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    with target:
        mod = DispatchSortScan()(Before)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_topk():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor(("m", 3), "float32", "llvm")):
            m = T.int64()
            with R.dataflow():
                lv = R.topk(x, k=2, axis=1, largest=True)
                gv = lv
                R.output(gv)
            return gv

    vdevices = [I.vdevice("llvm", 0)]
    m = tirx.Var("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()

    with bb.function("foo", (x,), {"global_symbol": "foo"}):
        with bb.dataflow():
            out = bb.emit_te(topi.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
            out = bb.emit_output(out)
        bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, expected_mod)


def test_dispatch_topk_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                lv = R.topk(x, k=2, axis=1, largest=True)
                gv = lv
                R.output(gv)
            return gv

    target = tvm.target.Target({"kind": "cuda", "libs": ["thrust"]}, host="llvm")

    vdevices = [I.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("foo", (x,), {"global_symbol": "foo"}):
            with bb.dataflow():
                out = bb.emit_te(topi.gpu.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    with target:
        mod = DispatchSortScan()(Before)
        expected_mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback())(expected_mod)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_topk_gpu():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("vulkan")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "vulkan")):
            with R.dataflow():
                # Two same calls should have only one PrimFunc
                lv0 = R.topk(x, k=2, axis=1, largest=True)
                lv1 = R.topk(x, k=2, axis=1, largest=True)
                gv = (lv0, lv1)
                R.output(gv)
            return gv

    target = tvm.target.Target("vulkan", host="llvm")

    vdevices = [I.vdevice("vulkan", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("foo", (x,), {"global_symbol": "foo"}):
            with bb.dataflow():
                lv0 = bb.emit_te(topi.gpu.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
                lv1 = bb.emit_te(topi.gpu.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
                out = (lv0, lv1)
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    with target:
        mod = DispatchSortScan()(Before)
        expected_mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback())(expected_mod)

    assert_structural_equal(mod, expected_mod)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "supports_int64": True}, marks=pytest.mark.gpu),
    ],
)
def test_dispatch_cumsum_gpu(target):
    """Test cumsum kernel dispatch and numerical correctness"""
    if not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor(("m", "n"), "int32")):
            with R.dataflow():
                gv = R.cumsum(x, axis=-1, exclusive=False)
                R.output(gv)
            return gv

    size = (8, 2000)
    np_data = np.random.randint(0, 10, size).astype("int32")
    np_cumsum = np.cumsum(np_data, axis=-1)
    with tvm.target.Target(target):
        mod = DispatchSortScan()(Module)
        ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm.device_from_target(target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        tvm_data = tvm.runtime.tensor(np_data, dev)
        cumsum = vm["main"](tvm_data)
        tvm.testing.assert_allclose(cumsum.numpy(), np_cumsum)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
def test_dispatch_cumprod_cuda_large_batch():
    """Test that GPU scan supports more batches than CUDA's grid-y limit."""
    target = "cuda"
    if not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    @I.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")):
            with R.dataflow():
                gv = R.cumprod(x, axis=1)
                R.output(gv)
            return gv

    np_data = np.ones((65536, 3), dtype="float32")
    np_data[:, 0] = np.arange(65536) % 7 + 1
    np_data[:, 1] = 2
    np_data[:, 2] = 3
    np_cumprod = np.cumprod(np_data, axis=1)

    with tvm.target.Target(target):
        mod = DispatchSortScan()(Module)
        ex = tvm.compile(mod, target)

    def run_and_check():
        dev = tvm.device_from_target(target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        tvm_data = tvm.runtime.tensor(np_data, dev)
        cumprod = vm["main"](tvm_data)
        tvm.testing.assert_allclose(cumprod.numpy(), np_cumprod)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize(
    "shape, axis, in_dtype, out_dtype, expected_kernel, expected_kernel_rank",
    [
        ((3, 5), 0, "float32", None, "gpu_3d_axis_1_cumsum", 3),
        ((2, 3, 4, 5), 1, "float32", None, "gpu_3d_axis_1_cumsum", 3),
        ((2, 3, 4, 5), -2, "int32", "float32", "gpu_3d_axis_1_cumsum", 3),
        # A short scan keeps total_rounds at zero in gpu_2d_continuous_cumsum.
        ((2, 3, 4, 5), -1, "float32", None, "gpu_2d_continuous_cumsum", 2),
    ],
)
def test_dispatch_cumsum_webgpu_axes_and_dtypes(
    shape, axis, in_dtype, out_dtype, expected_kernel, expected_kernel_rank
):
    """WebGPU dispatch collapses arbitrary-rank scans to the appropriate kernel."""

    vdevice = tvm.ir.VDevice("webgpu", 0)
    x = relax.Var("x", relax.TensorType(shape, in_dtype, vdevice=vdevice))
    bb = relax.BlockBuilder()
    with bb.function("main", (x,)):
        out = bb.emit(relax.op.cumsum(x, axis=axis, dtype=out_dtype))
        bb.emit_func_output(out)
    before = bb.finalize()
    before.update_global_info("vdevice", [vdevice])

    target = tvm.target.Target("webgpu", host="llvm")
    with target:
        mod = DispatchSortScan()(before)

    called_kernels = []
    permute_count = 0

    def collect_calls(expr):
        nonlocal permute_count
        if isinstance(expr, relax.Call) and getattr(expr.op, "name", None) == (
            "relax.permute_dims"
        ):
            permute_count += 1
        if isinstance(expr, relax.Call) and getattr(expr.op, "name", None) == "relax.call_tir":
            called_kernels.append(expr.args[0].name_hint)

    relax.analysis.post_order_visit(mod["main"], collect_calls)
    assert permute_count == 0
    assert called_kernels == [expected_kernel]

    cumsum = mod[expected_kernel]
    buffers = [param for param in cumsum.params if tvm.tirx.is_buffer_var(param)]
    assert len(buffers) == 2
    assert all(len(buffer.shape) == expected_kernel_rank for buffer in buffers)
    assert str(buffers[0].dtype) == in_dtype
    assert str(buffers[1].dtype) == (out_dtype or in_dtype)

    if expected_kernel == "gpu_2d_continuous_cumsum":
        floor_divisors = []

        def collect_floor_divisors(node):
            if isinstance(node, tirx.FloorDiv):
                floor_divisors.append(node.b)

        tirx.stmt_functor.post_order_visit(cumsum.body, collect_floor_divisors)
        assert floor_divisors
        assert all(
            isinstance(divisor, tirx.IntImm)
            and divisor.value > 0
            and divisor.value & (divisor.value - 1) == 0
            for divisor in floor_divisors
        )

    with target:
        tvm.compile(mod, target)


def test_dispatch_cumsum_webgpu_symbolic_non_contiguous_axis():
    """The serial WebGPU fallback accepts a symbolic scan extent."""

    @I.ir_module
    class Symbolic:
        I.module_global_infos({"vdevice": [I.vdevice("webgpu", 0)]})

        @R.function
        def main(x: R.Tensor((1, "n", 9), "float32", "webgpu")):
            return R.cumsum(x, axis=1)

    target = tvm.target.Target("webgpu", host="llvm")
    with target:
        mod = DispatchSortScan()(Symbolic)
        tvm.compile(mod, target)

    called_kernels = []

    def collect_calls(expr):
        if isinstance(expr, relax.Call) and getattr(expr.op, "name", None) == "relax.call_tir":
            called_kernels.append(expr.args[0].name_hint)

    relax.analysis.post_order_visit(mod["main"], collect_calls)
    assert called_kernels == ["gpu_3d_axis_1_cumsum"]


@pytest.mark.parametrize(
    "target",
    [
        pytest.param("cuda", marks=pytest.mark.gpu),
        pytest.param({"kind": "vulkan", "supports_int64": True}, marks=pytest.mark.gpu),
        pytest.param("metal", marks=pytest.mark.gpu),
    ],
)
@pytest.mark.parametrize(
    "in_dtype, out_dtype",
    [("float32", "float32"), ("int32", "int32"), ("int32", "float32")],
)
def test_gpu_axis_1_cumsum_numerical(target, in_dtype, out_dtype):
    """The fallback matches a sequential cumsum for supported WebGPU dtypes."""
    if not tvm.testing.device_enabled(target):
        pytest.skip(f"{target} not enabled")

    from tvm.relax.backend.gpu_generic import (  # pylint: disable=import-outside-toplevel
        gpu_3d_axis_1_cumsum,
    )

    shape = (2, 5, 7)
    if in_dtype == "int32":
        np_data = np.random.randint(-4, 5, shape).astype(in_dtype)
    else:
        np_data = np.random.uniform(-2, 2, shape).astype(in_dtype)
    expected = np.cumsum(np_data, axis=1, dtype=out_dtype)

    func = gpu_3d_axis_1_cumsum(in_dtype=in_dtype, out_dtype=out_dtype).with_attr(
        "global_symbol", "main"
    )
    compiled = tvm.compile(func, target=target)

    def run_and_check():
        dev = tvm.device_from_target(target)
        input_tensor = tvm.runtime.tensor(np_data, dev)
        output_tensor = tvm.runtime.empty(shape, out_dtype, dev)
        compiled(input_tensor, output_tensor)
        tvm.testing.assert_allclose(output_tensor.numpy(), expected)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
