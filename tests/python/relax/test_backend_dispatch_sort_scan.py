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

import numpy as np
import pytest

import tvm
import tvm.script
import tvm.testing
from tvm import dlight, relax, tir, topi
from tvm.contrib.thrust import can_use_thrust
from tvm.ir.base import assert_structural_equal
from tvm.relax.backend import DispatchSortScan
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


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
    m = tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("main", (x,), {"global_symbol": "main"}):
            with bb.dataflow():
                lv = bb.emit_te(
                    topi.cuda.cumsum,
                    x,
                    axis=1,
                    exclusive=True,
                )
                out = bb.emit_te(
                    topi.cuda.cumprod,
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
    m = tir.Var("m", "int64")
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

    target = tvm.target.Target("cuda -libs=thrust", host="llvm")

    vdevices = [I.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("foo", (x,), {"global_symbol": "foo"}):
            with bb.dataflow():
                out = bb.emit_te(
                    topi.cuda.sort,
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
                        topi.cuda.sort_thrust,
                        y,
                        axis=0,
                        is_ascend=False,
                        workspace=workspace,
                    )
                else:
                    out = bb.emit_te(topi.cuda.sort, y, axis=0, is_ascend=False)
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
    m = tir.Var("m", "int64")
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

    target = tvm.target.Target("cuda -libs=thrust", host="llvm")

    vdevices = [I.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    y = relax.Var("y", R.Tensor((2, 3), "float32"))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("foo", (x,), {"global_symbol": "foo"}):
            with bb.dataflow():
                out = bb.emit_te(topi.cuda.argsort, x, axis=1, is_ascend=True, dtype="int32")
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
                        topi.cuda.argsort_thrust,
                        y,
                        axis=0,
                        is_ascend=False,
                        dtype="int64",
                        workspace=workspace,
                    )
                else:
                    out = bb.emit_te(topi.cuda.argsort, y, axis=0, is_ascend=False, dtype="int64")
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
    m = tir.Var("m", "int64")
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

    target = tvm.target.Target("cuda -libs=thrust", host="llvm")

    vdevices = [I.vdevice("cuda", 0)]
    x = relax.Var("x", R.Tensor((2, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("foo", (x,), {"global_symbol": "foo"}):
            with bb.dataflow():
                out = bb.emit_te(topi.cuda.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
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
                lv0 = bb.emit_te(topi.cuda.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
                lv1 = bb.emit_te(topi.cuda.topk, x, k=2, axis=1, is_ascend=False, dtype="int32")
                out = (lv0, lv1)
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    with target:
        mod = DispatchSortScan()(Before)
        expected_mod = dlight.ApplyDefaultSchedule(dlight.gpu.Fallback())(expected_mod)

    assert_structural_equal(mod, expected_mod)


@tvm.testing.parametrize_targets("cuda", "vulkan -supports_int64=1")
def test_dispatch_cumsum_gpu(target, dev):
    """Test cumsum kernel dispatch and numerical correctness"""

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
        ex = tvm.relax.build(mod, target)
        vm = tvm.relax.VirtualMachine(ex, dev)
        tvm_data = tvm.nd.array(np_data, dev)
        cumsum = vm["main"](tvm_data)
        tvm.testing.assert_allclose(cumsum.numpy(), np_cumsum)


if __name__ == "__main__":
    tvm.testing.main()
