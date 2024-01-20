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

import pytest

import tvm
from tvm import topi, relax, tir, dlight
import tvm.script
import tvm.testing
from tvm.script import relax as R, tir as T, ir as I
from tvm.contrib.thrust import can_use_thrust


from tvm.relax.backend import DispatchSortScan
from tvm.ir.base import assert_structural_equal


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
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0)]})

        @R.function
        def main(x: R.Tensor(("m", 3), "float32", "cuda")):
            with R.dataflow():
                lv0 = R.cumsum(x, axis=1)
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
                out = bb.emit_te(
                    topi.cuda.sort_thrust
                    if can_use_thrust(target, "tvm.contrib.thrust.sort")
                    else topi.cuda.sort,
                    y,
                    0,
                    False,
                )
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
                out = bb.emit_te(
                    topi.cuda.argsort_thrust
                    if can_use_thrust(target, "tvm.contrib.thrust.sort")
                    else topi.cuda.argsort,
                    y,
                    0,
                    False,
                    "int64",
                )
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


if __name__ == "__main__":
    tvm.testing.main()
