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
from tvm import topi, relax, tir
import tvm.script
import tvm.testing
from tvm.script import relax as R, tir as T, ir as I
from tvm.contrib.thrust import can_use_thrust


from tvm.relax.backend import DispatchSortScan
from tvm.ir.base import assert_structural_equal


def test_dispatch_cumsum():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "llvm")):
            with R.dataflow():
                gv = R.cumsum(x, axis=1, dtype="float64")
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @T.prim_func(private=True)
        def cumsum(var_A: T.handle, out_buf: T.Buffer((T.int64(2), T.int64(3)), "float64")):
            T.func_attr({"tir.noalias": T.bool(True)})
            A = T.match_buffer(var_A, (T.int64(2), T.int64(3)), offset_factor=1)
            with T.block("cumsum_generic"):
                T.reads(A[T.int64(0) : T.int64(2), T.int64(0) : T.int64(3)])
                T.writes(out_buf[T.int64(0) : T.int64(2), T.int64(0) : T.int64(3)])
                for fused in T.parallel(T.int64(2)):
                    out_buf[
                        fused * T.int64(3) // T.int64(3), fused * T.int64(3) % T.int64(3)
                    ] = T.Cast(
                        "float64",
                        A[fused * T.int64(3) // T.int64(3), fused * T.int64(3) % T.int64(3)],
                    )
                    for _k in range(T.int64(2)):
                        out_buf[
                            (fused * T.int64(3) + (_k + T.int64(1))) // T.int64(3),
                            (fused * T.int64(3) + (_k + T.int64(1))) % T.int64(3),
                        ] = out_buf[
                            (fused * T.int64(3) + (_k + T.int64(1) - T.int64(1))) // T.int64(3),
                            (fused * T.int64(3) + (_k + T.int64(1) - T.int64(1))) % T.int64(3),
                        ] + T.Cast(
                            "float64",
                            A[
                                (fused * T.int64(3) + (_k + T.int64(1))) // T.int64(3),
                                (fused * T.int64(3) + (_k + T.int64(1))) % T.int64(3),
                            ],
                        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), dtype="float32", vdevice="llvm")
        ) -> R.Tensor((2, 3), dtype="float64", vdevice="llvm"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.cumsum, (x,), out_sinfo=R.Tensor((2, 3), "float64", "llvm"))
                R.output(gv)
            return gv

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, Expected)


def test_dispatch_cumsum_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def main(x: R.Tensor(("m", 3), "float32", "cuda")):
            with R.dataflow():
                lv = R.cumsum(x, axis=1)
                gv = lv
                R.output(gv)
            return gv

    target = tvm.target.Target("cuda", host="llvm")

    vdevices = [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]
    m = tir.Var("m", "int64")
    x = relax.Var("x", R.Tensor((m, 3), "float32", vdevices[0]))
    bb = relax.BlockBuilder()
    with target:
        with bb.function("main", (x,), {"global_symbol": "main"}):
            with bb.dataflow():
                out = bb.emit_te(
                    topi.cuda.cumsum,
                    x,
                    axis=1,
                )
                out = bb.emit_output(out)
            bb.emit_func_output(out)
    expected_mod = bb.finalize()
    expected_mod.update_global_info("vdevice", vdevices)

    with target:
        mod = DispatchSortScan()(Before)

    assert_structural_equal(mod, expected_mod)


def test_dispatch_sort():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor(("m", 3), "float32", "llvm")):
            m = T.int64()
            with R.dataflow():
                gv = R.sort(x, axis=1, descending=False)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @T.prim_func(private=True)
        def sort(var_A: T.handle, var_sort_cpu: T.handle):
            T.func_attr({"tir.noalias": T.bool(True)})
            m = T.int64()
            data_buf = T.match_buffer(var_A, (m, T.int64(3)), align=8)
            out_buf = T.match_buffer(var_sort_cpu, (m, T.int64(3)), align=8)
            with T.block("sort_cpu"):
                T.reads(data_buf[T.int64(0) : m, T.int64(0) : T.int64(3)])
                T.writes(out_buf[T.int64(0) : m, T.int64(0) : T.int64(3)])
                T.call_packed(
                    "tvm.contrib.sort.sort",
                    T.tvm_stack_make_array(
                        data_buf.data,
                        T.tvm_stack_make_shape(m, T.int64(3)),
                        0,
                        2,
                        T.float32(0),
                        T.int64(0),
                    ),
                    T.tvm_stack_make_array(
                        out_buf.data,
                        T.tvm_stack_make_shape(m, T.int64(3)),
                        0,
                        2,
                        T.float32(0),
                        T.int64(0),
                    ),
                    1,
                    T.bool(True),
                )

        @R.function
        def foo(
            x: R.Tensor(("m", 3), dtype="float32", vdevice="llvm")
        ) -> R.Tensor(("m", 3), dtype="float32", vdevice="llvm"):
            m = T.int64()
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(
                    cls.sort, (x,), out_sinfo=R.Tensor((m, 3), dtype="float32", vdevice="llvm")
                )
                R.output(gv)
            return gv

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, Expected)


def test_dispatch_sort_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda"), I.vdevice("llvm")]})

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

    vdevices = [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]
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

    assert_structural_equal(mod, expected_mod, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
