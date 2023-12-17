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
import tvm.script
import tvm.testing
from tvm.script import relax as R, tir as T, ir as I

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
                gv = R.call_tir(cls.cumsum, (x,), out_sinfo=R.Tensor((2, 3), dtype="float64"))
                R.output(gv)
            return gv

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, Expected, map_free_vars=True)


@pytest.mark.skip("The emitted primfunc is not roundtripable, failed in build.")
def test_dispatch_cumsum_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def main(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                gv = R.cumsum(x, axis=1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @T.prim_func(private=True)
        def cumsum(var_A: T.handle, T_add: T.Buffer((T.int64(2), T.int64(3)), "float32")):
            T.func_attr({"tir.noalias": T.bool(True)})
            data_buf = T.match_buffer(var_A, (T.int64(2), T.int64(3)), align=8)
            output_buf = T.alloc_buffer((T.int64(2), T.int64(3)), align=8)
            with T.block("exclusive_scan"):
                T.reads(data_buf[T.int64(0) : T.int64(2), T.int64(0) : T.int64(3)])
                T.writes(output_buf[T.int64(0) : T.int64(2), T.int64(0) : T.int64(3)])
                if T.bool(False):
                    blockIdx_x = T.launch_thread("blockIdx.x", T.int64(2))
                    if blockIdx_x < T.int64(2):
                        T.evaluate(0)
                else:
                    with T.launch_thread("threadIdx.x", T.int64(1024)) as threadIdx_x:
                        blockIdx_x = T.launch_thread("blockIdx.x", T.int64(1))
                        blockIdx_y = T.launch_thread("blockIdx.y", T.int64(2))
                        if blockIdx_x * T.int64(1024) + threadIdx_x < T.int64(3):
                            output_buf[
                                (
                                    blockIdx_y * T.int64(3)
                                    + (blockIdx_x * T.int64(1024) + threadIdx_x)
                                )
                                // T.int64(3),
                                (
                                    blockIdx_y * T.int64(3)
                                    + (blockIdx_x * T.int64(1024) + threadIdx_x)
                                )
                                % T.int64(3),
                            ] = data_buf[
                                (
                                    blockIdx_y * T.int64(3)
                                    + (blockIdx_x * T.int64(1024) + threadIdx_x)
                                )
                                // T.int64(3),
                                (
                                    blockIdx_y * T.int64(3)
                                    + (blockIdx_x * T.int64(1024) + threadIdx_x)
                                )
                                % T.int64(3),
                            ]
                    for i in range(T.Cast("int64", T.ceil(T.log2(T.float64(3))))):
                        threadIdx_x = T.launch_thread("threadIdx.x", 1024)
                        blockIdx_x = T.launch_thread(
                            "blockIdx.x",
                            T.max(
                                1,
                                T.Cast(
                                    "int32",
                                    (
                                        T.int64(3)
                                        + (T.int64(1024) * T.shift_left(T.int64(2), i) - T.int64(1))
                                    )
                                    // (T.int64(1024) * T.shift_left(T.int64(2), i)),
                                ),
                            ),
                        )
                        blockIdx_y = T.launch_thread("blockIdx.y", T.int64(2))
                        start = T.allocate([T.int64(1)], "int64", "local")
                        middle = T.allocate([T.int64(1)], "int64", "local")
                        end = T.allocate([T.int64(1)], "int64", "local")
                        start_1 = T.Buffer((1,), "int64", data=start, scope="local")
                        start_1[T.int64(0)] = T.shift_left(T.int64(2), i) * T.Cast(
                            "int64", blockIdx_x * 1024 + threadIdx_x
                        )
                        if start_1[T.int64(0)] < T.int64(3):
                            middle_1 = T.Buffer((1,), "int64", data=middle, scope="local")
                            middle_1[T.int64(0)] = start_1[T.int64(0)] + T.shift_left(
                                T.int64(2), i
                            ) // T.int64(2)
                            end_1 = T.Buffer((1,), "int64", data=end, scope="local")
                            end_1[T.int64(0)] = T.min(
                                start_1[T.int64(0)] + T.shift_left(T.int64(2), i), T.int64(3)
                            )
                            if middle_1[T.int64(0)] < T.int64(3):
                                output_buf[
                                    (blockIdx_y * T.int64(3) + end_1[T.int64(0)] - T.int64(1))
                                    // T.int64(3),
                                    (blockIdx_y * T.int64(3) + end_1[T.int64(0)] - T.int64(1))
                                    % T.int64(3),
                                ] = (
                                    output_buf[
                                        (blockIdx_y * T.int64(3) + end_1[T.int64(0)] - T.int64(1))
                                        // T.int64(3),
                                        (blockIdx_y * T.int64(3) + end_1[T.int64(0)] - T.int64(1))
                                        % T.int64(3),
                                    ]
                                    + output_buf[
                                        (
                                            blockIdx_y * T.int64(3)
                                            + middle_1[T.int64(0)]
                                            - T.int64(1)
                                        )
                                        // T.int64(3),
                                        (
                                            blockIdx_y * T.int64(3)
                                            + middle_1[T.int64(0)]
                                            - T.int64(1)
                                        )
                                        % T.int64(3),
                                    ]
                                )
                    with T.launch_thread("blockIdx.x", T.int64(2)) as blockIdx_x:
                        if blockIdx_x < T.int64(2):
                            output_buf[
                                ((blockIdx_x + T.int64(1)) * T.int64(3) - T.int64(1)) // T.int64(3),
                                ((blockIdx_x + T.int64(1)) * T.int64(3) - T.int64(1)) % T.int64(3),
                            ] = T.float32(0)
                    for j in range(T.Cast("int64", T.ceil(T.log2(T.float64(3))))):
                        threadIdx_x = T.launch_thread("threadIdx.x", 1024)
                        blockIdx_x = T.launch_thread(
                            "blockIdx.x",
                            T.max(
                                1,
                                T.Cast(
                                    "int32",
                                    (
                                        T.int64(3)
                                        + (
                                            T.int64(1024)
                                            * T.shift_left(
                                                T.int64(2),
                                                T.Cast("int64", T.ceil(T.log2(T.float64(3))))
                                                - j
                                                - T.int64(1),
                                            )
                                            - T.int64(1)
                                        )
                                    )
                                    // (
                                        T.int64(1024)
                                        * T.shift_left(
                                            T.int64(2),
                                            T.Cast("int64", T.ceil(T.log2(T.float64(3))))
                                            - j
                                            - T.int64(1),
                                        )
                                    ),
                                ),
                            ),
                        )
                        blockIdx_y = T.launch_thread("blockIdx.y", T.int64(2))
                        start = T.allocate([T.int64(1)], "int64", "local")
                        middle = T.allocate([T.int64(1)], "int64", "local")
                        end = T.allocate([T.int64(1)], "int64", "local")
                        end_1 = T.allocate([T.int64(1)], "float32", "local")
                        start_1 = T.Buffer((1,), "int64", data=start, scope="local")
                        start_1[T.int64(0)] = T.shift_left(
                            T.int64(2),
                            T.Cast("int64", T.ceil(T.log2(T.float64(3)))) - j - T.int64(1),
                        ) * T.Cast("int64", blockIdx_x * 1024 + threadIdx_x)
                        if start_1[T.int64(0)] < T.int64(3):
                            middle_1 = T.Buffer((1,), "int64", data=middle, scope="local")
                            middle_1[T.int64(0)] = start_1[T.int64(0)] + T.shift_left(
                                T.int64(2),
                                T.Cast("int64", T.ceil(T.log2(T.float64(3)))) - j - T.int64(1),
                            ) // T.int64(2)
                            end_2 = T.Buffer((1,), "int64", data=end, scope="local")
                            end_2[T.int64(0)] = T.min(
                                start_1[T.int64(0)]
                                + T.shift_left(
                                    T.int64(2),
                                    T.Cast("int64", T.ceil(T.log2(T.float64(3)))) - j - T.int64(1),
                                ),
                                T.int64(3),
                            )
                            if middle_1[T.int64(0)] < T.int64(3):
                                end_3 = T.Buffer((1,), data=end_1, scope="local")
                                end_3[T.int64(0)] = output_buf[
                                    (blockIdx_y * T.int64(3) + middle_1[T.int64(0)] - T.int64(1))
                                    // T.int64(3),
                                    (blockIdx_y * T.int64(3) + middle_1[T.int64(0)] - T.int64(1))
                                    % T.int64(3),
                                ]
                                output_buf[
                                    (blockIdx_y * T.int64(3) + middle_1[T.int64(0)] - T.int64(1))
                                    // T.int64(3),
                                    (blockIdx_y * T.int64(3) + middle_1[T.int64(0)] - T.int64(1))
                                    % T.int64(3),
                                ] = output_buf[
                                    (blockIdx_y * T.int64(3) + end_2[T.int64(0)] - T.int64(1))
                                    // T.int64(3),
                                    (blockIdx_y * T.int64(3) + end_2[T.int64(0)] - T.int64(1))
                                    % T.int64(3),
                                ]
                                output_buf[
                                    (blockIdx_y * T.int64(3) + end_2[T.int64(0)] - T.int64(1))
                                    // T.int64(3),
                                    (blockIdx_y * T.int64(3) + end_2[T.int64(0)] - T.int64(1))
                                    % T.int64(3),
                                ] = (
                                    output_buf[
                                        (blockIdx_y * T.int64(3) + end_2[T.int64(0)] - T.int64(1))
                                        // T.int64(3),
                                        (blockIdx_y * T.int64(3) + end_2[T.int64(0)] - T.int64(1))
                                        % T.int64(3),
                                    ]
                                    + end_3[T.int64(0)]
                                )
            for ax0, ax1 in T.grid(T.int64(2), T.int64(3)):
                with T.block("T_add"):
                    v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                    T.reads(data_buf[v_ax0, v_ax1], output_buf[v_ax0, v_ax1])
                    T.writes(T_add[v_ax0, v_ax1])
                    T_add[v_ax0, v_ax1] = data_buf[v_ax0, v_ax1] + output_buf[v_ax0, v_ax1]

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32", vdevice="cuda")
        ) -> R.Tensor((2, 3), dtype="float32", vdevice="cuda"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.cumsum, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
                R.output(gv)
            return gv

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, Expected, map_free_vars=True)


def test_dispatch_sort():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "llvm")):
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
            data_buf = T.match_buffer(var_A, (T.int64(2), T.int64(3)), align=8)
            out_buf = T.match_buffer(var_sort_cpu, (T.int64(2), T.int64(3)), align=8)
            with T.block("sort_cpu"):
                T.reads(data_buf[T.int64(0) : T.int64(2), T.int64(0) : T.int64(3)])
                T.writes(out_buf[T.int64(0) : T.int64(2), T.int64(0) : T.int64(3)])
                T.call_packed(
                    "tvm.contrib.sort.sort",
                    T.tvm_stack_make_array(
                        data_buf.data,
                        T.tvm_stack_make_shape(T.int64(2), T.int64(3)),
                        0,
                        2,
                        T.float32(0),
                        T.int64(0),
                    ),
                    T.tvm_stack_make_array(
                        out_buf.data,
                        T.tvm_stack_make_shape(T.int64(2), T.int64(3)),
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
            x: R.Tensor((2, 3), dtype="float32", vdevice="llvm")
        ) -> R.Tensor((2, 3), dtype="float32", vdevice="llvm"):
            cls = Expected
            with R.dataflow():
                gv = R.call_tir(cls.sort, (x,), out_sinfo=R.Tensor((2, 3), dtype="float32"))
                R.output(gv)
            return gv

    mod = DispatchSortScan()(Before)
    assert_structural_equal(mod, Expected, map_free_vars=True)


def test_dispatch_sort_cuda():
    @I.ir_module
    class Before:
        I.module_global_infos({"vdevice": [I.vdevice("cuda"), I.vdevice("llvm")]})

        @R.function
        def foo(x: R.Tensor((2, 3), "float32", "cuda")):
            with R.dataflow():
                gv = R.sort(x, axis=1, descending=False)
                R.output(gv)
            return gv

        @R.function
        def foo2(x: R.Tensor((2, 3), "float32")):
            with R.dataflow():
                gv = R.sort(x, axis=0, descending=True)
                R.output(gv)
            return gv

    target = tvm.target.Target("cuda -libs=thrust", host="llvm")
    with target:
        mod = DispatchSortScan()(Before)

    mod_text = mod.script()
    # The primfunc has thousands loc, simply check it has the sort_gpu
    assert 'T.block("sort_gpu")' in mod_text
    # Verify "tvm.contrib.thrust.sort" will be used
    assert "tvm.contrib.thrust.sort" in mod_text


if __name__ == "__main__":
    tvm.testing.main()
