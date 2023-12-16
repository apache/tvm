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

import tvm
import tvm.script
import tvm.testing
from tvm.script import relax as R, tir as T, ir as I

from tvm.relax.backend import DispatchOps
from tvm.ir.base import assert_structural_equal


def test_dispatch_cumsum():
    @I.ir_module
    class Cumsum:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def cumsum(x: R.Tensor((3, 2, 3), "float32", "llvm")):
            with R.dataflow():
                lv0 = R.add(x, x)
                lv1 = R.cumsum(lv0, axis=None)
                gv = R.cumsum(lv1, axis=1, dtype="float64")
                R.output(gv)
            return gv

    @I.ir_module
    class CumsumGPU:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def cumsum_gpu(x: R.Tensor((3, 2, 3), "float32", "cuda")):
            with R.dataflow():
                lv0 = R.add(x, x)
                lv1 = R.cumsum(lv0, axis=None)
                gv = R.cumsum(lv1, axis=1, dtype="float64")
                R.output(gv)
            return gv

    @I.ir_module
    class CumsumAfter:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def cumsum(
            x: R.Tensor((3, 2, 3), dtype="float32", vdevice="llvm")
        ) -> R.Tensor((18,), dtype="float64", vdevice="llvm"):
            with R.dataflow():
                lv0: R.Tensor((3, 2, 3), dtype="float32", vdevice="llvm") = R.add(x, x)
                lv1: R.Tensor((18,), dtype="float32", vdevice="llvm") = R.cumsum(
                    lv0, axis=None, dtype="void"
                )
                gv: R.Tensor((18,), dtype="float64", vdevice="llvm") = R.cumsum(
                    lv1, axis=1, dtype="float64"
                )
                R.output(gv)
            return gv

    @I.ir_module
    class CumsumGPUAfter:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def cumsum_gpu(
            x: R.Tensor((3, 2, 3), dtype="float32", vdevice="cuda")
        ) -> R.Tensor((18,), dtype="float64", vdevice="cuda"):
            with R.dataflow():
                lv0 = R.add(x, x)
                lv1 = R.reshape(lv0, R.shape([18]))
                lv1_1 = R.call_dps_packed(
                    "tvm.contrib.thrust.sum_scan",
                    (lv1, R.prim_value(0)),
                    out_sinfo=R.Tensor((18,), vdevice="cuda"),
                )
                gv = R.call_dps_packed(
                    "tvm.contrib.thrust.sum_scan",
                    (lv1_1, R.prim_value(1)),
                    out_sinfo=R.Tensor((18,), vdevice="cuda"),
                )
                R.output(gv)
            return gv

    mod = DispatchOps()(Cumsum)
    mod_gpu = DispatchOps()(CumsumGPU)

    assert_structural_equal(mod, CumsumAfter, map_free_vars=True)
    assert_structural_equal(mod_gpu, CumsumGPUAfter, map_free_vars=True)


def test_dispatch_sort():
    @I.ir_module
    class Sort:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def sort(x: R.Tensor((3, 2, 3), "float32", "llvm")):
            with R.dataflow():
                lv = R.add(x, x)
                gv = R.sort(lv, axis=1, is_ascend=True)  # R.cumsum(lv1, axis=1, dtype="int32")
                R.output(gv)
            return gv

    @I.ir_module
    class SortGPU:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def sort_gpu(x: R.Tensor((3, 2, 3), "float32", "cuda")):
            with R.dataflow():
                lv0 = R.add(x, x)
                lv1 = R.sort(lv0, axis=1, is_ascend=True)  # R.cumsum(lv1, axis=1, dtype="int32")
                gv = R.sort(lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class SortAfter:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def sort(
            x: R.Tensor((3, 2, 3), dtype="float32", vdevice="llvm")
        ) -> R.Tensor((3, 2, 3), dtype="float32", vdevice="llvm"):
            with R.dataflow():
                lv: R.Tensor((3, 2, 3), dtype="float32", vdevice="llvm") = R.add(x, x)
                gv = R.call_dps_packed(
                    "tvm.contrib.sort.sort",
                    (lv, R.prim_value(1), R.prim_value(True)),
                    out_sinfo=R.Tensor((3, 2, 3), dtype="float32", vdevice="llvm"),
                )
                R.output(gv)
            return gv

    @I.ir_module
    class SortGPUAfter:
        I.module_global_infos({"vdevice": [I.vdevice("cuda", 0), I.vdevice("llvm", 0)]})

        @R.function
        def sort_gpu(
            x: R.Tensor((3, 2, 3), dtype="float32", vdevice="cuda")
        ) -> R.Tensor((3, 2, 3), dtype="float32", vdevice="cuda"):
            with R.dataflow():
                lv0: R.Tensor((3, 2, 3), dtype="float32", vdevice="cuda") = R.add(x, x)
                lv1: R.Tensor((3, 3, 2), dtype="float32", vdevice="cuda") = R.permute_dims(
                    lv0, axes=[0, 2, 1]
                )
                lv2 = R.call_dps_packed(
                    "tvm.contrib.thrust.sort",
                    (lv1, R.prim_value(1)),
                    out_sinfo=R.Tensor((3, 3, 2), dtype="float32", vdevice="cuda"),
                )
                lv1_1: R.Tensor((3, 2, 3), dtype="float32", vdevice="cuda") = R.permute_dims(
                    lv2, axes=[0, 2, 1]
                )
                gv = R.call_dps_packed(
                    "tvm.contrib.thrust.sort",
                    (lv1_1, R.prim_value(1)),
                    out_sinfo=R.Tensor((3, 2, 3), dtype="float32", vdevice="cuda"),
                )
                R.output(gv)
            return gv

    mod = DispatchOps()(Sort)
    mod_gpu = DispatchOps()(SortGPU)

    assert_structural_equal(mod, SortAfter, map_free_vars=True)
    assert_structural_equal(mod_gpu, SortGPUAfter, map_free_vars=True)


if __name__ == "__main__":
    tvm.testing.main()
