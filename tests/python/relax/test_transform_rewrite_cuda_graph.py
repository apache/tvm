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
from tvm import relax
from tvm.script import tir as T, relax as R, ir as I
import tvm.testing


def test_rewrite_cuda_graph():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "exp"})
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1], dtype="float32")


        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # force_pure is expected because purity checking should be disabled before this pass
            R.func_attr({"relax.force_pure": True})
            cls = Before
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _1: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([2, 4]), "float32")
            _2: R.Tuple = cls.exp(alloc, alloc1)
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _4: R.Tuple = cls.exp(alloc1, alloc2)
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            storage2: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc3: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage2, 0, R.shape([2, 4]), "float32")
            _6: R.Tuple = cls.exp(alloc2, alloc3)
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            alloc4: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), "float32", 0)
            _8 = cls.exp(alloc3, alloc4)
            _9: R.Tuple = R.memory.kill_tensor(alloc3)
            _10: R.Tuple = R.memory.kill_storage(storage)
            _11: R.Tuple = R.memory.kill_storage(storage1)
            _12: R.Tuple = R.memory.kill_storage(storage2)
            return alloc4


    @I.ir_module
    class Expected:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "exp"})
            # body
            # with T.block("root")
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        T.reads(rxplaceholder[i0, i1])
                        T.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1], dtype="float32")

        @R.function
        def cuda_graph_alloc() -> R.Tuple(R.Object, R.Object, R.Object):
            R.func_attr({"relax.force_pure": True})
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage2: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv: R.Tuple(R.Object, R.Object, R.Object) = (storage, storage1, storage2)
            return gv

        @R.function
        def cuda_graph_capture(alloc: R.Tensor((2, 4), dtype="float32"), alloc1: R.Tensor((2, 4), dtype="float32"), storage: R.Object, storage2: R.Object) -> R.Tuple(R.Tensor((2, 4), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            _2: R.Tuple = cls.exp(alloc, alloc1)
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _4: R.Tuple = cls.exp(alloc1, alloc2)
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage2, 0, R.shape([2, 4]), "float32")
            _6: R.Tuple = cls.exp(alloc2, alloc3)
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            gv: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (alloc3,)
            return gv

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # this comes after RemovePurityChecking, so we expect purity to be forced
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Object, R.Object, R.Object) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), sinfo_args=(R.Tuple(R.Object, R.Object, R.Object),))
            storage: R.Object = gv[0]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _1: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = gv[1]
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            storage2: R.Object = gv[2]
            gv1: R.Tuple(R.Tensor((2, 4), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.cuda_graph_capture, (alloc, alloc1, storage, storage2), R.prim_value(0)), sinfo_args=(R.Tuple(R.Tensor((2, 4), dtype="float32")),))
            alloc3: R.Tensor((2, 4), dtype="float32") = gv1[0]
            alloc4: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _6: R.Tuple = cls.exp(alloc3, alloc4)
            _7: R.Tuple = R.memory.kill_tensor(alloc3)
            _8: R.Tuple = R.memory.kill_storage(storage)
            _9: R.Tuple = R.memory.kill_storage(storage1)
            _10: R.Tuple = R.memory.kill_storage(storage2)
            return alloc4
    # fmt: on

    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_tuple():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "exp"})
            # body
            # with T.block("root")
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        T.reads(rxplaceholder[i0, i1])
                        T.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1], dtype="float32")


        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Before
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([2, 4]), "float32")
            _: R.Tuple = cls.exp(alloc, alloc1)
            lv0 = (alloc1,)
            lv1 = (lv0,)
            lv2 = lv1[0]
            lv3 = lv2[0]
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _1: R.Tuple = cls.exp(lv3, alloc2)
            _2: R.Tuple = R.memory.kill_tensor(alloc)
            _3: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = cls.exp(alloc2, alloc3)
            _5: R.Tuple = R.memory.kill_tensor(alloc2)
            _6: R.Tuple = R.memory.kill_storage(storage)
            _7: R.Tuple = R.memory.kill_storage(storage1)
            return alloc2

    @I.ir_module
    class Expected:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"global_symbol": "exp", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        T.reads(rxplaceholder[i0, i1])
                        T.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function
        def cuda_graph_alloc() -> R.Tuple(R.Object, R.Object):
            R.func_attr({"relax.force_pure": True})
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv: R.Tuple(R.Object, R.Object) = (storage, storage1)
            return gv

        @R.function
        def cuda_graph_capture(alloc: R.Tensor((2, 4), dtype="float32"), alloc1: R.Tensor((2, 4), dtype="float32"), storage: R.Object) -> R.Tuple(R.Tensor((2, 4), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            _: R.Tuple = cls.exp(alloc, alloc1)
            lv0: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (alloc1,)
            lv1: R.Tuple(R.Tuple(R.Tensor((2, 4), dtype="float32"))) = (lv0,)
            lv2: R.Tuple(R.Tensor((2, 4), dtype="float32")) = lv1[0]
            lv3: R.Tensor((2, 4), dtype="float32") = lv2[0]
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _1: R.Tuple = cls.exp(lv3, alloc2)
            _2: R.Tuple = R.memory.kill_tensor(alloc)
            _3: R.Tuple = R.memory.kill_tensor(alloc1)
            gv: R.Tuple(R.Tensor((2, 4), dtype="float32")) = (alloc2,)
            return gv

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Object, R.Object) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), sinfo_args=(R.Tuple(R.Object, R.Object),))
            storage: R.Object = gv[0]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = gv[1]
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            gv1: R.Tuple(R.Tensor((2, 4), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.cuda_graph_capture, (alloc, alloc1, storage), R.prim_value(0)), sinfo_args=(R.Tuple(R.Tensor((2, 4), dtype="float32")),))
            alloc2: R.Tensor((2, 4), dtype="float32") = gv1[0]
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = cls.exp(alloc2, alloc3)
            _5: R.Tuple = R.memory.kill_tensor(alloc2)
            _6: R.Tuple = R.memory.kill_storage(storage)
            _7: R.Tuple = R.memory.kill_storage(storage1)
            return alloc2
    # fmt: on

    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


def test_vm_builtin():
    # fmt: off
    @I.ir_module
    class Before:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            # function attr dict
            T.func_attr({"tir.noalias": True, "global_symbol": "exp"})
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1], dtype="float32")


        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # force_pure is expected because purity checking should be disabled before this pass
            R.func_attr({"relax.force_pure": True})
            cls = Before
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            _1: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), 0, "global", "float32")
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([2, 4]), "float32")
            _2: R.Tuple = cls.exp(alloc, alloc1)
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), "float32")
            lv: R.Tensor((2, 4), dtype="float32") = alloc2
            _4: R.Tuple = R.call_packed("vm.builtin.dummy", (x, lv), sinfo_args=R.Tuple())
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), "float32", 0)
            _6 = cls.exp(alloc2, alloc3)
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            _8: R.Tuple = R.memory.kill_storage(storage)
            return alloc3

    @I.ir_module
    class Expected:
        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            T.func_attr({"global_symbol": "exp", "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(8), thread="threadIdx.x"):
                    with T.block("compute"):
                        i0 = T.axis.spatial(T.int64(2), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) // T.int64(4))
                        i1 = T.axis.spatial(T.int64(4), (i0_i1_fused_0 * T.int64(8) + i0_i1_fused_1) % T.int64(4))
                        T.reads(rxplaceholder[i0, i1])
                        T.writes(compute[i0, i1])
                        compute[i0, i1] = T.exp(rxplaceholder[i0, i1])

        @R.function
        def cuda_graph_alloc() -> R.Tuple(R.Object, R.Object):
            R.func_attr({"relax.force_pure": True})
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            storage1: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            gv: R.Tuple(R.Object, R.Object) = (storage, storage1)
            return gv

        @R.function
        def cuda_graph_capture(alloc: R.Tensor((2, 4), dtype="float32"), alloc1: R.Tensor((2, 4), dtype="float32"), storage: R.Object) -> R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            _2: R.Tuple = cls.exp(alloc, alloc1)
            _3: R.Tuple = R.memory.kill_tensor(alloc)
            alloc2: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            lv: R.Tensor((2, 4), dtype="float32") = alloc2
            gv: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")) = (lv, alloc2)
            return gv

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            gv: R.Tuple(R.Object, R.Object) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.get_cached_alloc", (cls.cuda_graph_alloc, R.prim_value(0)), sinfo_args=(R.Tuple(R.Object, R.Object),))
            storage: R.Object = gv[0]
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _1: R.Tuple = cls.exp(x, alloc)
            storage1: R.Object = gv[1]
            alloc1: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            gv1: R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")) = R.call_builtin_with_ctx("vm.builtin.cuda_graph.run_or_capture", (cls.cuda_graph_capture, (alloc, alloc1, storage), R.prim_value(0)), sinfo_args=(R.Tuple(R.Tensor((2, 4), dtype="float32"), R.Tensor((2, 4), dtype="float32")),))
            alloc2: R.Tensor((2, 4), dtype="float32") = gv1[1]
            lv: R.Tensor((2, 4), dtype="float32") = gv1[0]
            _4: R.Tuple = R.call_packed("vm.builtin.dummy", (x, lv), sinfo_args=(R.Tuple,))
            _5: R.Tuple = R.memory.kill_tensor(alloc1)
            alloc3: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), R.dtype("float32"), R.prim_value(0))
            _6: R.Tuple = cls.exp(alloc2, alloc3)
            _7: R.Tuple = R.memory.kill_tensor(alloc2)
            _8: R.Tuple = R.memory.kill_storage(storage)
            return alloc3
    # fmt: on

    after = relax.transform.RewriteCUDAGraph()(Before)
    tvm.ir.assert_structural_equal(after, Expected)


if __name__ == "__main__":
    tvm.testing.main()
