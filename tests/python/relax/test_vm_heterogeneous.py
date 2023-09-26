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
"""Test eliminate common subexpr pass"""
import tvm
from tvm import relax
import tvm.testing
from tvm.ir import VDevice
from tvm.relax.transform import RealizeVDevice
from tvm.script.parser import ir as I, relax as R, tir as T
import numpy as np


def run_cpu(mod):
    mod = relax.transform.LegalizeOps()(mod)
    target_cpu = tvm.target.Target("llvm")
    ex = relax.build(mod, target_cpu)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    ipt0 = tvm.nd.array(np.random.rand(2, 3).astype(np.float32))
    ipt1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32))
    ipt2 = tvm.nd.array(np.random.rand(4, 5).astype(np.float32))
    res = vm["foo"](ipt0, ipt1, ipt2)
    print("runtime result: ", res)


def run_cpu_gpu(mod):
    mod = relax.transform.RealizeVDevice()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    target_cuda = tvm.target.Target("cuda")
    target_cpu = tvm.target.Target("llvm")
    with target_cuda:
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    ex = relax.build(mod)

    dev_cpu = tvm.cpu()
    dev_cuda = tvm.cuda(0)
    dev_cuda_1 = tvm.cuda(1)
    print(ex.as_text())
    vm = relax.VirtualMachine(ex, [dev_cuda, dev_cpu])

    ipt0 = tvm.nd.array(np.random.rand(2, 3).astype(np.float32), dev_cpu)
    ipt1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32), dev_cpu)
    ipt2 = tvm.nd.array(np.random.rand(4, 5).astype(np.float32), dev_cuda)

    res = vm["foo"](ipt0, ipt1, ipt2)
    print("runtime result: ", res)


def run_gpu(mod):
    mod = relax.transform.LegalizeOps()(mod)
    mod.show()
    target_cuda = tvm.target.Target("cuda")
    with target_cuda:
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    mod.show()
    ex = relax.build(mod, target_cuda)
    dev = tvm.cuda()
    vm = relax.VirtualMachine(ex, dev)

    ipt0 = tvm.nd.array(np.random.rand(2, 3).astype(np.float32), dev)
    ipt1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32), dev)
    ipt2 = tvm.nd.array(np.random.rand(4, 5).astype(np.float32), dev)

    res = vm["foo"](ipt0, ipt1, ipt2)
    print("runtime result: ", res)


vdevices = [
    VDevice("llvm"),
    VDevice("cuda", 0),
    VDevice("metal", 0, "global"),
    VDevice("cuda -arch=sm_80", 0),
]


def test_single_device():
    @I.ir_module
    class Example:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                    I.vdevice("cuda", 0),
                    I.vdevice("metal", 0, "global"),
                    I.vdevice("cuda -arch=sm_80", 0),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((3, 4), "float32"),
            z: R.Tensor((4, 5), "float32"),
        ) -> R.Tensor((2, 5), "float32"):
            with R.dataflow():
                lv0: R.Tensor((2, 4), "float32") = R.matmul(x, y)
                # lv1: R.Tensor((2, 3), "float32", "cuda") = R.to_vdevice(lv0, "cuda")
                lv2: R.Tensor((2, 5), "float32") = R.matmul(lv0, z)
                R.output(lv2)
            return lv2

    run_gpu(Example)
    run_cpu(Example)


def test_multi_device():
    @I.ir_module
    class Example:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("cuda", 0),
                    I.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((3, 4), "float32"),
            z: R.Tensor((4, 5), "float32"),
        ) -> R.Tensor((2, 5), "float32"):
            with R.dataflow():
                lv0: R.Tensor((2, 4), "float32", "llvm") = R.matmul(x, y)  # cpu
                lv1: R.Tensor((2, 4), "float32", "cuda") = R.to_vdevice(lv0, "cuda")
                gv: R.Tensor((2, 4), "float32", "cuda") = R.matmul(lv1, z)  # cuda
                R.output(gv)
            return gv

    run_cpu_gpu(Example)


def test_tvm_build():
    @I.ir_module
    class ModGPU:
        I.module_attrs({"attr": 10})

        @T.prim_func
        def matmul1(
            A: T.Buffer((T.int64(2), T.int64(4)), "float32"),
            B: T.Buffer((T.int64(4), T.int64(5)), "float32"),
            matmul: T.Buffer((T.int64(2), T.int64(5)), "float32"),
        ):
            T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0_i1_fused_0 in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for i0_i1_fused_1 in T.thread_binding(T.int64(10), thread="threadIdx.x"):
                    for k in range(T.int64(4)):
                        with T.block("matmul"):
                            v_i0 = T.axis.spatial(
                                T.int64(2),
                                (i0_i1_fused_0 * T.int64(10) + i0_i1_fused_1) // T.int64(5),
                            )
                            v_i1 = T.axis.spatial(
                                T.int64(5),
                                (i0_i1_fused_0 * T.int64(10) + i0_i1_fused_1) % T.int64(5),
                            )
                            v_k = T.axis.reduce(T.int64(4), k)
                            T.reads(A[v_i0, v_k], B[v_k, v_i1])
                            T.writes(matmul[v_i0, v_i1])
                            with T.init():
                                matmul[v_i0, v_i1] = T.float32(0)
                            matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

    @I.ir_module
    class ModCPU:
        I.module_attrs({"attr": 10})

        @T.prim_func
        def matmul(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(3), T.int64(4)), "float32"),
            matmul_1: T.Buffer((T.int64(2), T.int64(4)), "float32"),
        ):
            T.func_attr({"tir.is_scheduled": T.bool(True), "tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i0, i1, k in T.grid(T.int64(2), T.int64(4), T.int64(3)):
                with T.block("matmul"):
                    v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                    T.reads(A[v_i0, v_k], B[v_k, v_i1])
                    T.writes(matmul_1[v_i0, v_i1])
                    with T.init():
                        matmul_1[v_i0, v_i1] = T.float32(0)
                    matmul_1[v_i0, v_i1] = matmul_1[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

    @I.ir_module
    class ModCPU2:
        I.module_attrs({"attr": 10})

        @T.prim_func
        def copy(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            for i0, i1 in T.grid(2, 3):
                with T.block("block"):
                    vi0, vi1 = T.axis.remap("SS", [i0, i1])
                    B[vi0, vi1] = A[vi0, vi1]

    target = tvm.target.Target("cuda")
    tir_mod = {"llvm": ModCPU, "cuda": ModGPU}
    # tir_mod = {"cuda": ModGPU}
    # lib = tvm.build(ModGPU, target="cuda")
    lib = tvm.build(tir_mod, target="cuda")


if __name__ == "__main__":
    # tvm.testing.main()
    test_multi_device()
    # test_tvm_build()
    # test_single_device()
