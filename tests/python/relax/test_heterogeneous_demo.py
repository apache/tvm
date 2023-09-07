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
    mod.show()
    mod = relax.transform.LegalizeOps()(mod)
    mod.show()
    target_cuda = tvm.target.Target("cuda")
    target_cpu = tvm.target.Target("llvm")
    with target_cuda:
        mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    mod.show()
    ex = relax.build(mod, target_cpu)
    dev_cpu = tvm.cpu()
    dev_cuda = tvm.cuda()
    vm = relax.VirtualMachine(ex, tvm.cpu())

    ipt0 = tvm.nd.array(np.random.rand(2, 3).astype(np.float32), dev_cpu)
    ipt1 = tvm.nd.array(np.random.rand(3, 4).astype(np.float32), dev_cpu)
    ipt2 = tvm.nd.array(np.random.rand(4, 5).astype(np.float32), dev_cuda)
    res = vm["foo"](ipt0, ipt1, ipt2)
    print("runtime result: ", res)

    # We have two options to legalize ops to gpu specific tir primfunc
    # o0: update all the legalize_ops/*.py to register gpu specific logic there
    #     pros: quite straightforward
    #     cons: too many codes to touch, and affect the op logic
    #           also it might be difficult to emit gpu related code via emit_te
    # o1: update DefaultGPUSchedule
    #     pros: touch limited code
    #     cons: if vdevice is none, still bind since DefaultGPUSchedule was called explictly
    # Decision: Go with o1.

    # What happens if we apply RealizeVDevice after LegalizeOps? can't handle hint_on_device, realizevdevice first is a must


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
                lv0: R.Tensor((2, 4), "float32", "llvm") = R.matmul(x, y)
                lv1: R.Tensor((2, 4), "float32", "cuda") = R.to_vdevice(lv0, "cuda")
                lv2: R.Tensor((2, 4), "float32", "cuda") = R.matmul(lv1, z)
                R.output(lv2)
            return lv2

    run_cpu_gpu(Example)


if __name__ == "__main__":
    # tvm.testing.main()
    test_multi_device()
