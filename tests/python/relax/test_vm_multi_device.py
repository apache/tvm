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
from typing import List
import tvm
from tvm import relax
import tvm.testing
from tvm.ir.module import IRModule
from tvm.script.parser import ir as I, relax as R
from tvm._ffi.runtime_ctypes import Device
import numpy as np


def compile(
    mod: IRModule,
    device: List[Device] = [
        tvm.cpu(),
    ],
) -> relax.VirtualMachine:
    # compile the model
    mod = relax.transform.RealizeVDevice()(mod)
    mod = relax.transform.LegalizeOps()(mod)
    mod = tvm.tir.transform.DefaultGPUSchedule()(mod)
    # no need to feed target argument for mult-target compilation
    ex = relax.build(mod)

    return relax.VirtualMachine(ex, device)


def test_multi_cpu():
    @I.ir_module
    class Example:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm", 0),
                    I.vdevice("llvm", 1),
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
                lv0: R.Tensor((2, 4), "float32", "llvm:0") = R.matmul(x, y)  # noqa: F722
                lv1: R.Tensor((2, 4), "float32", "llvm:1") = R.to_vdevice(  # noqa: F722
                    lv0, "llvm:1"  # noqa: F722
                )
                gv = R.matmul(lv1, z)  # noqa: F722
                R.output(gv)
            return gv

    devices = [tvm.cpu(0), tvm.cpu(1)]
    vm = compile(Example, devices)

    np_ipt0 = np.random.rand(2, 3).astype(np.float32)
    np_ipt1 = np.random.rand(3, 4).astype(np.float32)
    np_ipt2 = np.random.rand(4, 5).astype(np.float32)
    np_res = np.matmul(np.matmul(np_ipt0, np_ipt1), np_ipt2)

    ipt0 = tvm.nd.array(np_ipt0, devices[0])
    ipt1 = tvm.nd.array(np_ipt1, devices[0])
    ipt2 = tvm.nd.array(np_ipt2, devices[1])
    res = vm["foo"](ipt0, ipt1, ipt2)
    tvm.testing.assert_allclose(res.numpy(), np_res)


@tvm.testing.requires_multi_gpu
def test_multi_gpu():
    @I.ir_module
    class Example:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("cuda", 1),
                    I.vdevice("cuda", 0),
                    I.vdevice("cuda", 2),
                ]
            }
        )

        @R.function
        def foo(
            a: R.Tensor((2, 3), "float32"),
            b: R.Tensor((3, 4), "float32"),
            c: R.Tensor((4, 5), "float32"),
            d: R.Tensor((5, 6), "float32"),
        ) -> R.Tensor((2, 6), "float32"):
            with R.dataflow():
                lv0: R.Tensor((2, 4), "float32", "cuda:0") = R.matmul(a, b)  # noqa: F722
                lv1: R.Tensor((2, 4), "float32", "cuda:1") = R.to_vdevice(  # noqa: F722
                    lv0, "cuda:1"  # noqa: F722
                )
                lv2: R.Tensor((2, 5), "float32", "cuda:1") = R.matmul(lv1, c)  # noqa: F722
                lv3: R.Tensor((2, 5), "float32", "cuda:2") = R.to_vdevice(  # noqa: F722
                    lv2, "cuda:2"  # noqa: F722
                )
                gv: R.Tensor((2, 6), "float32", "cuda:2") = R.matmul(lv3, d)  # noqa: F722
                R.output(gv)
            return gv

    # The number and ordering of devices should be identical with the vdevice list
    # defined in global_infos of ir_module
    devices = [tvm.cuda(1), tvm.cuda(0), tvm.cuda(2)]
    vm = compile(Example, devices)

    np_ipt0 = np.random.rand(2, 3).astype(np.float32)
    np_ipt1 = np.random.rand(3, 4).astype(np.float32)
    np_ipt2 = np.random.rand(4, 5).astype(np.float32)
    np_ipt3 = np.random.rand(5, 6).astype(np.float32)
    np_res = np.matmul(np.matmul(np.matmul(np_ipt0, np_ipt1), np_ipt2), np_ipt3)

    ipt0 = tvm.nd.array(np_ipt0, devices[0])
    ipt1 = tvm.nd.array(np_ipt1, devices[0])
    ipt2 = tvm.nd.array(np_ipt2, devices[1])
    ipt3 = tvm.nd.array(np_ipt3, devices[2])
    res = vm["foo"](ipt0, ipt1, ipt2, ipt3)
    tvm.testing.assert_allclose(res.numpy(), np_res)


@tvm.testing.requires_gpu
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
                lv0: R.Tensor((2, 4), "float32", "llvm") = R.matmul(x, y)
                lv1: R.Tensor((2, 4), "float32", "cuda") = R.to_vdevice(lv0, "cuda")
                gv: R.Tensor((2, 5), "float32", "cuda") = R.matmul(lv1, z)
                R.output(gv)
            return gv

    # The number and ordering of devices should be identical with the vdevice list
    # defined in global_infos of ir_module
    devices = [tvm.cuda(0), tvm.cpu(0)]
    vm = compile(Example, devices)

    np_ipt0 = np.random.rand(2, 3).astype(np.float32)
    np_ipt1 = np.random.rand(3, 4).astype(np.float32)
    np_ipt2 = np.random.rand(4, 5).astype(np.float32)
    np_res = np.matmul(np.matmul(np_ipt0, np_ipt1), np_ipt2)

    ipt0 = tvm.nd.array(np_ipt0, devices[1])
    ipt1 = tvm.nd.array(np_ipt1, devices[1])
    ipt2 = tvm.nd.array(np_ipt2, devices[0])
    res = vm["foo"](ipt0, ipt1, ipt2)
    tvm.testing.assert_allclose(res.numpy(), np_res, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tvm.testing.main()
