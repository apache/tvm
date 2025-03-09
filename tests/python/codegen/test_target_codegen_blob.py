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

import ctypes
import numpy as np
from tvm.contrib import cc, utils, popen_pool, tar
import tvm
import tvm.testing
from tvm.script import ir as I, tir as T


@tvm.testing.uses_gpu
def test_cuda_multi_lib():
    # test combining two system lib together
    # each contains a fatbin component in cuda
    dev = tvm.cuda(0)
    for device in ["llvm", "cuda"]:
        if not tvm.testing.device_enabled(device):
            print("skip because %s is not enabled..." % device)
            return

    @tvm.script.ir_module
    class ModA:
        I.module_attrs({"system_lib_prefix": "modA_"})

        @T.prim_func
        def my_inplace_update(x: T.Buffer((12), "float32")) -> None:
            T.func_attr({"global_symbol": "modA_my_inplace_update"})
            for bx in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for tx in T.thread_binding(T.int64(12), thread="threadIdx.x"):
                    x[tx] = x[tx] + 1

    @tvm.script.ir_module
    class ModB:
        I.module_attrs({"system_lib_prefix": "modB_"})

        @T.prim_func
        def my_inplace_update(x: T.Buffer((12), "float32")) -> None:
            T.func_attr({"global_symbol": "modB_my_inplace_update"})
            for bx in T.thread_binding(T.int64(1), thread="blockIdx.x"):
                for tx in T.thread_binding(T.int64(12), thread="threadIdx.x"):
                    x[tx] = x[tx] + 2

    temp = utils.tempdir()
    target = tvm.target.Target("cuda", host="llvm")
    libA = tvm.compile(ModA, target=target)
    libB = tvm.compile(ModB, target=target)

    pathA = temp.relpath("libA.tar")
    pathB = temp.relpath("libB.tar")
    pathAll = temp.relpath("libAll.a")

    path_dso = temp.relpath("mylib.so")
    libA.export_library(pathA, fcompile=tar.tar)
    libB.export_library(pathB, fcompile=tar.tar)
    cc.create_staticlib(pathAll, [pathA, pathB])
    # package two static libs together
    cc.create_shared(path_dso, ["-Wl,--whole-archive", pathAll, "-Wl,--no-whole-archive"])

    def popen_check():
        # Load dll, will trigger system library registration
        ctypes.CDLL(path_dso)
        # Load the system wide library
        dev = tvm.cuda()
        a_np = np.random.uniform(size=12).astype("float32")
        a_nd = tvm.nd.array(a_np, dev)
        b_nd = tvm.nd.array(a_np, dev)
        syslibA = tvm.runtime.system_lib("modA_")
        syslibB = tvm.runtime.system_lib("modB_")
        # reload same lib twice
        syslibA = tvm.runtime.system_lib("modA_")
        syslibA["my_inplace_update"](a_nd)
        syslibB["my_inplace_update"](b_nd)
        np.testing.assert_equal(a_nd.numpy(), a_np + 1)
        np.testing.assert_equal(b_nd.numpy(), a_np + 2)

    # system lib should be loaded in different process
    worker = popen_pool.PopenWorker()
    worker.send(popen_check)
    worker.recv()


if __name__ == "__main__":
    test_synthetic()
    test_cuda_multilib()
