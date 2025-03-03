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
"""Test Naive allocator with memory scope for Relax VM"""

import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@I.ir_module
class Module:
    @T.prim_func
    def add(
        arg0: T.Buffer((2, 2), "float32"),
        arg1: T.Buffer((2, 2), "float32"),
        output: T.Buffer((2, 2), "float32"),
    ):
        T.func_attr({"operator_name": "relax.add"})
        for ax0 in range(2):
            for ax1 in range(2):
                with T.block("T_add"):
                    v_ax0 = T.axis.spatial(2, ax0)
                    v_ax1 = T.axis.spatial(2, ax1)
                    T.reads(arg0[v_ax0, v_ax1], arg1[v_ax0, v_ax1])
                    T.writes(output[v_ax0, v_ax1])
                    output[v_ax0, v_ax1] = arg0[v_ax0, v_ax1] + arg1[v_ax0, v_ax1]

    @R.function(pure=False)
    def main(x: R.Tensor((2, 2), dtype="float32")):
        cls = Module
        storage = R.vm.alloc_storage(
            R.shape([2 * 2]), runtime_device_index=0, dtype="float32", storage_scope="global"
        )
        alloc = R.vm.alloc_tensor(storage, offset=0, shape=R.shape([2, 2]), dtype="float32")
        _: R.Tuple = cls.add(x, x, alloc)
        out: R.Tensor((2, 2), dtype="float32") = alloc
        return out


def test_alloc_storage_with_scope_global():
    arg0 = np.random.uniform(size=(2, 2)).astype(np.float32)
    output_ref = arg0 + arg0
    mod = Module
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        lib = relax.build(mod, target, exec_mode="compiled")

    dev = tvm.cpu()
    # This is the important line which tests nd allocator
    vm_rt = relax.VirtualMachine(lib, dev, memory_cfg="naive")
    x = tvm.nd.array(arg0, dev)
    vm_rt.set_input("main", x)
    vm_rt.invoke_stateful("main")
    output = vm_rt.get_outputs("main").numpy()
    tvm.testing.assert_allclose(output_ref, output)
