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

"""
Test relax vm builtin to enable DMA copy and wait operations.
"""

import numpy as np
import tvm
import tvm.script
from tvm import relax
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
import tvm.contrib.hexagon
import tvm.testing

# pylint: disable=invalid-name, missing-class-docstring, missing-function-docstring, no-self-argument

data_type = "int32"


@I.ir_module
class Module_1D:
    @T.prim_func
    def compute_add_in_vtcm(a: T.handle, b: T.handle, c: T.handle) -> None:
        m = T.int32()
        A = T.match_buffer(a, (m,), data_type, scope="global.vtcm")
        B = T.match_buffer(b, (m,), data_type, scope="global.vtcm")
        C = T.match_buffer(c, (m,), data_type, scope="global.vtcm")
        for ax0 in T.grid(m):
            with T.block("T_add"):
                v_ax0 = T.axis.remap("S", [ax0])
                T.reads(A[v_ax0], B[v_ax0])
                T.writes(C[v_ax0])
                C[v_ax0] = A[v_ax0] + B[v_ax0]

    @R.function
    def main(
        x: R.Tensor((12800,), data_type),
        y: R.Tensor((12800,), data_type),
    ) -> R.Tensor((12800,), data_type):
        cls = Module_1D
        vtcm_obj: R.Object = R.vm.alloc_storage(
            R.shape(
                [
                    3 * 12800,  # 3 = 2 inputs + 1 output
                ]
            ),
            runtime_device_index=0,
            dtype=data_type,
            storage_scope="global.vtcm",
        )
        a: R.Tensor([12800,], dtype=data_type) = R.vm.alloc_tensor(
            vtcm_obj,
            offset=0,
            shape=R.shape(
                [
                    12800,
                ]
            ),
            dtype=data_type,
        )
        __: R.Tuple = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.dma_copy",
            [x, a, 0, True],
            sinfo_args=[],
        )
        b: R.Tensor([12800,], dtype=data_type) = R.vm.alloc_tensor(
            vtcm_obj,
            offset=12800 * 4,
            shape=R.shape(
                [
                    12800,
                ]
            ),
            dtype=data_type,
        )
        __: R.Tuple = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.dma_copy",
            [y, b, 1, True],
            sinfo_args=[],
        )
        c: R.Tensor([12800,], dtype=data_type) = R.vm.alloc_tensor(
            vtcm_obj,
            offset=2 * 12800 * 4,
            shape=R.shape(
                [
                    12800,
                ]
            ),
            dtype=data_type,
        )
        __: R.Tuple = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.dma_wait",
            [0, 2, x, a],
            sinfo_args=[],
        )
        __: R.Tuple = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.dma_wait",
            [1, 1, y, b],
            sinfo_args=[],
        )
        ___: R.Tuple = cls.compute_add_in_vtcm(a, b, c)
        ret_val: R.Tensor((12800,), dtype=data_type) = R.builtin.alloc_tensor(
            R.shape(
                [
                    12800,
                ]
            ),
            R.dtype(data_type),
            R.prim_value(0),
        )
        __: R.Tuple = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.dma_copy",
            [c, ret_val, 0, True],
            sinfo_args=[],
        )
        __: R.Tuple = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.dma_wait",
            [0, 1, c, ret_val],
            sinfo_args=[],
        )
        _t3: R.Tuple = R.vm.kill_object(vtcm_obj)
        _t6: R.Tuple = R.vm.kill_object(a)
        _t7: R.Tuple = R.vm.kill_object(b)
        _t8: R.Tuple = R.vm.kill_object(c)
        lv: R.Tensor((12800,), dtype=data_type) = ret_val
        return lv


class TestDMACopyWait:
    """Tests for Copy and wait"""

    mode = tvm.testing.parameter("bytecode", "compiled")
    module = tvm.testing.parameter(Module_1D)

    @tvm.testing.requires_hexagon
    def test_vtcm_alloc_compute(self, hexagon_launcher, mode, module):
        target_hexagon = tvm.target.hexagon("v69")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
        with tvm.transform.PassContext(opt_level=3, config=[]):
            ex = relax.build(mod=module, target=target, exec_mode=mode)
        with hexagon_launcher.create_session() as session:
            dev = session.device
            input_arg0_data = np.random.randint(0, 9, size=(12800,), dtype=data_type)
            input_arg1_data = np.random.randint(0, 9, size=(12800,), dtype=data_type)
            output_data = np.add(input_arg0_data, input_arg1_data)
            vm_mod = session.get_executor_from_factory(ex)
            vm_rt = relax.VirtualMachine(
                vm_mod, dev, "naive"
            )  # Use naive allocator to exercise VTCM allocation in relax
            data0 = tvm.nd.array(input_arg0_data, dev)
            data1 = tvm.nd.array(input_arg1_data, dev)
            vm_rt.set_input("main", data0, data1)
            vm_rt.invoke_stateful("main")
            hexagon_output = vm_rt.get_outputs("main").numpy()
            tvm.testing.assert_allclose(output_data, hexagon_output)
