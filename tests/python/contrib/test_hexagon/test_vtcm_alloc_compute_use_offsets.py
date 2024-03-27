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
# pylint: disable=unused-wildcard-import, invalid-name, missing-docstring,no-self-argument
"""Test Discontiguous allocation for hexagon"""

import numpy as np

import tvm
import tvm.contrib.hexagon
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T


@I.ir_module
class Module:
    @T.prim_func
    def compute_add_in_vtcm(a: T.handle, b: T.handle, c: T.handle) -> None:
        m, n = T.int32(), T.int32()
        A = T.match_buffer(a, (m, n), "int32", scope="global.vtcm")
        B = T.match_buffer(b, (m, n), "int32", scope="global.vtcm")
        C = T.match_buffer(c, (m, n), "int32", scope="global.vtcm")
        for ax0, ax1 in T.grid(m, n):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(C[v_ax0, v_ax1])
                C[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

    @T.prim_func
    def compute_mul_in_vtcm(a: T.handle, b: T.handle, c: T.handle) -> None:
        m, n = T.int32(), T.int32()
        A = T.match_buffer(a, (m, n), "int32", scope="global.vtcm")
        B = T.match_buffer(b, (m, n), "int32", scope="global.vtcm")
        C = T.match_buffer(c, (m, n), "int32", scope="global.vtcm")
        for ax0, ax1 in T.grid(m, n):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(C[v_ax0, v_ax1])
                C[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[v_ax0, v_ax1]

    @R.function
    def main(
        x: R.Tensor((4, 64), "int32"),
        y: R.Tensor((4, 64), "int32"),
        z: R.Tensor((4, 64), "int32"),
    ) -> R.Tensor((4, 64), "int32"):
        cls = Module
        vtcm_obj: R.Object = R.vm.alloc_storage(
            R.shape([4096]), runtime_device_index=0, dtype="uint8", storage_scope="global.vtcm"
        )
        a: R.Tensor([4, 64], dtype="int32") = R.vm.alloc_tensor(
            vtcm_obj, offset=0, shape=R.shape([4, 64]), dtype="int32"
        )
        __: R.Tuple = R.vm.copy_tensor_from_to(x, a)
        b: R.Tensor([4, 64], dtype="int32") = R.vm.alloc_tensor(
            vtcm_obj, offset=1024, shape=R.shape([4, 64]), dtype="int32"
        )
        _: R.Tuple = R.vm.copy_tensor_from_to(y, b)
        c: R.Tensor([4, 64], dtype="int32") = R.vm.alloc_tensor(
            vtcm_obj, offset=2048, shape=R.shape([4, 64]), dtype="int32"
        )
        ___: R.Tuple = cls.compute_add_in_vtcm(a, b, c)
        _t1: R.Tuple = R.vm.kill_object(a)
        _t2: R.Tuple = R.vm.kill_object(b)
        d: R.Tensor([4, 64], dtype="int32") = R.vm.alloc_tensor(
            vtcm_obj, offset=0, shape=R.shape([4, 64]), dtype="int32"
        )
        ___1: R.Tuple = R.vm.copy_tensor_from_to(z, d)
        e: R.Tensor([4, 64], dtype="int32") = R.vm.alloc_tensor(
            vtcm_obj, offset=1024, shape=R.shape([4, 64]), dtype="int32"
        )
        ___2: R.Tuple = cls.compute_mul_in_vtcm(c, d, e)
        _t2: R.Tuple = R.vm.kill_object(c)
        _t12: R.Tuple = R.vm.kill_object(d)
        f: R.Tensor([4, 64], dtype="int32") = R.vm.alloc_tensor(
            vtcm_obj, offset=2048, shape=R.shape([4, 64]), dtype="int32"
        )
        _t13: R.Tuple = R.vm.copy_tensor_from_to(e, f)
        _t14: R.Tuple = R.vm.kill_object(e)
        ret_val: R.Tensor([4, 64], dtype="int32") = R.builtin.alloc_tensor(
            R.shape([4, 64]), R.dtype("int32"), R.prim_value(0)
        )
        _1: R.Tuple = R.vm.copy_tensor_from_to(f, ret_val)
        _t15: R.Tuple = R.vm.kill_object(f)
        _t3: R.Tuple = R.vm.kill_object(vtcm_obj)
        lv: R.Tensor([4, 64], dtype="int32") = ret_val
        return lv


@I.ir_module
class Module_2d:
    @T.prim_func
    def compute_add_in_vtcm(a: T.handle, b: T.handle, c: T.handle) -> None:
        m, n = T.int32(), T.int32()
        A = T.match_buffer(a, (m, n), "int32", scope="global.vtcm", axis_separators=[1])
        B = T.match_buffer(b, (m, n), "int32", scope="global.vtcm", axis_separators=[1])
        C = T.match_buffer(c, (m, n), "int32", scope="global.vtcm", axis_separators=[1])
        for ax0, ax1 in T.grid(m, n):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(C[v_ax0, v_ax1])
                C[v_ax0, v_ax1] = A[v_ax0, v_ax1] + B[v_ax0, v_ax1]

    @T.prim_func
    def compute_mul_in_vtcm(a: T.handle, b: T.handle, c: T.handle) -> None:
        m, n = T.int32(), T.int32()
        A = T.match_buffer(a, (m, n), "int32", scope="global.vtcm", axis_separators=[1])
        B = T.match_buffer(b, (m, n), "int32", scope="global.vtcm", axis_separators=[1])
        C = T.match_buffer(c, (m, n), "int32", scope="global.vtcm", axis_separators=[1])
        for ax0, ax1 in T.grid(m, n):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(A[v_ax0, v_ax1], B[v_ax0, v_ax1])
                T.writes(C[v_ax0, v_ax1])
                C[v_ax0, v_ax1] = A[v_ax0, v_ax1] * B[v_ax0, v_ax1]

    @R.function
    def main(
        x: R.Tensor((4, 64), "int32"),
        y: R.Tensor((4, 64), "int32"),
        z: R.Tensor((4, 64), "int32"),
    ) -> R.Tensor((4, 64), "int32"):
        cls = Module_2d
        vtcm_obj: R.Object = R.vm.alloc_storage(
            R.shape([4096]), runtime_device_index=0, dtype="uint8", storage_scope="global.vtcm"
        )
        global_obj: R.Object = R.vm.alloc_storage(
            R.shape([64]), runtime_device_index=0, dtype="uint8", storage_scope="global"
        )
        a: R.Tensor([4, 64], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                0,
                vtcm_obj,
                R.shape([768, 256, 2304, 3072]),
                R.shape([4, 64]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        __: R.Tuple = R.vm.copy_tensor_from_to(x, a)
        b: R.Tensor([4, 64], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                16,
                vtcm_obj,
                R.shape([1536, 1280, 3328, 2560]),
                R.shape([4, 64]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        _: R.Tuple = R.vm.copy_tensor_from_to(y, b)

        c: R.Tensor([4, 64], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                32,
                vtcm_obj,
                R.shape([512, 0, 2048, 3840]),
                R.shape([4, 64]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        ___: R.Tuple = cls.compute_add_in_vtcm(a, b, c)
        _t1: R.Tuple = R.vm.kill_object(a)
        _t2: R.Tuple = R.vm.kill_object(b)

        d: R.Tensor([4, 64], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                0,
                vtcm_obj,
                R.shape([1536, 1280, 3328, 2560]),
                R.shape([4, 64]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        ___1: R.Tuple = R.vm.copy_tensor_from_to(z, d)
        vtcm_2d_obj: R.Object = R.vm.alloc_storage(
            R.shape([4, 64]), runtime_device_index=0, dtype="int32", storage_scope="global.vtcm"
        )
        vtcm_2d_tensor: R.Tensor([4, 64], dtype="int32") = R.vm.alloc_tensor(
            vtcm_2d_obj, offset=0, shape=R.shape([4, 64]), dtype="int32"
        )
        ___2: R.Tuple = cls.compute_mul_in_vtcm(c, d, vtcm_2d_tensor)
        _t2: R.Tuple = R.vm.kill_object(c)
        _t12: R.Tuple = R.vm.kill_object(d)

        e: R.Tensor([4, 64], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                16,
                vtcm_obj,
                R.shape([768, 256, 2304, 3072]),
                R.shape([4, 64]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        _t21: R.Tuple = R.vm.copy_tensor_from_to(vtcm_2d_tensor, e)
        _t22: R.Tuple = R.vm.kill_object(vtcm_2d_tensor)
        _t23: R.Tuple = R.vm.kill_object(vtcm_2d_obj)
        f: R.Tensor([4, 64], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                32,
                vtcm_obj,
                R.shape([1536, 1280, 3328, 2560]),
                R.shape([4, 64]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        _t13: R.Tuple = R.vm.copy_tensor_from_to(e, f)
        _t14: R.Tuple = R.vm.kill_object(e)
        ret_val: R.Tensor([4, 64], dtype="int32") = R.builtin.alloc_tensor(
            R.shape([4, 64]), R.dtype("int32"), R.prim_value(0)
        )
        _1: R.Tuple = R.vm.copy_tensor_from_to(f, ret_val)
        _t15: R.Tuple = R.vm.kill_object(f)
        _t16: R.Tuple = R.vm.kill_object(vtcm_obj)
        _t17: R.Tuple = R.vm.kill_object(global_obj)
        lv: R.Tensor([4, 64], dtype="int32") = ret_val
        return lv


@I.ir_module
class NDLogicalShapesModule:
    @T.prim_func
    def compute_add_in_vtcm(a: T.handle, b: T.handle, c: T.handle) -> None:
        m, n, o, p = T.int32(), T.int32(), T.int32(), T.int32()
        A = T.match_buffer(a, (m, n, o, p), "int32", scope="global.vtcm", axis_separators=[2])
        B = T.match_buffer(b, (m, n, o, p), "int32", scope="global.vtcm", axis_separators=[2])
        C = T.match_buffer(c, (m, n, o, p), "int32", scope="global.vtcm", axis_separators=[2])
        for ax0, ax1, ax2, ax3 in T.grid(m, n, o, p):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(C[v_ax0, v_ax1, v_ax2, v_ax3])
                C[v_ax0, v_ax1, v_ax2, v_ax3] = (
                    A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]
                )

    @T.prim_func
    def compute_mul_in_vtcm(a: T.handle, b: T.handle, c: T.handle) -> None:
        m, n, o, p = T.int32(), T.int32(), T.int32(), T.int32()
        A = T.match_buffer(a, (m, n, o, p), "int32", scope="global.vtcm", axis_separators=[2])
        B = T.match_buffer(b, (m, n, o, p), "int32", scope="global.vtcm", axis_separators=[2])
        C = T.match_buffer(c, (m, n, o, p), "int32", scope="global.vtcm", axis_separators=[2])
        for ax0, ax1, ax2, ax3 in T.grid(m, n, o, p):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
                T.writes(C[v_ax0, v_ax1, v_ax2, v_ax3])
                C[v_ax0, v_ax1, v_ax2, v_ax3] = (
                    A[v_ax0, v_ax1, v_ax2, v_ax3] * B[v_ax0, v_ax1, v_ax2, v_ax3]
                )

    @R.function
    def main(
        x: R.Tensor((2, 2, 8, 8), "int32"),
        y: R.Tensor((2, 2, 8, 8), "int32"),
        z: R.Tensor((2, 2, 8, 8), "int32"),
    ) -> R.Tensor((2, 2, 8, 8), "int32"):
        cls = NDLogicalShapesModule
        vtcm_obj: R.Object = R.vm.alloc_storage(
            R.shape([4096]), runtime_device_index=0, dtype="uint8", storage_scope="global.vtcm"
        )
        global_obj: R.Object = R.vm.alloc_storage(
            R.shape([64]), runtime_device_index=0, dtype="uint8", storage_scope="global"
        )
        a: R.Tensor([2, 2, 8, 8], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                0,
                vtcm_obj,
                R.shape([768, 256, 2304, 3072]),
                R.shape([2, 2, 8, 8]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        __: R.Tuple = R.vm.copy_tensor_from_to(x, a)
        b: R.Tensor([2, 2, 8, 8], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                16,
                vtcm_obj,
                R.shape([1536, 1280, 3328, 2560]),
                R.shape([2, 2, 8, 8]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        _: R.Tuple = R.vm.copy_tensor_from_to(y, b)
        c: R.Tensor([2, 2, 8, 8], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                32,
                vtcm_obj,
                R.shape([512, 0, 2048, 3840]),
                R.shape([2, 2, 8, 8]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        ___: R.Tuple = cls.compute_add_in_vtcm(a, b, c)
        _t1: R.Tuple = R.vm.kill_object(a)
        _t2: R.Tuple = R.vm.kill_object(b)
        d: R.Tensor([2, 2, 8, 8], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                0,
                vtcm_obj,
                R.shape([1536, 1280, 3328, 2560]),
                R.shape([2, 2, 8, 8]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        _: R.Tuple = R.vm.copy_tensor_from_to(z, d)
        e: R.Tensor([2, 2, 8, 8], dtype="int32") = R.call_builtin_with_ctx(
            "vm.builtin.hexagon.alloc_discontiguous_tensor",
            [
                global_obj,
                16,
                vtcm_obj,
                R.shape([768, 256, 2304, 3072]),
                R.shape([2, 2, 8, 8]),
                R.shape([4, 64]),
                "int32",
            ],
            sinfo_args=[],
        )
        ___2: R.Tuple = cls.compute_mul_in_vtcm(c, d, e)
        _t2: R.Tuple = R.vm.kill_object(c)
        _t12: R.Tuple = R.vm.kill_object(d)
        ret_val: R.Tensor([2, 2, 8, 8], dtype="int32") = R.builtin.alloc_tensor(
            R.shape([2, 2, 8, 8]), R.dtype("int32"), R.prim_value(0)
        )
        _1: R.Tuple = R.vm.copy_tensor_from_to(e, ret_val)
        _t14: R.Tuple = R.vm.kill_object(e)
        _t16: R.Tuple = R.vm.kill_object(vtcm_obj)
        _t17: R.Tuple = R.vm.kill_object(global_obj)
        lv: R.Tensor([2, 2, 8, 8], dtype="int32") = ret_val
        return lv


class TestVTCMAlloc:
    """Tests for VTCM Alloc, Compute and Copy"""

    mode = tvm.testing.parameter("bytecode", "compiled")
    (module, in_shape) = tvm.testing.parameters(
        (Module_2d, (4, 64)),
        (Module, (4, 64)),
        (NDLogicalShapesModule, (2, 2, 8, 8)),
    )

    @tvm.testing.requires_hexagon
    def test_vtcm_alloc_compute(self, hexagon_launcher, mode, module, in_shape):
        target_hexagon = tvm.target.hexagon("v69")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
        with tvm.transform.PassContext(opt_level=3, config=[], instruments=[]):
            ex = relax.build(mod=module, target=target, exec_mode=mode)

        with hexagon_launcher.create_session() as session:
            dev = session.device
            input_arg0_data = np.random.randint(0, 9, size=in_shape, dtype="int32")
            input_arg1_data = np.random.randint(0, 9, size=in_shape, dtype="int32")
            input_arg2_data = np.random.randint(0, 9, size=in_shape, dtype="int32")
            output_data = np.multiply(np.add(input_arg0_data, input_arg1_data), input_arg2_data)
            vm_mod = session.get_executor_from_factory(ex)
            vm_rt = relax.VirtualMachine(
                vm_mod, dev, "naive"
            )  # Use naive allocator to exercise VTCM allocation in relax
            data0 = tvm.nd.array(input_arg0_data, dev)
            data1 = tvm.nd.array(input_arg1_data, dev)
            data2 = tvm.nd.array(input_arg2_data, dev)
            vm_rt.set_input("main", data0, data1, data2)
            vm_rt.invoke_stateful("main")
            hexagon_output = vm_rt.get_outputs("main").numpy()
            tvm.testing.assert_allclose(output_data, hexagon_output)
