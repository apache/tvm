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
import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R, tir as T


def test_basic():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add(rxplaceholder: T.Buffer(T.int64(8), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer(T.int64(8), "float32")):
            T.evaluate(0)

        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer(T.int64(8), "float32")):
            T.evaluate(0)

        @T.prim_func
        def relu(rxplaceholder: T.Buffer(T.int64(8), "float32"), compute: T.Buffer(T.int64(8), "float32")):
            T.evaluate(0)

        @T.prim_func
        def log(rxplaceholder: T.Buffer(T.int64(10), "float32"), compute: T.Buffer(T.int64(10), "float32")):
            T.evaluate(0)

        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            T.evaluate(0)

        @T.prim_func
        def pad(rxplaceholder: T.Buffer(T.int64(8), "float32"), PadInput: T.Buffer(T.int64(10), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            # we expected RemovePurityChecking to have been invoked first
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 4), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, 4]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = cls.exp(x, alloc)
            lv: R.Tensor((2, 4), dtype="float32") = alloc
            lv1: R.Tensor((8,), dtype="float32") = R.reshape(lv, (8,))
            alloc1: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = cls.relu(lv1, alloc1)
            lv2: R.Tensor((8,), dtype="float32") = alloc1
            alloc2: R.Tensor((8,), dtype="float32") = R.builtin.alloc_tensor(R.shape([8]), dtype="float32", runtime_device_index=0)
            _2: R.Tuple() = cls.add(lv2, R.const(1, "float32"), alloc2)
            lv3: R.Tensor((8,), dtype="float32") = alloc2
            alloc3: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _3: R.Tuple() = cls.pad(lv3, alloc3)
            lv4: R.Tensor((10,), dtype="float32") = alloc3
            alloc4: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _4: R.Tuple() = cls.log(lv4, alloc4)
            gv: R.Tensor((10,), dtype="float32") = alloc4
            return gv

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add(rxplaceholder: T.Buffer(T.int64(8), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer(T.int64(8), "float32")):
            T.evaluate(0)

        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer(T.int64(8), "float32")):
            T.evaluate(0)

        @T.prim_func
        def relu(rxplaceholder: T.Buffer(T.int64(8), "float32"), compute: T.Buffer(T.int64(8), "float32")):
            T.evaluate(0)

        @T.prim_func
        def log(rxplaceholder: T.Buffer(T.int64(10), "float32"), compute: T.Buffer(T.int64(10), "float32")):
            T.evaluate(0)

        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            T.evaluate(0)

        @T.prim_func
        def pad(rxplaceholder: T.Buffer(T.int64(8), "float32"), PadInput: T.Buffer(T.int64(10), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), virtual_device_index=0, storage_scope="global", dtype="float32")
            alloc: R.Tensor((2, 4), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([2, 4]), dtype="float32")
            _: R.Tuple() = cls.exp(x, alloc)
            lv: R.Tensor((2, 4), dtype="float32") = alloc
            lv1: R.Tensor((8,), dtype="float32") = R.reshape(lv, (8,))
            storage1: R.Object = R.memory.alloc_storage(R.shape([40]), virtual_device_index=0, storage_scope="global", dtype="float32")
            alloc1: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([8]), dtype="float32")
            _1: R.Tuple() = cls.relu(lv1, alloc1)
            _2: R.Tuple() = R.memory.kill_tensor(alloc)
            _3: R.Tuple() = R.memory.kill_tensor(lv1)
            lv2: R.Tensor((8,), dtype="float32") = alloc1
            alloc2: R.Tensor((8,), dtype="float32") = R.memory.alloc_tensor(storage, 0, R.shape([8]), dtype="float32")
            _4: R.Tuple() = cls.add(lv2, R.const(1, "float32"), alloc2)
            _5: R.Tuple() = R.memory.kill_tensor(alloc1)
            lv3: R.Tensor((8,), dtype="float32") = alloc2
            alloc3: R.Tensor((10,), dtype="float32") = R.memory.alloc_tensor(storage1, 0, R.shape([10]), dtype="float32")
            _6: R.Tuple() = cls.pad(lv3, alloc3)
            _7: R.Tuple() = R.memory.kill_tensor(alloc2)
            lv4: R.Tensor((10,), dtype="float32") = alloc3
            alloc4: R.Tensor((10,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _8: R.Tuple() = cls.log(lv4, alloc4)
            _9: R.Tuple() = R.memory.kill_tensor(alloc3)
            gv5: R.Tensor((10,), dtype="float32") = alloc4
            _11: R.Tuple() = R.memory.kill_storage(storage)
            _10: R.Tuple() = R.memory.kill_storage(storage1)
            return gv5

    @I.ir_module
    class ExpectedLowered:
        @T.prim_func
        def add(rxplaceholder: T.Buffer((T.int64(8),), "float32"), rxplaceholder_1: T.Buffer((), "float32"), T_add: T.Buffer((T.int64(8),), "float32")):
            T.evaluate(0)

        @T.prim_func
        def exp(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), compute: T.Buffer((T.int64(2), T.int64(4)), "float32")):
            T.evaluate(0)

        @T.prim_func
        def log(rxplaceholder: T.Buffer((T.int64(10),), "float32"), compute: T.Buffer((T.int64(10),), "float32")):
            T.evaluate(0)

        @T.prim_func
        def pad(rxplaceholder: T.Buffer((T.int64(8),), "float32"), PadInput: T.Buffer((T.int64(10),), "float32")):
            T.evaluate(0)

        @T.prim_func
        def relu(rxplaceholder: T.Buffer((T.int64(8),), "float32"), compute: T.Buffer((T.int64(8),), "float32")):
            T.evaluate(0)

        @T.prim_func
        def reshape(rxplaceholder: T.Buffer((T.int64(2), T.int64(4)), "float32"), T_reshape: T.Buffer((T.int64(8),), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 4), dtype="float32")) -> R.Tensor((10,), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = ExpectedLowered
            storage: R.Object = R.vm.alloc_storage(R.shape([32]), R.prim_value(0), R.dtype("uint8"))
            alloc: R.Tensor((2, 4), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([2, 4]), R.dtype("float32"))
            _: R.Tuple = cls.exp(x, alloc)
            lv: R.Tensor((2, 4), dtype="float32") = alloc
            lv1: R.Tensor((8,), dtype="float32") = R.call_packed("vm.builtin.reshape", lv, R.shape([8]), sinfo_args=(R.Tensor((8,), dtype="float32"),))
            storage1: R.Object = R.vm.alloc_storage(R.shape([40]), R.prim_value(0), R.dtype("uint8"))
            alloc1: R.Tensor((8,), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([8]), R.dtype("float32"))
            _1: R.Tuple = cls.relu(lv1, alloc1)
            __1: R.Tuple = R.vm.kill_object(alloc)
            _1_1: R.Tuple = R.vm.kill_object(lv1)
            lv2: R.Tensor((8,), dtype="float32") = alloc1
            alloc2: R.Tensor((8,), dtype="float32") = R.vm.alloc_tensor(storage, R.prim_value(0), R.shape([8]), R.dtype("float32"))
            _2: R.Tuple = cls.add(lv2, R.const(1, "float32"), alloc2)
            _2_1: R.Tuple = R.vm.kill_object(alloc1)
            lv3: R.Tensor((8,), dtype="float32") = alloc2
            alloc3: R.Tensor((10,), dtype="float32") = R.vm.alloc_tensor(storage1, R.prim_value(0), R.shape([10]), R.dtype("float32"))
            _3: R.Tuple = cls.pad(lv3, alloc3)
            _3_1: R.Tuple = R.vm.kill_object(alloc2)
            lv4: R.Tensor((10,), dtype="float32") = alloc3
            storage_1: R.Object = R.vm.alloc_storage(R.shape([40]), R.prim_value(0), R.dtype("uint8"))
            alloc4: R.Tensor((10,), dtype="float32") = R.vm.alloc_tensor(storage_1, R.prim_value(0), R.shape([10]), R.dtype("float32"))
            _4: R.Tuple = cls.log(lv4, alloc4)
            _4_1: R.Tuple = R.vm.kill_object(alloc3)
            gv: R.Tensor((10,), dtype="float32") = alloc4
            _5: R.Tuple = R.vm.kill_object(storage)
            _6: R.Tuple = R.vm.kill_object(storage1)
            return gv
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)
    mod = relax.transform.VMBuiltinLower()(mod)
    tvm.ir.assert_structural_equal(mod, ExpectedLowered)


def test_different_dtype():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @T.prim_func
        def add1(
            A: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="int32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="int32", runtime_device_index=0
            )
            _1: R.Tuple() = cls.add1(y, y, alloc1)
            gv1: R.Tensor((2, 3), dtype="int32") = alloc1
            return x

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @T.prim_func
        def add1(
            A: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            _1: R.Tuple() = R.memory.kill_tensor(alloc)
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            storage1: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="int32"
            )
            alloc1: R.Tensor((2, 3), dtype="int32") = R.memory.alloc_tensor(
                storage1, 0, R.shape([2, 3]), dtype="int32"
            )
            _2: R.Tuple() = cls.add1(y, y, alloc1)
            _3: R.Tuple() = R.memory.kill_tensor(alloc1)
            gv12: R.Tensor((2, 3), dtype="int32") = alloc1
            _5: R.Tuple() = R.memory.kill_storage(storage)
            _4: R.Tuple() = R.memory.kill_storage(storage1)
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_dtype_bool():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add1(
            A: T.Buffer((T.int64(2), T.int64(3)), "bool"),
            B: T.Buffer((T.int64(2), T.int64(3)), "bool"),
            C: T.Buffer((T.int64(2), T.int64(3)), "bool"),
        ):
            T.evaluate(0)

        @R.function
        def main(y: R.Tensor((2, 3), dtype="bool")) -> R.Tensor((2, 3), dtype="bool"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="bool") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="bool", runtime_device_index=0
            )
            _1: R.Tuple() = cls.add1(y, y, alloc)
            gv1: R.Tensor((2, 3), dtype="bool") = alloc
            return y

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add1(
            A: T.Buffer((T.int64(2), T.int64(3)), "bool"),
            B: T.Buffer((T.int64(2), T.int64(3)), "bool"),
            C: T.Buffer((T.int64(2), T.int64(3)), "bool"),
        ):
            T.evaluate(0)

        @R.function
        def main(y: R.Tensor((2, 3), dtype="bool")) -> R.Tensor((2, 3), dtype="bool"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(
                R.shape([6]), virtual_device_index=0, storage_scope="global", dtype="bool"
            )
            alloc: R.Tensor((2, 3), dtype="bool") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="bool"
            )
            _2: R.Tuple() = cls.add1(y, y, alloc)
            _3: R.Tuple() = R.memory.kill_tensor(alloc)
            gv12: R.Tensor((2, 3), dtype="bool") = alloc
            _4: R.Tuple() = R.memory.kill_storage(storage)
            return y

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_same_dtype():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1: R.Tuple() = cls.add(y, y, alloc1)
            gv1: R.Tensor((2, 3), dtype="float32") = alloc1
            return x

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def add(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            _1: R.Tuple() = R.memory.kill_tensor(alloc)
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _2: R.Tuple() = cls.add(y, y, alloc1)
            _3: R.Tuple() = R.memory.kill_tensor(alloc1)
            gv12: R.Tensor((2, 3), dtype="float32") = alloc1
            _4: R.Tuple() = R.memory.kill_storage(storage)
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_if_cond():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def all_less_than_zero(A: T.Buffer((2, 3), "float32"), B: T.Buffer((), "bool")):
            T.evaluate(0)

        @T.prim_func
        def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((), dtype="bool") = R.builtin.alloc_tensor(
                R.shape([]), dtype="bool", runtime_device_index=0
            )
            _: R.Tuple() = cls.all_less_than_zero(x, alloc)
            x1: R.Tensor((), dtype="bool") = alloc
            if x1:
                y: R.Tensor((2, 3), dtype="float32") = x
            else:
                alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([2, 3]), dtype="float32", runtime_device_index=0
                )
                _1: R.Tuple() = cls.exp(x, alloc1)
                gv3: R.Tensor((2, 3), dtype="float32") = alloc1
                y: R.Tensor((2, 3), dtype="float32") = gv3
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Module)


def test_if_then_else():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = cls.exp(x, alloc)
            y: R.Tensor((2, 3), dtype="float32") = alloc
            if cond:
                z: R.Tensor((2, 3), dtype="float32") = y
            else:
                z: R.Tensor((2, 3), dtype="float32") = y
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Module)


def test_cross_block_use():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(
            cond: R.Tensor((), dtype="bool"), x: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = cls.exp(x, alloc)
            y: R.Tensor((2, 3), dtype="float32") = alloc
            if cond:
                alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([2, 3]), dtype="float32", runtime_device_index=0
                )
                _1: R.Tuple() = cls.exp(y, alloc1)
                y2: R.Tensor((2, 3), dtype="float32") = alloc1
                z: R.Tensor((2, 3), dtype="float32") = y2
            else:
                alloc2: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                    R.shape([2, 3]), dtype="float32", runtime_device_index=0
                )
                _2: R.Tuple() = cls.exp(y, alloc2)
                y2: R.Tensor((2, 3), dtype="float32") = alloc2
                z: R.Tensor((2, 3), dtype="float32") = y2
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Module)


def test_nested_tuple():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            cls = Module
            _: R.Tuple() = cls.exp(x, alloc)
            y1: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1: R.Tuple() = cls.exp(x, alloc1)
            y2: R.Tensor((2, 3), dtype="float32") = alloc1
            alloc2: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _2: R.Tuple() = cls.exp(x, alloc2)
            y3: R.Tensor((2, 3), dtype="float32") = alloc2
            t: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = (
                y1,
                y2,
            )
            nt: R.Tuple(
                R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")),
                R.Tensor((2, 3), dtype="float32"),
            ) = (t, y3)
            nt0: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = nt[
                0
            ]
            y1_: R.Tensor((2, 3), dtype="float32") = nt0[0]
            y2_: R.Tensor((2, 3), dtype="float32") = nt0[1]
            y3_: R.Tensor((2, 3), dtype="float32") = nt[1]
            alloc3: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _3: R.Tuple() = cls.exp(y1_, alloc3)
            z1: R.Tensor((2, 3), dtype="float32") = alloc3
            alloc4: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _4: R.Tuple() = cls.exp(y2_, alloc4)
            z2: R.Tensor((2, 3), dtype="float32") = alloc4
            alloc5: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _5: R.Tuple() = cls.exp(y3_, alloc5)
            z3: R.Tensor((2, 3), dtype="float32") = alloc5
            return x

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def exp(A: T.Buffer((2, 3), "float32"), B: T.Buffer((2, 3), "float32")):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = cls.exp(x, alloc)
            y1: R.Tensor((2, 3), dtype="float32") = alloc
            storage1: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc1: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage1, 0, R.shape([2, 3]), dtype="float32"
            )
            _1: R.Tuple() = cls.exp(x, alloc1)
            y2: R.Tensor((2, 3), dtype="float32") = alloc1
            storage2: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc2: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage2, 0, R.shape([2, 3]), dtype="float32"
            )
            _2: R.Tuple() = cls.exp(x, alloc2)
            y3: R.Tensor((2, 3), dtype="float32") = alloc2
            t: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = (
                y1,
                y2,
            )
            nt: R.Tuple(
                R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")),
                R.Tensor((2, 3), dtype="float32"),
            ) = (t, y3)
            nt0: R.Tuple(R.Tensor((2, 3), dtype="float32"), R.Tensor((2, 3), dtype="float32")) = nt[
                0
            ]
            y1_: R.Tensor((2, 3), dtype="float32") = nt0[0]
            y2_: R.Tensor((2, 3), dtype="float32") = nt0[1]
            y3_: R.Tensor((2, 3), dtype="float32") = nt[1]
            storage3: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc3: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage3, 0, R.shape([2, 3]), dtype="float32"
            )
            _3: R.Tuple() = cls.exp(y1_, alloc3)
            _4: R.Tuple() = R.memory.kill_tensor(alloc)
            _11: R.Tuple() = R.memory.kill_tensor(alloc3)
            z1: R.Tensor((2, 3), dtype="float32") = alloc3
            alloc4: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _41: R.Tuple() = cls.exp(y2_, alloc4)
            _21: R.Tuple() = R.memory.kill_tensor(alloc1)
            _31: R.Tuple() = R.memory.kill_tensor(alloc4)
            z2: R.Tensor((2, 3), dtype="float32") = alloc4
            alloc5: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage3, 0, R.shape([2, 3]), dtype="float32"
            )
            _5: R.Tuple() = cls.exp(y3_, alloc5)
            _42: R.Tuple() = R.memory.kill_tensor(alloc2)
            _51: R.Tuple() = R.memory.kill_tensor(alloc5)
            z3: R.Tensor((2, 3), dtype="float32") = alloc5
            _9: R.Tuple() = R.memory.kill_storage(storage)
            _7: R.Tuple() = R.memory.kill_storage(storage1)
            _8: R.Tuple() = R.memory.kill_storage(storage2)
            _6: R.Tuple() = R.memory.kill_storage(storage3)
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_call_func_other_than_primfunc():
    @tvm.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            R.func_attr({"relax.force_pure": True})
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _ = R.add(x, alloc)
            y: R.Tensor((2, 3), dtype="float32") = alloc
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Module)


def test_call_packed_external_func():
    @I.ir_module
    class Module:
        @R.function(pure=False)
        def main(x: R.Tensor((2, 3), "float32")):
            # the extern func may or may not be pure, depends on what we're calling
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _ = R.call_packed("extern_func", x, alloc, sinfo_args=[R.Tuple()])
            y: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1 = R.call_packed("extern_func", y, alloc1, sinfo_args=[R.Tuple()])
            z: R.Tensor((2, 3), dtype="float32") = alloc1
            return z

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(x: R.Tensor((2, 3), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
            storage: R.Object = R.memory.alloc_storage(
                R.shape([24]), R.prim_value(0), R.str("global"), R.dtype("float32")
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, R.prim_value(0), R.shape([2, 3]), R.dtype("float32")
            )
            _: R.Tuple = R.call_packed("extern_func", x, alloc, sinfo_args=(R.Tuple(),))
            y: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), R.dtype("float32"), R.prim_value(0)
            )
            _1: R.Tuple = R.call_packed("extern_func", y, alloc1, sinfo_args=(R.Tuple(),))
            _2: R.Tuple = R.memory.kill_tensor(alloc)
            z: R.Tensor((2, 3), dtype="float32") = alloc1
            _3: R.Tuple = R.memory.kill_storage(storage)
            return z

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_symbolic_shape():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def exp(var_A: T.handle, var_B: T.handle):
            m = T.int64()
            n = T.int64()
            A = T.match_buffer(var_A, (m, n), "float32")
            B = T.match_buffer(var_B, (m, n), "float32")
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor(("m", "n"), "float32")):
            R.func_attr({"relax.force_pure": True})
            m = T.int64()
            n = T.int64()
            alloc: R.Tensor((m, n), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([m, n]), dtype="float32", runtime_device_index=0
            )
            _ = Module.exp(x, alloc)
            y: R.Tensor((m, n), dtype="float32") = alloc
            return x

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Module)


def test_zero_reference():
    @tvm.script.ir_module
    class Module:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            R.func_attr({"relax.force_pure": True})
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            return x

    @tvm.script.ir_module
    class Expected:
        @R.function
        def main(x: R.Tensor((2, 3), "float32")):
            R.func_attr({"relax.force_pure": True})
            storage: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = R.memory.kill_storage(storage)
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_reshape_param():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add(
            A: T.Buffer((T.int64(2), T.int64(25), T.int64(2)), "float32"),
            B: T.Buffer((T.int64(2), T.int64(25), T.int64(2)), "float32"),
            C: T.Buffer((T.int64(2), T.int64(25), T.int64(2)), "float32"),
        ):
            T.evaluate(0)

        @R.function
        def main(
            x: R.Tensor((2, 50), dtype="float32"), y: R.Tensor((100,), dtype="float32")
        ) -> R.Tensor((2, 25, 2), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            lv: R.Tensor((2, 25, 2), dtype="float32") = R.reshape(x, (2, 25, 2))
            lv1: R.Tensor((2, 25, 2), dtype="float32") = R.reshape(y, (2, 25, 2))
            alloc: R.Tensor((2, 25, 2), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 25, 2]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = Module.add(lv, lv1, alloc)
            gv: R.Tensor((2, 25, 2), dtype="float32") = alloc
            return gv

    # The pass does no change.
    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Module)


def test_multiple_functions():
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @T.prim_func
        def add1(
            A: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def func1(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="int32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="int32", runtime_device_index=0
            )
            _1: R.Tuple() = cls.add1(y, y, alloc1)
            gv1: R.Tensor((2, 3), dtype="int32") = alloc1
            return x

        @R.function
        def func2(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            gv: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.builtin.alloc_tensor(
                R.shape([2, 3]), dtype="float32", runtime_device_index=0
            )
            _1: R.Tuple() = cls.add(y, y, alloc1)
            gv1: R.Tensor((2, 3), dtype="float32") = alloc1
            return x

    @I.ir_module
    class Expected:
        @T.prim_func
        def add(
            A: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "float32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "float32"),
        ):
            T.evaluate(0)

        @T.prim_func
        def add1(
            A: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            B: T.Buffer((T.int64(2), T.int64(3)), "int32"),
            C: T.Buffer((T.int64(2), T.int64(3)), "int32"),
        ):
            T.evaluate(0)

        @R.function
        def func1(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="int32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            _1: R.Tuple() = R.memory.kill_tensor(alloc)
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            storage1: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="int32"
            )
            alloc1: R.Tensor((2, 3), dtype="int32") = R.memory.alloc_tensor(
                storage1, 0, R.shape([2, 3]), dtype="int32"
            )
            _2: R.Tuple() = cls.add1(y, y, alloc1)
            _3: R.Tuple() = R.memory.kill_tensor(alloc1)
            gv12: R.Tensor((2, 3), dtype="int32") = alloc1
            _5: R.Tuple() = R.memory.kill_storage(storage)
            _4: R.Tuple() = R.memory.kill_storage(storage1)
            return x

        @R.function
        def func2(
            x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")
        ) -> R.Tensor((2, 3), dtype="float32"):
            R.func_attr({"relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(
                R.shape([24]), virtual_device_index=0, storage_scope="global", dtype="float32"
            )
            alloc: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _: R.Tuple() = cls.add(x, x, alloc)
            _1: R.Tuple() = R.memory.kill_tensor(alloc)
            gv1: R.Tensor((2, 3), dtype="float32") = alloc
            alloc1: R.Tensor((2, 3), dtype="float32") = R.memory.alloc_tensor(
                storage, 0, R.shape([2, 3]), dtype="float32"
            )
            _2: R.Tuple() = cls.add(y, y, alloc1)
            _3: R.Tuple() = R.memory.kill_tensor(alloc1)
            gv12: R.Tensor((2, 3), dtype="float32") = alloc1
            _4: R.Tuple() = R.memory.kill_storage(storage)
            return x

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tir_var_upper_bound():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @T.prim_func
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @T.prim_func
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @T.prim_func
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @T.prim_func
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @T.prim_func
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, "n"), dtype="float32")) -> R.Tensor(("2 * n + 2",), dtype="float32"):
            R.func_attr({"tir_var_upper_bound": {"n": 4}, "relax.force_pure": True})
            n = T.int64()
            cls = Module
            alloc: R.Tensor((2, n), dtype="float32") = R.builtin.alloc_tensor(R.shape([2, n]), dtype="float32", runtime_device_index=0)
            _: R.Tuple() = cls.exp(x, alloc)
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, (2 * n,))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _1: R.Tuple() = cls.relu(lv1, alloc1)
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n]), dtype="float32", runtime_device_index=0)
            _2: R.Tuple() = cls.add(lv2, R.const(1, "float32"), alloc2)
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([2 * n + 2]), dtype="float32", runtime_device_index=0)
            _3: R.Tuple() = cls.pad(lv3, alloc3)
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), dtype="float32", runtime_device_index=0)
            _4: R.Tuple() = cls.log(lv4, alloc4)
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            return gv

    @I.ir_module
    class Expected:
        @T.prim_func
        def add(rxplaceholder: T.handle, rxplaceholder_1: T.handle, T_add: T.handle):
            T.evaluate(0)

        @T.prim_func
        def exp(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @T.prim_func
        def log(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @T.prim_func
        def pad(rxplaceholder: T.handle, PadInput: T.handle):
            T.evaluate(0)

        @T.prim_func
        def relu(rxplaceholder: T.handle, compute: T.handle):
            T.evaluate(0)

        @T.prim_func
        def reshape(rxplaceholder: T.handle, T_reshape: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor((2, "n"), dtype="float32")) -> R.Tensor(("2 * n + 2",), dtype="float32"):
            n = T.int64()
            R.func_attr({"tir_var_upper_bound": {"n": 4}, "relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(R.shape([32]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((2, n), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2, n]), R.dtype("float32"))
            _: R.Tuple = cls.exp(x, alloc)
            lv: R.Tensor((2, n), dtype="float32") = alloc
            lv1: R.Tensor((2 * n,), dtype="float32") = R.reshape(lv, R.shape([2 * n]))
            storage1: R.Object = R.memory.alloc_storage(R.shape([40]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _1: R.Tuple = cls.relu(lv1, alloc1)
            __1: R.Tuple = R.memory.kill_tensor(alloc)
            _1_1: R.Tuple = R.memory.kill_tensor(lv1)
            lv2: R.Tensor((2 * n,), dtype="float32") = alloc1
            alloc2: R.Tensor((2 * n,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([2 * n]), R.dtype("float32"))
            _2: R.Tuple = cls.add(lv2, R.const(1, "float32"), alloc2)
            _2_1: R.Tuple = R.memory.kill_tensor(alloc1)
            lv3: R.Tensor((2 * n,), dtype="float32") = alloc2
            alloc3: R.Tensor((2 * n + 2,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([2 * n + 2]), R.dtype("float32"))
            _3: R.Tuple = cls.pad(lv3, alloc3)
            _3_1: R.Tuple = R.memory.kill_tensor(alloc2)
            lv4: R.Tensor((2 * n + 2,), dtype="float32") = alloc3
            alloc4: R.Tensor((2 * n + 2,), dtype="float32") = R.builtin.alloc_tensor(R.shape([10]), R.dtype("float32"), R.prim_value(0))
            _4: R.Tuple = cls.log(lv4, alloc4)
            _4_1: R.Tuple = R.memory.kill_tensor(alloc3)
            gv: R.Tensor((2 * n + 2,), dtype="float32") = alloc4
            _5: R.Tuple = R.memory.kill_storage(storage)
            _6: R.Tuple = R.memory.kill_storage(storage1)
            return gv
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_tir_var_decreasing_monotone():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor(("n", "m", "T.max(n - m, 1)"), dtype="float32")) -> R.Tensor(("n", "m", "T.max(n - m, 1)"), dtype="float32"):
            n = T.int64()
            m = T.int64()
            R.func_attr({"tir_var_upper_bound": {"m": 5, "n": 20}, "relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _: R.Tuple = cls.tir_exp(x, alloc)
            y: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc
            alloc1: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = cls.tir_exp(y, alloc1)
            z: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc1
            alloc2: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = cls.tir_exp(z, alloc2)
            r: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc2
            return r

    @I.ir_module
    class Expected:
        @T.prim_func
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(x: R.Tensor(("n", "m", "T.max(n - m, 1)"), dtype="float32")) -> R.Tensor(("n", "m", "T.max(n - m, 1)"), dtype="float32"):
            n = T.int64()
            m = T.int64()
            R.func_attr({"tir_var_upper_bound": {"m": 5, "n": 20}, "relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(R.shape([8000]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"))
            _: R.Tuple = cls.tir_exp(x, alloc)
            y: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc
            storage1: R.Object = R.memory.alloc_storage(R.shape([8000]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"))
            _1: R.Tuple = cls.tir_exp(y, alloc1)
            __1: R.Tuple = R.memory.kill_tensor(alloc)
            z: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc1
            alloc2: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = R.builtin.alloc_tensor(R.shape([n, m, T.max(n - m, 1)]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = cls.tir_exp(z, alloc2)
            _1_1: R.Tuple = R.memory.kill_tensor(alloc1)
            r: R.Tensor((n, m, T.max(n - m, 1)), dtype="float32") = alloc2
            _2_1: R.Tuple = R.memory.kill_storage(storage)
            _3: R.Tuple = R.memory.kill_storage(storage1)
            return r
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


def test_call_tir_dyn():
    # fmt: off
    @I.ir_module
    class Module:
        @T.prim_func
        def tir_full(var_full: T.handle, n: T.int64):
            T.evaluate(0)

        @T.prim_func
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape(["n"])) -> R.Tensor(("n",), dtype="float32"):
            n = T.int64()
            R.func_attr({"tir_var_upper_bound": {"n": 20}, "relax.force_pure": True})
            cls = Module
            alloc: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n])))
            full: R.Tensor((n,), dtype="float32") = alloc
            alloc1: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _1: R.Tuple = cls.tir_exp(full, alloc1)
            lv2: R.Tensor((n,), dtype="float32") = alloc1
            alloc2: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = cls.tir_exp(lv2, alloc2)
            lv3: R.Tensor((n,), dtype="float32") = alloc2
            return lv3

    @I.ir_module
    class Expected:
        @T.prim_func
        def tir_exp(var_rxplaceholder: T.handle, var_compute: T.handle):
            T.evaluate(0)

        @T.prim_func
        def tir_full(var_full: T.handle, n: T.int64):
            T.evaluate(0)

        @R.function
        def main(s: R.Shape(["n"])) -> R.Tensor(("n",), dtype="float32"):
            n = T.int64()
            R.func_attr({"tir_var_upper_bound": {"n": 20}, "relax.force_pure": True})
            cls = Expected
            storage: R.Object = R.memory.alloc_storage(R.shape([80]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc: R.Tensor((n,), dtype="float32") = R.memory.alloc_tensor(storage, R.prim_value(0), R.shape([n]), R.dtype("float32"))
            _: R.Tuple = R.vm.call_tir_dyn(cls.tir_full, (alloc, R.shape([n])))
            full: R.Tensor((n,), dtype="float32") = alloc
            storage1: R.Object = R.memory.alloc_storage(R.shape([80]), R.prim_value(0), R.str("global"), R.dtype("float32"))
            alloc1: R.Tensor((n,), dtype="float32") = R.memory.alloc_tensor(storage1, R.prim_value(0), R.shape([n]), R.dtype("float32"))
            _1: R.Tuple = cls.tir_exp(full, alloc1)
            __1: R.Tuple = R.memory.kill_tensor(alloc)
            lv2: R.Tensor((n,), dtype="float32") = alloc1
            alloc2: R.Tensor((n,), dtype="float32") = R.builtin.alloc_tensor(R.shape([n]), R.dtype("float32"), R.prim_value(0))
            _2: R.Tuple = cls.tir_exp(lv2, alloc2)
            _1_1: R.Tuple = R.memory.kill_tensor(alloc1)
            lv3: R.Tensor((n,), dtype="float32") = alloc2
            _2_1: R.Tuple = R.memory.kill_storage(storage)
            _3: R.Tuple = R.memory.kill_storage(storage1)
            return lv3
    # fmt: on

    mod = relax.transform.StaticPlanBlockMemory()(Module)
    tvm.ir.assert_structural_equal(mod, Expected)


if __name__ == "__main__":
    tvm.testing.main()
