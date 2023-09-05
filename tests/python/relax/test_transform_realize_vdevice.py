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
import tvm.testing
from tvm.ir import VDevice
from tvm.relax.transform import RealizeVDevice
from tvm.script.parser import ir as I, relax as R, tir as T


def verify(input, expected):
    tvm.ir.assert_structural_equal(RealizeVDevice()(input), expected)


vdevices = [
    VDevice("llvm"),
    VDevice("cuda", 0),
    VDevice("metal", 0, "global"),
    VDevice("cuda -arch=sm_80", 0),
]


def test_dataflow_binding():
    @I.ir_module
    class Input:
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
            y: R.Tensor((2, 3), "float32"),
            z: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32"):
            with R.dataflow():
                x1 = x
                y1 = y
                x2 = x1
                y2 = y1
                lv0: R.Tensor((2, 3), "float32", "llvm") = R.add(x2, y2)
                gv: R.Tensor((2, 3), "float32", "llvm") = R.multiply(lv0, z)
                R.output(gv)
            return gv

    @I.ir_module
    class Expect:
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
            x: R.Tensor((2, 3), "float32", "llvm"),
            y: R.Tensor((2, 3), "float32", "llvm"),
            z: R.Tensor((2, 3), "float32", "llvm"),
        ) -> R.Tensor((2, 3), "float32", "llvm"):
            with R.dataflow():
                x1: R.Tensor((2, 3), "float32", "llvm") = x
                y1: R.Tensor((2, 3), "float32", "llvm") = y
                x2: R.Tensor((2, 3), "float32", "llvm") = x1
                y2: R.Tensor((2, 3), "float32", "llvm") = y1
                lv0: R.Tensor((2, 3), "float32", "llvm") = R.add(x2, y2)
                gv: R.Tensor((2, 3), "float32", "llvm") = R.multiply(lv0, z)
                R.output(gv)
            return gv

    verify(Input, Expect)


def test_binding():
    @I.ir_module
    class Input:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((2, 3), "float32"),
            z: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32"):
            x1 = x
            y1 = y
            x2 = x1
            y2 = y1
            s: R.Tensor((2, 3), "float32", "llvm") = R.add(x2, y2)
            m = R.multiply(s, z)
            return m

    @I.ir_module
    class Expect:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("llvm"),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32", "llvm"),
            y: R.Tensor((2, 3), "float32", "llvm"),
            z: R.Tensor((2, 3), "float32", "llvm"),
        ) -> R.Tensor((2, 3), "float32", "llvm"):
            x1: R.Tensor((2, 3), "float32", "llvm") = x
            y1: R.Tensor((2, 3), "float32", "llvm") = y
            x2: R.Tensor((2, 3), "float32", "llvm") = x1
            y2: R.Tensor((2, 3), "float32", "llvm") = y1
            s: R.Tensor((2, 3), "float32", "llvm") = R.add(x2, y2)
            m: R.Tensor((2, 3), "float32", "llvm") = R.multiply(s, z)
            return m

    verify(Input, Expect)


def test_func_ret():
    @I.ir_module
    class Input:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("cuda"),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32"),
            y: R.Tensor((2, 3), "float32"),
            z: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32", "cuda"):
            with R.dataflow():
                lv0 = R.add(x, y)
                gv = R.multiply(lv0, z)
                R.output(gv)
            return gv

    @I.ir_module
    class Expect:
        I.module_attrs({"attr": 10})
        I.module_global_infos(
            {
                "vdevice": [
                    I.vdevice("cuda"),
                ]
            }
        )

        @R.function
        def foo(
            x: R.Tensor((2, 3), "float32", "cuda"),
            y: R.Tensor((2, 3), "float32", "cuda"),
            z: R.Tensor((2, 3), "float32", "cuda"),
        ) -> R.Tensor((2, 3), "float32", "cuda"):
            with R.dataflow():
                lv0: R.Tensor((2, 3), "float32", "cuda") = R.add(x, y)
                gv: R.Tensor((2, 3), "float32", "cuda") = R.multiply(lv0, z)
                R.output(gv)
            return gv

    verify(Input, Expect)


def test_multi_device():
    @I.ir_module
    class Input:
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
            y: R.Tensor((2, 3), "float32"),
            z: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32", "cuda"):
            with R.dataflow():
                lv0: R.Tensor((2, 3), "float32", "llvm") = R.add(x, y)
                lv1 = R.to_vdevice(lv0, "cuda")
                lv2 = R.add(z, z)
                gv: R.Tensor((2, 3), "float32", "cuda") = R.multiply(lv1, lv2)
                R.output(gv)
            return gv

    @I.ir_module
    class Expect:
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
            x: R.Tensor((2, 3), "float32", "llvm"),
            y: R.Tensor((2, 3), "float32", "llvm"),
            z: R.Tensor((2, 3), "float32", "cuda"),
        ) -> R.Tensor((2, 3), "float32", "cuda"):
            with R.dataflow():
                lv0: R.Tensor((2, 3), "float32", "llvm") = R.add(x, y)
                lv1: R.Tensor((2, 3), "float32", "cuda") = R.to_vdevice(lv0, "cuda")
                lv2: R.Tensor((2, 3), "float32", "cuda") = R.add(z, z)
                gv: R.Tensor((2, 3), "float32", "cuda") = R.multiply(lv1, lv2)
                R.output(gv)
            return gv

    verify(Input, Expect)


def test_insert_to_vdevice():
    @I.ir_module
    class Input:
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
            y: R.Tensor((2, 3), "float32"),
            z: R.Tensor((2, 3), "float32"),
        ) -> R.Tensor((2, 3), "float32"):
            with R.dataflow():
                lv0 = R.hint_on_device(y, tvm.cpu())
                lv1 = R.add(x, lv0)
                lv2 = R.hint_on_device(lv1, tvm.cuda())
                lv3 = R.add(lv2, lv2)
                lv4 = R.hint_on_device(z, tvm.cuda())
                gv = R.multiply(lv3, lv4)
                R.output(gv)
            return gv

    @I.ir_module
    class Expect:
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
            x: R.Tensor((2, 3), "float32", "llvm"),
            y: R.Tensor((2, 3), "float32", "llvm"),
            z: R.Tensor((2, 3), "float32", "cuda"),
        ) -> R.Tensor((2, 3), "float32", "cuda"):
            with R.dataflow():
                lv0: R.Tensor((2, 3), "float32", "llvm") = y
                lv1: R.Tensor((2, 3), "float32", "llvm") = R.add(x, lv0)
                lv2: R.Tensor((2, 3), "float32", "cuda") = R.to_vdevice(lv1, "cuda")
                lv3: R.Tensor((2, 3), "float32", "cuda") = R.add(lv2, lv2)
                lv4: R.Tensor((2, 3), "float32", "cuda") = z
                gv: R.Tensor((2, 3), "float32", "cuda") = R.multiply(lv3, lv4)
                R.output(gv)
            return gv

    verify(Input, Expect)


if __name__ == "__main__":
    tvm.testing.main()
