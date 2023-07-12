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

import tvm.testing
from tvm.ir import Range
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm.relax.distributed import DeviceMesh, DTensorStructInfo, Placement
from tvm.relax import TensorStructInfo


def _assert_print(obj, expected):
    if not isinstance(obj, str):
        obj = obj.script(verbose_expr=True)
    obj = obj.strip()
    assert obj == expected.strip(), "\n" + obj


def test_constant():
    constant = R.dist.const(
        1,
        struct_info=R.DTensor(
            (), "float32", device_mesh=DeviceMesh((2, 2), Range(0, 4)), placement="R, R"
        ),
    )
    assert (
        constant.__str__()
        == """R.dist.const(1, R.DTensor((), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "R, R"))"""
    )


def test_dtensor_struct_info():
    tensor_sinfo1 = TensorStructInfo((32, 32), "float32")
    tensor_sinfo2 = TensorStructInfo((32, 32), "void")
    obj0 = DTensorStructInfo(
        tensor_sinfo1, DeviceMesh((2, 2), Range(0, 4)), Placement.from_text("S[1], R")
    )
    assert (
        obj0.__str__()
        == """R.DTensor((32, 32), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[1], R")"""
    )

    obj1 = DTensorStructInfo(
        tensor_sinfo2, DeviceMesh((2, 2), Range(0, 4)), Placement.from_text("S[1], R")
    )
    assert (
        obj1.__str__()
        == """R.DTensor((32, 32), device_mesh=R.device_mesh((2, 2), R.Range(0, 4)), placement="S[1], R")"""
    )

    obj2 = DTensorStructInfo(
        tensor_sinfo2, DeviceMesh((2, 2), [0, 1, 2, 3]), Placement.from_text("S[1], R")
    )
    assert (
        obj2.__str__()
        == """R.DTensor((32, 32), device_mesh=R.device_mesh((2, 2), [0, 1, 2, 3]), placement="S[1], R")"""
    )


@I.ir_module
class TestModule:
    I.module_attrs({"device_num": 10})
    I.module_global_infos(
        {
            "mesh": [
                R.device_mesh((2, 2), I.Range(0, 4)),  # mesh[0]
                R.device_mesh((1,), I.Range(4, 5)),  # mesh[1]
            ]
        }
    )

    @T.prim_func
    def tir_func(
        x: T.Buffer((T.int64(128), T.int64(128)), "float32"),
        y: T.Buffer((T.int64(128), T.int64(128)), "float32"),
    ):
        T.func_attr({"tir.noalias": True})
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                y[vi, vj] = x[vi, vj] + 1.0

    @R.function
    def foo(
        x: R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"),
    ) -> R.DTensor((128, 128), "float32", device_mesh="mesh[0]", placement="S[0], R"):
        gv0 = R.dist.call_tir(
            TestModule.tir_func,
            x,
            R.DTensor(
                shape=(128, 128), dtype="float32", device_mesh="mesh[0]", placement="S[0], R"
            ),
        )
        return gv0


def test_func():
    _assert_print(
        TestModule["foo"],
        """
# from tvm.script import relax as R

@R.function
def foo(x: R.DTensor((128, 128), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[0], R")) -> R.DTensor((128, 128), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[0], R"):
    gv0 = R.dist.call_tir(tir_func, (x,), out_sinfo=R.DTensor((128, 128), "float32", R.device_mesh((2, 2), R.Range(0, 4)), "S[0], R"))
    return gv0
            """,
    )


def test_module():
    _assert_print(
        TestModule,
        """
# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    I.module_attrs({"device_num": 10})
    I.module_global_infos({"mesh": [R.device_mesh((2, 2), I.Range(0, 4)), R.device_mesh((1,), I.Range(4, 5))]})
    @T.prim_func
    def tir_func(x: T.Buffer((T.int64(128), T.int64(128)), "float32"), y: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with T.block(""):
                vi, vj = T.axis.remap("SS", [i, j])
                T.reads(x[vi, vj])
                T.writes(y[vi, vj])
                y[vi, vj] = x[vi, vj] + T.float32(1)

    @R.function
    def foo(x: R.DTensor((128, 128), "float32", "mesh[0]", "S[0], R")) -> R.DTensor((128, 128), "float32", "mesh[0]", "S[0], R"):
        cls = Module
        gv0 = R.dist.call_tir(cls.tir_func, (x,), out_sinfo=R.DTensor((128, 128), "float32", "mesh[0]", "S[0], R"))
        return gv0
    """,
    )


if __name__ == "__main__":
    tvm.testing.main()
