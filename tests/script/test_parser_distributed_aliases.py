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
"""Distributed source spellings share concrete annotation constructors."""

import pytest
import tvm_ffi

import tvm
from tvm.relax.script import builder as R
from tvm.script import parser
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import ir as I


@pytest.mark.parametrize("spelling", ["R.dist", "D"])
def test_distributed_annotation_aliases(spelling):
    from tvm.script import relax as public_R

    source = f"""
@I.ir_module
class Module:
    I.module_global_infos({{"mesh": [{spelling}.device_mesh((2,), I.Range(0, 2))]}})

    @R.function
    def main(x: {spelling}.DTensor(("n", 16), "float32", "mesh[0]", "R")):
        return x
"""
    captures = {"D": public_R.dist}
    with IRBuilder() as builder:
        with I.ir_module():
            mesh = R.device_mesh((2,), tvm.ir.Range(0, 2))
            I.module_global_infos({"mesh": [mesh]})
            with R.function():
                R.func_name("main")
                n = tvm.tirx.Var("n", "int64")
                x = R.arg("x", R.DTensor((n, 16), "float32", mesh, "R"))
                R.func_ret_value(x)
    expected = builder.get()
    actual = parser.parse(source, extra_vars=captures)
    tvm.ir.assert_structural_equal(expected, actual)
    mesh = actual.global_infos["mesh"][0]
    assert tvm_ffi.Object.same_as(actual["main"].params[0].ty.device_mesh, mesh)
    assert str(actual["main"].params[0].ty.tensor_ty.dtype) == "float32"


def test_distributed_constructors_keep_identity():
    from tvm.script import relax as public_R

    assert R.dist.DTensor is R.DTensor
    assert R.dist.device_mesh is R.device_mesh
    assert parser.parse is not None
    assert public_R.dist.DTensor is public_R.DTensor is R.DTensor
    assert public_R.dist.device_mesh is public_R.device_mesh is R.device_mesh
