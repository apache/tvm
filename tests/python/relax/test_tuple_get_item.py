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
import tvm.script
import tvm.testing
from tvm import relax
from tvm.script import relax as R, tir as T

import pytest

exec_mode = tvm.testing.parameter("bytecode", "compiled")

tuple_type_annotation = tvm.testing.parameter(
    by_dict={
        "tuple_of_obj": R.Tuple([R.Object, R.Object]),
        "tuple_of_known_types": R.Tuple([R.Prim("int64"), R.Prim("float32")]),
    }
)

tuple_index_type = tvm.testing.parameter("static", "dynamic")

syntax_sugar = tvm.testing.parameter(by_dict={"sugared": True, "unsugared": False})


def test_vm_tuple_get_item(exec_mode, tuple_type_annotation, tuple_index_type):
    def access_tuple(tuple_obj, dyn_index):
        if tuple_index_type == "static":
            return tuple_obj[0]
        elif tuple_index_type == "dynamic":
            return tuple_obj[dyn_index]

    @R.function(private=True)
    def func(arg: tuple_type_annotation, index_param: R.Prim(value="index_var")):
        index_var = T.int64()
        # Trivial binding provides a usage of
        # `tuple_type_annotation` within the body of the function,
        # which is required to expose it as a meta-variable for
        # TVMScript.
        arg: tuple_type_annotation = arg
        return access_tuple(arg, index_param)

    mod = tvm.IRModule({"main": func})

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    res = vm["main"]((17, 42.5), 0)
    assert res == 17


def test_dynamic_index_printing(syntax_sugar: bool):
    """Check syntax-sugar for dynamic tuple indices

    The "relax.tuple_get_item_dyn" operator should be printed as
    `my_tuple[my_index]` by default, which will regenerate the
    original operator when parsed.  If syntax sugar is disabled, it
    should display the `R.tuple_get_item_dyn` directly.
    """

    @R.function(private=True)
    def func(
        arg_tuple: R.Tuple([R.Prim("int64"), R.Prim("float32")]),
        arg_index: R.Prim(value="index_var"),
    ):
        return arg_tuple[arg_index]

    script = func.script(syntax_sugar=syntax_sugar)

    if syntax_sugar:
        assert "arg_tuple[arg_index]" in script
        assert "tuple_get_item_dyn" not in script
    else:
        assert "arg_tuple[arg_index]" not in script
        assert "tuple_get_item_dyn" in script

    roundtrip = tvm.script.from_source(script)

    tvm.ir.assert_structural_equal(func, roundtrip)


def test_tuple_get_item_simple():
    exec_mode = "bytecode"

    @R.function(private=True)
    def func(arg: R.Tuple([R.Prim("int64"), R.Prim("float32")])):
        return arg[0]

    mod = tvm.IRModule({"main": func})

    target = tvm.target.Target("llvm", host="llvm")
    ex = tvm.relax.build(mod, target, exec_mode=exec_mode)
    vm = tvm.relax.VirtualMachine(ex, tvm.cpu())

    res = vm["main"]((17, 42.5))
    assert res == 17


if __name__ == "__main__":
    tvm.testing.main()
