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
from tvm.script import relax as R, ir as I

exec_mode = tvm.testing.parameter("bytecode", "compiled")

tuple_type_annotation = tvm.testing.parameter(
    by_dict={
        "tuple_of_obj": R.Tuple([R.Object, R.Object]),
        "tuple_of_known_types": R.Tuple([R.Prim("int64"), R.Prim("float32")]),
    }
)

tuple_index_type = tvm.testing.parameter("static", "dynamic")


def test_vm_tuple_get_item(exec_mode, tuple_type_annotation, tuple_index_type):
    index_var = tvm.tir.Var("index", "int64")

    def access_tuple(tuple_obj, dyn_index):
        if tuple_index_type == "static":
            return tuple_obj[0]
        elif tuple_index_type == "dynamic":
            return tuple_obj[dyn_index]

    @R.function(private=True)
    def func(arg: tuple_type_annotation, index_param: R.Prim(value=index_var)):
        # Trivial binding provides a usage of
        # `tuple_type_annotation` within the body of the function,
        # which is required to expose it as a meta-variable for
        # TVMScript.
        arg: tuple_type_annotation = arg
        index_param: R.Prim(value=index_var) = index_param
        return access_tuple(arg, index_param)

    mod = tvm.IRModule({"main": func})

    target = tvm.target.Target("llvm", host="llvm")
    ex = relax.build(mod, target, exec_mode=exec_mode)
    vm = relax.VirtualMachine(ex, tvm.cpu())

    res = vm["main"]((17, 42.5), 0)
    assert res == 17


if __name__ == "__main__":
    tvm.testing.main()
