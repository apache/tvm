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
from tvm import te
import tvm.runtime._ffi_api
import tvm.target._ffi_api


def checker(mod, expected):
    assert mod.is_binary_serializable == expected["is_binary_serializable"]
    assert mod.is_runnable == expected["is_runnable"]
    assert mod.is_dso_exportable == expected["is_dso_exportable"]


def create_csource_module():
    return tvm.runtime._ffi_api.CSourceModuleCreate("", "cc", [], None)


def create_llvm_module():
    A = te.placeholder((1024,), name="A")
    B = te.compute(A.shape, lambda *i: A(*i) + 1.0, name="B")
    s = te.create_schedule(B.op)
    return tvm.build(s, [A, B], "llvm", name="myadd0")


def create_aot_module():
    return tvm.get_global_func("relay.build_module._AOTExecutorCodegen")()


def test_property():
    checker(
        create_csource_module(),
        expected={"is_binary_serializable": True, "is_runnable": False, "is_dso_exportable": True},
    )

    checker(
        create_llvm_module(),
        expected={"is_binary_serializable": False, "is_runnable": True, "is_dso_exportable": True},
    )

    checker(
        create_aot_module(),
        expected={"is_binary_serializable": False, "is_runnable": True, "is_dso_exportable": False},
    )


if __name__ == "__main__":
    tvm.testing.main()
