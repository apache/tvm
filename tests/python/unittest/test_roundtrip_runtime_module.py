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
""" Test roundtrip of runtime modules """
# pylint: disable=missing-docstring

import pytest
import tvm
import tvm.testing
from tvm import TVMError
from tvm import relay


def test_csource_module():
    mod = tvm.runtime._ffi_api.CSourceModuleCreate("", "cc", [], None)
    # source module that is not binary serializable.
    # Thus, it would raise an error.
    assert not mod.is_binary_serializable
    with pytest.raises(TVMError):
        tvm.ir.load_json(tvm.ir.save_json(mod))


def test_aot_module():
    mod = tvm.get_global_func("relay.build_module._AOTExecutorCodegen")()
    # aot module that is not binary serializable.
    # Thus, it would raise an error.
    assert not mod.is_binary_serializable
    with pytest.raises(TVMError):
        tvm.ir.load_json(tvm.ir.save_json(mod))


def test_recursive_imports():
    x = relay.var("x", shape=(1, 10))
    y = relay.var("y", shape=(1, 10))
    z = relay.add(x, y)
    func = relay.Function([x, y], z)
    mod = relay.build_module._build_module_no_factory(func, target="cuda")

    mod.imported_modules[0].imported_modules[0]
    assert mod.is_binary_serializable
    # GraphExecutorFactory Module contains LLVM Module and LLVM Module contains cuda Module.
    assert mod.type_key == "GraphExecutorFactory"
    assert mod.imported_modules[0].type_key == "llvm"
    assert mod.imported_modules[0].imported_modules[0].type_key == "cuda"

    json = tvm.ir.save_json(mod)
    print(json)


if __name__ == "__main__":
    test_recursive_imports()
    # tvm.testing.main()
