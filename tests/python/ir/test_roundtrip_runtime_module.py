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
    mod = tvm.runtime._ffi_api.CSourceModuleCreate("", "cc", [], [])
    assert mod.type_key == "c"
    assert mod.is_binary_serializable
    new_mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    assert new_mod.type_key == "c"
    assert new_mod.is_binary_serializable


def test_aot_module():
    mod = tvm.get_global_func("relay.build_module._AOTExecutorCodegen")()
    # aot module that is not binary serializable.
    # Thus, it would raise an error.
    assert not mod.is_binary_serializable
    with pytest.raises(TVMError):
        tvm.ir.load_json(tvm.ir.save_json(mod))


def get_test_mod():
    x = relay.var("x", shape=(1, 10), dtype="float32")
    y = relay.var("y", shape=(1, 10), dtype="float32")
    z = relay.add(x, y)
    func = relay.Function([x, y], z)
    return relay.build_module._build_module_no_factory(func, target="cuda")


def get_cuda_mod():
    # Get Cuda module which is binary serializable
    return get_test_mod().imported_modules[0].imported_modules[0]


@tvm.testing.requires_cuda
def test_cuda_module():
    mod = get_cuda_mod()
    assert mod.type_key == "cuda"
    assert mod.is_binary_serializable
    new_mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    assert new_mod.type_key == "cuda"
    assert new_mod.is_binary_serializable


@tvm.testing.requires_cuda
def test_valid_submodules():
    mod, mod2, mod3, mod4 = get_cuda_mod(), get_cuda_mod(), get_cuda_mod(), get_cuda_mod()

    # Create the nested cuda module
    mod.import_module(mod2)
    mod2.import_module(mod3)
    mod2.import_module(mod4)

    # Root module and all submodules should be binary serializable since they are cuda module
    assert mod.type_key == "cuda"
    assert mod.is_binary_serializable
    assert mod.imported_modules[0].type_key == "cuda"
    assert mod.imported_modules[0].is_binary_serializable
    assert mod.imported_modules[0].imported_modules[0].type_key == "cuda"
    assert mod.imported_modules[0].imported_modules[1].type_key == "cuda"
    assert mod.imported_modules[0].imported_modules[0].is_binary_serializable
    assert mod.imported_modules[0].imported_modules[1].is_binary_serializable

    # The roundtripped mod should have the same structure
    new_mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    assert new_mod.type_key == "cuda"
    assert new_mod.is_binary_serializable
    assert new_mod.imported_modules[0].type_key == "cuda"
    assert new_mod.imported_modules[0].is_binary_serializable
    assert new_mod.imported_modules[0].imported_modules[0].type_key == "cuda"
    assert new_mod.imported_modules[0].imported_modules[1].type_key == "cuda"
    assert new_mod.imported_modules[0].imported_modules[0].is_binary_serializable
    assert new_mod.imported_modules[0].imported_modules[1].is_binary_serializable


@tvm.testing.requires_cuda
def test_invalid_submodules():
    mod, mod2, mod3 = get_cuda_mod(), get_cuda_mod(), get_cuda_mod()
    mod4 = tvm.get_global_func("relay.build_module._AOTExecutorCodegen")()

    # Create the nested cuda module
    mod.import_module(mod2)
    mod2.import_module(mod3)
    mod2.import_module(mod4)

    # One of submodules is not binary serializable.
    assert mod.is_binary_serializable
    assert mod.imported_modules[0].is_binary_serializable
    assert mod.imported_modules[0].imported_modules[0].is_binary_serializable
    assert not mod.imported_modules[0].imported_modules[1].is_binary_serializable

    # Therefore, we cannot roundtrip.
    with pytest.raises(TVMError):
        tvm.ir.load_json(tvm.ir.save_json(mod))


if __name__ == "__main__":
    tvm.testing.main()
