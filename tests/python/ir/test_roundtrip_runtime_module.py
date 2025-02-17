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


def test_csource_module():
    mod = tvm.runtime._ffi_api.CSourceModuleCreate("", "cc", [], [])
    assert mod.type_key == "c"
    assert mod.is_binary_serializable
    new_mod = tvm.ir.load_json(tvm.ir.save_json(mod))
    assert new_mod.type_key == "c"
    assert new_mod.is_binary_serializable


if __name__ == "__main__":
    tvm.testing.main()
