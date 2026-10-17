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
# ruff: noqa: F401
"""Test roundtrip of runtime modules"""
# pylint: disable=missing-docstring

import pytest

import tvm_tirx
import tvm_tirx.testing


def test_csource_module():
    mod = tvm_tirx.runtime._ffi_api.CSourceModuleCreate("", "cc", [], [])
    assert mod.kind == "c"
    assert mod.is_binary_serializable()
    new_mod = tvm_tirx.ir.load_json(tvm_tirx.ir.save_json(mod))
    assert new_mod.kind == "c"
    assert new_mod.is_binary_serializable()


if __name__ == "__main__":
    tvm_tirx.testing.main()
