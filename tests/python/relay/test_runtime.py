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

import pytest

from tvm import TVMError
from tvm.relay.backend import Runtime


def test_create():
    runtime = Runtime("cpp")
    assert str(runtime) == "cpp"


def test_create_runtime_with_options():
    runtime = Runtime("crt", {"system-lib": True})
    assert str(runtime) == "crt"
    assert runtime["system-lib"]


def test_attr_check():
    runtime = Runtime("crt", {"system-lib": True})
    assert "woof" not in runtime
    assert "system-lib" in runtime


def test_create_runtime_not_found():
    with pytest.raises(TVMError, match='Runtime "woof" is not defined'):
        Runtime("woof", {})


def test_create_runtime_attr_not_found():
    with pytest.raises(TVMError, match='Attribute "woof" is not available on this Runtime'):
        Runtime("crt", {"woof": "bark"})


def test_create_runtime_attr_type_incorrect():
    with pytest.raises(
        TVMError,
        match='Attribute "system-lib" should have type "runtime.BoxBool"'
        ' but instead found "runtime.String"',
    ):
        Runtime("crt", {"system-lib": "woof"})


def test_list_runtimes():
    assert "crt" in Runtime.list_registered()


@pytest.mark.parametrize("runtime", [Runtime("crt"), "crt"])
def test_list_runtime_options(runtime):
    aot_options = Runtime.list_registered_options(runtime)
    assert "system-lib" in aot_options
    assert aot_options["system-lib"] == "runtime.BoxBool"


def test_list_runtime_options_not_found():
    with pytest.raises(TVMError, match='Runtime "woof" is not defined'):
        Runtime.list_registered_options("woof")
