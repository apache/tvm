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
from tvm.relay.backend import Executor


def test_create_executor():
    executor = Executor("aot")
    assert executor.name == "aot"


def test_create_executor_with_options():
    executor = Executor("aot", {"interface-api": "c"})
    assert executor.name == "aot"
    assert executor["interface-api"] == "c"


def test_create_executor_with_default():
    executor = Executor("graph")
    assert not executor["link-params"]


def test_attr_check():
    executor = Executor("aot", {"interface-api": "c"})
    assert "woof" not in executor
    assert "interface-api" in executor


def test_create_executor_not_found():
    with pytest.raises(TVMError, match='Executor "woof" is not defined'):
        Executor("woof", {})


def test_create_executor_attr_not_found():
    with pytest.raises(TVMError, match='Attribute "woof" is not available on this Executor'):
        Executor("aot", {"woof": "bark"})


def test_create_executor_attr_type_incorrect():
    with pytest.raises(
        TVMError,
        match='Attribute "interface-api" should have type "runtime.String"'
        ' but instead found "runtime.BoxBool"',
    ):
        Executor("aot", {"interface-api": True})


def test_list_executors():
    assert "aot" in Executor.list_registered()


@pytest.mark.parametrize("executor", [Executor("aot").name, "aot"])
def test_list_executor_options(executor):
    aot_options = Executor.list_registered_options(executor)
    assert "interface-api" in aot_options
    assert aot_options["interface-api"] == "runtime.String"


def test_list_executor_options_not_found():
    with pytest.raises(TVMError, match='Executor "woof" is not defined'):
        Executor.list_registered_options("woof")
