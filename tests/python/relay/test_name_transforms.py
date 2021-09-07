# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License" you may not use this file except in compliance
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

from tvm import TVMError
from tvm.relay.backend.name_transforms import (
    to_c_function_style,
    to_c_variable_style,
    prefix_name,
    prefix_generated_name,
    sanitise_name,
)
import pytest


def test_to_c_function_style():
    assert to_c_function_style("TVM_Woof") == "TVMWoof"
    assert to_c_function_style("TVM_woof") == "TVMWoof"
    assert to_c_function_style("TVM_woof_woof") == "TVMWoofWoof"
    assert to_c_function_style("TVMGen_woof_woof") == "TVMGenWoofWoof"

    with pytest.raises(TVMError):
        to_c_function_style("")


def test_to_c_variable_style():
    assert to_c_variable_style("TVM_Woof") == "tvm_woof"
    assert to_c_variable_style("TVM_woof") == "tvm_woof"
    assert to_c_variable_style("TVM_woof_Woof") == "tvm_woof_woof"

    with pytest.raises(TVMError):
        to_c_variable_style("")


def test_prefix_name():
    assert prefix_name("Woof") == "TVM_Woof"
    assert prefix_name(["Woof"]) == "TVM_Woof"
    assert prefix_name(["woof"]) == "TVM_woof"
    assert prefix_name(["woof", "moo"]) == "TVM_woof_moo"

    with pytest.raises(TVMError):
        prefix_name("")
    with pytest.raises(TVMError):
        prefix_name([])
    with pytest.raises(TVMError):
        prefix_name([""])


def test_prefix_generated_name():
    assert prefix_generated_name("Woof") == "TVMGen_Woof"
    assert prefix_generated_name(["Woof"]) == "TVMGen_Woof"
    assert prefix_generated_name(["Woof"]) == "TVMGen_Woof"
    assert prefix_generated_name(["woof"]) == "TVMGen_woof"
    assert prefix_generated_name(["woof", "moo"]) == "TVMGen_woof_moo"

    with pytest.raises(TVMError):
        prefix_generated_name("")
    with pytest.raises(TVMError):
        prefix_generated_name([])
    with pytest.raises(TVMError):
        prefix_generated_name([""])


def test_sanitise_name():
    assert sanitise_name("+_+ ") == "_"
    assert sanitise_name("input+") == "input_"
    assert sanitise_name("input-") == "input_"
    assert sanitise_name("input++") == "input_"
    assert sanitise_name("woof:1") == "woof_1"

    with pytest.raises(TVMError):
        sanitise_name("")


def test_combined_logic():
    assert (
        to_c_function_style(prefix_name(["Device", "target", "Invoke"])) == "TVMDeviceTargetInvoke"
    )
    assert to_c_function_style(prefix_generated_name(["model", "Run"])) == "TVMGenModelRun"
    assert to_c_variable_style(prefix_name(["Device", "target", "t"])) == "tvm_device_target_t"
    assert (
        to_c_variable_style(prefix_generated_name(["model", "Devices"])) == "tvmgen_model_devices"
    )
