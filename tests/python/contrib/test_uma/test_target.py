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
from typing import Union

import pytest
import tvm
from tests.python.contrib.test_uma.test_uma_vanilla_accelerator import VanillaAcceleratorBackend
from tvm.relay.backend.contrib.uma import uma_available

pytestmark = pytest.mark.skipif(not uma_available(), reason="UMA not available")


@pytest.mark.parametrize(
    "target_name,target_attrs,target_args",
    [
        ("my_hwa", {}, {}),
        (
            "my_hwa2",
            {
                "local_memory_size": 128 * 1024,
                "variant": "version1",
            },
            {"local_memory_size": 256 * 1024, "variant": "version2"},
        ),
    ],
)
def test_uma_target(target_name, target_attrs, target_args):
    registration_func = tvm.get_global_func("relay.backend.contrib.uma.RegisterTarget")
    registration_func(target_name, target_attrs)

    # Test Defaults
    my_target = tvm.target.Target(target_name)

    assert str(my_target.kind) == target_name

    for attr in target_attrs.keys():
        assert my_target.attrs[attr] == target_attrs[attr]

    # Test with parameters overwritten
    args = " ".join((f"--{k}={v}" for k, v in target_args.items()))
    my_target = tvm.target.Target(f"{target_name} {args}")

    for attr in target_args.keys():
        assert my_target.attrs[attr] == target_args[attr]


@pytest.mark.parametrize(
    "attr_name, target_attr",
    [
        ("float_attr", 3.14),
        ("none_attr", None),
        ("model", "my_model"),
    ],
)
def test_invalid_attr_option(attr_name: str, target_attr: Union[str, int, bool, float, None]):
    registration_func = tvm.get_global_func("relay.backend.contrib.uma.RegisterTarget")
    if target_attr is None:
        # None cannot be caught as TVMError, as it causes a SIGKILL, therefore it must be prevented to be
        # entered into relay.backend.contrib.uma.RegisterTarget at Python level.
        with pytest.raises(ValueError, match=r"Target attribute None is not supported."):
            uma_backend = VanillaAcceleratorBackend()
            uma_backend._target_attrs = {attr_name: target_attr}
            uma_backend.register()
    elif "model" in attr_name:
        target_name = f"{attr_name}_{target_attr}"
        target_attr = {attr_name: target_attr}
        with pytest.raises(tvm.TVMError, match=r"Attribute is already registered: .*"):
            registration_func(target_name, target_attr)
    else:
        target_name = f"{attr_name}_{target_attr}"
        target_attr = {attr_name: target_attr}
        with pytest.raises(TypeError, match=r"Only String, Integer, or Bool are supported. .*"):
            registration_func(target_name, target_attr)


@pytest.mark.parametrize(
    "target_name",
    [
        "llvm",
        "c",
    ],
)
def test_target_duplication(target_name: str):
    with pytest.raises(tvm.TVMError, match=r"TVM UMA Error: Target is already registered: .*"):
        registration_func = tvm.get_global_func("relay.backend.contrib.uma.RegisterTarget")
        registration_func(target_name, {})


if __name__ == "__main__":
    tvm.testing.main()
