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
import tvm

@pytest.mark.parametrize(
    "target_name,target_attrs,target_args",
    [
        ("my_hwa", {}, {}),
        (
            "my_hwa2", 
            {
                "local_memory_size": 128*1024,
                "variant": "version1",
            }, 
            {
                "local_memory_size": 256*1024, 
                "variant": "version2"
            }
        )
    ]
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
    args = " ".join((F"--{k}={v}" for k,v in target_args.items()))
    my_target = tvm.target.Target(f"{target_name} {args}")

    for attr in target_args.keys():
        assert my_target.attrs[attr] == target_args[attr]


if __name__ == "__main__":
    tvm.testing.main()