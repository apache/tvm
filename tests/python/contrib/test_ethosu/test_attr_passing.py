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

pytest.importorskip("ethosu.vela")
import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu import util


def test_compiler_attr():
    config = {
        "accelerator_config": "ethos-u55-32",
    }
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.ethosu.options": config}):
        with tvm.target.Target("c -device=micro_dev"):
            assert util.get_accelerator_config() == config["accelerator_config"]


def test_compiler_attr_default():
    default_config = {
        "accelerator_config": "ethos-u55-256",
    }
    with tvm.transform.PassContext(opt_level=3):
        with tvm.target.Target("c -device=micro_dev"):
            assert util.get_accelerator_config() == default_config["accelerator_config"]


if __name__ == "__main__":
    pytest.main([__file__])
