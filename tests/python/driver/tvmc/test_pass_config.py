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

from tvm.contrib.target.vitis_ai import vitis_ai_available
from tvm.driver import tvmc

from tvm.driver.tvmc.common import TVMCException


def test_config_invalid_format():
    with pytest.raises(TVMCException):
        _ = tvmc.common.parse_configs(["relay.backend.use_auto_scheduler.missing.value"])


def test_config_missing_from_tvm():
    with pytest.raises(TVMCException):
        _ = tvmc.common.parse_configs(["relay.backend.use_auto_scheduler.missing.value=1234"])


def test_config_unsupported_tvmc_config():
    with pytest.raises(TVMCException):
        _ = tvmc.common.parse_configs(["tir.LoopPartition=value"])


def test_config_empty():
    with pytest.raises(TVMCException):
        _ = tvmc.common.parse_configs([""])


def test_config_valid_config_bool():
    configs = tvmc.common.parse_configs(["relay.backend.use_auto_scheduler=true"])

    assert len(configs) == 1
    assert "relay.backend.use_auto_scheduler" in configs.keys()
    assert configs["relay.backend.use_auto_scheduler"] == True


@pytest.mark.skipif(
    not vitis_ai_available(),
    reason="--target vitis-ai is not available. TVM built with 'USE_VITIS_AI OFF'",
)
def test_config_valid_multiple_configs():
    configs = tvmc.common.parse_configs(
        [
            "relay.backend.use_auto_scheduler=false",
            "tir.detect_global_barrier=10",
            "relay.ext.vitis_ai.options.build_dir=mystring",
        ]
    )

    assert len(configs) == 3
    assert "relay.backend.use_auto_scheduler" in configs.keys()
    assert configs["relay.backend.use_auto_scheduler"] == False
    assert "tir.detect_global_barrier" in configs.keys()
    assert configs["tir.detect_global_barrier"] == 10
    assert "relay.ext.vitis_ai.options.build_dir" in configs.keys()
    assert configs["relay.ext.vitis_ai.options.build_dir"] == "mystring"
