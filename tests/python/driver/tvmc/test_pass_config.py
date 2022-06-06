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
from unittest import mock

from tvm.contrib.target.vitis_ai import vitis_ai_available

from tvm.driver.tvmc import TVMCException
from tvm.driver.tvmc.pass_config import parse_configs
from tvm.tir.transform import PrimFuncPass
from tvm.ir.transform import Sequential


def test_config_invalid_format():
    with pytest.raises(TVMCException):
        _ = parse_configs(["relay.backend.use_auto_scheduler.missing.value"])


def test_config_missing_from_tvm():
    with pytest.raises(TVMCException):
        _ = parse_configs(["relay.backend.use_auto_scheduler.missing.value=1234"])


def test_config_unsupported_tvmc_config():
    with pytest.raises(TVMCException):
        _ = parse_configs(["tir.LoopPartition=value"])


def test_config_empty():
    with pytest.raises(TVMCException):
        _ = parse_configs([""])


def test_config_valid_config_bool():
    configs = parse_configs(["relay.backend.use_auto_scheduler=true"])

    assert len(configs) == 1
    assert "relay.backend.use_auto_scheduler" in configs.keys()
    assert configs["relay.backend.use_auto_scheduler"] == True


@pytest.mark.skipif(
    not vitis_ai_available(),
    reason="--target vitis-ai is not available. TVM built with 'USE_VITIS_AI OFF'",
)
def test_config_valid_multiple_configs():
    configs = parse_configs(
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


def test_add_lower_pass_multi_built_in_pass():
    configs = parse_configs(
        [
            "tir.add_lower_pass=1,tir.transform.UnrollLoop",
            "tir.add_lower_pass=1,tir.transform.HoistIfThenElse,2,tir.transform.LoopPartition",
        ]
    )

    assert len(configs["tir.add_lower_pass"]) == 3
    # opt_level: 1, pass: tir.transform.UnrollLoop
    assert configs["tir.add_lower_pass"][0][0] == 1
    assert isinstance(configs["tir.add_lower_pass"][0][1], PrimFuncPass)
    # opt_level: 1, pass: tir.transform.HoistIfThenElse
    assert configs["tir.add_lower_pass"][1][0] == 1
    assert isinstance(configs["tir.add_lower_pass"][1][1], Sequential)
    assert configs["tir.add_lower_pass"][1][1].pass_info.name == "tir.HoistIfThenElse"
    # opt_level: 2, pass: tir.transform.LoopPartition
    assert configs["tir.add_lower_pass"][2][0] == 2
    assert isinstance(configs["tir.add_lower_pass"][2][1], PrimFuncPass)


def test_add_lower_pass_multi_external_pass():
    fake_pass_1 = mock.MagicMock()
    fake_pass_2 = mock.MagicMock()
    fake_pass_3 = mock.MagicMock()
    with mock.patch.dict(
        "sys.modules",
        {"fake_module": fake_pass_1, "fake_module": fake_pass_2, "fake_module": fake_pass_3},
    ):
        configs = parse_configs(
            [
                "tir.add_lower_pass=1,fake_module.fake_pass_1,2,fake_module.fake_pass2",
                "tir.add_lower_pass=3,fake_module.fake_pass_3",
            ]
        )
        assert len(configs["tir.add_lower_pass"]) == 3
        # opt_level: 1, pass: fake_module.fake_pass_1
        assert configs["tir.add_lower_pass"][0][0] == 1
        # opt_level: 2, pass: fake_module.fake_pass_2
        assert configs["tir.add_lower_pass"][1][0] == 2
        # opt_level: 3, pass: fake_module.fake_pass_3
        assert configs["tir.add_lower_pass"][2][0] == 3


def test_add_lower_pass_multi_mix_pass():
    fake_pass_1 = mock.MagicMock()
    fake_pass_2 = mock.MagicMock()
    with mock.patch.dict("sys.modules", {"fake_module": fake_pass_1, "fake_module": fake_pass_2}):
        configs = parse_configs(
            [
                "tir.add_lower_pass=1,fake_module.fake_pass_1,1,tir.transform.UnrollLoop",
                "tir.add_lower_pass=2,fake_module.fake_pass_2,2,tir.transform.LoopPartition",
            ]
        )
        assert len(configs["tir.add_lower_pass"]) == 4
        # opt_level: 1, pass: fake_module.fake_pass_1
        assert configs["tir.add_lower_pass"][0][0] == 1
        # opt_level: 1, pass: tir.transform.UnrollLoop
        assert configs["tir.add_lower_pass"][1][0] == 1
        assert isinstance(configs["tir.add_lower_pass"][1][1], PrimFuncPass)
        # opt_level: 2, pass: fake_module.fake_pass_2
        assert configs["tir.add_lower_pass"][2][0] == 2
        # opt_level: 2, pass: tir.transform.LoopPartition
        assert configs["tir.add_lower_pass"][3][0] == 2
        assert isinstance(configs["tir.add_lower_pass"][3][1], PrimFuncPass)


def test_add_lower_pass_invalid_format():
    # wrong format
    with pytest.raises(TVMCException):
        _ = parse_configs(["tir.add_lower_pass=tir.transform.UnrollLoop,1"])
    # missing pass name
    with pytest.raises(TVMCException):
        _ = parse_configs(["tir.add_lower_pass=1,tir.transform.UnrollLoop,3"])
    # wrong opt level
    with pytest.raises(TVMCException):
        _ = parse_configs(["tir.add_lower_pass=a,tir.transform.UnrollLoop"])
    # fake module
    with pytest.raises(ModuleNotFoundError):
        _ = parse_configs(
            ["tir.add_lower_pass=1,tir.transform.UnrollLoop,2,path.to.module.fake_func"]
        )
    # real module and fake func
    with pytest.raises(TVMCException):
        _ = parse_configs(["tir.add_lower_pass=1,tir.transform.UnrollLoop,2,tvm.tir.fake_func"])
