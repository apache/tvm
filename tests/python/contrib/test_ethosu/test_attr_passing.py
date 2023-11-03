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
        "enable_cascader": True,
        "enable_striping": True,
        "disable_copying_constants": True,
        "dev_force_block_config": "2x4x16",
        "dev_max_open_plans": "256",
        "dev_max_closed_plans": "128",
        "dev_select_proposal_idx": "1",
        "dev_disable_pareto_plans": True,
        "dev_disable_pareto_proposals": True,
        "dev_disable_block_culling": True,
        "dev_cascader_logging": True,
    }
    with tvm.transform.PassContext(opt_level=3, config={"relay.ext.ethos-u.options": config}):
        with tvm.target.Target("c"):
            compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
            assert compiler_attrs.accelerator_config == config["accelerator_config"]
            assert compiler_attrs.enable_cascader == config["enable_cascader"]
            assert compiler_attrs.enable_striping == config["enable_striping"]
            assert compiler_attrs.disable_copying_constants == config["disable_copying_constants"]
            assert compiler_attrs.dev_force_block_config == config["dev_force_block_config"]
            assert compiler_attrs.dev_max_open_plans == config["dev_max_open_plans"]
            assert compiler_attrs.dev_max_closed_plans == config["dev_max_closed_plans"]
            assert compiler_attrs.dev_select_proposal_idx == config["dev_select_proposal_idx"]
            assert compiler_attrs.dev_disable_pareto_plans == config["dev_disable_pareto_plans"]
            assert (
                compiler_attrs.dev_disable_pareto_proposals
                == config["dev_disable_pareto_proposals"]
            )
            assert compiler_attrs.dev_disable_block_culling == config["dev_disable_block_culling"]
            assert compiler_attrs.dev_cascader_logging == config["dev_cascader_logging"]


def test_compiler_attr_default():
    default_config = {
        "accelerator_config": "ethos-u55-256",
        "enable_cascader": False,
        "enable_striping": False,
        "disable_copying_constants": False,
        "dev_force_block_config": "",
        "dev_max_open_plans": "8",
        "dev_max_closed_plans": "32",
        "dev_select_proposal_idx": "-1",
        "dev_disable_pareto_plans": False,
        "dev_disable_pareto_proposals": False,
        "dev_disable_block_culling": False,
        "dev_cascader_logging": False,
    }
    with tvm.transform.PassContext(opt_level=3):
        with tvm.target.Target("c"):
            compiler_attrs = tvm.get_global_func("relay.ext.ethos-u.get_compiler_attrs")()
            assert compiler_attrs.accelerator_config == default_config["accelerator_config"]
            assert compiler_attrs.enable_cascader == default_config["enable_cascader"]
            assert compiler_attrs.enable_striping == default_config["enable_striping"]
            assert (
                compiler_attrs.disable_copying_constants
                == default_config["disable_copying_constants"]
            )
            assert compiler_attrs.dev_force_block_config == default_config["dev_force_block_config"]
            assert compiler_attrs.dev_max_open_plans == default_config["dev_max_open_plans"]
            assert compiler_attrs.dev_max_closed_plans == default_config["dev_max_closed_plans"]
            assert (
                compiler_attrs.dev_select_proposal_idx == default_config["dev_select_proposal_idx"]
            )
            assert (
                compiler_attrs.dev_disable_pareto_plans
                == default_config["dev_disable_pareto_plans"]
            )
            assert (
                compiler_attrs.dev_disable_pareto_proposals
                == default_config["dev_disable_pareto_proposals"]
            )
            assert (
                compiler_attrs.dev_disable_block_culling
                == default_config["dev_disable_block_culling"]
            )
            assert compiler_attrs.dev_cascader_logging == default_config["dev_cascader_logging"]


if __name__ == "__main__":
    tvm.testing.main()
