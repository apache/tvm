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
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor, pipeline_executor


def get_mannual_mod():
    """
    # get list of module that represent a subgraph
    """
    mods = []
    dshape = (3, 3)
    data = relay.var("data_0", relay.TensorType(dshape, "float32"))
    data21 = relay.var("data_1", relay.TensorType(dshape, "float32"))
    data_net1_output_1 = relay.var("data_0", relay.TensorType(dshape, "float32"))
    data_net1_output_2 = relay.var("data_1", relay.TensorType(dshape, "float32"))
    data_net2_output_1 = relay.var("data_0", relay.TensorType(dshape, "float32"))
    mvalue1 = np.full((1), 1).astype("float32")
    mvalue2 = np.full((1), 2).astype("float32")
    mvalue3 = np.full((1), 3).astype("float32")
    mv1 = relay.Constant(tvm.nd.array(mvalue1))
    mv2 = relay.Constant(tvm.nd.array(mvalue2))
    mv3 = relay.Constant(tvm.nd.array(mvalue3))

    """
    # net1 have three output, output3 is final output.
    """

    net_output1 = relay.add(data, mv1)
    net_output2 = relay.subtract(data, mv2)
    net_output3 = relay.multiply(data, mv3)

    """
    # net2 use net1 output1 as input.
    """
    net2 = relay.add(data_net1_output_1, mv2)
    net2 = relay.add(net2, data21)
    net2 = relay.add(net2, mv3)

    """
    # net3 use net2 output1 and net1 outpu2 as input.
    """
    net3 = relay.multiply(data_net2_output_1, mv3)
    net3 = relay.add(net3, data_net1_output_2)

    mods.append(
        tvm.IRModule.from_expr(
            relay.Function([data], relay.Tuple([net_output1, net_output2, net_output3]))
        )
    )
    mods.append(tvm.IRModule.from_expr(relay.Function([data_net1_output_1, data21], net2)))
    mods.append(
        tvm.IRModule.from_expr(relay.Function([data_net1_output_2, data_net2_output_1], net3))
    )

    return mods, dshape


def get_manual_conf(mods, target):
    """
    # This function use to generate manual pipe line configueration,
    # the result use to verify if the pipe configuration can generate
    # correct result.
    """
    mod_config = {}
    """
    # set configure
    """

    """
    # third output is final output, second output for mod3, first for mod2
    # input
    """
    pipe_config1 = {
        "mod_indx": 1,
        "output": [
            {"output_indx": 0, "dependent": [{"mod_indx": 2, "input_name": "data_0"}]},
            {"output_indx": 1, "dependent": [{"mod_indx": 3, "input_name": "data_0"}]},
            {"output_indx": 2, "dependent": [{"mod_indx": 0, "input_name": "0"}]},
        ],
    }
    mod_config[mods[0]] = {"pipeline": pipe_config1,
                           "target_host": None,
                           "mod_name": "default",
			   "build": None,
                           "params": None,
                           "target": target[0],
                           "dev": target[1],
                          }

    pipe_config2 = {
        "mod_indx": 2,
        "output": [
            {"output_indx": 0, "dependent": [{"mod_indx": 3, "input_name": "data_1"}]},
        ],
    }
    mod_config[mods[1]] = {"pipeline": pipe_config2,
                           "target_host": None,
                           "mod_name": "default",
			   "build": None,
                           "params": None,
                           "target": "llvm",
                           "dev": tvm.cpu(0),
                          }

    pipe_config3 = {
        "mod_indx": 3,
        "output": [{"output_indx": 0, "dependent": [{"mod_indx": 0, "input_name": "1"}]}],
    }
    mod_config[mods[2]] = {"pipeline": pipe_config3,
                           "target_host": None,
                           "mod_name": "default",
			   "build": None,
                           "params": None,
                           "target": "llvm",
                           "dev": tvm.cpu(0),
                          }
    return mod_config


def pipeline(target):
    """
    #Get 3 pipeline module.
    """
    (mod1, mod2, mod3), dshape = get_mannual_mod()

    # Prepare batch data for pipeline feeding
    datas = []
    for i in range(5):
        datas.append(np.full(dshape, 3 + i).astype("float32"))

    pipe_config = pipeline_executor.PipelineConfig([mod1, mod2, mod3])

    # Create pipeline compute input/output and subgraph dependent relation.

    # pipeline compute input "data_0" would get forward to mod1 as input "data_0"
    pipe_config.connect(pipe_config.pipe_input("data_0"), pipe_config[mod1].input("data_0"))

    # pipeline compute input "data_1" would get forward to mod2 as input "data_1"
    pipe_config.connect(pipe_config.pipe_input("data_1"), pipe_config[mod2].input("data_1"))

    # mod1 output(0) would get forward to mod2 as input "data_0"
    pipe_config.connect(pipe_config[mod1].output(0), pipe_config[mod2].input("data_0"))

    # mod1 output(1) would get forward to mod3 as input "data_0"
    pipe_config.connect(pipe_config[mod1].output(1), pipe_config[mod3].input("data_0"))

    # mod2 output(0) would get forward to mod3 as input "data_1"
    pipe_config.connect(pipe_config[mod2].output(0), pipe_config[mod3].input("data_1"))

    # mod1 output(2) would get forward as final pipeline compute output(1)
    pipe_config.connect(pipe_config[mod1].output(2), pipe_config.pipe_output("0"))

    # mod3 output(0) would get forward as final pipeline compute output(2)
    pipe_config.connect(pipe_config[mod3].output(0), pipe_config.pipe_output("1"))
    """
    # print configueration (print(pipe_config)), the expect result like following.
    #
    #Inputs
    #  |data_0: mod1:data_0
    #  |data_1: mod2:data_1
    #
    #output
    #  |output(1) : mod1.output(2)
    #  |output(2) : mod3.output(0)
    #
    #connections
    #  |mod1.output(0)-> mod2.data_0
    #  |mod1.output(1)-> mod3.data_0
    #  |mod2.output(0)-> mod3.data_1
    """

    """
    # connection correctness veify
    """

    """
    # try wrong module order connection check, expect assert.
    """

    with pytest.raises(AssertionError):
        pipe_config.connect(pipe_config[mod2].output(0), pipe_config[mod1].input("data_0"))

    """
    # try pipeline module input with module output connection check, expect assert.
    """

    with pytest.raises(AssertionError):
        pipe_config.connect(pipe_config.pipe_input("data_0"), pipe_config[mod1].output(0))
        assert 0, f"wrong global input connect check not pass!"

    """
    # set other parameter.
    """
    pipe_config[mod1].set_target(target[0])
    pipe_config[mod1].set_dev(target[1])

    pipe_config[mod2].set_target("llvm")
    pipe_config[mod2].set_dev(tvm.cpu(0))

    pipe_config[mod3].set_target("llvm")
    pipe_config[mod3].set_dev(tvm.cpu(0))

    """
    # check if the configuration match expectation.
    """
    assert pipe_config.get_config() == get_manual_conf([mod1, mod2, mod3], target)

    """
    # Test build and create pipeline module
    """
    with relay.build_config(opt_level=3):
        pipeline_mod_config = pipeline_executor.build(pipe_config)

    pipeline_module = pipeline_executor.create(pipeline_mod_config)
    assert pipeline_module


def test_pipeline():
    if pipeline_executor.pipeline_executor_enabled():
        target_list = tvm.testing.enabled_targets()
        for target in target_list:
            pipeline(target)


if __name__ == "__main__":
    test_pipeline()
