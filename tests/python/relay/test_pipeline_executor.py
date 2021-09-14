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
    # Get list of module that represent a subgraph.
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

    # The first model has three output.

    net1_output1 = relay.add(data, mv1)
    net1_output2 = relay.subtract(data, mv2)
    net1_output3 = relay.multiply(data, mv3)

    """The second model use first model net1_output1 as first input,
    here data_net1_output_1 represent net_output1, the second input
    of this model is data21.
    """
    net2 = relay.add(data_net1_output_1, mv2)
    net2 = relay.add(net2, data21)
    net2_output = relay.add(net2, mv3)

    """The third model use the second model net2_output as first input
    and use the first model net1_output2 as second input.
    """
    net3 = relay.multiply(data_net2_output_1, mv3)
    net3 = relay.add(net3, data_net1_output_2)

    mods.append(
        tvm.IRModule.from_expr(
            relay.Function([data], relay.Tuple([net1_output1, net1_output2, net1_output3]))
        )
    )
    mods.append(tvm.IRModule.from_expr(relay.Function([data_net1_output_1, data21], net2_output)))
    mods.append(
        tvm.IRModule.from_expr(relay.Function([data_net1_output_2, data_net2_output_1], net3))
    )

    return mods, dshape


def get_manual_conf(mods, target):
    # This function is used to generate manual pipeline configuration.
    mod_config = {}
    """The third output is the final output, the second output is for mod3, the first is for mod2
    input.
    """
    pipe_config1 = {
        "mod_idx": 1,
        "output": [
            {"output_idx": 0, "dependent": [{"mod_idx": 2, "input_name": "data_0"}]},
            {"output_idx": 1, "dependent": [{"mod_idx": 3, "input_name": "data_0"}]},
            {"output_idx": 2, "dependent": [{"mod_idx": 0, "input_name": "0"}]},
        ],
    }
    mod_config[mods[0]] = {
        "pipeline": pipe_config1,
        "target_host": None,
        "mod_name": "default",
        "build": None,
        "params": None,
        "target": target[0],
        "dev": target[1],
    }

    pipe_config2 = {
        "mod_idx": 2,
        "output": [
            {"output_idx": 0, "dependent": [{"mod_idx": 3, "input_name": "data_1"}]},
        ],
    }
    mod_config[mods[1]] = {
        "pipeline": pipe_config2,
        "target_host": None,
        "mod_name": "default",
        "build": None,
        "params": None,
        "target": "llvm",
        "dev": tvm.cpu(0),
    }

    pipe_config3 = {
        "mod_idx": 3,
        "output": [{"output_idx": 0, "dependent": [{"mod_idx": 0, "input_name": "1"}]}],
    }
    mod_config[mods[2]] = {
        "pipeline": pipe_config3,
        "target_host": None,
        "mod_name": "default",
        "build": None,
        "params": None,
        "target": "llvm",
        "dev": tvm.cpu(0),
    }
    return mod_config


def test_pipe_config_check():
    # This function is used to trigger runtime error by applying wrong logic connection.

    # Get three pipeline modules here.
    (mod1, mod2, mod3), dshape = get_mannual_mod()

    # The input/output name is illegal and expects a runtime error.
    pipe_error = pipeline_executor.PipelineConfig()
    with pytest.raises(RuntimeError):
        pipe_error[mod1]["output"][9]

    with pytest.raises(RuntimeError):
        pipe_error[mod1]["input"]["data_9"]

    # The connection will cause a cycle in DAG and expects runtime error.
    with pytest.raises(RuntimeError):
        pipe_error[mod1]["output"][0].connect(pipe_error[mod2]["input"]["data_0"])
        pipe_error[mod2]["output"][0].connect(pipe_error[mod1]["input"]["data_0"])

    # The module connection is illegal and expects runtime error.

    with pytest.raises(RuntimeError):
        pipe_error[mod1]["output"][0].connect(pipe_error[mod1]["input"]["data_0"])

    with pytest.raises(RuntimeError):
        pipe_error[mod1]["input"]["data_0"].connect(pipe_error[mod1]["input"]["data_0"])

    with pytest.raises(RuntimeError):
        pipe_error[mod1]["input"]["data_0"].connect(pipe_error[mod2]["input"]["data_0"])

    with pytest.raises(RuntimeError):
        pipe_error[mod1]["output"][0].connect(pipe_error["input"]["data_0"])

    with pytest.raises(RuntimeError):
        pipe_error["input"]["data_0"].connect(pipe_error[mod1]["output"][0])

    with pytest.raises(RuntimeError):
        pipe_error["output"]["0"].connect(pipe_error[mod1]["output"][0])


def test_pipeline():
    if pipeline_executor.pipeline_executor_enabled():
        target_list = tvm.testing.enabled_targets()
        for target in target_list:
            # Get three pipeline modules here.
            (mod1, mod2, mod3), dshape = get_mannual_mod()

            # Batch data is prepared for pipeline feeding.
            datas = []
            for i in range(5):
                datas.append(np.full(dshape, 3 + i).astype("float32"))

            pipe_config = pipeline_executor.PipelineConfig()

            """ The global input named "data_0" will be connected to a input
            named "data_0" of mod1.
            """
            pipe_config["input"]["data_0"].connect(pipe_config[mod1]["input"]["data_0"])

            # The global Input named "data_1" will be connected to a input named "data_1" of mod2.
            pipe_config["input"]["data_1"].connect(pipe_config[mod2]["input"]["data_1"])

            # The mod1 output[0] will be connected to a input named "data_0" of mod2.
            pipe_config[mod1]["output"][0].connect(pipe_config[mod2]["input"]["data_0"])

            # The mod1 output[1] will be connected to a input named "data_0" of mod3.
            pipe_config[mod1]["output"][1].connect(pipe_config[mod3]["input"]["data_0"])

            # The mod2 output[2] will be connected to a input named "data_1" of mod3.
            pipe_config[mod2]["output"][0].connect(pipe_config[mod3]["input"]["data_1"])

            # The mod1 output[2] will be connected to global output[1].
            pipe_config[mod1]["output"][2].connect(pipe_config["output"]["0"])

            # The mod3 output[0] will be connected to global output[2].
            pipe_config[mod3]["output"][0].connect(pipe_config["output"]["1"])
            """Print configueration (print(pipe_config)), the result looks like following.

            Inputs
              |data_0: mod1:data_0
              |data_1: mod2:data_1
            
            output
              |output(1) : mod1.output(2)
              |output(2) : mod3.output(0)
            
            connections
              |mod1.output(0)-> mod2.data_0
              |mod1.output(1)-> mod3.data_0
              |mod2.output(0)-> mod3.data_1
            """

            # Set other parameter.
            pipe_config[mod1].target = target[0]
            pipe_config[mod1].dev = target[1]

            pipe_config[mod2].target = "llvm"
            pipe_config[mod2].dev = tvm.cpu(0)

            pipe_config[mod3].target = "llvm"
            pipe_config[mod3].dev = tvm.cpu(0)

            # Here is to check correctness for configuration generated by API.
            assert pipe_config.get_config() == get_manual_conf([mod1, mod2, mod3], target)

            # Build and create pipeline module.
            with tvm.transform.PassContext(opt_level=3):
                pipeline_mod_config = pipeline_executor.build(pipe_config)

            pipeline_module = pipeline_executor.PipelineModule(pipeline_mod_config)
            assert pipeline_module


if __name__ == "__main__":
    pytest.main([__file__])
