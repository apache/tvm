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
import os
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform
from tvm.contrib import graph_executor, pipeline_executor


def get_mannual_mod():
    # Get a list of modules representing subgraphs.
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

    # There are three outputs in the first model.

    net1_output1 = relay.add(data, mv1)
    net1_output2 = relay.subtract(data, mv2)
    net1_output3 = relay.multiply(data, mv3)

    # The second model use output named net1_output1 of the first model as the first input,
    # the second input of the second model is data21.
    net2 = relay.add(data_net1_output_1, mv2)
    net2 = relay.add(net2, data21)
    net2_output = relay.add(net2, mv3)

    # The third model use the output named net2_output of the second model as the first input
    # and use the output named net1_output2 of the first model as the second input.
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


def recreate_parameters(mod):
    # Get the binding parameters from a module, then create a same parameters which have different
    # data value. This function can be used to test "set_param" function.
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, "llvm")

    mod1_customized_params = {}
    for key, value in lib.params.items():
        new_value = value.numpy() + np.full(value.shape, 10).astype(value.dtype)
        mod1_customized_params[key] = tvm.nd.array(new_value)
    return mod1_customized_params, mod


def get_manual_conf(mods, target):
    # This function is used to generate manual pipeline configuration.
    mod_config = {}
    # The third output is the final output, the second output is for mod3, the first output
    # is for mod2 input.
    pipe_config1 = {
        "mod_idx": 0,
        "output": [
            {"output_idx": 0, "dependencies": [{"mod_idx": 1, "input_name": "data_0"}]},
            {"output_idx": 1, "dependencies": [{"mod_idx": 2, "input_name": "data_0"}]},
            {"output_idx": 2, "dependencies": [{"global_output_index": 0}]},
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
        "mod_idx": 1,
        "output": [
            {"output_idx": 0, "dependencies": [{"mod_idx": 2, "input_name": "data_1"}]},
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
        "mod_idx": 2,
        "output": [{"output_idx": 0, "dependencies": [{"global_output_index": 1}]}],
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


def run_modules(
    mod_configs,
    dev,
    target,
    global_input_name,
    global_input_data,
    mod_set_input,
    input_name,
    input_data,
    params_mod=None,
    params=None,
):
    # Use graph executor to run these modules in serialized. The resturn data is used to
    # verify the result comming from pipeline executor.
    mod_input = {}
    final_output = {}
    idx = 0
    for mod in mod_configs:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)

        m = graph_executor.GraphModule(lib["default"](dev))
        # Get the input information from "mod_input" then set the input to the module.
        if idx in mod_input:
            for input in mod_input[idx]:
                input = mod_input[idx][input]
                m.set_input(input["index"], input["data"])
        else:
            m.set_input(global_input_name, global_input_data)

        # Set input data to the module which is "mod_set_input".
        if mod == mod_set_input:
            m.set_input(input_name, input_data)
        # If the module is "params_mod" then set the parameters to this module.
        if params_mod == mod:
            m.set_input(None, None, **params)

        m.run()
        n = m.get_num_outputs()
        # Set current output as inputs of next module.
        mconfig = mod_configs[mod]
        for output in mconfig["pipeline"]["output"]:
            output_data = m.get_output(output["output_idx"]).numpy()
            for dep in output["dependencies"]:
                is_global = False
                if "global_output_index" in dep:
                    is_global = True
                    name = dep["global_output_index"]
                else:
                    mod_idx = dep["mod_idx"]
                    name = dep["input_name"]
                if is_global:
                    final_output[name] = output_data
                else:
                    if mod_idx in mod_input:
                        mod_input[mod_idx][name] = {"index": name, "data": output_data}
                    else:
                        mod_input[mod_idx] = {name: {"index": name, "data": output_data}}
        idx = idx + 1

    return final_output


def test_pipe_config_check():
    # This function is used to trigger runtime error by applying wrong logic connection.

    # Get three pipeline modules here.
    (mod1, mod2, mod3), dshape = get_mannual_mod()

    # The input or output name is illegal and expects a runtime error.
    pipe_error = pipeline_executor.PipelineConfig()
    with pytest.raises(RuntimeError):
        pipe_error[mod1]["output"][9]

    with pytest.raises(RuntimeError):
        pipe_error[mod1]["input"]["data_9"]

    # The module connection will cause a cycle in DAG and expects runtime error.
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
            # Get the three pipeline modules here.
            (mod1, mod2, mod3), dshape = get_mannual_mod()
            # Choose a module to create new parameters.
            customized_parameters, customized_mod = recreate_parameters(mod1)
            # Prepare batch data for pipeline computation.
            datas = []
            for i in range(5):
                datas.append(np.full(dshape, 3 + i).astype("float32"))

            pipe_config = pipeline_executor.PipelineConfig()

            # The pipeline param named "param_0" will be connected to the param of mod1.
            pipe_config["param"]["param_0"].connect(pipe_config[customized_mod]["param"])
            # The pipeline input named "data_0" will be connected to a input named "data_0"
            # of mod1.
            pipe_config["input"]["data_0"].connect(pipe_config[mod1]["input"]["data_0"])

            # The pipeline Input named "data_1" will be connected to a input named "data_1"
            # of mod2.
            pipe_config["input"]["data_1"].connect(pipe_config[mod2]["input"]["data_1"])

            # The mod1 output[0] will be connected to a input named "data_0" of mod2.
            pipe_config[mod1]["output"][0].connect(pipe_config[mod2]["input"]["data_0"])

            # The mod1 output[1] will be connected to a input named "data_0" of mod3.
            pipe_config[mod1]["output"][1].connect(pipe_config[mod3]["input"]["data_0"])

            # The mod2 output[2] will be connected to a input named "data_1" of mod3.
            pipe_config[mod2]["output"][0].connect(pipe_config[mod3]["input"]["data_1"])

            # The mod1 output[2] will be connected to pipeline output[0].
            pipe_config[mod1]["output"][2].connect(pipe_config["output"]["0"])

            # The mod3 output[0] will be connected to pipeline output[1].
            pipe_config[mod3]["output"][0].connect(pipe_config["output"]["1"])
            # Print configueration (print(pipe_config)), the result looks like following.
            #
            # Inputs
            #   |data_0: mod1:data_0
            #   |data_1: mod2:data_1
            #
            # output
            #   |output(1) : mod1.output(2)
            #   |output(2) : mod3.output(0)
            #
            # connections
            #   |mod1.output(0)-> mod2.data_0
            #   |mod1.output(1)-> mod3.data_0
            #   |mod2.output(0)-> mod3.data_1

            # Set other parameters.
            pipe_config[mod1].target = target[0]
            pipe_config[mod1].dev = target[1]

            pipe_config[mod2].target = "llvm"
            pipe_config[mod2].dev = tvm.cpu(0)

            pipe_config[mod3].target = "llvm"
            pipe_config[mod3].dev = tvm.cpu(0)

            # Here is to check the correctness of the configuration generated by API.
            mconfig = pipe_config.get_config()
            assert mconfig["module_connection"] == get_manual_conf([mod1, mod2, mod3], target)

            # Build and create a pipeline module.
            with tvm.transform.PassContext(opt_level=3):
                pipeline_mod_factory = pipeline_executor.build(pipe_config)

            # Export the parameter configuration to a file.
            directory_path = tvm.contrib.utils.tempdir().temp_dir
            # If the directory does not exist, create it.
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            config_file_name = pipeline_mod_factory.export_library(directory_path)

            # Use the output of build to create and initialize PipelineModule.
            pipeline_module = pipeline_executor.PipelineModule(pipeline_mod_factory)
            assert pipeline_module

            # Use the import function to create and initialize PipelineModule.
            pipeline_module_test = pipeline_executor.PipelineModule.load_library(config_file_name)
            assert pipeline_module_test.num_outputs == 2
            # Set customized params
            pipeline_module_test.set_params("param_0", customized_parameters)
            for data in datas:
                # Get the result from the graph executor execution without setting customized
                # parameters.
                wrong_output = run_modules(
                    mconfig["module_connection"],
                    tvm.cpu(),
                    "llvm",
                    "data_0",
                    data,
                    mod2,
                    "data_1",
                    data,
                )
                # Get the result from the graph executor execution with setting customized
                # parameters.
                normal_output = run_modules(
                    mconfig["module_connection"],
                    tvm.cpu(),
                    "llvm",
                    "data_0",
                    data,
                    mod2,
                    "data_1",
                    data,
                    customized_mod,
                    customized_parameters,
                )
                # Get the result from the pipeline executor
                pipeline_module_test.set_input("data_0", data)
                pipeline_module_test.set_input("data_1", data)
                # Run the pipeline executor in serialized.
                pipeline_module_test.run(True)
                outputs = pipeline_module_test.get_output()
                for i in range(len(outputs)):
                    tvm.testing.assert_allclose(normal_output[i], outputs[i].numpy())
                    assert not (normal_output[i] == wrong_output[i]).all()

            pipeline_module_test.stop()


if __name__ == "__main__":
    pytest.main([__file__])
