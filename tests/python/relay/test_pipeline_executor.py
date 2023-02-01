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
import time
import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform, build_module
from tvm.relay.testing import run_opt_pass
from tvm.contrib import graph_executor, pipeline_executor, pipeline_executor_build
from tvm._ffi import get_global_func
from tvm.contrib import cc as _cc


def graph_split(expr, split_conf, params=None):
    """Splitting the graph into a list of subgraphs"""

    def get_dep_var(sub_var_dep):
        return [var for var in sub_var_dep[len(sub_var_dep) - 1]["ref_nodes"]]

    def parse_dependency(value, snode_dep, new_input_idx):
        new_args = []
        need_update = False
        for var in value.args:
            is_free_var = False
            for dep in snode_dep[:-1]:
                if var in dep["nodes"]:
                    # Mark the previous subgraph node as a dependency.
                    dep["nodes"][var] += 1
                    dep["ref_nodes"][var] = dep["nodes"][var]
                    # The var of this call is a free_var
                    is_free_var = True
            # if the var of this call is a free_var, recreate it and give it a fixed input name.
            if is_free_var:
                need_update = True
                new_args.append(relay.var(f"data_n_{new_input_idx}", var.checked_type))
                new_input_idx += 1
            else:
                new_args.append(var)
        # if the 'tvm.relay.expr.Call' has a free_var, recreate it with new name as 'data_n_*'.
        if need_update:
            value = tvm.relay.expr.Call(
                value.op, new_args, value.attrs, value.type_args, value.span
            )
        return value, snode_dep, new_input_idx

    def merge_constant_expr(constant_expr, expr):
        # merge constant express with a express
        if not isinstance(constant_expr.body, tvm.relay.expr.Let):
            return tvm.relay.expr.Let(constant_expr.var, constant_expr.value, expr)

        return tvm.relay.expr.Let(
            constant_expr.var, constant_expr.value, merge_constant_expr(constant_expr.body, expr)
        )

    def _recursion(anf, pipeline_mods, split_conf, constant_expr):
        # Enumurate all operators of compute graph, then split the compute graph into a group of
        # subgraph.
        nonlocal operator_index_map
        nonlocal new_input_idx
        nonlocal snode_dep
        cur_node_dep = snode_dep[len(snode_dep) - 1]
        if isinstance(anf, tvm.relay.Function):
            return tvm.relay.Function(
                anf.params,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
                anf.ret_type,
                anf.type_params,
                anf.attrs,
            )
        if isinstance(anf, tvm.relay.expr.Let):
            value = anf.value
            # record the constant expr to make sure all sugraphs can find correct constant.
            if isinstance(value, tvm.relay.expr.Constant):
                if not constant_expr:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, anf.var)
                else:
                    constant_expr = tvm.relay.expr.Let(anf.var, value, constant_expr)
            if isinstance(value, tvm.relay.expr.Call):
                new_args = []
                # build current var list
                cur_node_dep["nodes"][anf.var] = 0
                # Get the dependency information of the nodes.
                value, snode_dep, new_input_idx = parse_dependency(value, snode_dep, new_input_idx)
                if isinstance(value.op, tvm.ir.Op):
                    if value.op.name in operator_index_map:
                        operator_index_map[value.op.name] += 1
                    else:
                        operator_index_map[value.op.name] = 0
                    split_operator_name = split_conf[0]["op_name"] if split_conf else ""
                    split_operator_index = split_conf[0]["op_index"] if split_conf else ""
                    # if a operator name and repeating count in the network match with the values
                    # of the 'split configuration', then this place is where we should do the
                    # graph splitting.
                    if (
                        split_conf
                        and split_operator_name in operator_index_map
                        and operator_index_map[split_operator_name] >= split_operator_index
                    ):
                        # Do graph splitting.
                        split_conf.pop(0)
                        snode_dep.append({"nodes": {}, "ref_nodes": {}})
                        ann = _recursion(
                            anf.body,
                            pipeline_mods,
                            split_conf,
                            constant_expr,
                        )
                        snode_dep.pop()
                        dep_vars = get_dep_var(snode_dep)
                        # When the nodes of the current subgraph are the depedency node of another
                        # subgraph, we need to set them as the output of current subgraph.
                        body = relay.Tuple(dep_vars) if len(dep_vars) > 1 else anf.var
                        # when the operator of current subgraph uses previous subgraph constant
                        # as the argument of a "relay.expr.call", such constant may become a free
                        # varaible if the constant does not exist in the current subgraph.
                        # merge the previous constant with current subgraph to avoid such issue.
                        if constant_expr:
                            ann = merge_constant_expr(constant_expr, ann)
                        ann = run_opt_pass(ann, transform.ToGraphNormalForm())
                        mod = tvm.IRModule.from_expr(ann)
                        pipeline_mods.insert(0, mod)
                        # Return the last node of the current subgraph.
                        return tvm.relay.expr.Let(anf.var, value, body)
            return tvm.relay.expr.Let(
                anf.var,
                value,
                _recursion(anf.body, pipeline_mods, split_conf, constant_expr),
            )
        else:
            return anf

    snode_dep = [{"nodes": {}, "ref_nodes": {}}]
    pipeline_mods = []
    operator_index_map = {}
    # Used to tracking new input which caused by graph splitting.
    new_input_idx = 0
    constant_expr = None
    subgraph_split_conf = split_conf.copy()
    # Binding the parameters.
    if params:
        expr = build_module.bind_params_by_name(expr, params)
    anf = run_opt_pass(expr, transform.ToANormalForm())
    anf = run_opt_pass(anf, transform.InferType())
    ann = _recursion(
        anf,
        pipeline_mods,
        subgraph_split_conf,
        constant_expr,
    )
    ann = run_opt_pass(ann.body, transform.ToGraphNormalForm())
    mod = tvm.IRModule.from_expr(ann)
    pipeline_mods.insert(0, mod)
    return pipeline_mods


def get_network():
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
    net1_output3 = relay.concatenate((net1_output1, net1_output2), axis=0)
    (net1_output3, _) = relay.split(net1_output3, indices_or_sections=2, axis=0)
    net1_output3 = relay.add(net1_output3, mv2)
    # The second model uses the output named net1_output3 of the first model as the first input,
    # the second input of the second model is data21.
    net2 = relay.add(net1_output3, mv2)
    net2 = relay.add(net2, data21)
    net2_output = relay.add(net2, mv3)
    # The third model uses the output named net2_output of the second model as the first input
    # and uses the output named net1_output2 of the first model as the second input.
    net3 = relay.multiply(net2_output, mv3)
    net3 = relay.add(net3, net1_output2)
    return tvm.IRModule.from_expr(relay.Function([data, data21], relay.Tuple([net3]))), dshape


def get_split_mod():
    mod, dshape = get_network()
    split_conf = [{"op_name": "add", "op_index": 1}, {"op_name": "add", "op_index": 4}]
    mods = graph_split(mod["main"], split_conf)
    return mods, dshape


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


def get_manual_conf(mods, target):
    # This function is used to generate manual pipeline configuration.
    mod_config = {}
    # The third output is the final output, the second output is for mod3, the first output
    # is for mod2 input.
    pipe_config1 = {
        "mod_idx": 0,
        "cpu_affinity": "0",
        "output": [
            {"output_idx": 0, "dependencies": [{"mod_idx": 1, "input_name": "data_n_0"}]},
            {"output_idx": 1, "dependencies": [{"mod_idx": 2, "input_name": "data_n_2"}]},
        ],
    }
    mod_config[mods[0]] = {
        "pipeline": pipe_config1,
        "target_host": None,
        "mod_name": "default",
        "build": None,
        "params": None,
        "target": target[0],
        "fcompile": _cc.create_shared,
        "dev": target[1],
    }

    pipe_config2 = {
        "mod_idx": 1,
        "cpu_affinity": "0",
        "output": [
            {"output_idx": 0, "dependencies": [{"mod_idx": 2, "input_name": "data_n_1"}]},
        ],
    }
    mod_config[mods[1]] = {
        "pipeline": pipe_config2,
        "target_host": None,
        "mod_name": "default",
        "build": None,
        "params": None,
        "target": "llvm",
        "fcompile": None,
        "dev": tvm.cpu(0),
    }

    pipe_config3 = {
        "mod_idx": 2,
        "cpu_affinity": "0",
        "output": [{"output_idx": 0, "dependencies": [{"global_output_index": 0}]}],
    }
    mod_config[mods[2]] = {
        "pipeline": pipe_config3,
        "target_host": None,
        "mod_name": "default",
        "build": None,
        "params": None,
        "target": "llvm",
        "fcompile": None,
        "dev": tvm.cpu(0),
    }
    return mod_config


def recreate_parameters(mod):
    # Get the binding parameters from a module, then create the same parameters with different data.
    # This function is used to test the "parameter" connection.
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, "llvm")

    mod_customized_params = {}
    for key, value in lib.params.items():
        new_value = value.numpy() + np.full(value.shape, 10).astype(value.dtype)
        mod_customized_params[key] = tvm.nd.array(new_value)
    return mod_customized_params, mod


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
    # Running modules in serialized model. The returnning data are used to verify the pipeline
    # executor result.
    mod_input = {}
    final_output = {}
    idx = 0
    for mod in mod_configs:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)

        m = graph_executor.GraphModule(lib["default"](dev))
        # Getting the input data then setting the input data into the module.
        if idx in mod_input:
            for input in mod_input[idx]:
                input = mod_input[idx][input]
                m.set_input(input["index"], input["data"])
        else:
            m.set_input(global_input_name, global_input_data)

        # Setting the "input_data" into the module.
        if mod == mod_set_input:
            m.set_input(input_name, input_data)
        # If the module is "params_mod" then setting the parameters to this module.
        if params_mod == mod:
            m.set_input(None, None, **params)

        m.run()
        n = m.get_num_outputs()
        # Setting current output data as  the input of next module.
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


def reset_cpu_affinity(affinity):
    # Restore the CPU affinity into the default value.
    config_threadpool = get_global_func("runtime.config_threadpool")
    config_threadpool(-2, 0)
    os.sched_setaffinity(0, affinity)


def test_pipe_runtime_error_check():
    # This function is used to trigger runtime error by applying wrong logic.
    if pipeline_executor_build.pipeline_executor_build_enabled():
        # Get three pipeline modules here.
        (mod1, mod2, mod3), dshape = get_split_mod()

        # The input or output name is illegal and expects a runtime error.
        pipe_error = pipeline_executor_build.PipelineConfig()
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

        # Create pipeline executor to check the executor runtime errors.
        pipe_config = pipeline_executor_build.PipelineConfig()
        pipe_config[mod1].target = "llvm"
        pipe_config[mod1].dev = tvm.cpu(0)
        pipe_config["param_group"]["param_0"].connect(pipe_config[mod1]["param"])
        pipe_config[mod1]["output"][0].connect(pipe_config["output"]["0"])
        # Build and create a pipeline module.
        with tvm.transform.PassContext(opt_level=3):
            pipeline_mod_factory = pipeline_executor_build.build(pipe_config)
        pipeline_module = pipeline_executor.PipelineModule(pipeline_mod_factory)
        customized_parameters, _ = recreate_parameters(mod1)

        # Checking the pipeline executor runtime errors.
        with pytest.raises(RuntimeError):
            pipeline_module.set_params("param_0", None)

        with pytest.raises(RuntimeError):
            pipeline_module.set_params("param_1", customized_parameters)


def test_pipeline():
    if pipeline_executor_build.pipeline_executor_build_enabled():
        target_list = tvm.testing.enabled_targets()
        for target in target_list:
            affinity = os.sched_getaffinity(0)
            # Get the three pipeline modules here.
            (mod1, mod2, mod3), dshape = get_split_mod()

            # Prepare batch data for pipeline computation.
            datas = []
            for i in range(5):
                datas.append(np.full(dshape, 3 + i).astype("float32"))

            pipe_config = pipeline_executor_build.PipelineConfig()

            customized_parameters, customized_parameters_mod = recreate_parameters(mod1)
            assert customized_parameters_mod == mod1
            # The global parameters group named "param_0" will be connected to "mod1" as parameters.
            pipe_config["param_group"]["param_0"].connect(pipe_config[mod1]["param"])
            # The pipeline input named "data_a" will be connected to a input named "data_0"
            # of mod1.
            pipe_config["input"]["data_a"].connect(pipe_config[mod1]["input"]["data_0"])

            # The pipeline Input named "data_b" will be connected to a input named "data_1"
            # of mod2.
            pipe_config["input"]["data_b"].connect(pipe_config[mod2]["input"]["data_1"])

            # The mod1 output[0] will be connected to a input named "data_n_0" of mod2.
            pipe_config[mod1]["output"][0].connect(pipe_config[mod2]["input"]["data_n_0"])

            # The mod1 output[1] will be connected to a input named "data_n_2" of mod3.
            pipe_config[mod1]["output"][1].connect(pipe_config[mod3]["input"]["data_n_2"])

            # The mod2 output[2] will be connected to a input named "data_n_1" of mod3.
            pipe_config[mod2]["output"][0].connect(pipe_config[mod3]["input"]["data_n_1"])

            # The mod3 output[0] will be connected to pipeline output[0].
            pipe_config[mod3]["output"][0].connect(pipe_config["output"]["0"])
            # Print configuration (print(pipe_config)), the result looks like following.
            #
            # Params
            #   |param_0: mod0:param
            #
            # Inputs
            #   |data_a: mod0:data_0
            #   |data_b: mod1:data_1
            #
            # output
            #   |output(0) : mod2.output(0)
            #
            # connections
            #   |mod0.output(0)-> mod1.data_n_0
            #   |mod0.output(1)-> mod2.data_n_2
            #   |mod1.output(0)-> mod2.data_n_1

            # Set other parameters.
            pipe_config[mod1].target = target[0]
            pipe_config[mod1].dev = target[1]
            pipe_config[mod1].cpu_affinity = "0"
            pipe_config[mod1].fcompile = _cc.create_shared

            pipe_config[mod2].target = "llvm"
            pipe_config[mod2].dev = tvm.cpu(0)
            pipe_config[mod2].cpu_affinity = "0"

            pipe_config[mod3].target = "llvm"
            pipe_config[mod3].dev = tvm.cpu(0)
            pipe_config[mod3].cpu_affinity = "0"
            # Checking the configuration of modules dependency.
            mconfig = pipe_config.get_config()
            assert mconfig["module_connection"] == get_manual_conf([mod1, mod2, mod3], target)

            # Build and create a pipeline module.
            with tvm.transform.PassContext(opt_level=3):
                pipeline_mod_factory = pipeline_executor_build.build(pipe_config)

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
            assert pipeline_module_test.num_outputs == 1

            input_map = pipeline_module_test.get_input_pipeline_map("data_b")
            assert input_map[0] == "1" and input_map[1] == "data_1"
            input_map = pipeline_module_test.get_input_pipeline_map("data_a")
            assert input_map[0] == "0" and input_map[1] == "data_0"
            module_index = pipeline_module_test.get_params_group_pipeline_map("param_0")
            assert module_index == 0
            # Using the parameters group name to set parameters.
            pipeline_module_test.set_params("param_0", customized_parameters)
            normal_outputs = []
            for round in range(0, len(datas)):
                data = datas[round]
                # Getting the result without setting customized parameters.
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
                # Getting the result with setting customized parameters.
                normal_output = run_modules(
                    mconfig["module_connection"],
                    tvm.cpu(),
                    "llvm",
                    "data_0",
                    data,
                    mod2,
                    "data_1",
                    data,
                    customized_parameters_mod,
                    customized_parameters,
                )
                # Appending the normal output into the list in order to do future correctness
                # checking.
                normal_outputs.append(normal_output)
                # Setting the input data into the pipeline executor.
                pipeline_module_test.set_input("data_a", tvm.nd.array(data))
                pipeline_module_test.set_input("data_b", tvm.nd.array(data))
                input_map = pipeline_module_test.get_input_pipeline_map("data_a")
                # Checking whether the input setting of the first runtime is successful.
                # The input of the rest of runtime will go into a queue and we can not check
                # these input data here.
                if input_map[0] == "0":
                    input_data = pipeline_module_test.get_input("data_a")
                    tvm.testing.assert_allclose(data, input_data.numpy())

                assert pipeline_module_test.num_inputs == 2
                # Running the pipeline executor in the pipeline mode.
                pipeline_module_test.run()

            for k in range(0, len(datas)):
                statistic_time = 0
                outputs = pipeline_module_test.get_output()
                while len(outputs) == 0:
                    outputs = pipeline_module_test.get_output()
                    statistic_time = statistic_time + 1
                    # Setting the timeout to 10 seconds.
                    assert statistic_time < 5
                    time.sleep(1)

                for i in range(len(outputs)):
                    tvm.testing.assert_allclose(normal_outputs[k][i], outputs[i].numpy())
                    assert not (normal_output[i] == wrong_output[i]).all()

                    assert pipeline_module_test.num_executing_pipeline == round + 1

            # Reset the cpu affinity after a test.
            reset_cpu_affinity(affinity)


if __name__ == "__main__":
    tvm.testing.main()
