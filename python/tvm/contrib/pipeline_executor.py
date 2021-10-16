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
"""Pipeline executor that executes pipeline containing TVM PackedFunc."""
import json
import tvm._ffi
from tvm import relay
from tvm.contrib import graph_executor


def pipeline_executor_enabled():
    """check if pipeline executor enabled.
    Return
    ------
    enable: bool
        return pipeline executor get enabled or not
    """
    pipeline_enabled = False
    try:
        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create")
        assert pipelinecreate
        pipeline_enabled = True
    except ValueError:
        print("pipeline executor not enabled!")

    return pipeline_enabled


def write_file(file_name, data, mode):
    """write data into file

    Parameters
    ----------
    file_name: str
        file name
    data: str
        data
    mode: str
        file open mode
    """
    if file_name:
        with open(file_name, mode) as file_handle:
            file_handle.write(data)


def build_pipeline(mod_n_configs, export_path=None):
    """build module list that can use for pipeline execution.

    Parameters
    ----------
    mod_n_configs: Dict[IRModule, Dict[str, Any]]
        build configuration informaton, structure like following.
        {IRModule: {"target":target,
                    "target_host":target_host,
                    "params":params,
                    "mod_name"mod_name,
                    "build":build}}
    export_path: str
        export build result into file

    Returns
    -------
    ret: List[IRModule]
        list of IRModule
    string_config: Dict[int, Dict[str, any]]
        pipeline configuration
    """
    mods = {}
    config_len = len(mod_n_configs)
    string_config = [{} for _ in range(config_len)]
    for _, (ir_mod, mod_config) in enumerate(mod_n_configs.items()):
        # init lib_name and json_name params with empty
        lib_name = ""
        json_name = ""
        params_name = ""
        # Get module configuration
        assert "pipeline" in mod_config and "mod_indx" in mod_config["pipeline"]
        # Get module index in pipeline configuration
        mconf = mod_config["pipeline"].copy()
        # Get mod device config
        dev = mod_config["dev"]
        mod_indx = mconf["mod_indx"]
        assert mod_indx < config_len
        build_func = relay.build
        # if there is a self defined build function then use it.
        if "build" in mod_config and mod_config["build"]:
            build_func = mod_config["build"]

        # build IRModule
        mod = build_func(
            ir_mod,
            mod_config["target"],
            params=mod_config["params"],
            target_host=mod_config["target_host"],
            mod_name=mod_config["mod_name"],
        )

        if export_path:
            graph, lib, params = mod
            lib_name = "{}/lib{}.so".format(export_path, mod_indx)
            json_name = "{}/json{}".format(export_path, mod_indx)
            params_name = "{}/params{}".format(export_path, mod_indx)
            lib.export_library(lib_name)
            write_file(json_name, graph, "w")
            write_file(params_name, relay.save_param_dict(params), "wb")

        mconf["lib_name"] = lib_name
        mconf["json_name"] = json_name
        mconf["params_name"] = params_name
        mconf["dev"] = "{},{}".format(dev.device_type, dev.device_id)
        # Create pipeline configuration
        string_config[mod_indx] = mconf
        # associate mod with device
        mods[mod] = {"dev": dev}

    if export_path:
        write_file("{}/config".format(export_path), json.dumps(string_config), "w")
        # with open("{}/config".format(export_path), "w") as config_file:
        # config_file.write(json.dumps(string_config))

    # return IRModule list and pipeline configuration
    return mods, string_config


def create(pipeline_mods, mod_config):
    """Create a pipeline runtime executor.

    Parameters
    ----------
    pipeline_mods : List[IRModule]
        list of IRModule

    mod_config : Dict[int, Dict[str, Any]]
        modules and modules dependency configuration informaiton.

    Returns
    -------
    submodule : PipelineModule
        Runtime pipeline module.
    """

    mods = []
    for pipeline_mod in pipeline_mods:
        mod = graph_executor.GraphModule(
            pipeline_mod["default"](pipeline_mods[pipeline_mod]["dev"])
        )
        mods.append(mod.module)

    submodule = PipelineModule(mods, json.dumps(mod_config))
    # submodule = PipelineModule(pipeline_mods, json.dumps(mod_config))
    return submodule


class PipelineModule(object):
    """Wrapper runtime module. This is a thin wrapper of the underlying TVM module.
    you can also directly call set_input, run, and get_output of underlying module functions.

    Parameters
    ----------
    graph_module : List[GraphModule]
        The internal tvm module that holds the actual graph functions.

    pipeline_config : Dict[IRModule, Dict[str, Any]]
        modules and modules dependency configuration informaiton.

    """

    def __init__(self, modules, pipeline_config):
        mods = []
        for module in modules:
            mods.append(module)

        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create")
        assert pipelinecreate
        module = pipelinecreate(mods, pipeline_config)

        self.graph_modules_ = modules

        self._set_input = module["set_input"]
        self._run = module["run"]
        self._stop = module["stop"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_num_inputs = module["get_num_inputs"]

    def set_input(self, key, value, mod_idx=0, params=None):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : array_like
            The input key

        value : array_like.
            The input key

        mod_idx : int
            the submodule index

        params : dict of str to NDArray
            Additional arguments
        """
        assert mod_idx >= 0
        self._set_input(key, tvm.nd.array(value, tvm.cpu()), mod_idx)

        if params:
            for param in params:
                self.graph_modules_[mod_idx].set_input(**param)

    def run(self):
        """Run forward execution of the graph"""
        self._run()

    def stop(self):
        """Stop pipeline run"""
        self._stop()

    def get_num_outputs(self):
        """Get the number of outputs from the graph

        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    def get_num_inputs(self):
        """Get the number of inputs to the graph

        Returns
        -------
        count : int
            The number of inputs.
        """
        return self._get_num_inputs()

    def get_input(self, input_indx, runtime_index=0, out=None):
        """Get index-th input to out

        Parameters
        ----------
        index : int
            The input index

        out : NDArray
            The output array container
        """
        if out:
            self._get_input(input_indx, runtime_index).copyto(out)
            return out

        return self._get_input(input_indx, runtime_index)

    def get_output(self):
        """Get index-th output to out

        Parameters
        ----------
        index : int
            The output index
        """
        return self._get_output()

    def __getitem__(self, key):
        """Get internal module function

        Parameters
        ----------
        key : str
            The key to the module.
        """
        return self.module[key]
