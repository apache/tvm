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
    """check if pipeline executor enabled."""
    pipeline_enabled = False
    try:
        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create")
        assert pipelinecreate
        pipeline_enabled = True
    except ValueError:
        print("pipeline executor not enabled!")

    return pipeline_enabled


def build_pipeline(config):
    """build module list that can use for pipeline execution.

    Parameters
    ----------

    config: Dict[IRModule, Dict[str, Any]]
        build configuration informaton, structure like following.
        {IRModule: {"target":target,
                    "target_host":target_host,
                    "params":params,
                    "mod_name"mod_name,
                    "build":build}}

    Returns
    -------
    ret: List[IRModule]
        list of IRModule
    string_config: Dict[int, Dict[str, any]]
        pipeline configuration
    """
    mods = {}
    config_len = len(config)
    string_config = [{} for _ in range(config_len)]
    for ir_mod in config:
        # Get module configuration
        mod_config = config[ir_mod]
        assert "pipeline" in mod_config and "mod_indx" in mod_config["pipeline"]
        # Get module index in pipeline configuration
        mod_indx = mod_config["pipeline"]["mod_indx"] - 1
        assert mod_indx < config_len
        # Create pipeline configuration
        string_config[mod_indx] = mod_config["pipeline"]
        build_func = relay.build
        # if there is a self defined build function then use it.
        if "build" in mod_config and mod_config["build"]:
            build_func = mod_config.build

        # build IRModule
        mod = build_func(
            ir_mod,
            mod_config["target"],
            params=mod_config["params"],
            target_host=mod_config["target_host"],
            mod_name=mod_config["mod_name"],
        )

        mods[mod] = {"dev": mod_config["dev"]}

    # return IRModule list and pipeline configuration
    return mods, string_config


def create(mods, mod_config):
    """Create a pipeline runtime executor.

    Parameters
    ----------
    mods : List[IRModule]
        list of IRModule

    mod_config : Dict[IRModule, Dict[str, Any]]
        modules and modules dependency configuration informaiton.

    Returns
    -------
    submodule : PipelineModule
        Runtime pipeline module.
    """
    pipeline_mods, string_config = build_pipeline(mod_config)

    mods = []
    for pipeline_mod in pipeline_mods:
        mod = graph_executor.GraphModule(
            pipeline_mod["default"](pipeline_mods[pipeline_mod]["dev"])
        )

        mods.append(mod)

    submodule = PipelineModule(mods, json.dumps(string_config))
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

    def __init__(self, graph_modules, pipeline_config):
        mods = []
        for module in graph_modules:
            mods.append(module.module)

        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create")
        assert pipelinecreate
        module = pipelinecreate(mods, pipeline_config)

        self.graph_modules_ = graph_modules

        self._set_input = module["set_input"]
        self._run = module["run"]
        self._stop = module["stop"]
        self._get_output = module["get_output"]
        self._get_input = module["get_input"]
        self._get_num_outputs = module["get_num_outputs"]
        self._get_num_inputs = module["get_num_inputs"]

    def set_input(self, key, value, modindx=1, params=None):
        """Set inputs to the module via kwargs

        Parameters
        ----------
        key : array_like
           The input key

        value : array_like.
           The input key

        params : dict of str to NDArray
           Additional arguments
        """
        assert modindx >= 1
        if key is not None:
            self._set_input(key, tvm.nd.array(value, tvm.cpu()), modindx)

        if params:
            for param in params:
                self.graph_modules_[modindx - 1].set_input(**param)

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
