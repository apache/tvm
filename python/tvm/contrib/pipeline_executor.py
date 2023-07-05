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
"""Pipeline executor that executes a series of modules in a pipeline fashion."""
import json
import os
import time
from tvm import runtime
from tvm._ffi import get_global_func
from tvm.contrib import graph_executor


def pipeline_executor_enabled():
    """Check if the pipeline executor is enabled.

    Return
    -------
    enable: bool
        Return whether the pipeline executor is enabled.
    """
    return get_global_func("tvm.pipeline_executor.create", allow_missing=True) is not None


class PipelineModule(object):
    """Wrapper of runtime module, caller can use this module to set parameters and get outputs.

    Parameters
    ----------
    module : Union[PipelineExecutorFactoryModule, Module]
        Common interface for pipeline executor factory modules or Module.
    """

    def __init__(self, module):
        if isinstance(module, PipelineExecutorFactoryModule):
            self.module = module.get_pipeline_executor_module()
        else:
            self.module = module
        # Get the packed functions from the pipeline executor.
        self._get_params_group_pipeline_map = self.module["get_params_group_pipeline_map"]
        self._run = self.module["run"]
        self._set_param = self.module["set_param"]
        self._set_input = self.module["set_input"]
        self._get_input = self.module["get_input"]
        self._get_output = self.module["get_output"]
        self._get_num_outputs = self.module["get_num_outputs"]
        self._get_num_inputs = self.module["get_num_inputs"]
        self._get_input_pipeline_map = self.module["get_input_pipeline_map"]
        self._get_pipe_execute_count = self.module["get_execute_count"]

    def run(self):
        """Run the pipeline executor."""
        self._run()

    def get_input_pipeline_map(self, name):
        """Using the "name" to get the corresponding subgraph index and also get the "input name"
        of the corresponding subgraph interface.
        Returns
        -------
        input map: Array[str]
            Returning the index and "input name" of the subgraph.
        """
        return self._get_input_pipeline_map(name)

    def get_params_group_pipeline_map(self, name):
        """Use the name of the parameters group to get the corresponding runtime module index.

        Parameters
        ----------
        name: str
            The parameter group name.

        Returns
        -------
        module_index: int
            The index of the runtime module.
        """
        return self._get_params_group_pipeline_map(name)

    def set_input(self, key, value):
        """Set the input via input name.

        Parameters
        ----------
        key : str
            The input name
        value : array_like.
            The input value
        """
        self._set_input(key, value)

    def set_params(self, params_group_name, params_data):
        """Set the parameter group value given the parameter group name. Note that the parameter
        group name is declared in the pipeline executor config.

        Parameters
        ----------
        params_group_name : str
            The parameters group name.

        params_data : Dict[str, NDArray]
            A map from parameter name to data.
        """
        if not params_data:
            raise RuntimeError('"params_data is empty!"')

        for key, val in params_data.items():
            self._set_param(params_group_name, key, val)

    def get_input(self, key):
        """Get the input via an input name.
        Parameters
        ----------
        key : str
            The input key
        Returns
        -------
        data : NDArray
            The input data.
        """
        return self._get_input(key)

    def get_output(self, synchronize=True, sleep_interval=0.001):
        """Get the output.
        Returns
        -------
        data : Array[NDArray]
            A list of output data.
        synchronize : BOOL
            Whether to do a synchronize poll.
        sleep_interval : Float32
            When doing the synchronize loop poll, how many seconds the loop should sleep for yield.
        """
        outputs = []
        if not synchronize:
            outputs = self._get_output()
        else:
            while not outputs:
                outputs = self._get_output()
                time.sleep(sleep_interval)

        return outputs

    @property
    def num_executing_pipeline(self):
        """Getting the count of running pipeline.
        Returns
        -------
        count : int
            The count of running pipeline.
        """
        return self._get_pipe_execute_count()

    @property
    def num_outputs(self):
        """Get the number of outputs.
        Returns
        -------
        count : int
            The number of outputs.
        """
        return self._get_num_outputs()

    @property
    def num_inputs(self):
        """Get the number of inputs
        Returns
        -------
        count : int
            The number of inputs
        """
        return self._get_num_inputs()

    @staticmethod
    def load_library(config_file_name):
        """Import files to create a pipeline executor.

        Parameters
        ----------
        config_file_name : str
            Path and name of the configuration file, the configuration file contains the
            disk path of the parameter file, library file, and JSON file.
        """
        with open(config_file_name, "r") as file_handle:
            config = file_handle.read()
        config = json.loads(config)
        if "load_config" not in config or "pipeline_config" not in config:
            raise RuntimeError(
                f'"load_config" or "pipeline_config" is missing in {config_file_name}'
            )

        # The config file used to load library, prameters, and JSON files.
        with open(config["load_config"], "r") as file_handle:
            load_config = file_handle.read()

        # The config file used to load pipeline compute config.
        with open(config["pipeline_config"], "r") as file_handle:
            pipeline_config = file_handle.read()

        # Load a PipelineExecutor from the disk files.
        load_library = get_global_func("tvm.pipeline_executor.load", allow_missing=False)
        module = load_library(load_config, pipeline_config)

        return PipelineModule(module)


class PipelineExecutorFactoryModule(object):
    """Common interface for pipeline executor factory modules.

    Parameters
    ----------
    pipeline_mods : List[GraphExecutorFactoryModule]
        List of GraphExecutorFactoryModule.

    mod_config : Dict[int, Dict[str, Any]]
        Modules dependency configuration information.

    """

    def __init__(self, pipeline_mods, mods_config):
        self.pipeline_mods = pipeline_mods
        self.mods_config = mods_config
        self.module = None

    def get_pipeline_executor_module(self):
        """Get the pipeline executor module.

        Returns
        -------
        module : Module
            Common interface for pipeline executor factory Module.
        """
        if not self.module:
            graph_executors, config = self.graph_executor_create(
                self.pipeline_mods, self.mods_config
            )
            self.pipeline_create = get_global_func(
                "tvm.pipeline_executor.create", allow_missing=False
            )
            self.module = self.pipeline_create(graph_executors, config)
        return self.module

    def graph_executor_create(self, pipeline_mods, mod_config):
        """Create graph_executor list and return configuration as a json string.

        Parameters
        ----------
        pipeline_mods : List[GraphExecutorFactoryModule]
          List of GraphExecutorFactoryModule

        mod_config : Dict[str, Any]
            Modules dependency configuration information.

        Returns
        -------
        mods : List[Module]
            The Module list.

        mod_config : str
            The Modudle configuration.
        """
        # Should store modules in the list named 'mods' in index order.
        mods = [None for _ in range(len(pipeline_mods))]
        for lib_index in pipeline_mods:
            pipeline_lib = pipeline_mods[lib_index]["lib"]
            dev = pipeline_mods[lib_index]["dev"]
            lib = graph_executor.GraphModule(pipeline_lib["default"](dev))
            # Return a module list sorted by lib_index.
            mods[lib_index] = lib.module

        return mods, json.dumps(mod_config)

    def export_library(self, directory_path):
        """Export the pipeline executor into disk files.

        Parameters
        ----------
        directory_path : str
            Export the files to this directory.
        """
        if not self.pipeline_mods:
            raise RuntimeError("The pipeline executor has not been initialized.")

        # Check if the directory_path exists.
        if not os.path.exists(directory_path):
            raise RuntimeError("The directory {directory_path} does not exist.")
        # Create an load configuration.
        load_config_file_name = f"{directory_path}/load_config"
        pipeline_config_file_name = f"{directory_path}/pipeline_config"
        config = {}
        config["load_config"] = load_config_file_name
        config["pipeline_config"] = pipeline_config_file_name
        load_config = []
        # Export the library, JSON, and parameter into files, then export these files path
        # into a configuration file.
        for lib_index in self.pipeline_mods:
            mconfig = {}
            mconfig["mod_idx"] = lib_index
            mconfig["lib_name"] = f"{directory_path}/lib{lib_index}.so"
            mconfig["json_name"] = f"{directory_path}/json{lib_index}"
            mconfig["params_name"] = f"{directory_path}/params{lib_index}"
            mconfig["dev"] = (
                f"{self.pipeline_mods[lib_index]['dev'].device_type},"
                f"{self.pipeline_mods[lib_index]['dev'].device_id}"
            )
            # Get the graph, lib, and parameters from GraphExecutorFactoryModule.
            lib = self.pipeline_mods[lib_index]["lib"]
            # Export the lib, graph, and parameters to disk.
            if self.pipeline_mods[lib_index]["export_cc"]:
                lib.export_library(
                    mconfig["lib_name"], cc=self.pipeline_mods[lib_index]["export_cc"]
                )
            else:
                lib.export_library(mconfig["lib_name"])

            with open(mconfig["json_name"], "w") as file_handle:
                file_handle.write(lib.graph_json)
            with open(mconfig["params_name"], "wb") as file_handle:
                file_handle.write(runtime.save_param_dict(lib.params))

            load_config.append(mconfig)

        with open(load_config_file_name, "w") as file_handle:
            json.dump(load_config, file_handle)

        with open(pipeline_config_file_name, "w") as file_handle:
            json.dump(self.mods_config, file_handle)

        config_file_name = f"{directory_path}/config"
        with open(config_file_name, "w") as file_handle:
            json.dump(config, file_handle)

        return config_file_name
