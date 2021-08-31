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
import tvm._ffi
from tvm import relay
from tvm.contrib import graph_executor


def pipeline_executor_enabled():
    """check if pipeline executor is enabled.

    Return
    -------
    enable: bool
        Return pipeline executor is enabled or not.
    """
    return tvm._ffi.get_global_func("tvm.pipeline_executor.create",
                                    allow_missing=True) is not None


def build(pipe_configs):
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

    Returns
    -------
    ret: List[IRModule]
        list of IRModule
    string_config: Dict[int, Dict[str, any]]
        pipeline configuration
    """
    mods = {}
    mod_n_configs = pipe_configs.get_config()
    config_len = len(mod_n_configs)
    string_config = [{} for _ in range(config_len)]
    #for _, (ir_mod, mod_config) in enumerate(mod_n_configs.items()):
    for ir_mod, mod_config in mod_n_configs.items():
        mconf = mod_config["pipeline"].copy()
        mod_indx = mconf["mod_indx"] - 1
        # Get mod device config
        dev = mod_config["dev"]
        target = mod_config["target"]
        build_func = relay.build
        # if there is a self defined build function then use it.
        if "build" in mod_config and mod_config["build"]:
            build_func = mod_config["build"]

        # build IRModule
        mod = build_func(
            ir_mod,
            target,
            params=mod_config["params"],
            target_host=mod_config["target_host"],
            mod_name=mod_config["mod_name"],
        )

        mconf["dev"] = "{},{}".format(dev.device_type, dev.device_id)
        # Create pipeline configuration
        string_config[mod_indx] = mconf
        # associate mod with device
        mods[mod] = {"dev": dev}

    # return PipeModuleConfig
    return PipeModuleConfig(mods, string_config)


def create(pipe_mod_config):
    """Create a pipeline runtime executor.

    Parameters
    ----------

    pipe_mod_config : PipeModuleConfig
        class to storage IRModule list and pipeline configuration.

    Returns
    -------
    submodule : PipelineModule
        Runtime pipeline module.
    """

    return PipelineModule(pipe_mod_config)

class PipelineModule(object):
    """Wrapper runtime module. This is a thin wrapper of the underlying TVM module.

    Parameters
    ----------
    pipeline_mods : List[GraphModule]
        The internal tvm module that holds the actual graph functions.
    pipeline_config : Dict[IRModule, Dict[str, Any]]
        modules and modules dependency configuration informaiton.
    """

    def __init__(self, pipe_mod_config):
        self.pipeline_mods_ = pipe_mod_config.pipeline_mods_
        self.mod_config_ = pipe_mod_config.mods_config_
        mods, config = self.graph_executor_create(self.pipeline_mods_, self.mod_config_)
        assert pipeline_executor_enabled(), \
              "Pipeline executor is not enabled. Please \
              re-build TVM with USE_PIPELINE_EXECUTOR=ON"
        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create",
                                                  allow_missing=False)
        assert pipelinecreate
        module = pipelinecreate(mods, config)

        self.module_ = module

    def graph_executor_create(self, pipeline_mods, mod_config):
        """Create graph_executor list and return string format config.

        Parameters
        ----------

        pipeline_mods : List[IRModule]
          list of IRModule

        mod_config : Dict[int, Dict[str, Any]]
            modules and modules dependency configuration informaiton.

        Returns
        -------
        mods : List[GraphModule]
            Runtime graph module.

	mod_config : str
	    mods configuration
        """

        mods = []
        for pipeline_mod in pipeline_mods:
            mod = graph_executor.GraphModule(
                pipeline_mod["default"](pipeline_mods[pipeline_mod]["dev"])
            )
            mods.append(mod.module)

        return mods, json.dumps(mod_config)

class PipelineConfig(object):
    """The wrapper of each module to be pipelined. The wrapper mainly includes the
    module itself as well as the binding that represents the connections of this
    module's inputs and outputs to other modules.
    """

    class ModuleWrapper:
        """The class use use to represent Module and storage module index and
        Binding information.
        """

        class Binding:
            """The class that use to storage module connection information.
               There are 2 types Binding Input:1 Output:2
            Parameters
            ----------

            owner : ModuleWrapper
                The class that own this interface, in such class there are
                Module information like index, module name

            io_type : str
                The type of this binding. It can be either "input" or "output".

            name : str/integer
                Binding name, for input that is string for example "data0"
                for output that is integer for example 0.
            """

            def __init__(self, owner, stype, name):
                self.io_owner = owner
                self.io_type = stype
                self.name = str(name)
                self.bindings = []

            def get_name(self):
                owner_name = ""
                if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                    owner_name = self.io_owner.name

                return owner_name, self.name

            def get_owner_indx(self):
                owner_indx = 0
                if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                    return self.io_owner.indx_
                return 0

            def get_bindings_str(self):
                name = ""
                for dependent in self.bindings:
                    mname, dname = dependent.get_name()
                    name += (mname + ":output(" + dname \
                          if self.io_type == "output" else "")
                    name += (")" if self.io_type == "output" else mname + ":" + dname)
                return name

            def connect(self, dependent):
                """
                # check if the dependency setting correct.
                # correct connection are following
                # 1. global input to module input
                # 2. module output to next module input
                # 3. module output to global output
                """
                '''
                owner_indx = self.get_owner_indx()
                dep_owner_indx = dependent.get_owner_indx()
                assert owner_indx != dep_owner_indx, f"can not set self as dependent."
                assert not (
                    owner_indx > dep_owner_indx
                    and not (dependent.io_type == "output" and dep_owner_indx == 0)
                ), f"dependent only can be next module interface or global output."
                assert not (
                    owner_indx == 0 and dependent.io_type != "input"
                ), f"global input only can set dependent with module input."
                '''
                self.bindings.append(dependent)

        def __init__(self, index=0):
            self.indx_ = index
            self.name = "mod{}".format(str(index) if index else "")
            self.input_bindings = PipelineConfig.BindingList(self, "input")
            self.output_bindings = PipelineConfig.BindingList(self, "output")
            self.target_host_ = None
            self.build_func_ = None
            self.params_ = None
            self.target_ = None
            self.dev_ = None

        def __getitem__(self, key):
            if (isinstance(key, str)):
                if (key == "input"):
                    return self.input_bindings

                if (key == "output"):
                    return self.output_bindings
            assert 0, "key not found"

        def input(self, name):
            if name not in self.input_bindings:
                self.input_bindings[name] = self.Binding(self, "input", name)

            return self.input_bindings[name]

        def output(self, index):
            if index not in self.output_bindings:
                self.output_bindings[index] = self.Binding(self, "output", index)

            return self.output_bindings[index]

        def set_target_host(self, host):
            self.target_host_ = host

        def set_build_func(self, build_func):
            self.build_func_ = build_func

        def set_params(self, params):
            self.params_ = params

        def set_target(self, target):
            self.target_ = target

        def set_dev(self, dev):
            self.dev_ = dev

    class BindingList:
        def __init__(self, owner, type_name):
            self.bindings = {}
            self.io_owner = owner
            self.binding_type = type_name

        def __getitem__(self, key):
            if key not in self.bindings:
                self.bindings[key] = \
                    PipelineConfig.ModuleWrapper.Binding(self.io_owner,
                                                         self.binding_type, key)

            return self.bindings[key]

    def __init__(self, mods):
        self.mod_wrapper = {m: self.ModuleWrapper(i + 1) for m, i in zip(mods, range(len(mods)))}
        self.input_bindings = self.BindingList(self, "input")
        self.output_bindings = self.BindingList(self, "output")

    def __str__(self):
        """ Get configuration in string type"""
        # get input
        input_dump = "Inputs\n"
        for input_name in self.input_bindings.bindings:
            inf = self.input_bindings.bindings[input_name]
            input_dump += "  |" + input_name + ": " + inf.get_bindings_str() + "\n"

        # get connections
        output = {}
        connections_dump = "\nconnections\n"
        for mod in self.mod_wrapper:
            for _, interface in self.mod_wrapper[mod].output_bindings.bindings.items():
                if interface.bindings:
                    mname, dname = interface.get_name()
                    iname = mname + ".output(" + dname + ")->"
                    for dep in interface.bindings:
                        dep_mname, dep_dname = dep.get_name()
                        if isinstance(dep.io_owner, PipelineConfig.ModuleWrapper):
                            iname += " " + dep_mname + "." + dep_dname
                            connections_dump += "  |" + iname + "\n"
                        else:
                            output[dep_dname] = mname + ".output(" + dname + ")"

        # get output
        output_dump = "\noutput\n"
        for name in sorted(output.keys()):
            output_dump += "  |output(" + name + ") : " + output[name] + "\n"

        return input_dump + output_dump + connections_dump

    def get_config(self):
        """ Get configuration in dictionary format."""
        mconfig = {}
        for mod in self.mod_wrapper:
            # get pipeline configure
            mconf = {}
            output_conf = []
            module = self.mod_wrapper[mod]
            for _, binding in module.output_bindings.bindings.items():
                dep_conf = []
                output = {}
                if binding.bindings:
                    for dep in binding.bindings:
                        dep_item = {}
                        _, dname = dep.get_name()
                        dep_item["mod_indx"] = dep.get_owner_indx()
                        dep_item["input_name"] = dname
                        dep_conf.append(dep_item)

                # ouput_indx start from 0.

                output["output_indx"] = int(binding.name)
                output["dependent"] = dep_conf
                output_conf.append(output)
            mconf["mod_indx"] = module.indx_
            mconf["output"] = output_conf

            # build module configuration with pipeline and other parameters.
            mconfig[mod] = {"pipeline": mconf,
                            "target_host": module.target_host_,
                            "mod_name": "default",
			    "build": module.build_func_,
                            "params": module.params_,
                            "target": module.target_,
                            "dev": module.dev_,
                           }

        return mconfig

    def __getitem__(self, key):
        if (isinstance(key, tvm.ir.module.IRModule)):
            return self.mod_wrapper[key]

        if (isinstance(key, str)):
            if (key == "input"):
                return self.input_bindings
            if (key == "output"):
                return self.output_bindings

        assert 0, "key not find!"

    def get_mod_indx(self, mod):
        indx = self.mod_wrapper[mod].indx_
        return indx

    def pipe_input(self, name):
        return self.input_bindings[name]

    def pipe_output(self, index):
        return self.output_bindings[index]

    def connect(self, left: ModuleWrapper.Binding, right: ModuleWrapper.Binding):
        left.add_dependent(right)


class PipeModuleConfig(object):
    """This class use to storage pipeline IRModule and configurations.

    Parameters
    ----------

    pipeline_mods : List[IRModule]
        list of IRModule

    mod_config : Dict[int, Dict[str, Any]]
        modules and modules dependency configuration informaiton.

    """

    def __init__(self, pipeline_mods, mods_config):
        self.pipeline_mods_ = pipeline_mods
        self.mods_config_ = mods_config
