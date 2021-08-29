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
    """Pipeline Configuration Class, in this class there are 2 internal class,
    first is Module which use to represent Module, second is Interface which use
    to represent Module input/output and Pipeline Module input/output, by setting
    dependency relation between Interfaces this class can build the module
    connection relation.

    The class Hierarchical as following.
         PipelineConfig ---> ModuleWrapper ---> Interface(input/output)
    """

    class ModuleWrapper:
        """The class use use to represent Module and storage module index and
        Interface information.
        """

        class Interface:
            """The class that use to storage module connection information.
               There are 2 types Interface Input:1 Output:2
            Parameters
            ----------

            owner : ModuleWrapper
                The class that own this interface, in such class there are
                Module information like index, module name

            itype : integer
                Interface type, 1 is input interface, 2 is output interface

            name : str/integer
                Interface name, for input that is string for example "data0"
                for output that is integer for example 0.
            """

            def __init__(self, owner, itype, name):
                self.owner_ = owner
                self.itype_ = itype
                self.name_ = str(name)
                self.dependent_ = []

            def get_name(self):
                mname = ""
                if self.owner_:
                    mname = self.owner_.name_

                return mname, self.name_

            def get_owner_indx(self):
                return self.owner_.indx_

            def get_dependent_str(self):
                name = ""
                for dependent in self.dependent_:
                    mname, dname = dependent.get_name()
                    name = name + (mname + ":output(" + dname if self.itype_ == 2 else "")
                    name = name + (")" if self.itype_ == 2 else mname + ":" + dname)
                return name

            def add_dependent(self, dependent):
                """
                # check if the dependency setting correct.
                # correct connection are following
                # 1. global input to module input
                # 2. module output to next module input
                # 3. module output to global output
                """
                owner_indx = self.get_owner_indx()
                dep_owner_indx = dependent.get_owner_indx()
                assert owner_indx != dep_owner_indx, f"can not set self as dependent."
                assert not (
                    owner_indx > dep_owner_indx
                    and not (dependent.itype_ == 2 and dep_owner_indx == 0)
                ), f"dependent only can be next module interface or global output."
                assert not (
                    owner_indx == 0 and dependent.itype_ != 1
                ), f"global input only can set dependent with module input."

                self.dependent_.append(dependent)

        def __init__(self, indx=0):
            self.indx_ = indx
            self.name_ = "mod" + str(indx) if indx else ""
            self.interfaces_ = {1: {}, 2: {}}
            self.target_host_ = None
            self.mod_name_ = "default"
            self.build_func_ = None
            self.params_ = None
            self.target_ = None
            self.dev_ = None


        def get_interface(self, itype, name):
            if name not in self.interfaces_[itype]:
                self.interfaces_[itype][name] = self.Interface(self, itype, name)

            return self.interfaces_[itype][name]

        def input(self, name):
            return self.get_interface(1, name)

        def output(self, index):
            return self.get_interface(2, index)

        def set_target_host(self, host):
            self.target_host_ = host

        def set_mod_name(self, name):
            self.mod_name_ = name

        def set_build_func(self, build_func):
            self.build_func_ = build_func

        def set_params(self, params):
            self.params_ = params

        def set_target(self, target):
            self.target_ = target

        def set_dev(self, dev):
            self.dev_ = dev

    def __init__(self, mods):
        self.pipe_module_name_ = "pipeline_module"
        self.mod_wrapper = {m: self.ModuleWrapper(i + 1) for m, i in zip(mods, range(len(mods)))}
        self.mod_wrapper[self.pipe_module_name_] = self.ModuleWrapper(0)

    def __str__(self):
        """ Get configuration in string type"""
        # get input
        input_dump = "Inputs\n"
        for input_name in self.mod_wrapper["pipeline_module"].interfaces_[1]:
            inf = self.mod_wrapper["pipeline_module"].interfaces_[1][input_name]
            input_dump += "  |" + input_name + ": " + inf.get_dependent_str() + "\n"

        # get connections
        output = {}
        connections_dump = "\nconnections\n"
        for mod in self.mod_wrapper:
            for _, interface in self.mod_wrapper[mod].interfaces_[2].items():
                if interface.dependent_:
                    mname, dname = interface.get_name()
                    iname = mname + ".output(" + dname + ")->"
                    for dep in interface.dependent_:
                        dep_mname, dep_dname = dep.get_name()
                        if dep.owner_.indx_ > 0:
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
            if mod == self.pipe_module_name_:
                continue
            # get pipeline configure
            mconf = {}
            output_conf = []
            module = self.mod_wrapper[mod]
            for _, interface in module.interfaces_[2].items():
                dep_conf = []
                output = {}
                if interface.dependent_:
                    for dep in interface.dependent_:
                        dep_item = {}
                        _, dname = dep.get_name()
                        dep_item["mod_indx"] = dep.get_owner_indx()
                        dep_item["input_name"] = dname
                        dep_conf.append(dep_item)

                # ouput_indx start from 0.

                output["output_indx"] = int(interface.name_)
                output["dependent"] = dep_conf
                output_conf.append(output)
            mconf["mod_indx"] = module.indx_
            mconf["output"] = output_conf

            # build module configuration with pipeline and other parameters.
            mconfig[mod] = {"pipeline": mconf,
                            "target_host": module.target_host_,
                            "mod_name": module.mod_name_,
			    "build": module.build_func_,
                            "params": module.params_,
                            "target": module.target_,
                            "dev": module.dev_,
                           }

        return mconfig

    def __getitem__(self, key):
        return self.mod_wrapper[key]

    def get_mod_indx(self, mod):
        indx = self.mod_wrapper[mod].indx_
        return indx

    def pipe_input(self, name):
        return self.mod_wrapper[self.pipe_module_name_].input(name)

    def pipe_output(self, index):
        return self.mod_wrapper[self.pipe_module_name_].output(index)

    def connect(self, left: ModuleWrapper.Interface, right: ModuleWrapper.Interface):
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
