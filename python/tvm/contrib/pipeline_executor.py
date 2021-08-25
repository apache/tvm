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


def build_pipeline(mod_n_configs):
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
        mod_indx = mconf["mod_indx"] - 1
        target = mod_config["target"]
        assert mod_indx < config_len
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

        mconf["lib_name"] = lib_name
        mconf["json_name"] = json_name
        mconf["params_name"] = params_name
        mconf["dev"] = "{},{}".format(dev.device_type, dev.device_id)
        # Create pipeline configuration
        string_config[mod_indx] = mconf
        # associate mod with device
        mods[mod] = {"dev": dev}

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

    submodule = PipelineModule(pipeline_mods, mod_config)
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

    def graph_executor_create(self, pipeline_mods, mod_config):
        """Create a pipeline runtime executor.

        Parameters
        ----------
        pipeline_mods : List[IRModule]
          list of IRModule

        mod_config : Dict[int, Dict[str, Any]]
            modules and modules dependency configuration informaiton.

        Returns
        -------
        mods : GreaphModule
            Runtime graph module.
        """

        mods = []
        for pipeline_mod in pipeline_mods:
            mod = graph_executor.GraphModule(
                pipeline_mod["default"](pipeline_mods[pipeline_mod]["dev"])
            )
            mods.append(mod.module)

        return mods, json.dumps(mod_config)

    def __init__(self, pipeline_mods, mod_config):
        self.pipeline_mods = pipeline_mods
        self.mod_config = mod_config
        mods, config = self.graph_executor_create(pipeline_mods, mod_config)

        pipelinecreate = tvm._ffi.get_global_func("tvm.pipeline_executor.create")
        assert pipelinecreate
        module = pipelinecreate(mods, config)

        self.module_ = module


class PipelineModuleConfig:
    class interface:
        def __init__(self, owner, itype, name):
            self.owner_ = owner
            self.itype_ = itype
            self.name_ = str(name)
            self.dependent_ = []
            return

        def get_name(self):
            mname = ""
            mindx = 0
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

        def addDependent(self, dependent):
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
            assert not (owner_indx > dep_owner_indx and \
                    not (dependent.itype_ == 2 and dep_owner_indx == 0)), \
                    f"dependent only can be next module interface or global output."
            assert not (owner_indx == 0 and dependent.itype_ != 1), \
                    f"global input only can set dependent with module input."

            self.dependent_.append(dependent)

    class instance:
        def __init__(self, indx = 0):
            self.indx_ = indx
            self.name_ = "mod" + str(indx) if indx else ""
            self.interfaces_ = {1:{}, 2:{}}
            return

        def get_interface(self, itype,  name):
            if name not in self.interfaces_[itype]:
                self.interfaces_[itype][name] = PipelineModuleConfig.interface(self, itype, name)

            return self.interfaces_[itype][name]

        def input(self, name):
            return self.get_interface(1, name)

        def output(self, index):
            return self.get_interface(2, index)


    def __init__(self, mods):
        """
        # input
        """
        self.pipe_instance = self.instance(0)
        self.mod_instance = {
            m:self.instance(i + 1) for m, i in zip(mods, range(len(mods)))}
        return

    def __str__(self):
        # get input
        input_dump = "Inputs\n"
        for input_name in self.pipe_instance.interfaces_[1]:
            inf = self.pipe_instance.interfaces_[1][input_name]
            input_dump += "  |" +input_name + ": " + inf.get_dependent_str() + "\n"

        # connections
        output = {}
        connections_dump = "\nconnections\n"
        for mod in self.mod_instance:
            for _, interface in self.mod_instance[mod].interfaces_[2].items():
                if len(interface.dependent_):
                    mname, dname = interface.get_name()
                    iname = mname + ".output(" + dname + ")->"
                    for dep in interface.dependent_:
                        dep_mname, dep_dname = dep.get_name()
                        if dep.owner_.indx_ > 0:
                            iname += " " + dep_mname + "." + dep_dname
                            connections_dump += "  |" + iname +"\n"
                        else:
                            output[dep_dname] = mname + ".output(" + dname + ")"         

        # get output
        output_dump = "\noutput\n"
        for name in sorted(output.keys()):
            output_dump += "  |output(" + name + ") : " + output[name] + "\n"


        return input_dump + output_dump + connections_dump

    def get_config(self):
        mconfig = {}
        for mod in self.mod_instance:
            mconf = {}
            output_conf = []
            instance = self.mod_instance[mod]
            for _, interface in instance.interfaces_[2].items():
                dep_conf = []
                output = {}
                if len(interface.dependent_):
                    for dep in interface.dependent_:
                        dep_item = {}
                        _, dname = dep.get_name()
                        dep_item["mod_indx"] = dep.get_owner_indx()
                        dep_item["input_name"] = dname
                        dep_conf.append(dep_item)

                """
                # in configuration the ouput_indx start from 1
                """
                output["output_indx"] = int(interface.name_) + 1
                output["dependent"] = dep_conf
                output_conf.append(output)
            mconf["mod_indx"] = interface.get_owner_indx()
            mconf["output"] = output_conf
            mconfig[mod] = {"pipeline" : mconf}

        return mconfig


    def __getitem__(self, key):
        return self.mod_instance[key]

    def get_mod_indx(self, mod):
        indx = self.mod_instance[mod].indx_   
        return indx

    def pipe_input(self, name):
        return self.pipe_instance.input(name)

    def pipe_output(self, index):
        return self.pipe_instance.output(index)

    def connect(self, left:interface, right:interface):
        left.addDependent(right)
        return
