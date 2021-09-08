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
from tvm.relay.transform import InferType
from tvm.contrib import graph_executor


def pipeline_executor_enabled():
    """check if pipeline executor is enabled.

    Return
    -------
    enable: bool
        Return pipeline executor is enabled or not.
    """
    return tvm._ffi.get_global_func("tvm.pipeline_executor.create", allow_missing=True) is not None


def build(pipe_configs):
    """build module list that can use for pipeline execution.

    Parameters
    ----------
    pipe_configs: PipelineConfig
        build configuration informaton.

    Returns
    -------
    ret: PipelineExecutorFactoryModule
        the class that wrap module list and configuration.
    """
    mods = {}
    mod_n_configs = pipe_configs.get_config()
    config_len = len(mod_n_configs)
    string_config = [{} for _ in range(config_len)]
    for ir_mod, mod_config in mod_n_configs.items():
        mconf = mod_config["pipeline"].copy()
        mod_idx = mconf["mod_idx"] - 1
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
        string_config[mod_idx] = mconf
        # associate mod with device
        mods[mod] = {"dev": dev}

    return PipelineExecutorFactoryModule(mods, string_config)


def create(pipe_executor_factory_module):
    """Create a pipeline runtime executor.

    Parameters
    ----------
    pipe_executor_factory_module : PipelineExecutorFactoryModule
        Executor factory to storage IRModule list and pipeline configuration.

    Returns
    submodule : PipelineModule
        Runtime pipeline module.
    """

    return PipelineModule(pipe_executor_factory_module)


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
        self.pipeline_mods = pipe_mod_config.pipeline_mods
        self.mod_config = pipe_mod_config.mods_config
        mods, config = self.graph_executor_create(self.pipeline_mods, self.mod_config)
        assert (
            pipeline_executor_enabled()
        ), "Pipeline executor is not enabled. Please \
              re-build TVM with USE_PIPELINE_EXECUTOR=ON"
        pipeline_create = tvm._ffi.get_global_func(
            "tvm.pipeline_executor.create", allow_missing=False
        )
        assert pipeline_create
        module = pipeline_create(mods, config)

        self.module_ = module

    def graph_executor_create(self, pipeline_mods, mod_config):
        """Create graph_executor list and return string format config.

        Parameters
        ----------
        pipeline_mods : List[GraphExecutorFactoryModule]
          list of GraphExecutorFactoryModule

        mod_config : Dict[str, Any]
            modules dependency configuration informaiton.

        Returns
        -------
        mods : List[Module]
            Module list.

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

    class Binding:
        """The class that use to storage module connection information.
        The binding can be either "input" or "output".

        Parameters
        ----------
        owner : ModuleWrapper
            The class that own this interface, in such class there are
            Module information like idx, module name

        io_type : str
            The type of this binding. It can be either "input" or "output".

        name : str/integer
            Binding name, for input it is string such as "data0";
            for output it is the idx integer such as 0.
        """

        def __init__(self, owner, stype, name, data_type=None):
            self.io_owner = owner
            self.io_type = stype
            self.name = str(name)
            # These item that have dependency relation with self
            self.bindings = []
            # The item that self depend
            self.parents = []

            self.data_type = data_type

        def get_name(self):
            """get owner name and self name"""
            owner_name = ""
            if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                owner_name = self.io_owner.name

            return owner_name, self.name

        def get_owner_idx(self):
            """return idx if owner is ModuleWrapper, if not return 0"""
            if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                return self.io_owner.idx

            # if not ModuleWrapper then owner is PipelineConfig, return 0
            # to identify this is global interface
            return 0

        def is_global_interface(self):
            """check if this interface is global"""
            return not isinstance(self.io_owner, PipelineConfig.ModuleWrapper)

        def __repr__(self):
            """Get all binding(input data), exepect like |data_0: mod1:data_0"""
            ret = "  |{}: ".format(self.name)
            for binding in self.bindings:
                mname, dname = binding.get_name()
                ret += "{0}:{1} ".format(mname, dname)
            return ret

        def check_dag_acyclic(self, start, inputs):
            """check if the DAG that current binding stay is acircle"""
            for binding in inputs.values():
                if start == binding.io_owner:
                    return False
                for p in binding.parents:
                    if not self.check_dag_acyclic(start, p.io_owner.input_bindings.bindings):
                        return False

            return True

        def connect(self, binding):
            """
            # check if the bindendency setting correct.
            # correct connection are following
            # 1. global input to module input
            # 2. module output to global output
            # 3. module output to moudle input
            """
            if self.io_owner == binding.io_owner:
                raise RuntimeError(f"can not set self as binding.")

            if not self.is_global_interface() and self.io_type == "input":
                raise RuntimeError(f"Module only can start binding from output!")

            if (
                not self.is_global_interface()
                and not binding.is_global_interface()
                and binding.io_type == "output"
            ):
                raise RuntimeError(f"Module output can not binding with module output!")

            if (
                not self.is_global_interface()
                and binding.is_global_interface()
                and binding.io_type == "input"
            ):
                raise RuntimeError(f"Module output can not binding with global input!")

            if self.is_global_interface() and self.io_type != "input":
                raise RuntimeError(f"Global only can start binding from input!")

            if self.is_global_interface() and binding.io_type != "input":
                raise RuntimeError(f"Global input only can set binding with module input.")

            self.bindings.append(binding)
            if not self.is_global_interface():
                # check if the source and target data_type same
                if (
                    isinstance(binding.io_owner, PipelineConfig.ModuleWrapper)
                    and self.data_type != binding.data_type
                ):
                    raise RuntimeError(
                        f"Illegal type (%s vs. %s): binding type is not same!"
                        % (self.data_type, binding.data_type)
                    )

                binding.parents.append(self)
                # Do acyclic check after increase the in-degree.
                if not self.check_dag_acyclic(
                    binding.io_owner, self.io_owner.input_bindings.bindings
                ):
                    raise RuntimeError(f"Illegal connection: cause a circle!")

    class BindingList:
        """Container for bindings(input or output interface).

        Parameters
        ----------
        owner : ModuleWrapper/PipelineConfig
            The owner of this list, it can be ModuleWrapper or PipelineConfig

        type_name : str
            The type of this binding list. It can be either "input" or "output".
        """

        def __init__(self, owner, type_name):
            self.bindings = {}
            self.io_owner = owner
            self.binding_type = type_name

        def get_binding_data_type(self, key):
            """return binding data type"""
            if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                return self.io_owner.get_data_type(key, self.binding_type)
            return None

        def __getitem__(self, key):
            """return item by key"""
            if key not in self.bindings:
                data_type = self.get_binding_data_type(key)
                if not data_type and isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                    raise RuntimeError(f"Cannot find {key} in binding list {self.binding_type}")

                self.bindings[key] = PipelineConfig.Binding(
                    self.io_owner, self.binding_type, key, data_type
                )

            return self.bindings[key]

    class ModuleWrapper:
        """Module Wrapper with information like module index, binding information
        and building informations.
        """
        def __init__(self, mod=None):
            """init class"""
            self.target_host = None
            self.build_func = None
            self.params = None
            self.target = None
            self.name = None
            self.dev = None
            self.idx = None
            self.mod = mod
            self.input_params = InferType()(mod)["main"].params
            self.output_values = InferType()(mod)["main"].checked_type.ret_type
            self.input_bindings = PipelineConfig.BindingList(self, "input")
            self.output_bindings = PipelineConfig.BindingList(self, "output")

        def __eq__(self, other):
            """check if self equl other"""
            if isinstance(other, PipelineConfig.ModuleWrapper):
                return self.mod == other.mod

            return False

        def __getitem__(self, key):
            """get item by key"""
            if isinstance(key, str):
                if key == "input":
                    return self.input_bindings

                if key == "output":
                    return self.output_bindings

            raise RuntimeError(f"{key} not found!")

        def get_data_type(self, key, stype):
            """get module input/output data type."""
            if stype == "input":
                for param in self.input_params:
                    if param.name_hint == key:
                        return param._checked_type_

            if stype == "output":
                if isinstance(self.output_values, tvm.ir.type.TupleType):
                    if int(key) < len(self.output_values.fields):
                        return self.output_values.fields[int(key)]
                elif int(key) == 0:
                    return self.output_values

            return None

        def set_idx_name(self, idx):
            """Set index and generating name by index"""
            self.idx = idx
            self.name = "mod{}".format(str(idx))

        def is_root_mod(self):
            """Identify if this item is root item, used by DAG topological sort."""
            return all([not b.parents for b in self.input_bindings.bindings.values()])

        def remove_self_from_bindings(self):
            """Remove self from binding to reduce child in-degree, used by DAG topological sort."""
            for binding in self.output_bindings.bindings.values():
                for child in binding.bindings:
                    if binding in child.parents:
                        child.parents.remove(binding)


    def __init__(self):
        self.mod_wrapper = {}
        self.input_bindings = self.BindingList(self, "input")
        self.output_bindings = self.BindingList(self, "output")

    def __str__(self):
        """ Get configuration in string type"""
        # topological sort to get correct module order in list.
        self.dag_topology_sort()
        # get input
        input_dump = "Inputs\n"
        for input_name in self.input_bindings.bindings:
            inf = self.input_bindings.bindings[input_name]
            input_dump += inf.__repr__() + "\n"

        # get connections
        output = {}
        connections_dump = "\nconnections\n"
        for mod in self.mod_wrapper:
            for interface in self.mod_wrapper[mod].output_bindings.bindings.values():
                if interface.bindings:
                    mname, dname = interface.get_name()
                    iname = mname + ".output(" + dname + ")->"
                    for dep in interface.bindings:
                        dep_mname, dep_dname = dep.get_name()
                        if isinstance(dep.io_owner, PipelineConfig.ModuleWrapper):
                            iname += f" {dep_mname}.{dep_dname}"
                            connections_dump += f"  |{iname}\n"
                        else:
                            output[dep_dname] = f"{mname}.output({dname})"

        # get output
        output_dump = "\noutput\n"
        for name in sorted(output.keys()):
            output_dump += f"  |output({name}) : {output[name]}\n"

        return input_dump + output_dump + connections_dump

    def __getitem__(self, key):
        """return item by key"""
        if isinstance(key, tvm.ir.module.IRModule):
            if key not in self.mod_wrapper:
                self.mod_wrapper[key] = self.ModuleWrapper(key)
            return self.mod_wrapper[key]

        if isinstance(key, str):
            if key == "input":
                return self.input_bindings
            if key == "output":
                return self.output_bindings

        raise RuntimeError(f"{key} not found.")

    def get_config(self):
        """ Get configuration in dictionary format."""

        # topological sort to get correct module order in list.
        self.dag_topology_sort()
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
                        dep_item["mod_idx"] = dep.get_owner_idx()
                        dep_item["input_name"] = dname
                        dep_conf.append(dep_item)

                # ouput_idx start from 0.
                output["output_idx"] = int(binding.name)
                output["dependent"] = dep_conf
                output_conf.append(output)

            mconf["mod_idx"] = module.idx
            mconf["output"] = output_conf

            # build module configuration with pipeline and other parameters.
            mconfig[mod] = {
                "pipeline": mconf,
                "target_host": module.target_host,
                "mod_name": "default",
                "build": module.build_func,
                "params": module.params,
                "target": module.target,
                "dev": module.dev,
            }

        return mconfig

    def dag_topology_sort(self):
        """ Do topological sort to get pipeline module order."""
        mlist = []
        mod_wrapper = self.mod_wrapper.copy()
        while mod_wrapper:
            temp_list = []
            for mod, wrapper in mod_wrapper.items():
                if wrapper.is_root_mod():
                    temp_list.append(mod)
                    wrapper.remove_self_from_bindings()

            for mod in temp_list:
                mod_wrapper.pop(mod, None)

            mlist += temp_list

        for mod, i in zip(mlist, range(len(mlist))):
            self.mod_wrapper[mod].set_idx_name(i + 1)

    def get_mod_idx(self, mod):
        """return idx for specify mod"""
        idx = self.mod_wrapper[mod].idx
        return idx

    def pipe_input(self, name):
        """return input binding by name"""
        return self.input_bindings[name]

    def pipe_output(self, idx):
        """return output binding by idx"""
        return self.output_bindings[idx]


class PipelineExecutorFactoryModule(object):
    """This class use to storage pipeline IRModule and configurations.

    Parameters
    ----------
    pipeline_mods : List[IRModule]
        list of IRModule

    mod_config : Dict[int, Dict[str, Any]]
        modules and modules dependency configuration informaiton.

    """

    def __init__(self, pipeline_mods, mods_config):
        self.pipeline_mods = pipeline_mods
        self.mods_config = mods_config
