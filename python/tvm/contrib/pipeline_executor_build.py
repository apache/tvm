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
import tvm._ffi
from tvm import relay
from tvm.relay.transform import InferType
from tvm.contrib.pipeline_executor import PipelineExecutorFactoryModule


def pipeline_executor_build_enabled():
    """Check if the pipeline executor build is enabled.

    Return
    -------
    enable: bool
        Return whether the pipeline executor is enabled.
    """
    return tvm.contrib.pipeline_executor.pipeline_executor_enabled()


def build(pipe_configs):
    """Build modules used in the pipeline executor, then use these modules and configuration
    to create a pipeline executor.

    Parameters
    ----------
    pipe_configs: PipelineConfig
        Build Configuration information.

    Returns
    -------
    ret: PipelineExecutorFactoryModule
        Common interface for pipeline executor factory modules.
    """
    libs = {}
    config = pipe_configs.get_config()
    if "module_connection" not in config:
        raise RuntimeError('"module_connection" is missing')
    if "input_connection" not in config:
        raise RuntimeError('"input_connection" is missing')
    if "param_connection" not in config:
        raise RuntimeError('"param_connection" is missing')

    mod_n_configs = config["module_connection"]
    config_len = len(mod_n_configs)
    module_string_config = [{} for _ in range(config_len)]
    # Use hardware configurations to build backend modules for each subgraph.
    for ir_mod, mod_config in mod_n_configs.items():
        pipe_config = mod_config["pipeline"].copy()
        mod_idx = pipe_config["mod_idx"]
        dev = mod_config["dev"]
        target = mod_config["target"]
        build_func = relay.build
        # Callers may need to use a customized building function to wrap the pre-building logic
        # and the backend building logic. For example, in order to support a backend which only
        # can do "int8" computation, the caller may need to merge the "quantization" logic
        # into the building logic to creat a customized building function.
        if "build" in mod_config and mod_config["build"]:
            build_func = mod_config["build"]

        lib = build_func(
            ir_mod,
            target,
            params=mod_config["params"],
            target_host=mod_config["target_host"],
            mod_name=mod_config["mod_name"],
        )

        pipe_config["dev"] = "{},{}".format(dev.device_type, dev.device_id)
        # Use "mod_idx" as the key to create a "module_connection" map which is not only
        # for the module index but also for the module connection used to build the pipeline.
        module_string_config[mod_idx] = pipe_config
        libs[mod_idx] = {
            "lib": lib,
            "dev": dev,
            "fcompile": mod_config["fcompile"],
            "export_cc": mod_config["export_cc"],
        }

    # Creating a text form configuration to record the "input_connection" and the
    # "module_connection" information. The "input_connection" is used to record the
    # map of global input and subgraph input, and the "module_connection" is used to
    # record module dependency.
    string_config = {}
    string_config["param_connection"] = config["param_connection"]
    string_config["input_connection"] = config["input_connection"]
    string_config["module_connection"] = module_string_config

    return PipelineExecutorFactoryModule(libs, string_config)


def export_library(factory, directory_path):
    """Export the pipeline executor into disk files.

    Parameters
    ----------
    factory : PipelineExecutorFactoryModule
        The pipeline executor factory
    directory_path : str
        Export the files to this directory.
    """
    if not factory.pipeline_mods:
        raise RuntimeError("The pipeline executor has not been initialized.")

    # Check if the directory_path exists.
    if not directory_path or not os.path.exists(directory_path):
        raise RuntimeError("The directory {directory_path} does not exist.")
    # Create an load configuration.
    load_config_file_name = "{}/load_config".format(directory_path)
    pipeline_config_file_name = "{}/pipeline_config".format(directory_path)
    config = {}
    config["load_config"] = load_config_file_name
    config["pipeline_config"] = pipeline_config_file_name
    load_config = []
    # Export the library, JSON, and parameter into files, then export these files path
    # into a configuration file.
    for lib_index in factory.pipeline_mods:
        mconfig = {}
        mconfig["mod_idx"] = lib_index
        mconfig["lib_name"] = "{}/lib{}.so".format(directory_path, lib_index)
        mconfig["json_name"] = "{}/json{}".format(directory_path, lib_index)
        mconfig["params_name"] = "{}/params{}".format(directory_path, lib_index)
        lib_config = factory.pipeline_mods[lib_index]
        mconfig["dev"] = "{},{}".format(lib_config["dev"].device_type, lib_config["dev"].device_id)
        fcompile = lib_config["fcompile"]
        if not fcompile:
            fcompile = False

        # Get the graph, lib, and parameters from GraphExecutorFactoryModule.
        lib = factory.pipeline_mods[lib_index]["lib"]
        # Export the lib, graph, and parameters to disk.
        lib.export_library(mconfig["lib_name"], fcompile)
        with open(mconfig["json_name"], "w") as file_handle:
            file_handle.write(lib.graph_json)
        with open(mconfig["params_name"], "wb") as file_handle:
            file_handle.write(relay.save_param_dict(lib.params))

        load_config.append(mconfig)

    with open(load_config_file_name, "w") as file_handle:
        json.dump(load_config, file_handle)

    with open(pipeline_config_file_name, "w") as file_handle:
        json.dump(factory.mods_config, file_handle)

    config_file_name = "{}/config".format(directory_path)
    with open(config_file_name, "w") as file_handle:
        json.dump(config, file_handle)

    return config_file_name


class PipelineConfig(object):
    """Pipeline configuration information, this class contains the DAG that expresses
    the dependency of each module involved in a pipeline and the parameters for building
    each module.
    """

    class Binding:
        """This class defines the module connections information.
        The type can only be "input" or "output".

        Parameters
        ----------
        owner : ModuleWrapper
            The class who owns this interface.

        io_type : str
            The I/O type of this interface. It can only be "input" or "output".

        name : str/integer
            Name, for input it is string such as "data0", for output it is an integer such as 0.

        data_type: TensorType
            The data type of this interface.
        """

        def __init__(self, owner, io_type, name, data_type=None):
            self.io_owner = owner
            self.io_type = io_type
            self.name = str(name)
            # Child interfaces that depend on this interface.
            self.bindings = []
            # Parents interfaces that this interface depend on.
            self.parents = []

            self.data_type = data_type

        def get_name(self):
            # Return name of this interface and the name of owner who owns this interface.
            owner_name = ""
            if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                owner_name = self.io_owner.name

            return owner_name, self.name

        def get_owner_idx(self):
            # If the owner is ModuleWrapper return the owner index, if not return 0.
            if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                return self.io_owner.idx

            return -1

        def is_pipeline_executor_interface(self):
            """The pipeline interface is used to interact with the caller. There are two types
            of interfaces, one is 'input' another is 'output'. The pipeline input interface
            is responsible for passing parameters to the internal module interface, and the
            pipeline output interface is responsible for outputting the results computed by
            the pipeline executor to the caller.
            """
            return not isinstance(self.io_owner, PipelineConfig.ModuleWrapper)

        def __repr__(self):
            # Geting the binding information in the form of text.
            str_format = "  |{}: ".format(self.name)
            for binding in self.bindings:
                mname, dname = binding.get_name()
                str_format += "{0}:{1} ".format(mname, dname)

            return str_format

        def check_binding_dict(self, connection_dict):
            """Checking the binding dictionary.
            Parameter
            ---------
            connection_dict : Dict[str, Any]
                It is a dictionary of module connections.
            """
            if "interface_name" not in connection_dict:
                raise RuntimeError('"inteface_name" is missing in global config!"')
            if "connection" not in connection_dict:
                raise RuntimeError(f'"connection" is missing!"')
            # The global interface mapping should be one-to-one.
            if not connection_dict["connection"]:
                raise RuntimeError("The global interface map is empty!")
            if len(connection_dict["connection"]) > 1:
                raise RuntimeError("A global interface maps multiple module interfaces!")
            if "mod_idx" not in connection_dict["connection"][0]:
                raise RuntimeError('"mod_idx" is missing!')

        def get_binding_dict(self):
            """Returning the binding information in the form of dictionary.
            Returns
            -------
            data : Dict[str, Any]
                The binding information is in the form of dictionary.
            """
            dict_format = {"interface_name": self.name, "connection": []}
            for binding in self.bindings:
                _, dname = binding.get_name()
                midx = binding.get_owner_idx()
                dict_format["connection"].append({"mod_idx": midx, "interface_name": dname})

            self.check_binding_dict(dict_format)
            return dict_format

        def check_dag_acyclic(self, start, inputs):
            """This is to check whether the DAG containing these input interfaces is acyclic.
            Parameters
            ----------
            start: ModuleWrapper
                The starting node of the cycle check algorithm.

            inputs: Binding
                These interfaces are used to connect to each other to build DAG.

            Return
            ------
                Return true if there is no cycle in the DAG.
            """
            for binding in inputs.values():
                if start == binding.io_owner:
                    return False
                for p in binding.parents:
                    if not self.check_dag_acyclic(start, p.io_owner.input_bindings.bindings):
                        return False

            return True

        def connect(self, binding):
            """Connect the current interface to the destination interface.
            Correct connections are as follows: 1. the pipeline input connected to a module input,
            2. the module output connected to a pipeline output, 3. the module output connected to
            a module input.

            Parameters
            ----------
            binding: Binding
                The destination of this connection.
            """

            # Check whether the binding setting is correct or not.
            if self.io_owner == binding.io_owner:
                raise RuntimeError("Can not bind itself.")

            if self.io_type == "param" and not self.is_pipeline_executor_interface():
                raise RuntimeError(
                    'The "param" binding can only be used by a pipeline executor interface!'
                )

            if not self.is_pipeline_executor_interface() and self.io_type == "input":
                raise RuntimeError("Module can only bind from output interface!")

            if self.io_type == "param" and binding.io_type != "param":
                raise RuntimeError(
                    'A global "param" interface can only be bind with a module "param" interface!'
                )

            if (
                not self.is_pipeline_executor_interface()
                and not binding.is_pipeline_executor_interface()
                and binding.io_type == "output"
            ):
                raise RuntimeError("Can not bind module output with another module output!")

            if (
                not self.is_pipeline_executor_interface()
                and binding.is_pipeline_executor_interface()
                and binding.io_type == "input"
            ):
                raise RuntimeError("Can not bind module output with pipeline input!")

            if self.is_pipeline_executor_interface() and self.io_type == "output":
                raise RuntimeError("Global output can not be used as binding start point.")

            if (
                self.is_pipeline_executor_interface()
                and self.io_type == "input"
                and binding.io_type != "input"
            ):
                raise RuntimeError("Global input can only bind with module input.")

            self.bindings.append(binding)
            if not self.is_pipeline_executor_interface():
                # Check whether the data types of the source and destination are the same.
                if (
                    isinstance(binding.io_owner, PipelineConfig.ModuleWrapper)
                    and self.data_type != binding.data_type
                ):
                    raise RuntimeError(
                        f"Illegal type (%s vs. %s): binding type is not same!"
                        % (self.data_type, binding.data_type)
                    )

                binding.parents.append(self)

                # Do acyclic check after increasing the in-degree of child node by setting
                # current interface as a parent of the child node.

                if not self.check_dag_acyclic(
                    binding.io_owner, self.io_owner.input_bindings.bindings
                ):
                    raise RuntimeError("Illegal connection: Cause a cycle!")

    class BindingList:
        """Container for bindings(input or output interface).

        Parameters
        ----------
        owner : ModuleWrapper/PipelineConfig
            The owner of this class can be ModuleWrapper or PipelineConfig.

        io_type : str
            The type of this class can be "input" or "output".
        """

        def __init__(self, owner, io_type):
            self.bindings = {}
            self.io_owner = owner
            self.binding_type = io_type

        def get_binding_data_type(self, key):
            if isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                return self.io_owner.get_data_type(key, self.binding_type)
            return None

        def __getitem__(self, key):
            if key not in self.bindings:
                data_type = self.get_binding_data_type(key)
                if not data_type and isinstance(self.io_owner, PipelineConfig.ModuleWrapper):
                    raise RuntimeError(f"Can not find {key} in binding list {self.binding_type}.")

                self.bindings[key] = PipelineConfig.Binding(
                    self.io_owner, self.binding_type, key, data_type
                )

            return self.bindings[key]

    class ModuleWrapper:
        """This class is a wrapper representing the module and contains information such as
        module information, binding information and building information.
        """

        def __init__(self, mod=None):
            self.target_host = None
            self.build_func = None
            self.params = None
            self.target = None
            self.fcompile = None
            self.name = None
            self.dev = None
            self.export_cc = None
            self.cpu_affinity = ""
            self.idx = None
            self.mod = mod
            self.input_params = InferType()(mod)["main"].params
            self.output_type = InferType()(mod)["main"].checked_type.ret_type
            self.input_bindings = PipelineConfig.BindingList(self, "input")
            self.output_bindings = PipelineConfig.BindingList(self, "output")
            self.param_binding = PipelineConfig.Binding(self, "param", "param")

        def __eq__(self, other):
            if isinstance(other, PipelineConfig.ModuleWrapper):
                return self.mod == other.mod

            return False

        def __getitem__(self, key):
            if isinstance(key, str):
                if key == "input":
                    return self.input_bindings

                if key == "output":
                    return self.output_bindings

                if key == "param":
                    return self.param_binding

                raise RuntimeError(f"{key} not found!")

            raise RuntimeError('The data type of "key" is not supported!')

        def get_data_type(self, key, interface_type):
            """Get the module interface data type according to the key value and interface type.
            Parameters
            ----------
            key: str
                The interface name.

            interface_type:
                The interface type.

            Return
            -------
                Return data type.
            """
            if interface_type == "input":
                for param in self.input_params:
                    if param.name_hint == key:
                        return param._checked_type_

            if interface_type == "output":
                if isinstance(self.output_type, tvm.ir.type.TupleType):
                    if int(key) < len(self.output_type.fields):
                        return self.output_type.fields[int(key)]
                elif int(key) == 0:
                    return self.output_type

            return None

        def set_idx_name(self, idx):
            # Set the index value and generate the module name.
            self.idx = idx
            self.name = "mod{}".format(str(idx))

        def is_root_mod(self):
            """Check whether this node is the root node in DAG, this function is used
            in topological sort.
            """
            return all([not b.parents for b in self.input_bindings.bindings.values()])

        def remove_self_from_bindings(self):
            """Remove the current node from child dependencies to reduce the in-degree
            of child node, this function is used in topological sort.
            """
            for binding in self.output_bindings.bindings.values():
                for child in binding.bindings:
                    if binding in child.parents:
                        child.parents.remove(binding)

    def __init__(self):
        self.mod_wrapper = {}
        self.input_bindings = self.BindingList(self, "input")
        self.output_bindings = self.BindingList(self, "output")
        # There is a map of global parameters group and module index.
        self.param_group_bindings = self.BindingList(self, "param")

    def __str__(self):
        # Get configuration information as a string.

        # Use topological sort to get correct module order.
        self.dag_topology_sort()
        # Getting the parameters dependencies.
        param_dump = "Params\n"
        for param_name in self.param_group_bindings.bindings:
            inf = self.param_group_bindings.bindings[param_name]
            param_dump += str(inf) + "\n"
        # Get the input dependencies.
        input_dump = "\nInputs\n"
        for input_name in self.input_bindings.bindings:
            inf = self.input_bindings.bindings[input_name]
            input_dump += str(inf) + "\n"

        # Get the connections information of each module.
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

        # Get the output dependencies.
        output_dump = "\noutput\n"
        for name in sorted(output.keys()):
            output_dump += f"  |output({name}) : {output[name]}\n"

        return param_dump + input_dump + output_dump + connections_dump

    def __getitem__(self, key):
        if isinstance(key, tvm.ir.module.IRModule):
            if key not in self.mod_wrapper:
                self.mod_wrapper[key] = self.ModuleWrapper(key)
            return self.mod_wrapper[key]

        if isinstance(key, str):
            if key == "input":
                return self.input_bindings
            if key == "output":
                return self.output_bindings
            if key == "param_group":
                return self.param_group_bindings

            raise RuntimeError(f"{key} not found!")

        raise RuntimeError(f'The key type "{type(key)}" is not supported!')

    def get_config(self):
        """Get the configuration information in dictionary form, this configuration
        will be used to create pipeline executor.
        """

        # Use topological sort to get the correct order of modules.
        self.dag_topology_sort()
        mconfig = {}
        module_connection = {}
        for mod in self.mod_wrapper:
            # Generate pipeline configuration.
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
                        if dep.is_pipeline_executor_interface():
                            dep_item["global_output_index"] = int(dname)
                        else:
                            dep_item["mod_idx"] = dep.get_owner_idx()
                            dep_item["input_name"] = dname
                        dep_conf.append(dep_item)

                # The value of output_idx start from 0.
                output["output_idx"] = int(binding.name)
                output["dependencies"] = dep_conf
                output_conf.append(output)

            mconf["mod_idx"] = module.idx
            mconf["cpu_affinity"] = module.cpu_affinity
            mconf["output"] = output_conf

            module_connection[mod] = {
                "pipeline": mconf,
                "target_host": module.target_host,
                "mod_name": "default",
                "build": module.build_func,
                "params": module.params,
                "target": module.target,
                "fcompile": module.fcompile,
                "dev": module.dev,
                "export_cc": module.export_cc,
            }

        # Creating a map including pipeline inputs and subgraph inputs.
        input_connection = []
        for input_name in self.input_bindings.bindings:
            input_dict = self.input_bindings.bindings[input_name].get_binding_dict()
            if "interface_name" not in input_dict["connection"][0]:
                raise RuntimeError("interface_name is missing in connection config!")
            # Creating the map including global interfaces and subgraph interfaces.
            input_map = {
                "global_interface_name": input_dict["interface_name"],
                "mod_idx": input_dict["connection"][0]["mod_idx"],
                "module_interface_name": input_dict["connection"][0]["interface_name"],
            }
            input_connection.append(input_map)

        # Create a map including global parameters groups and modules.
        param_connection = []
        for param_name in self.param_group_bindings.bindings:
            param_dict = self.param_group_bindings.bindings[param_name].get_binding_dict()
            param_map = {
                "global_param_name": param_dict["interface_name"],
                "mod_idx": param_dict["connection"][0]["mod_idx"],
            }
            param_connection.append(param_map)

        mconfig["module_connection"] = module_connection
        mconfig["input_connection"] = input_connection
        mconfig["param_connection"] = param_connection
        return mconfig

    def dag_topology_sort(self):
        """Use topological sort to get order of pipeline modules."""
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

        mod_wrapper_sort = {}
        for mod, i in zip(mlist, range(len(mlist))):
            self.mod_wrapper[mod].set_idx_name(i)
            mod_wrapper_sort[mod] = self.mod_wrapper[mod]

        self.mod_wrapper = mod_wrapper_sort

    def get_mod_idx(self, mod):
        # Return the module index.
        idx = self.mod_wrapper[mod].idx
        return idx

    def pipe_input(self, name):
        # Return the input interface according to the name.
        return self.input_bindings[name]

    def pipe_output(self, idx):
        # Return the output interface according to the name.
        return self.output_bindings[idx]
