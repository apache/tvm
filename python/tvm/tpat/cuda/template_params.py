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

import re

from .type_mapping import plugin_type_size, python_to_trt_type_mapping, tvm_to_c_type_mapping


class PluginTemplateParams(object):
    """
    Generate useable params for TensorRT plugin.
    """

    def __init__(self, kernel, model, output_shapes, tunning_node, name):
        self._kernel = kernel
        self._model = model
        self._tunning_name = name
        self._tunning_node = tunning_node

        self._input_dict = {}

        self._cuda_source_code = None

        self._workspace_size = []  # eid -> workspace size
        self._workspace_dtype = []  # eid -> workspace dtype
        self._total_workspace_size = 0  # total workspace size need by plugin

        # Kernel related params
        self._device_function_params = (
            {}
        )  # kernel -> index for params of host function or address based on workspace
        self._device_thread_config = {}  # kernel -> thread dim
        self._device_function_list = []  # kernel invoke order
        self._device_allocate_memory_size = {}  # address -> (dtype, extent), intermediate variable

        # Host side function attrs
        self._host_function_params = {}  # function -> eid of params (firstly inputs, then outputs)

        self._nums_inputs = 0  # number of inputs
        self._nums_outputs = 0  # number of outputs
        self._output_dtype = []  # dtype of outputs
        self._output_shape = output_shapes  # shape of outputs
        self._constant_params = {}  # constant params, storage_id -> data
        self._trt_workspace_constant = {}

        self._tensor_type = []  # tensor type of inputs and outputs

        self._storage_id = []  # eid -> storage id
        self._device_function_configuration = None

        self._parse_tensor_type()
        self._parse_kernel_params()
        self._prepare_input_dict()
        self._prepare_device_function_config()

    def _describe(self):
        """Use for debug."""
        print(f"Cuda source code >>> {self._cuda_source_code}")
        print(f"Constant params >>> {self._constant_params}")
        print(f"Device Function Param >>> {self._device_function_params}")
        print(f"Device Thread Config >>> {self._device_thread_config}")
        print(f"Device Function List >>> {self._device_function_list}")
        print(f"Nums Input >>> {self._nums_inputs}")
        print(f"Nums Output >>> {self._nums_outputs}")
        print(f"Workspace Data Type >>> {self._workspace_dtype}")
        print(f"Workspace Size >>> {self._workspace_size}")
        print(f"Host Function Params >>> {self._host_function_params}")
        print(f"Storage Id >>> {self._storage_id}")
        print(f"Device Memory Size >>> {self._device_allocate_memory_size}")

    # Parse Constant.
    def _parse_constant_params(self, constant_params):
        tvm_constant = {}
        for key, value in constant_params.items():
            tvm_constant[key] = value.flatten()
        return tvm_constant

    def _parse_device_function_list(self, device_function_thread_config):
        function_list = []
        for item in device_function_thread_config.split("\n"):
            if len(item) == 0:
                continue
            item = item.split()

            function_list.append(item[0])

        return function_list

    # Parse device functions params order.
    def _parse_device_function_params(self, device_function_list):
        frequency = {}
        result = {}
        for device_function in device_function_list.split("\n"):
            if len(device_function) == 0:
                continue
            item = device_function.split()
            name = item[0]
            params = item[1:]

            if name not in result.keys():
                result[name] = params
                frequency[name] = 0
            else:
                frequency[name] += 1
                func_name = f"{name}_{frequency[name]}"
                result[func_name] = params
        return result

    # Parse device functions thread config.
    def _parse_device_function_thread_config(self, device_function_thread_config):
        frequency = {}
        kernel_thread_config = {}
        for item in device_function_thread_config.split("\n"):
            if len(item) == 0:
                continue
            config = item.split()
            kernel_name = config[0]
            params = config[1:]

            if kernel_name not in kernel_thread_config.keys():
                kernel_thread_config[kernel_name] = params
                frequency[kernel_name] = 0
            else:
                frequency[kernel_name] += 1
                func_name = f"{kernel_name}_{frequency[kernel_name]}"
                kernel_thread_config[func_name] = params
        return kernel_thread_config

    # Parse global memory allocated in device side.
    def _parse_device_allocate_memory_size(self, device_allocate_global_memory):
        allocate_global_memory = {}
        for allocate_memory in device_allocate_global_memory.split("\n"):
            if len(allocate_memory) == 0:
                continue
            allocate = allocate_memory.split()
            allocate_global_memory[allocate[0]] = allocate[1:]
        return allocate_global_memory

    # Parse variables storage index.
    def _parse_storageid(self, storageid):
        storage_id = []
        for sid in storageid.split("\n"):
            if len(sid) == 0:
                continue
            storage_id = sid.split()
        return storage_id

    # Parse numbers of input, only variable.
    def _parse_nums_input(self, nums_input):
        real_nums_input = int(nums_input) - int(len(self._constant_params))
        return real_nums_input

    # Parse numbers of output.
    def _parse_nums_output(self, nums_output):
        real_nums_output = int(nums_output)
        return real_nums_output

    # Parse datatype of variables in memory.
    def _parse_workspace_dtype(self, workspaces_dtype):
        return workspaces_dtype.split()

    # Parse size of variables in memory.
    def _parse_workspace_size(self, workspace_size):
        return workspace_size.split()

    def _parse_host_function_params(self, host_function_list):
        """
        Parse the list of host functions.
        """
        frequency = {}
        result = {}
        for function in host_function_list.split("\n"):
            if len(function) == 0:
                continue
            data = function.split()
            name = data[0]
            params = data[1:]

            if name not in result.keys():
                result[name] = params
                frequency[name] = 0
            else:
                frequency[name] += 1
                func_name = f"{name}_{frequency[name]}"
                result[func_name] = params
        return result

    def _parse_kernel_params(self):
        self._cuda_source_code = self._kernel.cuda_source_code
        self._constant_params = self._parse_constant_params(self._kernel.constant_params)
        self._device_function_params = self._parse_device_function_params(
            self._kernel.device_function_list
        )
        self._device_function_list = self._parse_device_function_list(
            self._kernel.device_function_thread_config
        )
        self._device_thread_config = self._parse_device_function_thread_config(
            self._kernel.device_function_thread_config
        )
        self._device_allocate_memory_size = self._parse_device_allocate_memory_size(
            self._kernel.device_allocate_memory_size
        )
        self._nums_inputs = self._parse_nums_input(self._kernel.num_inputs)
        self._nums_outputs = self._parse_nums_output(self._kernel.num_outputs)
        self._workspace_dtype = self._parse_workspace_dtype(self._kernel.workspace_dtype)
        self._workspace_size = self._parse_workspace_size(self._kernel.workspace_size)
        self._host_function_params = self._parse_host_function_params(
            self._kernel.host_function_list
        )
        self._storage_id = self._parse_storageid(self._kernel.storageid)

        self._describe()

    def _parse_tensor_type(self):
        """
        Infer for input and output shape.
        """
        tunning_node = self._tunning_node

        for inp in tunning_node.inputs:
            self._tensor_type.append(python_to_trt_type_mapping[inp.dtype.name])

        for oup in tunning_node.outputs:
            self._tensor_type.append(python_to_trt_type_mapping[oup.dtype.name])

    def _prepare_input_dict(self):
        """
        The memory address used by functions params.
        """
        workspace_size = 0
        input_slot_dict = {}  # storageid -> xx

        # 1. for outputs
        for i in range(self._nums_outputs):
            # given index of outputs, return entry id
            eid = self._kernel.graph_module.get_output_eid(i)
            sid = int(self._storage_id[eid])
            # resolve output type given entry id
            self._output_dtype.append(python_to_trt_type_mapping[self._workspace_dtype[eid]])
            self._input_dict[str(eid)] = f"outputs[{i}]"
            input_slot_dict[sid] = f"outputs[{i}]"

        # 2. for inputs, including variable and constants
        storage_id_to_workspace_size = {}  # different entry id may map to same storage id
        for eid in range(len(self._workspace_size)):
            sid = int(self._storage_id[eid])
            if sid not in storage_id_to_workspace_size.keys():
                storage_id_to_workspace_size[sid] = 0
            storage_id_to_workspace_size[sid] = max(
                int(self._workspace_size[eid]), int(storage_id_to_workspace_size[sid])
            )

        for eid in range(len(self._workspace_size)):
            sid = int(self._storage_id[eid])
            if sid in input_slot_dict.keys():
                self._input_dict[str(eid)] = input_slot_dict[sid]
                continue
            if eid < self._nums_inputs:
                # it must be variable
                self._input_dict[str(eid)] = "inputs[" + str(eid) + "]"
            elif eid < len(self._workspace_size) - self._nums_outputs:
                # it must be constant
                if eid == self._nums_inputs:
                    # the first one
                    self._input_dict[str(eid)] = "workspace"
                else:
                    self._input_dict[str(eid)] = f"(workspace + {workspace_size})"
                workspace_size += int(storage_id_to_workspace_size[sid])

                key = self._input_dict[str(eid)]
                if (
                    not key in self._trt_workspace_constant.keys()
                    and str(sid) in self._constant_params.keys()
                ):
                    self._trt_workspace_constant[key] = (
                        self._constant_params[str(sid)],  # value
                        tvm_to_c_type_mapping[self._workspace_dtype[eid]],  # type
                        int(eid),  # id
                    )
            input_slot_dict[sid] = self._input_dict[str(eid)]

        if len(self._device_allocate_memory_size) != 0:
            for key, value in self._device_allocate_memory_size.items():
                self._input_dict[key] = (
                    "("
                    + tvm_to_c_type_mapping[value[0]]
                    + "*)(workspace + "
                    + str(workspace_size)
                    + ")"
                )
                workspace_size += int(value[1]) * plugin_type_size[value[0]]

        self._total_workspace_size = workspace_size

    def _prepare_device_function_config(self):
        """
        Grid, Block Layout, etc.
        """
        configuration = {}
        frequency = {}

        for i in range(len(self._device_function_list)):
            device_function_name = self._device_function_list[i]
            host_function_name = re.sub(r"_kernel_?\d*", "", device_function_name, count=1)

            if device_function_name not in configuration.keys():
                configuration[device_function_name] = {}
                frequency[device_function_name] = 0
            else:
                frequency[device_function_name] += 1
                host_function_name = f"{host_function_name}_{frequency[device_function_name]}"
                device_function_name = f"{device_function_name}_{frequency[device_function_name]}"
                configuration[device_function_name] = {}

            # grid and block dim
            configuration[device_function_name]["grid_dim"] = self._device_thread_config[
                device_function_name
            ][0].strip("grid=")
            configuration[device_function_name]["block_dim"] = self._device_thread_config[
                device_function_name
            ][1].strip("block=")

            device_params = self._device_function_params[device_function_name]
            host_params = self._host_function_params[host_function_name]  # eid of params

            enqueue_params = ""
            for j in range(len(device_params)):
                if device_params[j].isdigit():  # correspond to eid
                    eid = host_params[int(device_params[j])]
                    dtype = self._workspace_dtype[int(eid)]
                    enqueue_params += (
                        "(" + tvm_to_c_type_mapping[dtype] + "*)" + self._input_dict[str(eid)]
                    )
                else:
                    if (
                        device_params[j] in self._input_dict.keys()
                    ):  # correspond to device memory, intermediate variable
                        enqueue_params += self._input_dict[device_params[j]]

                if j != len(device_params) - 1:
                    enqueue_params += ", "
            configuration[device_function_name]["enqueue_params"] = enqueue_params
        self._device_function_configuration = configuration

    @property
    def device_function_list(self):
        return self._device_function_list

    @property
    def device_function_configuration(self):
        return self._device_function_configuration

    @property
    def total_workspace_size(self):
        return self._total_workspace_size

    @property
    def num_outputs(self):
        return self._nums_outputs

    @property
    def output_dtype(self):
        return self._output_dtype

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def tensor_type(self):
        return self._tensor_type

    @property
    def workspace_constant(self):
        return self._trt_workspace_constant

    @property
    def cuda_source_code(self):
        return self._cuda_source_code

    @property
    def plugin_name(self):
        return self._kernel.plugin_name

    @property
    def onnx_op_type(self):
        return self._kernel.onnx_op_type
