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

    def __init__(self, kernel, model, graph, tunning_node, name):
        self._kernel = kernel
        self._model = model
        self._graph = graph
        self._tunning_name = name
        self._tunning_node = tunning_node

        self._input_dict = {}

        self._cuda_source_code = None

        self._workspace_size = []  # eid -> workspace size
        self._workspace_dtype = []  # eid -> workspace dtype
        self._total_workspace_size = 0  # total workspace size need by plugin

        # Kernel related params
        self._device_function_list = (
            {}
        )  # kernel -> index for params of host function or address based on address
        self._device_thread_config = {}  # kernel -> thread dim
        self._device_function_order = []  # kernel invoke order
        self._device_allocate_memory_size = {}  # address -> (dtype, extent)

        # Host side function attrs
        self._host_function_list = {}  # function -> eid of params (firstly inputs, then outputs)
        self._host_function_order = []  # host function order

        self._nums_inputs = 0  # number of inputs
        self._nums_outputs = 0  # number of outputs
        self._output_dtype = []  # dtype of outputs
        self._output_shape = []  # shape of outputs
        self._constant_params = {}  # constant params, storage_id -> data
        self._tvm_workspace_constant = {}

        self._tensor_type = []  # tensor type of inputs and outputs

        self._storage_id = []  # eid -> storage id
        self._device_function_configuration = None

        self._parse_shape_and_type()
        self._parse_kernel_params()
        self._parse_device_function_inputs()
        self._parse_device_function_config()

    def _describe(self):
        """Use for debug."""
        print(f"Cuda source code >>> {self._cuda_source_code}")
        print(f"Constant params >>> {self._constant_params}")
        print(f"Device Function List >>> {self._device_function_list}")
        print(f"Device Thread Config >>> {self._device_thread_config}")
        print(f"Device Function Order >>> {self._device_function_order}")
        print(f"Nums Input >>> {self._nums_inputs}")
        print(f"Nums Output >>> {self._nums_outputs}")
        print(f"Workspace Data Type >>> {self._workspace_dtype}")
        print(f"Workspace Size >>> {self._workspace_size}")
        print(f"Host Function List >>> {self._host_function_list}")
        print(f"Host Function Order >>> {self._host_function_order}")
        print(f"Storage Id >>> {self._storage_id}")
        print(f"Device Memory Size >>> {self._device_allocate_memory_size}")

    # Parse Constant.
    def _parse_constant_params(self, constant_params):
        tvm_constant = {}
        for key, value in constant_params.items():
            tvm_constant[key] = value.flatten()
        return tvm_constant

    # Parse device functions params order.
    def _parse_device_function_list(self, device_function_list):
        _device_function_list = {}
        for device_function in device_function_list.split("\n"):
            if len(device_function) == 0:
                continue
            item = device_function.split()

            _device_function_list[item[0]] = item[1:]
        return _device_function_list

    # Parse device functions thread config.
    def _parse_device_function_thread_config(self, device_function_thread_config):
        kernel_thread_config = {}
        kernel_order = []
        for item in device_function_thread_config.split("\n"):
            if len(item) == 0:
                continue
            config = item.split()
            kernel_name = config[0]
            kernel_thread_config[kernel_name] = config[1:]
            kernel_order.append(kernel_name)
        return kernel_thread_config, kernel_order

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

    def _parse_host_function_list(self, host_function_list):
        """
        Parse the list of host functions.
        """
        func_call = {}
        host_executor_order = {}
        host_func_order = []
        for host_func_inorder in host_function_list.split("\n"):
            if len(host_func_inorder) == 0:
                continue
            tvm_host_func = host_func_inorder.split()
            if tvm_host_func[0] not in host_executor_order.keys():
                host_executor_order[tvm_host_func[0]] = tvm_host_func[1:]
                host_func_order.append(tvm_host_func[0])
                func_call[tvm_host_func[0]] = 0
            else:
                func_call[tvm_host_func[0]] += 1
                func_name = tvm_host_func[0] + "_" + str(func_call[tvm_host_func[0]])
                host_executor_order[func_name] = tvm_host_func[1:]
                host_func_order.append(func_name)
        return host_executor_order, host_func_order

    def _parse_kernel_params(self):
        self._cuda_source_code = self._kernel.cuda_source_code
        self._constant_params = self._parse_constant_params(self._kernel.constant_params)
        self._device_function_list = self._parse_device_function_list(
            self._kernel.device_function_list
        )
        (
            self._device_thread_config,
            self._device_function_order,
        ) = self._parse_device_function_thread_config(self._kernel.device_function_thread_config)
        self._device_allocate_memory_size = self._parse_device_allocate_memory_size(
            self._kernel.device_allocate_memory_size
        )
        self._nums_inputs = self._parse_nums_input(self._kernel.num_inputs)
        self._nums_outputs = self._parse_nums_output(self._kernel.num_outputs)
        self._workspace_dtype = self._parse_workspace_dtype(self._kernel.workspace_dtype)
        self._workspace_size = self._parse_workspace_size(self._kernel.workspace_size)
        self._host_function_list, self._host_function_order = self._parse_host_function_list(
            self._kernel.host_function_list
        )
        self._storage_id = self._parse_storageid(self._kernel.storageid)

    def _parse_shape_and_type(self):
        """
        Infer for input and output shape.
        """
        tunning_node = self._tunning_node

        for inp in tunning_node.inputs:
            self._tensor_type.append(python_to_trt_type_mapping[inp.dtype.name])

        for oup in tunning_node.outputs:
            self._tensor_type.append(python_to_trt_type_mapping[oup.dtype.name])

        self._output_shape = [oup.shape for oup in tunning_node.outputs]

    def _parse_device_function_inputs(self):
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
        storage_id_to_allocate_size = {}
        for eid in range(len(self._workspace_size)):
            sid = int(self._storage_id[eid])
            if sid not in storage_id_to_allocate_size.keys():
                storage_id_to_allocate_size[sid] = 0
            storage_id_to_allocate_size[sid] = max(
                int(self._workspace_size[eid]), int(storage_id_to_allocate_size[sid])
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
                workspace_size += int(storage_id_to_allocate_size[sid])

                if (
                    self._input_dict[str(eid)] not in self._tvm_workspace_constant.keys()
                    and str(sid) in self._constant_params.keys()
                ):
                    self._tvm_workspace_constant[self._input_dict[str(eid)]] = (
                        self._constant_params[str(sid)],
                        tvm_to_c_type_mapping[self._workspace_dtype[eid]],
                        int(eid),
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

    def _parse_device_function_config(self):
        """
        Grid, Block Layout, etc.
        """
        output_json = {}
        kernel_call_times = {}
        for i in range(len(self._device_function_order)):
            device_funtion_name = self._device_function_order[i]
            host_function_name = re.sub(r"_kernel_?\d*", "", device_funtion_name, count=1)

            if device_funtion_name not in output_json.keys():
                output_json[device_funtion_name] = {}
                kernel_call_times[device_funtion_name] = 0
                unique_device_function_name = device_funtion_name
            else:
                kernel_call_times[device_funtion_name] += 1
                host_function_name = (
                    host_function_name + "_" + str(kernel_call_times[device_funtion_name])
                )
                unique_device_function_name = (
                    device_funtion_name + "_" + str(kernel_call_times[device_funtion_name])
                )
                output_json[unique_device_function_name] = {}

            # grid and block dim
            output_json[unique_device_function_name]["grid_dim"] = self._device_thread_config[
                device_funtion_name
            ][0].strip("grid=")
            output_json[unique_device_function_name]["block_dim"] = self._device_thread_config[
                device_funtion_name
            ][1].strip("block=")

            device_param_order = self._device_function_list[device_funtion_name]
            host_param_order = self._host_function_list[host_function_name]  # eid

            enqueue_params = ""
            for j in range(len(device_param_order)):
                if device_param_order[j].isdigit():
                    eid = host_param_order[int(device_param_order[j])]
                    enqueue_params += (
                        "("
                        + tvm_to_c_type_mapping[self._workspace_dtype[int(eid)]]
                        + "*)"
                        + self._input_dict[str(eid)]
                    )
                else:
                    if device_param_order[j] in self._input_dict.keys():
                        enqueue_params += self._input_dict[device_param_order[j]]

                if j != len(device_param_order) - 1:
                    enqueue_params += ", "
            output_json[unique_device_function_name]["enqueue_params"] = enqueue_params
        self._device_function_configuration = output_json

    @property
    def host_func_order(self):
        return self._host_function_order

    @property
    def device_function_order(self):
        return self._device_function_order

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
        return self._tvm_workspace_constant

    @property
    def cuda_source_code(self):
        return self._cuda_source_code

    @property
    def plugin_name(self):
        return self._kernel.plugin_name

    @property
    def onnx_op_type(self):
        return self._kernel.onnx_op_type
