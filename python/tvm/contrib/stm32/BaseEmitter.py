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

# pylint: disable=line-too-long

"""Code emission for the STM32 targets."""

import json
import os
import re
import tarfile

from datetime import datetime

import numpy as np

import tvm
from tvm.contrib import utils

AI_TOOLS_VERSION_MAJOR = 1
AI_TOOLS_VERSION_MINOR = 8
AI_TOOLS_VERSION_MICRO = 0

AI_API_VERSION_MAJOR = 1
AI_API_VERSION_MINOR = 0
AI_API_VERSION_MICRO = 0

AI_TOOLS_REVISION = "v1"

DBAR = "=" * 60


def _fix_name(node_name):
    """ Replace ':' with '_' in names like 'InputImg:0' """
    return node_name.replace(":", "_")


def get_input_tensor_name(node_name):
    return _fix_name(node_name)


def get_output_tensor_name(node_name, idx):
    return _fix_name(node_name) + "_" + str(idx)


def get_tensor_data_name(tensor_name):
    return tensor_name + ".dltensor.data"


def get_node_args_name(node_name):
    return _fix_name(node_name) + "_args"


def get_node_arg_types_name(node_name):
    return _fix_name(node_name) + "_arg_type_ids"


def _get_type_size(dltype):

    if dltype in ("uint64", "int64"):
        return 8
    if dltype in ("uint32", "int32", "float32"):
        return 4
    if dltype in ("uint16", "int16"):
        return 2
    if dltype in ("uint8", "int8"):
        return 1

    raise ValueError(f"Data type {dltype} is not supported")


def _get_type_data(dltype):

    if dltype == "uint64":
        return "kDLUInt, 64, 1"
    if dltype == "int64":
        return "kDLInt, 64, 1"
    if dltype == "float32":
        return "kDLFloat, 32, 1"
    if dltype == "uint32":
        return "kDLUInt, 32, 1"
    if dltype == "int32":
        return "kDLInt, 32, 1"
    if dltype == "uint16":
        return "kDLUInt, 16, 1"
    if dltype == "int16":
        return "kDLInt, 16, 1"
    if dltype == "uint8":
        return "kDLUInt, 8, 1"
    if dltype == "int8":
        return "kDLInt, 8, 1"

    raise ValueError(f"Data type {dltype} is not supported")


def get_aligned_offset(offset, dltype):

    align = _get_type_size(dltype)

    if offset % align != 0:
        offset = offset + (align - offset % align)

    return offset


def _get_tensor_elts(dims):
    size = 1
    for dim in dims:
        size = size * dim
    return size


def get_tensor_size(dims, dltype):
    size = _get_tensor_elts(dims)
    return size * _get_type_size(dltype)


def _emit_tensor_shape(dl_tensor_name, ndim, shape, strides, out_c):
    out_c.write(f"AI_STATIC int64_t {dl_tensor_name}_shape[{ndim}] = {{{shape[1:-1]}}}; \n")
    assert strides is None, f"###Error: non-compact tensors are not handled yet."
    out_c.write(f"AI_STATIC int64_t {dl_tensor_name}_strides[{ndim}] = {{}}; \n")


def _emit_tensor_quant(dl_tensor_name, quantization_map, out_c):

    if dl_tensor_name in quantization_map:
        quantization = quantization_map[dl_tensor_name]
    else:
        quantization = None

    if quantization is not None:
        # minval = quantization['min']
        # maxval = quantization['max']
        scale = quantization["scale"]
        zero_point = quantization["zero_point"]
        # dim = quantization['dim']

        #
        # Sometimes we get a scalar with ScaleAsNumpy.
        # This seem to mean not quantized ?
        #
        if not isinstance(scale, np.ndarray):
            assert scale == 0.0, f"Non-quantized tensor with scale != 0.0"
            assert (
                not isinstance(zero_point, np.ndarray) and zero_point == 0
            ), f"Non-quantized tensor with zero_point != 0"
            return None

        scale_size = len(scale)
        zero_point_size = len(zero_point)

        assert len(scale) == len(
            zero_point
        ), f"Inconsistent quantizations scale:{scale} vs zero-point:{zero_point}"

        if len(scale) == 1:
            quant_name = dl_tensor_name + "_quant"
            out_c.write(f"AI_STATIC float {quant_name}_scale[{scale_size}] = {{ ")
            for val in scale:
                out_c.write(f"{val}, ")
            out_c.write(f"}};\n")
            out_c.write(f"AI_STATIC int32_t {quant_name}_zero_point[{zero_point_size}] = {{ ")
            for val in zero_point:
                out_c.write(f"{val}, ")
            out_c.write(f"}};\n")
            out_c.write(f"AI_STATIC ai_quantization_info {quant_name} = {{\n")
            out_c.write(f"  .scale = {quant_name}_scale,\n")
            out_c.write(f"  .zero_point = {quant_name}_zero_point,\n")
            out_c.write(f"  .dim = -1\n")
            out_c.write(f"}}; \n")

            return quant_name

    return None


def emit_tensor_init(dl_tensor_name, tensor, quantization_map, out_c):
    """Emit tensor definition."""
    dltype = tensor["dltype"]
    dims = tensor["dims"]
    strides = tensor["strides"]
    if "device" in tensor:
        device = tensor["device"]
    else:
        device = "kDLCPU"
    # storage_id = tensor['storage_id']
    byte_offset = tensor["byte_offset"]
    # size = tensor['size']
    dtype = _get_type_data(dltype)
    ndim = len(dims)
    shape = str(dims)
    _emit_tensor_shape(dl_tensor_name, ndim, shape, strides, out_c)

    #
    # Quantization
    #
    if quantization_map is not None:
        quant_name = _emit_tensor_quant(dl_tensor_name, quantization_map, out_c)
    else:
        quant_name = None

    #
    # Contents
    #
    out_c.write(f"AI_ALIGNED({8}) AI_STATIC ai_tensor {dl_tensor_name} = {{ \n")
    out_c.write(f"  .dltensor = {{ \n")
    out_c.write(f"  .data = (ai_ptr)(NULL), \n")
    #
    # TODO: use the 'storage_id'
    #
    out_c.write(f"  .device = {{ {device}, 0 }}, \n")
    out_c.write(f"  .ndim = {ndim}, \n")
    out_c.write(f"  .dtype = {{ {dtype} }}, \n")
    out_c.write(f"  .shape = {dl_tensor_name}_shape, \n")
    if strides is not None:
        out_c.write(f"  .strides = {dl_tensor_name}_strides, \n")
    else:
        out_c.write(f"  .strides = NULL, \n")
    out_c.write(f"  .byte_offset = {byte_offset} \n")
    out_c.write(f"  }}, \n")
    #
    # Figure out quantization, if exists
    #
    if quant_name is not None:
        out_c.write(f"  .quant = &{quant_name} \n")
    else:
        out_c.write(f"  .quant = NULL \n")
    out_c.write(f"}}; \n")


class BaseEmitter:
    """Base code emitter class."""

    def __init__(
        self, model_name, include_activations=True, include_inputs=True, include_outputs=True
    ):
        """Initialize the Emitter instance.

        Parameters
        ----------
        include_activations:
            The Emitter allocates the storage for the activations data
            and places it in a specific data section. If Falsr, the
            main application is responsible for allocating the activations
            storage. Default: True.

        include_inputs/include_outputs:
            The Emitter allocates the storage for the input/output data.
            This storage is shared with the activations and placed in the
            specific activations data section. If False, the main
            application is responsible for allocating the input/output
            data storage. Default: True.

        Returns
        -------
            CodeEmitter object.

        """

        #
        # Fix the model name
        #
        model_name = re.sub("[^0-9a-zA-Z_]+", "_", model_name)
        model_name = model_name.lower()
        self.model_name_ = model_name

        #
        # Constants:
        #
        self.data_alignment = 8

        #
        # Static model: activations placed into a nn_data_act section
        # Dynamic model: activations need to be malloc'ed by the
        #   applications.
        #
        if include_activations is True:
            self.activations_static = 1
        else:
            self.activations_static = 0

        #
        # Inputs/outputs may be allocated within the activations or
        # separately.
        # TODO: Separate the inputs from activations inside TVM.
        #
        if include_inputs is True:
            assert (
                self.activations_static == 1
            ), "###Error: Static inputs are not allowed without activations."
            self.inputs_static = 1
        else:
            self.inputs_static = 0

        if include_outputs is True:
            assert (
                self.activations_static == 1
            ), "###Error: Static outputs are not allowed without activations."
            self.outputs_static = 1
        else:
            self.outputs_static = 0

        #
        # Parsed graph
        #
        self.nodes_ = []
        self.arg_nodes_ = []
        self.outputs_ = []
        self.attrs_ = {}
        self.node_row_ptr_ = []
        #
        # Filled by data_placement()
        #
        self.weights_ = {}
        self.activations_ = {}
        self.input_data_ = {}
        self.output_data_ = {}
        self.nodes_size_ = 0
        self.weights_size_ = 0
        self.activations_size_ = 0

        self.graph_ = None
        self.params_ = {}
        self.lib_ = None

    def _parse_model(self):
        """Parse the module graph. Build internal data structures."""

        for key in self.graph_:
            if key == "nodes":
                self.nodes_ = self.graph_["nodes"]
            elif key == "arg_nodes":
                self.arg_nodes_ = self.graph_["arg_nodes"]
            elif key == "node_row_ptr":
                self.node_row_ptr_ = self.graph_["node_row_ptr"]
            elif key == "heads":
                self.outputs_ = self.graph_["heads"]
            elif key == "attrs":
                self.attrs_ = self.graph_["attrs"]
            elif key == "metadata":
                continue
            else:
                print("### Error: JSON key {} not supported".format(key))
                assert False

    def parse_library_format(self, model_library_format_path):
        """Parse the module. Build internal data structures.

        Parameters
        ----------
        model_name :
           Name for this model.

        model_library_format_path :
           The module to parse
        """

        temp_dir = utils.tempdir()
        extract_path = temp_dir.relpath("extract")
        os.mkdir(extract_path)
        with tarfile.TarFile(model_library_format_path) as f:
            f.extractall(extract_path)

        #
        # Extract informations from the Model Library Format
        #
        graph_file = os.path.join(extract_path, "runtime-config", "graph", "graph.json")
        with open(graph_file, "r") as f:
            # returns JSON object as a dictionary
            graph_dict = json.load(f)

        param_file = os.path.join(extract_path, "parameters", "default.params")
        with open(param_file, "rb") as f:
            params = tvm.runtime.load_param_dict(f.read())
            #
            # Map -> Python Dict
            #
            tmp_dict = {}
            for (k, v) in params.items():
                tmp_dict[k] = v

            #
            # Sort params for debugging
            #
            for k in sorted(tmp_dict.keys()):
                self.params_[k] = tmp_dict[k]

        src_dir = os.path.join(extract_path, "codegen", "host", "src")
        # List of strings from Model Library Format C files
        src_files = []
        for filename in os.listdir(src_dir):
            with open(os.path.join(src_dir, filename), "r") as fin:
                src = fin.read()
                src_files.append(src)

        self.graph_ = graph_dict
        self.lib_ = src_files

        self._parse_model()

    def parse_model(self, module):
        """Parse the module. Build internal data structures.

        Parameters
        ----------
        model_name :
           Name for this model.

        module : TVM module
           The module to parse
        """

        graph = module.get_json()
        if not isinstance(graph, (str,)):
            try:
                graph = graph._tvm_graph_json()
            except AttributeError:
                raise ValueError("Type %s is not supported" % type(graph))

        #
        # Sort params for debugging
        #
        tmp_params = module.get_params()
        for k in sorted(tmp_params.keys()):
            self.params_[k] = tmp_params[k]

        self.graph_ = json.loads(graph)
        self.lib_ = module.get_lib()

        self._parse_model()

    def _tensor_is_output(self, nid, idx):

        for out in self.outputs_:
            out_nid = out[0]
            out_idx = out[1]
            if out_nid == nid and out_idx == idx:
                return True
        return False

    def _get_tensor_from_node(self, nid, idx):
        #
        # 'eid' is index into the dltype', 'shape', etc.
        #
        eid = self.node_row_ptr_[nid] + idx

        dltype = self.attrs_["dltype"][1][eid]
        dims = self.attrs_["shape"][1][eid]
        storage_id = self.attrs_["storage_id"][1][eid]
        ndim = len(dims)
        size = get_tensor_size(dims, dltype)
        tensor = {
            "dltype": dltype,
            "ndim": ndim,
            "dims": dims,
            "strides": None,
            "storage_id": storage_id,
            "byte_offset": 0,
            "offset": 0,
            "size": size,
        }

        return tensor

    def _get_node_input_name(self, arg):
        arg_nid = arg[0]
        arg_idx = arg[1]
        arg_node = self.nodes_[arg_nid]
        arg_name = self.nodes_[arg_nid]["name"]
        if arg_node["op"] == "null":
            # parameter
            dl_tensor_name = get_input_tensor_name(arg_name)
        elif arg_node["name"] == "reshape_nop":
            # Handle __nop
            src = arg_node["inputs"][0]
            dl_tensor_name = self._get_node_input_name(src)
        else:
            # activation
            dl_tensor_name = get_output_tensor_name(arg_name, arg_idx)
        return dl_tensor_name

    def compute_data_placement(self):
        """ Compute inputs, outputs, weight, activation sizes"""

        self.inputs_ = self.arg_nodes_.copy()

        #
        # weights:
        #
        offset = 0

        #
        # Order is important := the order of how the weights will be
        # written to memory.
        #
        for key in self.params_:
            #
            # First, find the node in graph
            #
            nid = 0
            for node in self.nodes_:
                if node["name"] == key:
                    break
                nid += 1

            #
            # May not find
            #
            if nid == len(self.nodes_):
                continue

            dl_tensor_name = get_input_tensor_name(key)
            tensor = self._get_tensor_from_node(nid, 0)
            #
            # Compute the offset
            #
            dltype = tensor["dltype"]
            aligned_offset = get_aligned_offset(offset, dltype)
            tensor["offset"] = aligned_offset

            for idx in self.arg_nodes_:
                node = self.nodes_[idx]
                node_name = node["name"]
                if node_name == key:
                    self.inputs_.remove(idx)

            self.weights_[dl_tensor_name] = tensor

            #
            # Next offset
            #
            offset = aligned_offset + tensor["size"]

        self.weights_size_ = offset

        #
        # activations:
        #

        buffer_list_ = {}

        nid = 0

        for node in self.nodes_:

            if node["op"] == "null":
                nid += 1
                continue

            if node["op"] != "tvm_op":
                raise ValueError(f"Only TVM ops are supported")

            node_name = node["name"]
            node_attrs = node["attrs"]
            func_name = node_attrs["func_name"]
            num_outputs = int(node_attrs["num_outputs"])

            if func_name == "__nop":
                assert node_name == "reshape_nop", f"Unsupported __nop operator {node_name}."
                assert num_outputs == 1
                assert not self._tensor_is_output(nid, 0)
                nid += 1
                continue

            for idx in range(num_outputs):
                #
                # Do not count the 'outputs_'
                #
                if self._tensor_is_output(nid, idx):
                    continue

                dl_tensor_name = get_output_tensor_name(node_name, idx)
                tensor = self._get_tensor_from_node(nid, idx)
                #
                # Remember this tensor with the storage id
                #
                storage_id = tensor["storage_id"]
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)

                self.activations_[dl_tensor_name] = tensor

            self.nodes_size_ = self.nodes_size_ + 1

            nid += 1

        #
        # Compute 'input_data_'
        #
        offset = 0

        for nid in self.inputs_:
            node = self.nodes_[nid]
            node_name = node["name"]
            #
            # Arthur: I suppose that input nodes only have a single
            #         output dependency
            #
            dl_tensor_name = get_input_tensor_name(node_name)
            #
            # This tensor is at some index inside 'input_data_' dictionary
            # depending on the 'inputs_' list order. We refer to this position
            # when generating the XXX.h file.
            #
            tensor = self._get_tensor_from_node(nid, 0)

            if self.inputs_static == 1:
                #
                # Remember this tensor with the storage id
                #
                storage_id = tensor["storage_id"]
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)
            else:
                #
                # Compute the offset
                #
                dltype = tensor["dltype"]
                aligned_offset = get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset
                #
                # Next offset
                #
                offset = aligned_offset + tensor["size"]

            self.input_data_[dl_tensor_name] = tensor

        #
        # Compute 'output_data_'
        #
        offset = 0

        for output in self.outputs_:

            nid = output[0]
            idx = output[1]

            node = self.nodes_[nid]
            node_name = node["name"]

            dl_tensor_name = get_output_tensor_name(node_name, idx)

            tensor = self._get_tensor_from_node(nid, idx)

            if self.outputs_static == 1:
                #
                # Remember this tensor with the storage id
                #
                storage_id = tensor["storage_id"]
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)
            else:
                #
                # Compute the offset
                #
                dltype = tensor["dltype"]
                aligned_offset = get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset
                #
                # Next offset
                #
                offset = aligned_offset + tensor["size"]

            self.output_data_[dl_tensor_name] = tensor

        #
        # Go over all storage IDs and compute offsets and activations_size_
        #
        offset = 0

        for storage_id in buffer_list_:
            buffer_entry = buffer_list_[storage_id]

            new_offset = offset
            for tensor in buffer_entry:
                assert tensor["storage_id"] == storage_id
                dltype = tensor["dltype"]
                aligned_offset = get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset
                size = tensor["size"]
                if (aligned_offset + size) > new_offset:
                    new_offset = aligned_offset + size
            offset = new_offset

        self.activations_size_ = offset

    def _emit_tensor_activation(self, dl_tensor_name, tensor, out_c):

        storage_id = tensor["storage_id"]
        offset = tensor["offset"]
        out_c.write(f"  // \n")
        out_c.write(f"  // {dl_tensor_name}: storage_id:{storage_id}\n")
        out_c.write(f"  // \n")
        out_c.write(f"  {dl_tensor_name}.dltensor.data = (ai_ptr)(activations + {offset}); \n")
        out_c.write(f"\n")

    def _emit_activation_init(self, name, out_c):
        """ Emits buffer initialization code for activation tensors."""
        out_c.write(f"// {DBAR} \n")
        out_c.write(f"//   {name}_configure_activations \n")
        out_c.write(f"// {DBAR} \n")
        out_c.write(f"AI_STATIC AI_INLINE\n")
        out_c.write(f"ai_status {name}_configure_activations ( \n")
        out_c.write(f"  const ai_ptr activations \n")
        out_c.write(f") \n")
        out_c.write(f"{{ \n")
        out_c.write(f"  if (activations == NULL) {{\n")
        out_c.write(
            f'    TVMAPISetLastError ("Non-null activations arena is required for this model.");\n'
        )
        out_c.write(f"    return AI_STATUS_ERROR;\n")
        out_c.write(f"  }}\n")
        out_c.write(f"\n")

        #
        # Allocate inputs with the static model
        #
        if self.inputs_static:
            for dl_tensor_name in self.input_data_:
                tensor = self.input_data_[dl_tensor_name]
                self._emit_tensor_activation(dl_tensor_name, tensor, out_c)

        #
        # Prepare activation buffers
        #
        for dl_tensor_name in self.activations_:
            tensor = self.activations_[dl_tensor_name]
            self._emit_tensor_activation(dl_tensor_name, tensor, out_c)

        #
        # Allocate outputs with the static model
        #
        if self.outputs_static:
            for dl_tensor_name in self.output_data_:
                tensor = self.output_data_[dl_tensor_name]
                self._emit_tensor_activation(dl_tensor_name, tensor, out_c)

        out_c.write(f"  return AI_STATUS_OK; \n")
        out_c.write(f"}} \n")
        out_c.write(f"\n")

    def _emit_params_init(self, name, out_c):
        """ Emits buffer initialization code for params tensors."""
        out_c.write(f"// {DBAR} \n")
        out_c.write(f"//   {name}_configure_weights \n")
        out_c.write(f"// {DBAR} \n")
        out_c.write(f"AI_STATIC AI_INLINE\n")
        out_c.write(f"ai_status {name}_configure_weights ( \n")
        out_c.write(f"  const ai_ptr weights \n")
        out_c.write(f") \n")
        out_c.write(f"{{ \n")
        out_c.write(f"  if (weights == NULL) {{\n")
        out_c.write(
            f'    TVMAPISetLastError ("Non-null weights arena is required for this model.");\n'
        )
        out_c.write(f"    return AI_STATUS_ERROR;\n")
        out_c.write(f"  }}\n")
        out_c.write(f"\n")

        for dl_tensor_name in self.weights_:
            tensor = self.weights_[dl_tensor_name]
            offset = tensor["offset"]
            out_c.write(f"  // \n")
            out_c.write(f"  // {dl_tensor_name}\n")
            out_c.write(f"  // \n")
            out_c.write(f"  {dl_tensor_name}.dltensor.data = (ai_ptr)(weights + {offset}); \n")
            out_c.write(f"\n")

        out_c.write(f"  return AI_STATUS_OK; \n")
        out_c.write(f"}} \n")
        out_c.write(f"\n")

    def emit_init(self, out_c):
        """ Emits buffer initialization code."""
        #
        # {name}_configure_activations
        #
        self._emit_activation_init(self.model_name_, out_c)
        #
        # {name}_configure_weights
        #
        self._emit_params_init(self.model_name_, out_c)

    def emit_operator_proto(self, node, out_c):
        """Emit operator C prototype."""

        node_attrs = node["attrs"]
        func_name = node_attrs["func_name"]

        if func_name == "__nop":
            return

        if func_name == "__copy":
            return

        out_c.write(
            f"TVM_DLL int32_t {func_name}(void * args, void * arg_type_ids, int32_t num_args); \n"
        )

    def emit_operator_call(self, node, out_c):
        """Emit operator code at call site."""

        node_name = node["name"]
        node_attrs = node["attrs"]
        func_name = node_attrs["func_name"]

        if func_name == "__nop":
            return

        if func_name == "__copy":
            return

        out_c.write(f"  // \n")
        out_c.write(f"  // {func_name}\n")
        out_c.write(f"  // \n")

        #
        # Prepare TVM packed function - this is the one called
        #

        #
        # Get function from the TVM module
        #
        #  void * args         : arg_values.data()
        #  void * arg_type_ids : arg_tcodes.data()
        #  int32_t num_args    : arg_values.size()
        #

        node_name_upper = node_name.upper()
        dl_args_name = get_node_args_name(node_name)
        dl_arg_types_name = get_node_arg_types_name(node_name)

        num_inputs = len(node["inputs"])
        num_outputs = int(node_attrs["num_outputs"])
        num_args = num_inputs + num_outputs

        out_c.write(f"  TVMValue {dl_args_name}[{num_args}]; \n")
        out_c.write(f"  int32_t {dl_arg_types_name}[{num_args}]; \n")

        curr_idx = 0

        for arg in node["inputs"]:
            dl_tensor_name = self._get_node_input_name(arg)
            #
            # If this input is not an activation or a parameter => find the input
            #
            if dl_tensor_name not in self.weights_ and dl_tensor_name not in self.activations_:
                assert (
                    dl_tensor_name in self.input_data_
                ), f"Tensor {dl_tensor_name} not registered ?"
                input_idx = 0
                for dl_entry_name in self.input_data_:
                    if dl_entry_name == dl_tensor_name:
                        break
                    input_idx += 1
                out_c.write(
                    f"  {dl_args_name}[{curr_idx}].v_handle = &inputs[{input_idx}]->dltensor; \n"
                )
            else:
                out_c.write(
                    f"  {dl_args_name}[{curr_idx}].v_handle = &{dl_tensor_name}.dltensor; \n"
                )
            out_c.write(f"  {dl_arg_types_name}[{curr_idx}] = kTVMNDArrayHandle; \n")
            if dl_tensor_name in self.weights_:
                out_c.write(f"#ifdef _DUMP_INPUTS_ \n")
                out_c.write(
                    f'  TVMDumpBuffer ("{dl_tensor_name}", &{dl_tensor_name}, DumpFile_p); \n'
                )
                out_c.write(f"#endif \n")
            elif dl_tensor_name in self.input_data_:
                input_idx = 0
                for dl_entry_name in self.input_data_:
                    if dl_entry_name == dl_tensor_name:
                        break
                    input_idx += 1
                out_c.write(f"#ifdef _DUMP_INPUTS_ \n")
                out_c.write(
                    f'  TVMDumpBuffer ("{dl_tensor_name}", inputs[{input_idx}], DumpFile_p); \n'
                )
                out_c.write(f"#endif \n")

            out_c.write(f"\n")

            curr_idx += 1

        for idx in range(num_outputs):
            dl_tensor_name = get_output_tensor_name(node_name, idx)
            #
            # If this output is not an activation => find the output
            #
            if dl_tensor_name not in self.activations_:

                assert dl_tensor_name in self.output_data_

                output_idx = 0
                for dl_exit_name in self.output_data_:
                    if dl_exit_name == dl_tensor_name:
                        break
                    output_idx += 1
                out_c.write(
                    f"  {dl_args_name}[{curr_idx}].v_handle = &outputs[{output_idx}]->dltensor; \n"
                )
            else:
                out_c.write(
                    f"  {dl_args_name}[{curr_idx}].v_handle = &{dl_tensor_name}.dltensor; \n"
                )
            out_c.write(f"  {dl_arg_types_name}[{curr_idx}] = kTVMNDArrayHandle; \n")
            out_c.write(f"\n")

            curr_idx += 1

        #
        # call this function
        #
        out_c.write(f"#if (_VERBOSE_ > 0) \n")
        out_c.write(f'  printf ("  {func_name}  ... \\r\\n"); \n')
        out_c.write(f"#endif \n")

        out_c.write(f"  if ({func_name} ({dl_args_name}, {dl_arg_types_name}, {num_args})) {{ \n")
        out_c.write(f'    TVMAPISetLastError("Invalid handle");\n')
        out_c.write(f"    return AI_STATUS_ERROR; \n")

        out_c.write(f"  }} \n")
        out_c.write(f"#if (_VERBOSE_ > 0) \n")
        out_c.write(f'  printf ("  {func_name}  Done.\\r\\n"); \n')
        out_c.write(f"#endif \n")
        out_c.write(f"\n")

        out_c.write(f"#ifdef _DUMP_{node_name_upper}_ \n")
        for idx in range(num_outputs):
            dl_tensor_name = get_output_tensor_name(node_name, idx)
            if dl_tensor_name in self.activations_:
                out_c.write(
                    f'  TVMDumpBuffer ("{dl_tensor_name}", &{dl_tensor_name}, DumpFile_p); \n'
                )
            else:
                assert dl_tensor_name in self.output_data_
                output_idx = 0
                for dl_exit_name in self.output_data_:
                    if dl_exit_name == dl_tensor_name:
                        break
                    output_idx += 1
                out_c.write(
                    f'  TVMDumpBuffer ("{dl_tensor_name}", outputs[{output_idx}], DumpFile_p); \n'
                )

        out_c.write(f"#endif \n")

    def emit_params_data(self, target_dir):
        """ Emits the network_data[c,h] files with parameters."""
        name = self.model_name_
        name_upper = self.model_name_.upper()

        data_h_name = os.path.join(target_dir, name + "_data.h")
        data_c_name = os.path.join(target_dir, name + "_data.c")

        out_h = open(data_h_name, "w")
        out_c = open(data_c_name, "w")

        #
        # XXX_data.h
        #
        out_h.write(f"#ifndef __{name_upper}_DATA_H_ \n")
        out_h.write(f"#define __{name_upper}_DATA_H_ \n")
        out_h.write(f"\n")
        out_h.write(f'#include "ai_runtime_api.h" \n')
        out_h.write(f"\n")

        out_h.write(f"AI_API_ENTRY \n")
        out_h.write(f"const ai_ptr ai_{name}_data_weights_get (void); \n")
        out_h.write(f"\n")

        out_h.write(f"#endif /* __{name_upper}_DATA_H_ */ \n")
        out_h.write(f"\n")

        #
        # XXX_data.cc
        #
        out_c.write(f'#include "{name}_data.h" \n')
        out_c.write(f"\n")

        out_c.write(f"const ai_ptr ai_{name}_data_weights_get (void) \n")
        out_c.write(f"{{\n")
        out_c.write(
            f'  AI_ALIGNED(4) static const __attribute__ ((section(".nn_weights"))) uint8_t s_{name}_weights[] = {{ \n'
        )

        offset = 0
        for key in self.params_:
            data = self.params_[key]  # ND Array
            npdata = data.asnumpy()
            blob = npdata.tobytes()

            out_c.write(f'// "{key}": \n')
            out_c.write(f"\t")

            count = 0

            #
            # Align by emitting garbage between un-aligned data
            #
            dl_tensor_name = get_input_tensor_name(key)
            tensor = self.weights_[dl_tensor_name]
            tensor_offset = tensor["offset"]
            tensor_size = tensor["size"]

            while offset < tensor_offset:
                count += 1
                out_c.write("0x{:02X}, ".format(0))
                if count == 12:
                    out_c.write("\n\t")
                    count = 0
                offset += 1

            for val in blob:
                count += 1
                out_c.write("0x{:02X}, ".format(val))
                if count == 12:
                    out_c.write("\n\t")
                    count = 0

            offset += tensor_size

            out_c.write(f"\n")

        out_c.write(f"  }}; \n")
        out_c.write(f"  return (const ai_ptr)s_{name}_weights; \n")
        out_c.write(f"}} \n")
        out_c.write(f"\n")

        out_h.close()
        out_c.close()

    def emit_open(self, out_h, out_c):
        """Emits the network.h file with a few network defines and
        writes the header part of the network.c file."""

        name = self.model_name_
        name_upper = name.upper()

        #
        # XXX.h
        #
        out_h.write(f"#ifndef __AI_{name_upper}_H__ \n")
        out_h.write(f"#define __AI_{name_upper}_H__ \n")
        out_h.write(f"\n")

        out_h.write(f'#include "ai_runtime_api.h" \n')
        out_h.write(f"\n")

        input_size = len(self.input_data_)
        output_size = len(self.output_data_)
        out_h.write(f"#define _{name_upper}_INPUTS_COUNT_ ({input_size})\n")
        out_h.write(f"#define _{name_upper}_OUTPUTS_COUNT_ ({output_size})\n")
        out_h.write(f"#define _{name_upper}_ACTIVATION_BYTES_ ({self.activations_size_})\n")
        out_h.write(f"\n")

        #
        # XXX.c
        #
        out_c.write(f"#include <stdio.h> \n")
        out_c.write(f"\n")

        out_c.write(f'#include "dlpack/dlpack.h" \n')
        # out_c.write(f'#include "tvm/runtime/c_runtime_api.h" \n')
        out_c.write(f'#include "tvm/runtime/c_backend_api.h" \n')
        out_c.write(f'#include "{name}.h" \n')
        out_c.write(f'#include "{name}_data.h" \n')
        out_c.write(f"\n")

    def emit_close(self, out_h, out_c):
        """ Emits the ai_model_info structure. """

        name = self.model_name_
        name_upper = name.upper()

        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        #
        # XXX.h
        #
        out_h.write(f"#endif /*__AI_{name_upper}_H__*/ \n")

        #
        # XXX.c
        #

        #
        # Emit network structure
        #
        num_inputs = len(self.input_data_)
        num_outputs = len(self.output_data_)

        out_c.write(
            f'AI_API_ENTRY  __attribute__ ((section(".nn_models"))) ai_model_info {name}_network = {{\n'
        )
        out_c.write(f'  .name = "{name}",\n')
        out_c.write(f'  .datetime = "{dt_string}",\n')
        out_c.write(f'  .revision = "{AI_TOOLS_REVISION}",\n')
        out_c.write(
            f'  .tool_version = "{AI_TOOLS_VERSION_MAJOR}.{AI_TOOLS_VERSION_MINOR}.{AI_TOOLS_VERSION_MICRO}.0",\n'
        )
        out_c.write(
            f'  .api_version = "{AI_API_VERSION_MAJOR}.{AI_API_VERSION_MINOR}.{AI_API_VERSION_MICRO}.0",\n'
        )
        out_c.write(f"  .n_nodes = {self.nodes_size_},\n")
        out_c.write(f"  .n_inputs = {num_inputs},\n")
        out_c.write(f"  .n_outputs = {num_outputs},\n")
        out_c.write(f"  .activations_size = {self.activations_size_},\n")
        out_c.write(f"  .params_size = {self.weights_size_},\n")
        if self.activations_static == 1:
            out_c.write(f"  .activations = {name}_activations, \n")
        else:
            out_c.write(f"  .activations = NULL, \n")
        out_c.write(f"  .inputs = _InputsList,\n")
        out_c.write(f"  .outputs = _OutputsList,\n")
        out_c.write(f"  .ai_get_params = &ai_{name}_data_weights_get,\n")
        out_c.write(f"  .ai_create = &ai_{name}_create,\n")
        out_c.write(f"  .ai_destroy = &ai_{name}_destroy,\n")
        out_c.write(f"  .ai_run = &ai_{name}_run\n")
        out_c.write(f"}};\n")
        out_c.write(f"\n")

    def emit_params_buffers(self, quantization_map, out_c):
        """ Emits all parameter tensors."""
        name = self.model_name_
        out_c.write(f"// \n")
        out_c.write(f"// Weights: {name}\n")
        out_c.write(f"// \n")
        for dl_tensor_name in self.weights_:
            tensor = self.weights_[dl_tensor_name]
            emit_tensor_init(dl_tensor_name, tensor, quantization_map, out_c)
        out_c.write(f"\n")

    def emit_activation_buffers(self, quantization_map, out_c):
        """ Emits activation tensors, including inputs/outputs."""

        name = self.model_name_

        #
        # Inputs:
        #
        out_c.write(f"// \n")
        out_c.write(f"// Inputs:\n")
        out_c.write(f"// \n")
        #
        # shape/buffer
        #
        for dl_tensor_name in self.input_data_:
            tensor = self.input_data_[dl_tensor_name]
            emit_tensor_init(dl_tensor_name, tensor, quantization_map, out_c)
            out_c.write(f"\n")
        out_c.write(f"\n")

        #
        # tensor
        #
        idx = 0
        out_c.write(f"AI_STATIC ai_tensor * _InputsList[] = {{ \n")
        for dl_tensor_name in self.input_data_:
            out_c.write(f"  &{dl_tensor_name}, // [{idx}]\n")
            idx = idx + 1
        out_c.write(f"}}; \n")
        out_c.write(f"\n")

        #
        # Activations:
        #
        out_c.write(f"// \n")
        out_c.write(f"// Activations: \n")
        out_c.write(f"// \n")
        for dl_tensor_name in self.activations_:
            tensor = self.activations_[dl_tensor_name]
            emit_tensor_init(dl_tensor_name, tensor, quantization_map, out_c)
        out_c.write(f"\n")

        #
        # Outputs:
        #
        out_c.write(f"// \n")
        out_c.write(f"// Outputs:\n")
        out_c.write(f"// \n")
        for dl_tensor_name in self.output_data_:
            tensor = self.output_data_[dl_tensor_name]
            emit_tensor_init(dl_tensor_name, tensor, quantization_map, out_c)
            out_c.write(f"\n")
        out_c.write(f"\n")

        idx = 0
        out_c.write(f"AI_STATIC ai_tensor * _OutputsList[] = {{ \n")
        for dl_tensor_name in self.output_data_:
            out_c.write(f"  &{dl_tensor_name}, // [{idx}]\n")
            idx = idx + 1
        out_c.write(f"}}; \n")
        out_c.write(f"\n")

        if self.activations_static == 1:
            out_c.write(
                f'AI_ALIGNED({self.data_alignment}) __attribute__ ((section(".{name}.nn_data_act"))) uint8_t {name}_activations[{self.activations_size_}];\n'
            )
            out_c.write(f"\n")


def _preprocess_code(src):
    """ Hack the C code implementing the model. """

    dst = "#include <stdio.h>\n" "#include <math.h>\n" '#include "stm32lib.h"\n\n' ""
    for line in src.splitlines():
        #
        # This is sort of hacking - when AoT is available, we will be
        # able to clean this ...
        #
        dst = dst + line + "\n"

    return dst


def write_operators_lib(model_name, lib, target_dir):
    """Write the C code into a file."""

    if isinstance(lib, list):
        # List of strings from Model Library Format C files
        for idx, src in enumerate(lib):
            code = _preprocess_code(src)
            filename = os.path.join(target_dir, f"{model_name}_lib{idx}.c")
            with open(filename, "w") as fout:
                fout.write(code)
    else:
        # a TVM RuntimeGraphFactory
        src = lib.get_source(fmt="c")
        code = _preprocess_code(src)
        filename = os.path.join(target_dir, f"{model_name}_lib.c")
        with open(filename, "w") as fout:
            fout.write(code)
