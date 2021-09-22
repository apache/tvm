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
import shutil
import tarfile
import textwrap

from datetime import datetime

import numpy as np

import tvm
from tvm.contrib import utils

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


def _get_node_args_name(node_name):
    return _fix_name(node_name) + "_args"


def _get_node_arg_types_name(node_name):
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


C_TYPE_TO_DLTYPE = {
    "uint64": "kDLUInt, 64, 1",
    "int64": "kDLInt, 64, 1",
    "float32": "kDLFloat, 32, 1",
    "uint32": "kDLUInt, 32, 1",
    "int32": "kDLInt, 32, 1",
    "uint16": "kDLUInt, 16, 1",
    "int16": "kDLInt, 16, 1",
    "uint8": "kDLUInt, 8, 1",
    "int8": "kDLInt, 8, 1",
}


def _get_type_data(dltype):
    try:
        return C_TYPE_TO_DLTYPE[dltype]
    except KeyError:
        raise ValueError(f"Data type {dltype} is not supported")


def _get_aligned_offset(offset, dltype):
    align = _get_type_size(dltype)
    if offset % align != 0:
        offset = offset + (align - offset % align)
    return offset


def _get_num_tensor_elts(shape):
    size = 1
    for dim in shape:
        size = size * dim
    return size


def _get_tensor_size_bytes(dims, dltype):
    size = _get_num_tensor_elts(dims)
    return size * _get_type_size(dltype)


def _preprocess_code(src):
    """ Hack the C code implementing the model. """

    # TODO: '#include "stm32lib.h"\n\n' - the 'magic option is not contributed yet
    dst = "#include <stdio.h>\n" "#include <math.h>\n\n"
    for line in src.splitlines():
        #
        # This is sort of hacking - when AoT is available, we will be
        # able to clean this ...
        #
        dst = dst + line + "\n"

    return dst


class CodeEmitter(object):
    """Code emitter class."""

    #
    # Constants:
    #
    DATA_ALIGNMENT_BYTES = 8

    def __init__(self, include_activations=True, include_inputs=True, include_outputs=True):
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
        # Static model: activations placed into a nn_data_act section
        # Dynamic model: activations need to be malloc'ed by the
        #   applications.
        #
        self.activations_static = include_activations

        #
        # Inputs/outputs may be allocated within the activations or
        # separately.
        # TODO: Separate the inputs from activations inside TVM.
        #
        if include_inputs is True:
            assert (
                self.activations_static == True
            ), "###Error: Static inputs are not allowed without activations."
        self.inputs_static = include_inputs

        if include_outputs is True:
            assert (
                self.activations_static == True
            ), "###Error: Static outputs are not allowed without activations."
        self.outputs_static = include_outputs

        #
        # Parsed graph
        #
        self.nodes_ = []
        self.arg_nodes_ = []
        self.outputs_ = []
        self.attrs_ = {}
        self.node_row_ptr_ = []
        #
        # Parameters
        #
        self.params_ = {}
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

        self.quantization_ = {}

    def __extract_quantization_info(self, quantization):
        """ Build dictionary with quantization infos."""

        for dl_tensor_name in self.input_data_:
            if dl_tensor_name in quantization:
                self.quantization_[dl_tensor_name] = quantization[dl_tensor_name]

        #
        # Matching outputs is more difficult because TVM does not preserve
        # output tensor names.
        # We only support models with a single output now.
        #
        assert len(self.output_data_) == 1, "Multiple outputs models are not yet supported."

        for dl_tensor_name in self.output_data_:
            for name in quantization:
                if name not in self.input_data_:
                    self.quantization_["output"] = quantization[name]
                    break

    def __get_node_arg_name(self, arg):
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
            dl_tensor_name = self.__get_node_arg_name(src)
        else:
            # activation
            dl_tensor_name = get_output_tensor_name(arg_name, arg_idx)
        return dl_tensor_name

    def __tensor_is_output(self, nid, idx):
        for out in self.outputs_:
            out_nid = out[0]
            out_idx = out[1]
            if out_nid == nid and out_idx == idx:
                return True
        return False

    def __get_tensor_from_node(self, nid, idx):
        # 'eid' is index into the dltype', 'shape', etc.
        eid = self.node_row_ptr_[nid] + idx
        dltype = self.attrs_["dltype"][1][eid]
        dims = self.attrs_["shape"][1][eid]
        storage_id = self.attrs_["storage_id"][1][eid]
        ndim = len(dims)
        #
        # Get tensor size
        #
        size = _get_tensor_size_bytes(dims, dltype)

        tensor = {
            "dltype": dltype,
            "ndim": ndim,
            "dims": dims,
            "strides": None,
            "storage_id": storage_id,
            #
            # What is this byte_offset really ?
            #
            "byte_offset": 0,
            "offset": 0,
            "size": size,
        }

        return tensor

    def __compute_data_placement(self):
        """ Compute inputs, outputs, weight, activation sizes"""

        self.inputs_ = self.arg_nodes_.copy()

        #
        # weights:
        #
        offset = 0

        for key in self.params_:
            #
            # First, find the node in graph
            #
            nid = 0
            for node in self.nodes_:
                if node["name"] == key:
                    break
                nid += 1

            dl_tensor_name = get_input_tensor_name(key)
            tensor = self.__get_tensor_from_node(nid, 0)
            #
            # Compute the offset
            #
            dltype = tensor["dltype"]
            aligned_offset = _get_aligned_offset(offset, dltype)
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
                assert not self.__tensor_is_output(nid, 0)
                nid += 1
                continue

            for idx in range(num_outputs):
                #
                # Do not count the 'outputs_'
                #
                if self.__tensor_is_output(nid, idx):
                    continue

                dl_tensor_name = get_output_tensor_name(node_name, idx)
                tensor = self.__get_tensor_from_node(nid, idx)
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
            tensor = self.__get_tensor_from_node(nid, 0)

            if self.inputs_static == True:
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
                aligned_offset = _get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset

            self.input_data_[dl_tensor_name] = tensor

            #
            # Next offset
            #
            offset = aligned_offset + tensor["size"]

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

            tensor = self.__get_tensor_from_node(nid, idx)

            if self.outputs_static == True:
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
                aligned_offset = _get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset

            self.output_data_[dl_tensor_name] = tensor

            #
            # Next offset
            #
            offset = aligned_offset + tensor["size"]

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
                aligned_offset = _get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset
                size = tensor["size"]
                if (aligned_offset + size) > new_offset:
                    new_offset = aligned_offset + size
            offset = new_offset

        self.activations_size_ = offset

    def _parse_model(self, quantization=None):
        """Parse the module. Build internal data structures.

        Parameters
        ----------
        module : TVM module or ModuleLibraryFormat object
           The module to parse

        quantization: Dictionary
           The quantization information for model inputs/outputs.
        """

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

        #
        # Build all tensor lists
        #
        self.__compute_data_placement()

        #
        # Extract quantization info for inputs/outputs
        #
        if quantization is not None:
            self.__extract_quantization_info(quantization)

    def parse_library_format(self, model_library_format_path, quantization=None):
        """Parse the module. Build internal data structures.

        Parameters
        ----------
        model_library_format_path :
           The ModuleLibraryFormat object to parse

        quantization: Dictionary
           The quantization information for model inputs/outputs.
        """

        temp_dir = utils.tempdir()
        extract_path = temp_dir.relpath("extract")
        os.mkdir(extract_path)
        with tarfile.TarFile(model_library_format_path) as f:
            f.extractall(extract_path)

        #
        # Extract informations from the Model Library Format
        #
        graph_file = os.path.join(extract_path, "executor-config", "graph", "graph.json")
        with open(graph_file, "r") as f:
            # returns JSON object as a dictionary
            graph_dict = json.load(f)

        params_dict = {}
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
                params_dict[k] = tmp_dict[k]

        src_dir = os.path.join(extract_path, "codegen", "host", "src")
        # List of strings from Model Library Format C files
        src_files = []
        for filename in os.listdir(src_dir):
            with open(os.path.join(src_dir, filename), "r") as fin:
                src = fin.read()
                src_files.append(src)

        self.graph_ = graph_dict
        self.params_ = params_dict
        self.lib_ = src_files

        self._parse_model(quantization)

    def parse_module(self, module, quantization=None):
        """Parse the module. Build internal data structures.

        Parameters
        ----------
        module : TVM Runtime Module
           The module to parse.

        quantization: Dictionary
           The quantization information for model inputs/outputs.
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
        params_dict = {}
        tmp_params = module.get_params()
        for k in sorted(tmp_params.keys()):
            params_dict[k] = tmp_params[k]

        self.graph_ = json.loads(graph)
        self.params_ = params_dict
        self.lib_ = module.get_lib()

        self._parse_model(quantization)

    def __emit_params_data(self, name, out_h, out_c):
        """ Emits the network_data[c,h] files with parameters."""

        name_upper = name.upper()

        #
        # XXX_data.h
        #
        out_h.write(
            textwrap.dedent(
                f"""\
        #ifndef __{name_upper}_DATA_H_
        #define __{name_upper}_DATA_H_
        
        #include \"ai_runtime_api.h\"

        AI_API_ENTRY
        const ai_ptr ai_{name}_data_weights_get (void);
        
        #endif /* __{name_upper}_DATA_H_ */
        """
            )
        )

        #
        # XXX_data.cc
        #
        out_c.write(
            textwrap.dedent(
                f"""
        #include \"{name}_data.h\"

        const ai_ptr ai_{name}_data_weights_get (void)
        {{
          AI_ALIGNED({self.DATA_ALIGNMENT_BYTES}) static const __attribute__ ((section(\".nn_weights\"))) uint8_t s_{name}_weights[] = {{
        """
            )
        )

        #
        # Weights are arranged in the order of 'params_'
        #
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

        out_c.write(
            textwrap.dedent(
                f"""\
          }};
          return (const ai_ptr)s_{name}_weights;
        }}
        """
            )
        )

    def __emit_open(self, name, out_h, out_c):
        """Emits the network.h file with a few network defines and
        writes the header part of the network.c file."""

        name_upper = name.upper()

        input_size = len(self.input_data_)
        output_size = len(self.output_data_)

        #
        # XXX.h
        #
        out_h.write(
            textwrap.dedent(
                f"""\
        #ifndef __AI_{name_upper}_H__
        #define __AI_{name_upper}_H__
        
        #include \"ai_runtime_api.h\"

        #define _{name_upper}_INPUTS_COUNT_ ({input_size})
        #define _{name_upper}_OUTPUTS_COUNT_ ({output_size})
        #define _{name_upper}_ACTIVATION_BYTES_ ({self.activations_size_})
        """
            )
        )

        #
        # XXX.c
        #
        out_c.write(
            textwrap.dedent(
                f"""\
        #include <stdio.h>
        
        #include \"dlpack/dlpack.h\"
        #include \"tvm/runtime/c_runtime_api.h\"
        #include \"{name}.h\"
        #include \"{name}_data.h\"
        """
            )
        )

    def __emit_close(self, name, out_h, out_c):
        """ Emits the ai_model_info structure. """

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

        if self.activations_static == True:
            out_c.write(
                f'AI_ALIGNED({self.DATA_ALIGNMENT_BYTES}) __attribute__ ((section(".{name}.nn_data_act"))) uint8_t {name}_activations[{self.activations_size_}];\n'
            )
        else:
            out_c.write(f"AI_STATIC ai_ptr {name}_activations = NULL;")

        #
        # Emit network structure
        #
        num_inputs = len(self.input_data_)
        num_outputs = len(self.output_data_)

        tool_version = tvm.__version__
        api_version = f"{AI_API_VERSION_MAJOR}.{AI_API_VERSION_MINOR}.{AI_API_VERSION_MICRO}.0"

        out_c.write(
            textwrap.dedent(
                f"""
        AI_API_ENTRY  __attribute__ ((section(".nn_models"))) ai_model_info {name}_network = {{
          .name = \"{name}\",
          .datetime = \"{dt_string}\",
          .revision = \"{AI_TOOLS_REVISION}\",
          .tool_version = \"{tool_version}\",
          .api_version = \"{api_version}\",
          .n_nodes = {self.nodes_size_},
          .n_inputs = {num_inputs},
          .n_outputs = {num_outputs},
          .activations_size = {self.activations_size_},
          .params_size = {self.weights_size_},
          .activations = {name}_activations,
          .inputs = _InputsList,
          .outputs = _OutputsList,
          .ai_get_params = &ai_{name}_data_weights_get,
          .ai_create = &ai_{name}_create,
          .ai_destroy = &ai_{name}_destroy,
          .ai_run = &ai_{name}_run
        }};
        """
            )
        )

    def __emit_tensor_shape(self, dl_tensor_name, ndim, shape, strides, out_c):
        out_c.write(f"AI_STATIC int64_t {dl_tensor_name}_shape[{ndim}] = {{{shape[1:-1]}}}; \n")
        assert strides is None, f"###Error: non-compact tensors are not handled yet."
        out_c.write(f"AI_STATIC int64_t {dl_tensor_name}_strides[{ndim}] = {{}}; \n")

    def __emit_tensor_quant(self, dl_tensor_name, out_c):

        if dl_tensor_name in self.quantization_:
            quantization = self.quantization_[dl_tensor_name]
        #
        # At this time, TVM only supports quantization info with
        # single output models.
        #
        elif dl_tensor_name in self.output_data_ and "output" in self.quantization_.keys():
            quantization = self.quantization_["output"]
        else:
            quantization = None

        if quantization is not None:
            scale = quantization["scale"]
            zero_point = quantization["zero_point"]

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
                out_c.write(f"}};")
                out_c.write(
                    textwrap.dedent(
                        f"""
                AI_STATIC ai_quantization_info {quant_name} = {{
                  .scale = {quant_name}_scale,
                  .zero_point = {quant_name}_zero_point,
                  .dim = -1
                }};
                """
                    )
                )

                return quant_name

        return None

    def __emit_tensor_init(self, dl_tensor_name, tensor, out_c):
        """ Emits the tensor instantiation code. """

        dltype = tensor["dltype"]
        dims = tensor["dims"]
        strides = tensor["strides"]
        byte_offset = tensor["byte_offset"]
        dtype = _get_type_data(dltype)
        ndim = len(dims)
        shape = str(dims)
        self.__emit_tensor_shape(dl_tensor_name, ndim, shape, strides, out_c)

        #
        # Quantization
        #
        quant_name = self.__emit_tensor_quant(dl_tensor_name, out_c)

        #
        # Contents
        #
        # TODO: use the 'storage_id':
        #   "    .ctx = {{ {} }}, \n".format(str(storage_id)[1:-1])
        #
        out_c.write(
            textwrap.dedent(
                f"""
        AI_ALIGNED({self.DATA_ALIGNMENT_BYTES}) AI_STATIC ai_tensor {dl_tensor_name} = {{
          .dltensor = {{
            .data = (ai_ptr)(NULL),
            .device = {{kDLCPU,0}},
            .ndim = {ndim},
            .dtype = {{{dtype}}},
            .shape = {dl_tensor_name}_shape,
            .strides = {dl_tensor_name}_strides,
            .byte_offset = {byte_offset}
          }},
        """
            )
        )

        #
        # Figure out quantization, if exists
        #
        if quant_name is not None:
            out_c.write(f"  .quant = &{quant_name} \n")
        else:
            out_c.write(f"  .quant = NULL \n")
        out_c.write(f"}}; \n")

    def __emit_activation_buffers(self, name, out_c):
        # pylint: disable=unused-argument
        """ Emits activation tensors, including inputs/outputs."""

        out_c.write(
            textwrap.dedent(
                f"""\
        //
        // Inputs:
        //
        """
            )
        )
        #
        # shape/buffer
        #
        for dl_tensor_name in self.input_data_:
            tensor = self.input_data_[dl_tensor_name]
            self.__emit_tensor_init(dl_tensor_name, tensor, out_c)
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

        out_c.write(
            textwrap.dedent(
                f"""\
        //
        // Activations:
        //
        """
            )
        )
        for dl_tensor_name in self.activations_:
            tensor = self.activations_[dl_tensor_name]
            self.__emit_tensor_init(dl_tensor_name, tensor, out_c)
        out_c.write(f"\n")

        #
        # Outputs:
        #
        out_c.write(
            textwrap.dedent(
                f"""\
        //
        // Outputs:
        //
        """
            )
        )
        for dl_tensor_name in self.output_data_:
            tensor = self.output_data_[dl_tensor_name]
            self.__emit_tensor_init(dl_tensor_name, tensor, out_c)
            out_c.write(f"\n")
        out_c.write(f"\n")

        idx = 0
        out_c.write(f"AI_STATIC ai_tensor * _OutputsList[] = {{ \n")
        for dl_tensor_name in self.output_data_:
            out_c.write(f"  &{dl_tensor_name}, // [{idx}]\n")
            idx = idx + 1
        out_c.write(f"}}; \n")
        out_c.write(f"\n")

    def __emit_params_buffers(self, name, out_c):
        """ Emits all parameter tensors."""

        out_c.write(
            textwrap.dedent(
                f"""
        //
        // Weights: {name}
        //
        """
            )
        )
        for dl_tensor_name in self.weights_:
            tensor = self.weights_[dl_tensor_name]
            self.__emit_tensor_init(dl_tensor_name, tensor, out_c)
        out_c.write(f"\n")

    def __emit_network(self, name, out_c):
        """ Emits prototypes for the network operator functions."""

        out_c.write(
            textwrap.dedent(
                f"""
        //
        // Network: {name}
        //
        """
            )
        )
        for node in self.nodes_:
            if node["op"] == "null":
                continue
            assert node["op"] == "tvm_op", f"###Error: Only TVM ops are supported."
            node_attrs = node["attrs"]
            func_name = node_attrs["func_name"]

            if func_name == "__nop":
                continue

            out_c.write(
                f"TVM_DLL int32_t {func_name}(void * args, void * arg_type_ids, int32_t num_args); \n"
            )
        out_c.write(f"\n")

    def __emit_tensor_activation(self, dl_tensor_name, tensor, out_c):

        storage_id = tensor["storage_id"]
        offset = tensor["offset"]
        out_c.write(
            textwrap.indent(
                textwrap.dedent(
                    f"""
        //
        // {dl_tensor_name}: storage_id:{storage_id}
        //
        {dl_tensor_name}.dltensor.data = (ai_ptr)(activations + {offset});
        """
                ),
                "  ",
            )
        )

    def __emit_activation_init(self, name, out_c):
        """ Emits buffer initialization code for activation tensors."""

        out_c.write(
            textwrap.dedent(
                f"""
        // {DBAR}
        //   {name}_configure_activations
        // {DBAR}
        AI_STATIC AI_INLINE
        ai_status {name}_configure_activations (
          const ai_ptr activations
        )
        {{
          if (activations == NULL) {{
            TVMAPISetLastError (\"Non-null activations arena is required for this model.\");
            return AI_STATUS_ERROR;
          }}
        """
            )
        )

        #
        # Allocate inputs with the static model
        #
        if self.inputs_static == True:
            for dl_tensor_name in self.input_data_:
                tensor = self.input_data_[dl_tensor_name]
                self.__emit_tensor_activation(dl_tensor_name, tensor, out_c)

        #
        # Prepare activation buffers
        #
        for dl_tensor_name in self.activations_:
            tensor = self.activations_[dl_tensor_name]
            self.__emit_tensor_activation(dl_tensor_name, tensor, out_c)

        #
        # Allocate outputs with the static model
        #
        if self.outputs_static == True:
            for dl_tensor_name in self.output_data_:
                tensor = self.output_data_[dl_tensor_name]
                self.__emit_tensor_activation(dl_tensor_name, tensor, out_c)

        out_c.write(
            textwrap.dedent(
                f"""
          return AI_STATUS_OK;
        }}
        """
            )
        )

    def __emit_params_init(self, name, out_c):
        """ Emits buffer initialization code for params tensors."""

        out_c.write(
            textwrap.dedent(
                f"""
        // {DBAR}
        //   {name}_configure_weights
        // {DBAR}
        AI_STATIC AI_INLINE
        ai_status {name}_configure_weights (
          const ai_ptr weights
        )
        {{
          if (weights == NULL) {{
            TVMAPISetLastError(\"Non-null weights arena is required for this model.\");
            return AI_STATUS_ERROR;
          }}
        """
            )
        )

        for dl_tensor_name in self.weights_:
            tensor = self.weights_[dl_tensor_name]
            offset = tensor["offset"]
            out_c.write(
                textwrap.indent(
                    textwrap.dedent(
                        f"""\
            //
            //  {dl_tensor_name}
            //
            {dl_tensor_name}.dltensor.data = (ai_ptr)(weights + {offset});
            """
                    ),
                    "  ",
                )
            )

        out_c.write(
            textwrap.dedent(
                f"""
          return AI_STATUS_OK;
        }}
        """
            )
        )

    def __emit_init(self, name, out_c):
        """ Emits buffer initialization code."""

        #
        # {name}_configure_activations
        #
        self.__emit_activation_init(name, out_c)
        #
        # {name}_configure_weights
        #
        self.__emit_params_init(name, out_c)

    def __emit_run(self, name, out_h, out_c):
        """ Emits the run function code."""

        out_h.write(
            textwrap.dedent(
                f"""
        AI_API_ENTRY
        ai_status ai_{name}_run (
          ai_tensor *inputs[],
          ai_tensor *outputs[]
        );
        """
            )
        )

        out_c.write(
            textwrap.dedent(
                f"""
        // {DBAR}
        //   ai_{name}_run
        // {DBAR}
        AI_API_ENTRY
        ai_status ai_{name}_run (
          ai_tensor *inputs[],
          ai_tensor *outputs[]
        )
        {{
        """
            )
        )

        out_c.write(f"#if defined(_DUMP_INPUTS_) ")
        for node in self.nodes_:
            node_name = node["name"]
            node_name_upper = node_name.upper()
            if node["op"] != "null":
                out_c.write(f"|| defined(_DUMP_{node_name_upper}_) ")
        out_c.write(f"\n")
        out_c.write(f'  FILE * DumpFile_p = fopen("dump.txt", "w"); \n')
        out_c.write(f"#endif \n")
        out_c.write(f"\n")

        #
        # Execute nodes one by one
        #
        nid = 0

        for node in self.nodes_:
            node_name = node["name"]
            node_name_upper = node_name.upper()

            nid += 1

            if node["op"] == "null":
                continue

            assert node["op"] == "tvm_op", f"###Error: Only TVM ops are supported."
            node_attrs = node["attrs"]
            func_name = node_attrs["func_name"]

            if func_name == "__nop":
                continue

            out_c.write(f"  // \n")
            out_c.write(f"  // {func_name}\n")
            out_c.write(f"  // \n")

            #
            # Prepare TVM packed function - this is the one called
            #
            if name == "__nop":
                print("      exec: __nop")
                continue

            if name == "__copy":
                print("      exec: __copy")
                continue

            #
            # Get function from the TVM module
            #
            #  void * args         : arg_values.data()
            #  void * arg_type_ids : arg_tcodes.data()
            #  int32_t num_args    : arg_values.size()
            #

            dl_args_name = _get_node_args_name(node_name)
            dl_arg_types_name = _get_node_arg_types_name(node_name)

            num_inputs = len(node["inputs"])
            num_outputs = int(node_attrs["num_outputs"])
            num_args = num_inputs + num_outputs

            out_c.write(f"  TVMValue {dl_args_name}[{num_args}]; \n")
            out_c.write(f"  int32_t {dl_arg_types_name}[{num_args}]; \n")

            curr_idx = 0

            for arg in node["inputs"]:
                dl_tensor_name = self.__get_node_arg_name(arg)
                #
                # If this input is not an activation or a parameter => find the input
                #
                if dl_tensor_name not in self.weights_ and dl_tensor_name not in self.activations_:

                    assert dl_tensor_name in self.input_data_, "Tensor {} not registered ?".format(
                        dl_tensor_name
                    )

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
            out_c.write(
                textwrap.dedent(
                    f"""
            #if (_VERBOSE_ > 0)
              printf (\"  {func_name}  ... \\r\\n\");
            #endif
              if ({func_name} ({dl_args_name}, {dl_arg_types_name}, {num_args})) {{
                TVMAPISetLastError("Invalid handle");
                return AI_STATUS_ERROR;
              }}
            #if (_VERBOSE_ > 0)
              printf (\"  {func_name}  Done.\\r\\n\");
            #endif
            """
                )
            )

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

        out_c.write(f"\n")

        out_c.write(f"#if defined(_DUMP_INPUTS_) ")
        for node in self.nodes_:
            node_name = node["name"]
            if node["op"] != "null":
                out_c.write(f"|| defined(_DUMP_{node_name_upper}_) ")
        out_c.write(f"\n")
        out_c.write(f"  fclose(DumpFile_p); \n")
        out_c.write(f"#endif \n")

        out_c.write(
            textwrap.dedent(
                f"""
          return AI_STATUS_OK;
        }}
        """
            )
        )

    def __emit_create_destroy(self, name, out_h, out_c):
        """ Emits the create/destroy functions."""

        out_h.write(
            textwrap.dedent(
                f"""
        AI_API_ENTRY
        ai_status ai_{name}_create (
          const ai_ptr weights,
          const ai_ptr activations
        );
        """
            )
        )

        out_h.write(
            textwrap.dedent(
                f"""
        AI_API_ENTRY
        ai_status ai_{name}_destroy ();
        """
            )
        )

        out_c.write(
            textwrap.dedent(
                f"""
        // {DBAR}
        //   ai_{name}_create
        // {DBAR}
        AI_API_ENTRY
        ai_status ai_{name}_create(
          const ai_ptr weights,
          const ai_ptr activations
        )
        {{
          ai_status status = AI_STATUS_OK;
          status = {name}_configure_weights (weights);
          if (status != AI_STATUS_OK) {{
            return status;
          }}
          status = {name}_configure_activations (activations);
          if (status != AI_STATUS_OK) {{
            return status;
          }}
          return AI_STATUS_OK;
        }}
        """
            )
        )

        out_c.write(
            textwrap.dedent(
                f"""
        // {DBAR}
        //   ai_{name}_destroy
        // {DBAR}
        AI_API_ENTRY
        ai_status ai_{name}_destroy ()
        {{
          return AI_STATUS_OK;
        }}
        """
            )
        )

    def emit_code(self, dest_dir, model_name):
        """ Emits the C code implementing the model. """

        #
        # Build the directory structure
        #
        if os.path.exists(dest_dir):
            raise ValueError(f"emit_code.Error: {dest_dir} exists.")

        # Make a new one
        os.makedirs(dest_dir)

        #
        # Fix the model name
        #
        model_name = re.sub("[^0-9a-zA-Z_]+", "_", model_name)
        model_name = model_name.lower()

        #
        # Write the C code: we can parse the string
        #
        if isinstance(self.lib_, list):
            # List of strings from Model Library Format C files
            for idx, src in enumerate(self.lib_):
                code = _preprocess_code(src)
                filename = os.path.join(dest_dir, f"{model_name}_lib{idx}.c")
                with open(filename, "w") as fout:
                    fout.write(code)
        else:
            # a TVM RuntimeGraphFactory
            src = self.lib_.get_source(fmt="c")
            code = _preprocess_code(src)
            filename = os.path.join(dest_dir, f"{model_name}_lib.c")
            with open(filename, "w") as fout:
                fout.write(code)

        #
        # Save params as binary data
        #
        saved_params = tvm.runtime.save_param_dict(self.params_)
        params_name = os.path.join(dest_dir, model_name + ".params")
        with open(params_name, "wb") as f:
            f.write(saved_params)

        #
        # Write the .json
        #
        graph_name = os.path.join(dest_dir, model_name + ".json")
        json_string = json.dumps(self.graph_, indent=4)
        with open(graph_name, "w") as f:
            print(json_string, file=f)

        #
        # emit X_data[c,h]
        #

        data_h_name = os.path.join(dest_dir, model_name + "_data.h")
        data_c_name = os.path.join(dest_dir, model_name + "_data.c")
        model_h_name = os.path.join(dest_dir, model_name + ".h")
        model_c_name = os.path.join(dest_dir, model_name + ".c")

        data_h = open(data_h_name, "w")
        data_c = open(data_c_name, "w")
        out_h = open(model_h_name, "w")
        out_c = open(model_c_name, "w")

        #
        # emit X[c,h]
        #

        self.__emit_params_data(model_name, data_h, data_c)

        self.__emit_open(model_name, out_h, out_c)
        self.__emit_params_buffers(model_name, out_c)
        self.__emit_activation_buffers(model_name, out_c)
        self.__emit_network(model_name, out_c)

        self.__emit_init(model_name, out_c)
        self.__emit_create_destroy(model_name, out_h, out_c)
        self.__emit_run(model_name, out_h, out_c)

        self.__emit_close(model_name, out_h, out_c)

        #
        # Close files
        #
        out_c.close()
        out_h.close()
        data_c.close()
        data_h.close()
