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

import contextlib
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
    dst = "#include <stdio.h>\n" "#include <math.h>\n\n"
    dst = dst + src
    return dst


class CodeEmitter(object):
    """Code emitter class."""

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

        # Static model: activations placed into a nn_data_act section
        # Dynamic model: activations need to be malloc'ed by the
        #   applications.
        self.activations_static = include_activations

        # Inputs/outputs may be allocated within the activations or
        # separately.
        # TODO: Separate the inputs from activations inside TVM.
        if include_inputs:
            assert (
                self.activations_static == True
            ), "###Error: Static inputs are not allowed without activations."
        self.inputs_static = include_inputs

        if include_outputs:
            assert (
                self.activations_static == True
            ), "###Error: Static outputs are not allowed without activations."
        self.outputs_static = include_outputs

        # Parsed graph
        self._nodes = []
        self._arg_nodes = []
        self._outputs = []
        self._attrs = {}
        self._node_row_ptr = []

        # Parameters
        self._params = {}

        # Filled by data_placement()
        self._weights = {}
        self._activations = {}
        self._input_data = {}
        self._output_data = {}
        self._nodes_size = 0
        self._weights_size = 0
        self._activations_size = 0

        self._quantization = {}

    def _extract_quantization_info(self, quantization):
        """ Build dictionary with quantization infos."""

        for dl_tensor_name in self._input_data:
            if dl_tensor_name in quantization:
                self._quantization[dl_tensor_name] = quantization[dl_tensor_name]

        # Matching outputs is more difficult because TVM does not preserve
        # output tensor names.
        # We only support models with a single output now.
        assert len(self._output_data) == 1, "Multiple outputs models are not yet supported."

        for dl_tensor_name in self._output_data:
            for name in quantization:
                if name not in self._input_data:
                    self._quantization["output"] = quantization[name]
                    break

    def _get_node_arg_name(self, arg):
        arg_nid = arg[0]
        arg_idx = arg[1]
        arg_node = self._nodes[arg_nid]
        arg_name = self._nodes[arg_nid]["name"]
        if arg_node["op"] == "null":
            # parameter
            dl_tensor_name = get_input_tensor_name(arg_name)
        elif arg_node["name"] == "reshape_nop":
            # Handle __nop
            src = arg_node["inputs"][0]
            dl_tensor_name = self._get_node_arg_name(src)
        else:
            # activation
            dl_tensor_name = get_output_tensor_name(arg_name, arg_idx)
        return dl_tensor_name

    def _tensor_is_output(self, nid, idx):
        for out in self._outputs:
            out_nid = out[0]
            out_idx = out[1]
            if out_nid == nid and out_idx == idx:
                return True
        return False

    def _get_tensor_from_node(self, nid, idx):
        # 'eid' is index into the dltype', 'shape', etc.
        eid = self._node_row_ptr[nid] + idx
        dltype = self._attrs["dltype"][1][eid]
        dims = self._attrs["shape"][1][eid]
        storage_id = self._attrs["storage_id"][1][eid]
        ndim = len(dims)
        size = _get_tensor_size_bytes(dims, dltype)

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

    def _compute_data_placement(self):
        """ Compute inputs, outputs, weight, activation sizes"""

        self._inputs = self._arg_nodes.copy()

        # weights:
        offset = 0

        for key in self._params:

            # First, find the node in graph
            nid = 0
            for node in self._nodes:
                if node["name"] == key:
                    break
                nid += 1

            dl_tensor_name = get_input_tensor_name(key)
            tensor = self._get_tensor_from_node(nid, 0)

            # Compute the offset
            dltype = tensor["dltype"]
            aligned_offset = _get_aligned_offset(offset, dltype)
            tensor["offset"] = aligned_offset

            for idx in self._arg_nodes:
                node = self._nodes[idx]
                node_name = node["name"]
                if node_name == key:
                    self._inputs.remove(idx)

            self._weights[dl_tensor_name] = tensor

            # Next offset
            offset = aligned_offset + tensor["size"]

        self._weights_size = offset

        # activations:
        buffer_list_ = {}

        nid = 0
        for node in self._nodes:

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

                # Do not count the '_outputs'
                if self._tensor_is_output(nid, idx):
                    continue

                dl_tensor_name = get_output_tensor_name(node_name, idx)
                tensor = self._get_tensor_from_node(nid, idx)

                # Remember this tensor with the storage id
                storage_id = tensor["storage_id"]
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)

                self._activations[dl_tensor_name] = tensor

            self._nodes_size = self._nodes_size + 1

            nid += 1

        # Compute '_input_data'
        offset = 0
        for nid in self._inputs:
            node = self._nodes[nid]
            node_name = node["name"]

            # Arthur: I suppose that input nodes only have a single
            #         output dependency
            dl_tensor_name = get_input_tensor_name(node_name)

            # This tensor is at some index inside '_input_data' dictionary
            # depending on the '_inputs' list order. We refer to this position
            # when generating the XXX.h file.
            tensor = self._get_tensor_from_node(nid, 0)

            if self.inputs_static:

                # Remember this tensor with the storage id
                storage_id = tensor["storage_id"]
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)
            else:

                # Compute the offset
                dltype = tensor["dltype"]
                aligned_offset = _get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset

            self._input_data[dl_tensor_name] = tensor

            # Next offset
            offset = aligned_offset + tensor["size"]

        # Compute '_output_data'
        offset = 0
        for output in self._outputs:
            nid = output[0]
            idx = output[1]

            node = self._nodes[nid]
            node_name = node["name"]

            dl_tensor_name = get_output_tensor_name(node_name, idx)

            tensor = self._get_tensor_from_node(nid, idx)

            if self.outputs_static:

                # Remember this tensor with the storage id
                storage_id = tensor["storage_id"]
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)
            else:

                # Compute the offset
                dltype = tensor["dltype"]
                aligned_offset = _get_aligned_offset(offset, dltype)
                tensor["offset"] = aligned_offset

            self._output_data[dl_tensor_name] = tensor

            # Next offset
            offset = aligned_offset + tensor["size"]

        # Go over all storage IDs and compute offsets and _activations_size
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

        self._activations_size = offset

    def _parse_model(self, quantization=None):
        """Parse the module. Build internal data structures.

        Parameters
        ----------
        module : TVM module or ModuleLibraryFormat object
           The module to parse

        quantization: Dictionary
           The quantization information for model inputs/outputs.
        """

        for key in self._graph:
            if key == "nodes":
                self._nodes = self._graph["nodes"]
            elif key == "arg_nodes":
                self._arg_nodes = self._graph["arg_nodes"]
            elif key == "node_row_ptr":
                self._node_row_ptr = self._graph["node_row_ptr"]
            elif key == "heads":
                self._outputs = self._graph["heads"]
            elif key == "attrs":
                self._attrs = self._graph["attrs"]
            elif key == "metadata":
                continue
            else:
                print("### Error: JSON key {} not supported".format(key))
                assert False

        # Build all tensor lists
        self._compute_data_placement()

        # Extract quantization info for inputs/outputs
        if quantization is not None:
            self._extract_quantization_info(quantization)

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

        # Extract informations from the Model Library Format
        graph_file = os.path.join(extract_path, "executor-config", "graph", "graph.json")
        with open(graph_file, "r") as f:
            # returns JSON object as a dictionary
            graph_dict = json.load(f)

        params_dict = {}
        param_file = os.path.join(extract_path, "parameters", "default.params")
        with open(param_file, "rb") as f:
            params = tvm.runtime.load_param_dict(f.read())

            # Map -> Python Dict
            tmp_dict = {}
            for (k, v) in params.items():
                tmp_dict[k] = v

            # Sort params for debugging
            for k in sorted(tmp_dict.keys()):
                params_dict[k] = tmp_dict[k]

        src_dir = os.path.join(extract_path, "codegen", "host", "src")
        # List of strings from Model Library Format C files
        src_files = []
        for filename in os.listdir(src_dir):
            with open(os.path.join(src_dir, filename), "r") as fin:
                src = fin.read()
                src_files.append(src)

        self._graph = graph_dict
        self._params = params_dict
        self._lib = src_files

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

        # Sort params for debugging
        params_dict = {}
        tmp_params = module.get_params()
        for k in sorted(tmp_params.keys()):
            params_dict[k] = tmp_params[k]

        self._graph = json.loads(graph)
        self._params = params_dict
        self._lib = module.get_lib()

        self._parse_model(quantization)

    def _emit_params_data(self, name, out_h, out_c):
        """ Emits the network_data[c,h] files with parameters."""

        name_upper = name.upper()

        # XXX_data.h

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

        # XXX_data.cc

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

        # Weights are arranged in the order of 'params_'
        offset = 0

        for key in self._params:
            data = self._params[key]  # ND Array
            npdata = data.asnumpy()
            blob = npdata.tobytes()

            out_c.write(f'// "{key}": \n')
            out_c.write(f"\t")

            count = 0

            # Align by emitting garbage between un-aligned data
            dl_tensor_name = get_input_tensor_name(key)
            tensor = self._weights[dl_tensor_name]
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

    def _emit_open(self, name, out_h, out_c):
        """Emits the network.h file with a few network defines and
        writes the header part of the network.c file."""

        name_upper = name.upper()

        input_size = len(self._input_data)
        output_size = len(self._output_data)

        # XXX.h

        out_h.write(
            textwrap.dedent(
                f"""\
        #ifndef __AI_{name_upper}_H__
        #define __AI_{name_upper}_H__
        
        #include \"ai_runtime_api.h\"

        #define _{name_upper}_INPUTS_COUNT_ ({input_size})
        #define _{name_upper}_OUTPUTS_COUNT_ ({output_size})
        #define _{name_upper}_ACTIVATION_BYTES_ ({self._activations_size})
        """
            )
        )

        # XXX.c

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

    def _emit_close(self, name, out_h, out_c):
        """ Emits the ai_model_info structure. """

        name_upper = name.upper()

        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        # XXX.h

        out_h.write(f"#endif /*__AI_{name_upper}_H__*/ \n")

        # XXX.c

        if self.activations_static:
            out_c.write(
                f'AI_ALIGNED({self.DATA_ALIGNMENT_BYTES}) __attribute__ ((section(".{name}.nn_data_act"))) uint8_t {name}_activations[{self._activations_size}];\n'
            )
        else:
            out_c.write(f"AI_STATIC ai_ptr {name}_activations = NULL;")

        # Emit network structure
        num_inputs = len(self._input_data)
        num_outputs = len(self._output_data)

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
          .n_nodes = {self._nodes_size},
          .n_inputs = {num_inputs},
          .n_outputs = {num_outputs},
          .activations_size = {self._activations_size},
          .params_size = {self._weights_size},
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

    def _emit_tensor_shape(self, dl_tensor_name, ndim, shape, strides, out_c):
        out_c.write(f"AI_STATIC int64_t {dl_tensor_name}_shape[{ndim}] = {{{shape[1:-1]}}}; \n")
        assert strides is None, f"###Error: non-compact tensors are not handled yet."
        out_c.write(f"AI_STATIC int64_t {dl_tensor_name}_strides[{ndim}] = {{}}; \n")

    def _emit_tensor_quant(self, dl_tensor_name, out_c):

        if dl_tensor_name in self._quantization:
            quantization = self._quantization[dl_tensor_name]

        # At this time, TVM only supports quantization info with
        # single output models.
        elif dl_tensor_name in self._output_data and "output" in self._quantization.keys():
            quantization = self._quantization["output"]
        else:
            quantization = None

        if quantization is not None:
            scale = quantization["scale"]
            zero_point = quantization["zero_point"]

            # Sometimes we get a scalar with ScaleAsNumpy.
            # This seem to mean not quantized ?
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

    def _emit_tensor_init(self, dl_tensor_name, tensor, out_c):
        """ Emits the tensor instantiation code. """

        dltype = tensor["dltype"]
        dims = tensor["dims"]
        strides = tensor["strides"]
        byte_offset = tensor["byte_offset"]
        dtype = _get_type_data(dltype)
        ndim = len(dims)
        shape = str(dims)
        self._emit_tensor_shape(dl_tensor_name, ndim, shape, strides, out_c)

        # Quantization
        quant_name = self._emit_tensor_quant(dl_tensor_name, out_c)

        # Contents
        #
        # TODO: use the 'storage_id':
        #   "    .ctx = {{ {} }}, \n".format(str(storage_id)[1:-1])
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

        # Figure out quantization, if exists
        if quant_name is not None:
            out_c.write(f"  .quant = &{quant_name} \n")
        else:
            out_c.write(f"  .quant = NULL \n")
        out_c.write(f"}}; \n")

    def _emit_activation_buffers(self, name, out_c):
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

        # shape/buffer
        for dl_tensor_name in self._input_data:
            tensor = self._input_data[dl_tensor_name]
            self._emit_tensor_init(dl_tensor_name, tensor, out_c)
            out_c.write(f"\n")
        out_c.write(f"\n")

        # tensor
        idx = 0
        out_c.write(f"AI_STATIC ai_tensor * _InputsList[] = {{ \n")
        for dl_tensor_name in self._input_data:
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
        for dl_tensor_name in self._activations:
            tensor = self._activations[dl_tensor_name]
            self._emit_tensor_init(dl_tensor_name, tensor, out_c)
        out_c.write(f"\n")

        # Outputs:
        out_c.write(
            textwrap.dedent(
                f"""\
        //
        // Outputs:
        //
        """
            )
        )
        for dl_tensor_name in self._output_data:
            tensor = self._output_data[dl_tensor_name]
            self._emit_tensor_init(dl_tensor_name, tensor, out_c)
            out_c.write(f"\n")
        out_c.write(f"\n")

        idx = 0
        out_c.write(f"AI_STATIC ai_tensor * _OutputsList[] = {{ \n")
        for dl_tensor_name in self._output_data:
            out_c.write(f"  &{dl_tensor_name}, // [{idx}]\n")
            idx = idx + 1
        out_c.write(f"}}; \n")
        out_c.write(f"\n")

    def _emit_params_buffers(self, name, out_c):
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
        for dl_tensor_name in self._weights:
            tensor = self._weights[dl_tensor_name]
            self._emit_tensor_init(dl_tensor_name, tensor, out_c)
        out_c.write(f"\n")

    def _emit_network(self, name, out_c):
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
        for node in self._nodes:
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

    def _emit_tensor_activation(self, dl_tensor_name, tensor, out_c):

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

    def _emit_activation_init(self, name, out_c):
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

        # Allocate inputs with the static model
        if self.inputs_static:
            for dl_tensor_name in self._input_data:
                tensor = self._input_data[dl_tensor_name]
                self._emit_tensor_activation(dl_tensor_name, tensor, out_c)

        # Prepare activation buffers
        for dl_tensor_name in self._activations:
            tensor = self._activations[dl_tensor_name]
            self._emit_tensor_activation(dl_tensor_name, tensor, out_c)

        # Allocate outputs with the static model
        if self.outputs_static:
            for dl_tensor_name in self._output_data:
                tensor = self._output_data[dl_tensor_name]
                self._emit_tensor_activation(dl_tensor_name, tensor, out_c)

        out_c.write(
            textwrap.dedent(
                f"""
          return AI_STATUS_OK;
        }}
        """
            )
        )

    def _emit_params_init(self, name, out_c):
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

        for dl_tensor_name in self._weights:
            tensor = self._weights[dl_tensor_name]
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

    def _emit_init(self, name, out_c):
        """ Emits buffer initialization code."""

        self._emit_activation_init(name, out_c)
        self._emit_params_init(name, out_c)

    def _emit_run(self, name, out_h, out_c):
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

        # Execute nodes one by one
        nid = 0

        for node in self._nodes:
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

            # Prepare TVM packed function - this is the one called
            if name == "__nop":
                print("      exec: __nop")
                continue

            if name == "__copy":
                print("      exec: __copy")
                continue

            # Get function from the TVM module
            #
            #  void * args         : arg_values.data()
            #  void * arg_type_ids : arg_tcodes.data()
            #  int32_t num_args    : arg_values.size()

            dl_args_name = _get_node_args_name(node_name)
            dl_arg_types_name = _get_node_arg_types_name(node_name)

            num_inputs = len(node["inputs"])
            num_outputs = int(node_attrs["num_outputs"])
            num_args = num_inputs + num_outputs

            out_c.write(f"  TVMValue {dl_args_name}[{num_args}]; \n")
            out_c.write(f"  int32_t {dl_arg_types_name}[{num_args}]; \n")

            curr_idx = 0

            for arg in node["inputs"]:
                dl_tensor_name = self._get_node_arg_name(arg)
                #
                # If this input is not an activation or a parameter => find the input
                #
                if dl_tensor_name not in self._weights and dl_tensor_name not in self._activations:

                    assert dl_tensor_name in self._input_data, "Tensor {} not registered ?".format(
                        dl_tensor_name
                    )

                    input_idx = 0
                    for dl_entry_name in self._input_data:
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

                curr_idx += 1

            for idx in range(num_outputs):
                dl_tensor_name = get_output_tensor_name(node_name, idx)

                # If this output is not an activation => find the output
                if dl_tensor_name not in self._activations:

                    assert dl_tensor_name in self._output_data

                    output_idx = 0
                    for dl_exit_name in self._output_data:
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

            # call this function
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
        out_c.write(f"\n")
        out_c.write(
            textwrap.dedent(
                f"""
          return AI_STATUS_OK;
        }}
        """
            )
        )
        out_c.write(f"\n")

    def _emit_create_destroy(self, name, out_h, out_c):
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

        # Build the directory structure
        if os.path.exists(dest_dir):
            raise ValueError(f"emit_code.Error: {dest_dir} exists.")

        # Make a new one
        os.makedirs(dest_dir)

        # Fix the model name
        model_name = re.sub("[^0-9a-zA-Z_]+", "_", model_name)
        model_name = model_name.lower()

        # Write the C code: we can parse the string
        if isinstance(self._lib, list):
            # List of strings from Model Library Format C files
            for idx, src in enumerate(self._lib):
                code = _preprocess_code(src)
                filename = os.path.join(dest_dir, f"{model_name}_lib{idx}.c")
                with open(filename, "w") as fout:
                    fout.write(code)
        else:
            # a TVM RuntimeGraphFactory
            src = self._lib.get_source(fmt="c")
            code = _preprocess_code(src)
            filename = os.path.join(dest_dir, f"{model_name}_lib.c")
            with open(filename, "w") as fout:
                fout.write(code)

        # Save params as binary data
        saved_params = tvm.runtime.save_param_dict(self._params)
        params_name = os.path.join(dest_dir, model_name + ".params")
        with open(params_name, "wb") as f:
            f.write(saved_params)

        # Write the .json
        graph_name = os.path.join(dest_dir, model_name + ".json")
        json_string = json.dumps(self._graph, indent=4)
        with open(graph_name, "w") as f:
            print(json_string, file=f)

        # emit X_data[c,h]
        data_h_name = os.path.join(dest_dir, model_name + "_data.h")
        data_c_name = os.path.join(dest_dir, model_name + "_data.c")
        model_h_name = os.path.join(dest_dir, model_name + ".h")
        model_c_name = os.path.join(dest_dir, model_name + ".c")

        with contextlib.ExitStack() as exit_stack:

            # emit X[c,h]

            data_h = exit_stack.enter_context(open(data_h_name, "w"))
            data_c = exit_stack.enter_context(open(data_c_name, "w"))
            out_h = exit_stack.enter_context(open(model_h_name, "w"))
            out_c = exit_stack.enter_context(open(model_c_name, "w"))

            self._emit_params_data(model_name, data_h, data_c)

            self._emit_open(model_name, out_h, out_c)
            self._emit_params_buffers(model_name, out_c)
            self._emit_activation_buffers(model_name, out_c)
            self._emit_network(model_name, out_c)

            self._emit_init(model_name, out_c)
            self._emit_create_destroy(model_name, out_h, out_c)
            self._emit_run(model_name, out_h, out_c)

            self._emit_close(model_name, out_h, out_c)
