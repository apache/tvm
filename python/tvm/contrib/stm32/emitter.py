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

"""Code emission for the STM32 targets."""

import os
import sys

import numpy as np

import json            # reading json graph

from string import Template

import tvm.relay as relay

import logging

from datetime import datetime

AI_TOOLS_VERSION_MAJOR = 1
AI_TOOLS_VERSION_MINOR = 8
AI_TOOLS_VERSION_MICRO = 0

AI_API_VERSION_MAJOR = 1
AI_API_VERSION_MINOR = 0
AI_API_VERSION_MICRO = 0

AI_TOOLS_REVISION = "v1"

DBar = '============================================================'

# ==========================================================
#   _fix_name
#
#   Replace ':' with '_' in names like 'InputImg:0'
# ==========================================================
def _fix_name (node_name):
    return node_name.replace(':','_')

# ==========================================================
#   get_input_tensor_name
# ==========================================================
def get_input_tensor_name (node_name):
    return _fix_name (node_name)

# ==========================================================
#   get_output_tensor_name
# ==========================================================
def get_output_tensor_name (node_name, idx):
    return _fix_name (node_name) + "_" + str(idx)

# ==========================================================
#   _get_node_args_name
# ==========================================================
def _get_node_args_name (node_name):
    return _fix_name (node_name) + "_args"

# ==========================================================
#   _get_node_arg_types_name
# ==========================================================
def _get_node_arg_types_name (node_name):
    return _fix_name (node_name) + "_arg_type_ids"

# ==========================================================
#   _get_type_size
# ==========================================================
def _get_type_size (dltype):
    
    if dltype == 'uint64' or dltype == 'int64' :
        return 8
    elif dltype == 'uint32' or dltype == 'int32' or dltype == 'float32':
        return 4
    elif dltype == 'uint16' or dltype == 'int16':
        return 2
    elif dltype == 'uint8' or dltype == 'int8':
        return 1
    else:
        print ("### Error: Data type {} not supported".format(dltype))
        assert(False)

# ==========================================================
#   _get_type_data
# ==========================================================
def _get_type_data (dltype):

    if dltype == 'uint64':
        return 'kDLUInt, 64, 1'
    elif dltype == 'int64':
        return 'kDLInt, 64, 1'
    elif dltype == 'float32':
        return 'kDLFloat, 32, 1'
    elif dltype == 'uint32':
        return 'kDLUInt, 32, 1'
    elif dltype == 'int32':
        return 'kDLInt, 32, 1'
    elif dltype == 'uint16':
        return 'kDLUInt, 16, 1'
    elif dltype == 'int16':
        return 'kDLInt, 16, 1'
    elif dltype == 'uint8':
        return 'kDLUInt, 8, 1'
    elif dltype == 'int8':
        return 'kDLInt, 8, 1'
    else:
        print ("### Error: Data type {} not supported".format(dltype))
        assert(False)
        
# ==========================================================
#   _get_aligned_offset
# ==========================================================
def _get_aligned_offset (offset, dltype):

    align = _get_type_size (dltype)

    if offset % align != 0:
        offset = offset + (align - offset % align)

    return offset

# ==========================================================
#   _get_tensor_elts
# ==========================================================
def _get_tensor_elts (dims):
    size = 1
    for i in range (0, len(dims)):
        size = size * dims[i]
    return size

# ==========================================================
#   _get_tensor_size
# ==========================================================
def _get_tensor_size (dims, dltype):
    size = _get_tensor_elts (dims)
    return size * _get_type_size(dltype)

# ==========================================================
#                       CodeEmitter
# ==========================================================
class CodeEmitter (object) :
    """Code emitter class/utility."""

    def __init__(self, include_activations=True, include_inputs=True, include_outputs=True):
        """Initialize the Emitter instance.

        Parameters
        ----------
        include_activations:
            The Emitter allocates the storage for the activations data
            and places it in a specific data section. If Falsr, the
            main application is responsible for allocating the activations
            storage. Default: True.

        include_feeds:
            The Emitter allocates the storage for the input/output data.
            This storage is shared with the activations and placed in the
            specific activations data section. If Falsr, the main 
            application is responsible for allocating the input/output
            data storage. Default: True.

        Returns
        -------
            CodeEmitter object.

        """

        #
        # Constants:
        #
        self.DATA_ALIGNMENT = 8

        #
        # Static model: activations placed into a nn_data_act section
        # Dynamic model: activations need to be malloc'ed by the 
        #   applications.
        #
        if include_activations == True:
            self.ACTIVATIONS_STATIC = 1
        else:
            self.ACTIVATIONS_STATIC = 0

        #
        # Inputs/outputs may be allocated within the activations or
        # separately.
        # TODO: Separate the inputs from activations inside TVM.
        #
        if include_inputs == True:
            assert self.ACTIVATIONS_STATIC == 1, ("###Error: Static inputs are not allowed without activations.")
            self.INPUTS_STATIC = 1
        else:
            self.INPUTS_STATIC = 0

        if include_outputs == True:
            assert self.ACTIVATIONS_STATIC == 1, ("###Error: Static outputs are not allowed without activations.")
            self.OUTPUTS_STATIC = 1
        else:
            self.OUTPUTS_STATIC = 0

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


    # ==================================================================
    #   __extract_quantization_info
    # ==================================================================
    def __extract_quantization_info (self, quantization):
        """ Build dictionary with quantization infos."""

        for dl_tensor_name in self.input_data_:
            if dl_tensor_name in quantization:
                self.quantization_[dl_tensor_name] = quantization[dl_tensor_name]

        #
        # Matching outputs is more difficult because TVM does not preserve
        # output tensor names.
        # We only support models with a single output now.
        #
        assert len(self.output_data_) == 1, ("Multiple outputs models are not yet supported.")
        
        for dl_tensor_name in self.output_data_:
            for name in quantization:
                if name not in self.input_data_:
                    self.quantization_['output'] = quantization[name]
                    break

    # ==================================================================
    #   _json_load_attrs
    # ==================================================================
    #def __json_load_attrs (self, attr_dict):

    #    param = None
    
    #    for key in attr_dict:
            
    #        if key == 'func_name':
                #param->func_name = value;
    #            print ("json_load_attrs: TODO.")
    #        elif key == 'num_inputs':
                #param->num_inputs = strtoul(value.c_str(), nullptr, 10);
    #            print ("json_load_attrs: TODO.")
    #        elif key == 'num_outputs':
                #param->num_outputs = strtoul(value.c_str(), nullptr, 10);
    #            print ("json_load_attrs: TODO.")
    #        elif key == 'flatten_data':
                #param->flatten_data = strtoul(value.c_str(), nullptr, 10);
    #            print ("json_load_attrs: TODO.")

    #    return param

    # ==========================================================
    #   _json_parse_nodes
    # ==========================================================
    def __json_parse_nodes (self, nodes):
        """ Parse the nodes part of the JSON graph. """
    
        for node_dict in nodes:

            #print ("    node: {}".format(node_dict))

            op = None
            name = None
            inputs = None
            attrs = None
            deps = None
        
            for key in node_dict:

                if key == 'op':
                    op = node_dict['op']
                    #print ("    op: {}".format(node_dict['op']))
                elif key == 'name':
                    name = node_dict['name']
                    #print ("    name: {}".format(node_dict['name']))
                elif key == 'inputs':
                    inputs = node_dict['inputs']
                    #print ("    inputs: {}".format(node_dict['inputs']))
                elif key == 'attr':
                    attrs = node_dict['attr']
                elif key == 'attrs':
                    attrs = node_dict['attrs']
                elif key == 'control_deps':
                    deps = node_dict['control_deps']
                else:
                    print ("### Error: JSON do not support key {}".format(key))
                    assert(False)

    # ==========================================================
    #   __tensor_is_output
    # ==========================================================
    def __tensor_is_output (self, nid, idx):
    
        for out in self.outputs_:
            out_nid = out[0]
            out_idx = out[1]
            out_version = out[2]
            if out_nid == nid and out_idx == idx:
                return True
        return False

    # ==========================================================
    #   __get_tensor_from_node
    # ==========================================================
    def __get_tensor_from_node (self, nid, idx):
        #
        # 'eid' is index into the dltype', 'shape', etc.
        #
        eid = self.node_row_ptr_[nid] + idx
                
        dltype = self.attrs_['dltype'][1][eid]
        dims = self.attrs_['shape'][1][eid]
        storage_id = self.attrs_['storage_id'][1][eid]
        ndim = len(dims)
        shape = str(dims)
        #
        # Get tensor size
        #
        size = _get_tensor_size (dims, dltype)

        tensor = {
            'dltype' : dltype,
            'ndim' : ndim,
            'dims' : dims,
            'strides' : None,
            'storage_id' : storage_id,
            #
            # What is this byte_offset really ?
            #
            'byte_offset' : 0,   
            'offset' : 0,
            'size' : size
            }

        return tensor
    

    # ==================================================================
    #   __compute_data_placement
    # ==================================================================
    def __compute_data_placement (self):
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
                if node['name'] == key:
                    break
                nid += 1
        
            dl_tensor_name = get_input_tensor_name (key);
            tensor = self.__get_tensor_from_node (nid, 0)
            #
            # Compute the offset
            #
            dltype = tensor['dltype']
            aligned_offset = _get_aligned_offset (offset, dltype)
            tensor['offset'] = aligned_offset
        
            for idx in self.arg_nodes_:
                node = self.nodes_[idx]
                node_name = node['name']
                if node_name == key:
                    self.inputs_.remove(idx)
            
            self.weights_[dl_tensor_name] = tensor

            #
            # Next offset
            #
            offset = aligned_offset + tensor['size']

        self.weights_size_ = offset

        #
        # activations:
        #

        buffer_list_ = {}
    
        nid = 0
    
        for node in self.nodes_:

            node_name = node['name']
        
            #print ("== [{}] op: {}".format(nid, node['name']))
        
            if node['op'] != 'null':

                if node['op'] != 'tvm_op':
                    print ("### Error: Only TVM ops are supported")
                    exit

                node_attrs = node['attrs']
                num_outputs = int(node_attrs['num_outputs'])
             
                for idx in range(num_outputs):
                    #
                    # Do not count the 'outputs_'
                    #
                    if self.__tensor_is_output(nid, idx):
                        continue

                    dl_tensor_name = get_output_tensor_name (node_name, idx);
                    tensor = self.__get_tensor_from_node (nid, idx)
                    #
                    # Remember this tensor with the storage id
                    #
                    storage_id = tensor['storage_id']
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
            node_name = node['name']
            #
            # Arthur: I suppose that input nodes only have a single
            #         output dependency
            #
            dl_tensor_name = get_input_tensor_name (node_name);
            #
            # This tensor is at some index inside 'input_data_' dictionary 
            # depending on the 'inputs_' list order. We refer to this position 
            # when generating the XXX.h file.
            #
            tensor = self.__get_tensor_from_node (nid, 0)

            if self.INPUTS_STATIC == 1:
                #
                # Remember this tensor with the storage id
                #
                storage_id = tensor['storage_id']
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)
            else:
                #
                # Compute the offset
                #
                dltype = tensor['dltype']
                aligned_offset = _get_aligned_offset (offset, dltype)
                tensor['offset'] = aligned_offset

            self.input_data_[dl_tensor_name] = tensor

            #
            # Next offset
            #
            offset = aligned_offset + tensor['size']


        #
        # Compute 'output_data_'
        #
        offset = 0
    
        for output in self.outputs_:

            nid = output[0]
            idx = output[1]
            version = output[2]
        
            node = self.nodes_[nid]
            node_name = node['name']

            dl_tensor_name = get_output_tensor_name (node_name, idx);

            tensor = self.__get_tensor_from_node (nid, idx)

            if self.OUTPUTS_STATIC == 1:
                #
                # Remember this tensor with the storage id
                #
                storage_id = tensor['storage_id']
                if storage_id not in buffer_list_:
                    buffer_list_[storage_id] = []
                buffer_entry = buffer_list_[storage_id]
                buffer_entry.append(tensor)
            else:
                #
                # Compute the offset
                #
                dltype = tensor['dltype']
                aligned_offset = _get_aligned_offset (offset, dltype)
                tensor['offset'] = aligned_offset

            self.output_data_[dl_tensor_name] = tensor
        
            #
            # Next offset
            #
            offset = aligned_offset + tensor['size']

        #
        # Go over all storage IDs and compute offsets and activations_size_
        #
        offset = 0
    
        for storage_id in buffer_list_:
            buffer_entry = buffer_list_[storage_id]

            new_offset = offset
            for tensor in buffer_entry:
                assert (tensor['storage_id'] == storage_id)
                dltype = tensor['dltype']
                aligned_offset = _get_aligned_offset (offset, dltype)
                tensor['offset'] = aligned_offset
                size = tensor['size']
                if (aligned_offset + size) > new_offset:
                    new_offset = aligned_offset + size
            offset = new_offset
        
        self.activations_size_ = offset


    # ==========================================================
    #   parse_module
    # ==========================================================
    def parse_module (self, module, quantization=None):
        """Parse the module. Build internal data structures.

        Parameters
        ----------
        module : TVM module
           The module to parse

        quantization: Dictionary
           The quantization information for model inputs/outputs.
        """

        self.graph_ = module.get_json()
        self.params_ = module.get_params()
        self.lib_ = module.get_lib()

        if not isinstance(self.graph_, (str,)):
            try:
                self.graph = self.graph._tvm_graph_json()
            except AttributeError:
                raise ValueError("Type %s is not supported" % type(self.graph))

        graph_dict = json.loads(self.graph_)
        
        for key in graph_dict:
            if key == 'nodes':
                self.nodes_ = graph_dict['nodes']
                #print (" -- nodes: {}".format(graph_dict['nodes']))
                self.__json_parse_nodes (self.nodes_)
            elif key == 'arg_nodes':
                self.arg_nodes_ = graph_dict['arg_nodes']
                #print (" -- inputs: {}".format(arg_nodes_))
            elif key == 'node_row_ptr':
                self.node_row_ptr_ = graph_dict['node_row_ptr']
                #print (" -- node_row_ptr: {}".format(graph_dict['node_row_ptr']))
            elif key == 'heads':
                self.outputs_ = graph_dict['heads']
                #print (" -- outputs: {}".format(outputs_))
            elif key == 'attrs':
                self.attrs_ = graph_dict['attrs']
                #print (" -- attrs: {}".format(attrs_))
            elif key == 'metadata':
                #print (" -- meta: {}".format(graph_dict['metadata']))
                continue
            else:
                print ("### Error: JSON key {} not supported".format(key))
                assert(False)

        #
        # Build all tensor lists
        #
        self.__compute_data_placement()

        #
        # Extract quantization info for inputs/outputs
        #
        if quantization != None:
            self.__extract_quantization_info (quantization)

    # ==========================================================
    #   __emit_params_data
    # ==========================================================
    def __emit_params_data (self, name, out_h, out_c):

        name_upper = name.upper()

        #
        # XXX_data.h
        #
        out_h.write (f'#ifndef __{name_upper}_DATA_H_ \n')
        out_h.write (f'#define __{name_upper}_DATA_H_ \n')
        out_h.write (f'\n')
        out_h.write (f'#include \"ai_runtime_api.h\" \n')
        out_h.write (f'\n')

        out_h.write (f'AI_API_ENTRY \n')
        out_h.write (f'const ai_ptr ai_{name}_data_weights_get (void); \n')
        out_h.write (f'\n')

        out_h.write (f'#endif /* __{name_upper}_DATA_H_ */ \n')
        out_h.write (f'\n')
    
        #
        # XXX_data.cc
        #
        out_c.write (f'#include \"{name}_data.h\" \n')
        out_c.write (f'\n')

        out_c.write (f'const ai_ptr ai_{name}_data_weights_get (void) \n')
        out_c.write (f'{{\n')
        out_c.write (f'  AI_ALIGNED({self.DATA_ALIGNMENT}) static const __attribute__ ((section(\".nn_weights\"))) uint8_t s_{name}_weights[] = {{ \n')

        #
        # Weights are arranged in the order of 'params_'
        #

        offset = 0
    
        for key in self.params_:
            data = self.params_[key]    # ND Array
            npdata = data.asnumpy()
            blob = npdata.tobytes()
        
            out_c.write (f'// \"{key}\": \n')
            out_c.write (f'\t')
        
            count = 0

            #
            # Align by emitting garbage between un-aligned data
            #
            dl_tensor_name = get_input_tensor_name (key);
            tensor = self.weights_[dl_tensor_name]
            tensor_offset = tensor['offset']
            tensor_size = tensor['size']

            while offset < tensor_offset:
                count += 1
                out_c.write ("0x{:02X}, ".format(0))
                if count == 12:
                    out_c.write ("\n\t")
                    count = 0
                offset += 1
        
            for val in blob:
                count += 1
                out_c.write ("0x{:02X}, ".format(val))
                if count == 12:
                    out_c.write ("\n\t")
                    count = 0

            offset += tensor_size
                
            out_c.write (f'\n')

        out_c.write (f'  }}; \n')
        out_c.write (f'  return (const ai_ptr)s_{name}_weights; \n')
        out_c.write (f'}} \n')
        out_c.write (f'\n')
    
    # ==========================================================
    #   __emit_open
    # ==========================================================
    def __emit_open (self, name, out_h, out_c):

        name_upper = name.upper()

        #
        # XXX.h
        #
        out_h.write (f'#ifndef __AI_{name_upper}_H__ \n')
        out_h.write (f'#define __AI_{name_upper}_H__ \n')
        out_h.write (f'\n')

        out_h.write (f'#include \"ai_runtime_api.h\" \n')
        out_h.write (f'\n')

        input_size = len(self.input_data_)
        output_size = len(self.output_data_)
        out_h.write (f'#define _{name_upper}_INPUTS_COUNT_ ({input_size})\n')
        out_h.write (f'#define _{name_upper}_OUTPUTS_COUNT_ ({output_size})\n')
        out_h.write (f'#define _{name_upper}_ACTIVATION_BYTES_ ({self.activations_size_})\n')
        out_h.write (f'\n')

        #
        # XXX.c
        #
        out_c.write (f'#include <stdio.h> \n')
        out_c.write (f'\n')
    
        out_c.write (f'#include \"dlpack/dlpack.h\" \n')
        out_c.write (f'#include \"c_runtime_api.h\" \n')
        out_c.write (f'#include \"{name}.h\" \n')
        out_c.write (f'#include \"{name}_data.h\" \n')
        out_c.write (f'\n')

    # ==========================================================
    #   __emit_close
    # ==========================================================
    def __emit_close (self, name, out_h, out_c):
    
        name_upper = name.upper()

        # datetime object containing current date and time
        now = datetime.now()
        # dd/mm/YY H:M:S
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        #print("date and time =", dt_string)

        #
        # XXX.h
        #
        out_h.write (f'#endif /*__AI_{name_upper}_H__*/ \n')

        #
        # XXX.c
        #

        #
        # Emit network structure
        #
        num_inputs = len(self.input_data_)
        num_outputs = len(self.output_data_)

        out_c.write (f'AI_API_ENTRY  __attribute__ ((section(\".nn_models\"))) ai_model_info {name}_network = {{\n')
        out_c.write (f'  .name = \"{name}\",\n')
        out_c.write (f'  .datetime = \"{dt_string}\",\n')
        out_c.write (f'  .revision = \"{AI_TOOLS_REVISION}\",\n')
        out_c.write (f'  .tool_version = \"{AI_TOOLS_VERSION_MAJOR}.{AI_TOOLS_VERSION_MINOR}.{AI_TOOLS_VERSION_MICRO}.0\",\n')
        out_c.write (f'  .api_version = \"{AI_API_VERSION_MAJOR}.{AI_API_VERSION_MINOR}.{AI_API_VERSION_MICRO}.0\",\n')
        #out_c.write ("  .n_macc = {},\n".format(0))
        out_c.write (f'  .n_nodes = {self.nodes_size_},\n')
        out_c.write (f'  .n_inputs = {num_inputs},\n')
        out_c.write (f'  .n_outputs = {num_outputs},\n')
        out_c.write (f'  .activations_size = {self.activations_size_},\n')
        out_c.write (f'  .params_size = {self.weights_size_},\n')
        if self.ACTIVATIONS_STATIC == 1:
            out_c.write (f'  .activations = {name}_activations, \n')
        else:
            out_c.write (f'  .activations = NULL, \n')
        out_c.write (f'  .inputs = _InputsList,\n')
        out_c.write (f'  .outputs = _OutputsList,\n')
        out_c.write (f'  .ai_get_params = &ai_{name}_data_weights_get,\n')
        out_c.write (f'  .ai_create = &ai_{name}_create,\n')
        out_c.write (f'  .ai_destroy = &ai_{name}_destroy,\n')
        out_c.write (f'  .ai_run = &ai_{name}_run\n')
        out_c.write (f'}};\n')
        out_c.write (f'\n')

    # ==========================================================
    #   __emit_tensor_shape
    # ==========================================================
    def __emit_tensor_shape (self, dl_tensor_name, ndim, shape, strides, out_c):
        out_c.write (f'AI_STATIC int64_t {dl_tensor_name}_shape[{ndim}] = {{{shape[1:-1]}}}; \n')
        assert strides == None, (f'###Error: non-compact tensors are not handled yet.')
        out_c.write (f'AI_STATIC int64_t {dl_tensor_name}_strides[{ndim}] = {{}}; \n')

    # ==========================================================
    #   __emit_tensor_quant
    # ==========================================================
    def __emit_tensor_quant (self, dl_tensor_name, out_c):

        if dl_tensor_name in self.quantization_:
            quantization = self.quantization_[dl_tensor_name]
        #
        # At this time, TVM only supports quantization info with
        # single output models.
        #
        elif dl_tensor_name in self.output_data_ and 'output' in self.quantization_.keys():
            quantization = self.quantization_['output']
        else:
            quantization = None

        if quantization != None:
            min = quantization['min']
            max = quantization['max']
            scale = quantization['scale']
            zero_point = quantization['zero_point']
            dim = quantization['dim']

            #
            # Sometimes we get a scalar with ScaleAsNumpy.
            # This seem to mean not quantized ?
            #
            if not isinstance(scale, np.ndarray):
                assert scale == 0.0, (f'Non-quantized tensor with scale != 0.0')
                assert not isinstance(zero_point, np.ndarray) and zero_point == 0,(f'Non-quantized tensor with zero_point != 0')
                return None
                
            scale_size = len(scale)
            zero_point_size = len(zero_point)

            assert len(scale) == len(zero_point), (f'Inconsistent quantizations scale:{scale} vs zero-point:{zero_point}')

            if len(scale) == 1:
                #print ("== Setting quantization for {}: {}".format(dl_tensor_name, quantization))
                quant_name = dl_tensor_name+"_quant"

                out_c.write (f'AI_STATIC float {quant_name}_scale[{scale_size}] = {{ ')
                for val in scale:
                    out_c.write (f'{val}, ')
                out_c.write (f'}};\n')
                out_c.write (f'AI_STATIC int32_t {quant_name}_zero_point[{zero_point_size}] = {{ ')
                for val in zero_point:
                    out_c.write (f'{val}, ')
                out_c.write (f'}};\n')
                out_c.write (f'AI_STATIC ai_quantization_info {quant_name} = {{\n')
                out_c.write (f'  .scale = {quant_name}_scale,\n')
                out_c.write (f'  .zero_point = {quant_name}_zero_point,\n')
                out_c.write (f'  .dim = -1\n')
                out_c.write (f'}}; \n')

                return quant_name

        return None

    # ==========================================================
    #   __emit_tensor_init
    # ==========================================================
    def __emit_tensor_init (self, dl_tensor_name, tensor, out_c, out_h):

        dltype = tensor['dltype']
        dims = tensor['dims']
        strides = tensor['strides']
        storage_id = tensor['storage_id']
        byte_offset = tensor['byte_offset']
        size = tensor['size']
        dtype = _get_type_data (dltype)
        ndim = len(dims)
        shape = str(dims)
        self.__emit_tensor_shape (dl_tensor_name, ndim, shape, strides, out_c)

        #
        # Quantization
        #
        quant_name = self.__emit_tensor_quant (dl_tensor_name, out_c)

        #
        # Contents
        #
        out_c.write (f'AI_ALIGNED({self.DATA_ALIGNMENT}) AI_STATIC ai_tensor {dl_tensor_name} = {{ \n')
        out_c.write (f'  .dltensor = {{ \n')
        out_c.write (f'  .data = (ai_ptr)(NULL), \n')
        #
        # TODO: use the 'storage_id'
        #
        #out_c.write ("    .ctx = {{ {} }}, \n".format(str(storage_id)[1:-1]))
        out_c.write (f'  .ctx = {{ kDLCPU, 0 }}, \n')
        out_c.write (f'  .ndim = {ndim}, \n')
        out_c.write (f'  .dtype = {{ {dtype} }}, \n')
        out_c.write (f'  .shape = {dl_tensor_name}_shape, \n')
        if strides != None:
            out_c.write (f'  .strides = {dl_tensor_name}_strides, \n')
        else:
            out_c.write (f'  .strides = NULL, \n')
        out_c.write (f'  .byte_offset = {byte_offset} \n')
        out_c.write (f'  }}, \n')
        #
        # Figure out quantization, if exists
        #
        if quant_name != None:
            out_c.write (f'  .quant = &{quant_name} \n')
        else:
            out_c.write (f'  .quant = NULL \n')
        out_c.write (f'}}; \n')

    # ==========================================================
    #   __emit_activation_buffers
    # ==========================================================
    def __emit_activation_buffers (self, name, out_h, out_c):
        #
        # Inputs:
        #
        out_c.write (f'// \n')
        out_c.write (f'// Inputs:\n')
        out_c.write (f'// \n')
        #
        # shape/buffer
        #
        for dl_tensor_name in self.input_data_:
            tensor = self.input_data_[dl_tensor_name]
            self.__emit_tensor_init (dl_tensor_name, tensor, out_c, out_h)
            out_c.write (f'\n')
        out_c.write (f'\n')

        #
        # tensor
        #
        idx = 0
        out_c.write (f'AI_STATIC ai_tensor * _InputsList[] = {{ \n')
        for dl_tensor_name in self.input_data_:
            out_c.write (f'  &{dl_tensor_name}, // [{idx}]\n')
            idx = idx+1
        out_c.write (f'}}; \n')
        out_c.write (f'\n')
        
        #
        # Activations:
        #
        out_c.write (f'// \n')
        out_c.write (f'// Activations: \n')
        out_c.write (f'// \n')
        for dl_tensor_name in self.activations_:
            tensor = self.activations_[dl_tensor_name]
            self.__emit_tensor_init (dl_tensor_name, tensor, out_c, out_h)
        out_c.write (f'\n')

        #
        # Outputs:
        #
        out_c.write (f'// \n')
        out_c.write (f'// Outputs:\n')
        out_c.write (f'// \n')
        for dl_tensor_name in self.output_data_:
            tensor = self.output_data_[dl_tensor_name]
            self.__emit_tensor_init (dl_tensor_name, tensor, out_c, out_h)
            out_c.write (f'\n')
        out_c.write (f'\n')

        idx = 0
        out_c.write (f'AI_STATIC ai_tensor * _OutputsList[] = {{ \n')
        for dl_tensor_name in self.output_data_:
            out_c.write (f'  &{dl_tensor_name}, // [{idx}]\n')
            idx = idx+1
        out_c.write (f'}}; \n')
        out_c.write (f'\n')

        if self.ACTIVATIONS_STATIC == 1:
            out_c.write (f'AI_ALIGNED({self.DATA_ALIGNMENT}) __attribute__ ((section(\".{name}.nn_data_act\"))) uint8_t {name}_activations[{self.activations_size_}];\n')
            out_c.write (f'\n')

    # ==========================================================
    #   __emit_params_buffers
    # ==========================================================
    def __emit_params_buffers (self, name, out_h, out_c):

        out_c.write (f'// \n')
        out_c.write (f'// Weights: \n')
        out_c.write (f'// \n')
        for dl_tensor_name in self.weights_:
            tensor = self.weights_[dl_tensor_name]
            self.__emit_tensor_init (dl_tensor_name, tensor, out_c, out_h)
        out_c.write (f'\n')

    # ==========================================================
    #   __emit_network
    # ==========================================================
    def __emit_network (self, name, out_h, out_c):

        out_c.write (f'// \n')
        out_c.write (f'// Network: \n')
        out_c.write (f'// \n')
        #
        # Emit prototypes for the TVM library functions
        #
        for node in self.nodes_:
            if node['op'] != 'null':
                assert node['op'] == 'tvm_op', (f'###Error: Only TVM ops are supported.')
                node_attrs = node['attrs']
                func_name = node_attrs['func_name']
                out_c.write (f'TVM_DLL int32_t {func_name}(void * args, void * arg_type_ids, int32_t num_args); \n')
        out_c.write (f'\n')
    
    # ==========================================================
    #   __emit_tensor_activation
    # ==========================================================
    def __emit_tensor_activation (self, name, dl_tensor_name, tensor, out_c):

        storage_id = tensor['storage_id']
        offset = tensor['offset']
        out_c.write (f'  // \n')
        out_c.write (f'  // {dl_tensor_name}: storage_id:{storage_id}\n')
        out_c.write (f'  // \n')
        out_c.write (f'  {dl_tensor_name}.dltensor.data = (ai_ptr)(activations + {offset}); \n')
        out_c.write (f'\n')

    # ==========================================================
    #   __emit_activation_init
    # ==========================================================
    def __emit_activation_init (self, name, out_h, out_c):

        out_c.write (f'// {DBar} \n')
        out_c.write (f'//   {name}_configure_activations \n')
        out_c.write (f'// {DBar} \n')
        out_c.write (f'AI_STATIC AI_INLINE\n')
        out_c.write (f'ai_status {name}_configure_activations ( \n')
        out_c.write (f'  const ai_ptr activations \n')
        out_c.write (f') \n')
        out_c.write (f'{{ \n')
        out_c.write (f'  if (activations == NULL) {{\n')
        out_c.write (f'    TVMAPISetLastError (\"Non-null activations arena is required for this model.\");\n')
        out_c.write (f'    return AI_STATUS_ERROR;\n')
        out_c.write (f'  }}\n')
        out_c.write (f'\n')

        #
        # Allocate inputs with the static model
        #
        if self.INPUTS_STATIC:
            for dl_tensor_name in self.input_data_:
                tensor = self.input_data_[dl_tensor_name]
                self.__emit_tensor_activation (name, dl_tensor_name, tensor, out_c)

        #
        # Prepare activation buffers
        #
        for dl_tensor_name in self.activations_:
            tensor = self.activations_[dl_tensor_name]
            self.__emit_tensor_activation (name, dl_tensor_name, tensor, out_c)

        #
        # Allocate outputs with the static model
        #
        if self.OUTPUTS_STATIC:
            for dl_tensor_name in self.output_data_:
                tensor = self.output_data_[dl_tensor_name]
                self.__emit_tensor_activation (name, dl_tensor_name, tensor, out_c)

        out_c.write (f'  return AI_STATUS_OK; \n')
        out_c.write (f'}} \n')
        out_c.write (f'\n')

    # ==========================================================
    #   __emit_params_init
    # ==========================================================
    def __emit_params_init (self, name, out_h, out_c):

        out_c.write (f'// {DBar} \n')
        out_c.write (f'//   {name}_configure_weights \n')
        out_c.write (f'// {DBar} \n')
        out_c.write (f'AI_STATIC AI_INLINE\n')
        out_c.write (f'ai_status {name}_configure_weights ( \n')
        out_c.write (f'  const ai_ptr weights \n')
        out_c.write (f') \n')
        out_c.write (f'{{ \n')
        out_c.write (f'  if (weights == NULL) {{\n')
        out_c.write (f'    TVMAPISetLastError (\"Non-null weights arena is required for this model.\");\n')
        out_c.write (f'    return AI_STATUS_ERROR;\n')
        out_c.write (f'  }}\n')
        out_c.write (f'\n')

        for dl_tensor_name in self.weights_:
            tensor = self.weights_[dl_tensor_name]
            offset = tensor['offset']
            out_c.write (f'  // \n')
            out_c.write (f'  // {dl_tensor_name}\n')
            out_c.write (f'  // \n')
            out_c.write (f'  {dl_tensor_name}.dltensor.data = (ai_ptr)(weights + {offset}); \n')
            out_c.write (f'\n')

        out_c.write (f'  return AI_STATUS_OK; \n')
        out_c.write (f'}} \n')
        out_c.write (f'\n')

    # ==========================================================
    #   __emit_init
    # ==========================================================
    def __emit_init (self, name, out_h, out_c):
        #
        # {name}_configure_activations
        #
        self.__emit_activation_init (name, out_h, out_c)
        #
        # {name}_configure_weights
        #
        self.__emit_params_init (name, out_h, out_c)

    # ==========================================================
    #   __emit_run
    # ==========================================================
    def __emit_run (self, name, out_h, out_c):

        out_h.write (f'AI_API_ENTRY \n')
        out_h.write (f'ai_status ai_{name}_run ( \n')
        out_h.write (f'  ai_tensor *inputs[], \n')
        out_h.write (f'  ai_tensor *outputs[] \n')
        out_h.write (f'); \n')
        out_h.write (f'\n')
    
        out_c.write (f'// {DBar} \n')
        out_c.write (f'//   ai_{name}_run \n')
        out_c.write (f'// {DBar} \n')
        out_c.write (f'AI_API_ENTRY \n')
        out_c.write (f'ai_status ai_{name}_run ( \n')
        out_c.write (f'  ai_tensor *inputs[], \n')
        out_c.write (f'  ai_tensor *outputs[] \n')
        out_c.write (f') \n')
        out_c.write (f'{{ \n')

        out_c.write (f'#if defined(_DUMP_INPUTS_) ')
        for node in self.nodes_:
            node_name = node['name']
            node_name_upper = node_name.upper()
            if node['op'] != 'null':
                out_c.write (f'|| defined(_DUMP_{node_name_upper}_) ')
        out_c.write (f'\n')
        out_c.write (f'  FILE * DumpFile_p = fopen(\"dump.txt\", \"w\"); \n')
        out_c.write (f'#endif \n')
        out_c.write (f'\n')
    
        #
        # Execute nodes one by one
        #
        nid = 0
        
        for node in self.nodes_:
            node_name = node['name']
            node_name_upper = node_name.upper()
        
            #print ("    [{}] op: {}".format(nid, node['name']))
        
            if node['op'] != 'null':
                assert node['op'] == 'tvm_op', (f'###Error: Only TVM ops are supported.')
                node_attrs = node['attrs']
                func_name = node_attrs['func_name']
                
                out_c.write (f'  // \n')
                out_c.write (f'  // {func_name}\n')
                out_c.write (f'  // \n')

                #
                # Prepare TVM packed function - this is the one called
                #
                if name == '__nop':
                    print ("      exec: __nop")
                    continue
        
                elif name == '__copy':
                    print ("      exec: __copy")
                    continue
        
                else:
                    #
                    # Get function from the TVM module
                    #
                    #  void * args         : arg_values.data()
                    #  void * arg_type_ids : arg_tcodes.data()
                    #  int32_t num_args    : arg_values.size()
                    #

                    dl_args_name = _get_node_args_name (node_name);
                    dl_arg_types_name = _get_node_arg_types_name (node_name);
                
                    num_inputs = len(node['inputs'])
                    num_outputs = int(node_attrs['num_outputs'])
                    num_args = num_inputs + num_outputs

                    out_c.write (f'  TVMValue {dl_args_name}[{num_args}]; \n')
                    out_c.write (f'  int32_t {dl_arg_types_name}[{num_args}]; \n')
                
                    curr_idx = 0

                    for input in node['inputs']:
                        src_nid = input[0]
                        src_idx = input[1]
                        src_version = input[2]
                        src_node = self.nodes_[src_nid]
                        src_name = self.nodes_[src_nid]['name']

                        if src_node['op'] == 'null':
                            dl_tensor_name = get_input_tensor_name (src_name);
                        else:
                            dl_tensor_name = get_output_tensor_name (src_name, src_idx);
                        #
                        # If this input is not an activation or a parameter => find the input
                        #
                        if not dl_tensor_name in self.weights_ and not dl_tensor_name in self.activations_:

                            assert (dl_tensor_name in self.input_data_), ("Tensor {} not registered ?".format(dl_tensor_name))

                            input_idx = 0
                            for dl_entry_name in self.input_data_:
                                if dl_entry_name == dl_tensor_name:
                                    break
                                input_idx += 1
                            out_c.write (f'  {dl_args_name}[{curr_idx}].v_handle = &inputs[{input_idx}]->dltensor; \n')
                        else:
                            out_c.write (f'  {dl_args_name}[{curr_idx}].v_handle = &{dl_tensor_name}.dltensor; \n')
                        out_c.write (f'  {dl_arg_types_name}[{curr_idx}] = kTVMNDArrayHandle; \n')

                        if dl_tensor_name in self.weights_:
                            out_c.write (f'#ifdef _DUMP_INPUTS_ \n')
                            out_c.write (f'  TVMDumpBuffer (\"{dl_tensor_name}\", &{dl_tensor_name}, DumpFile_p); \n')
                            out_c.write (f'#endif \n')
                        elif dl_tensor_name in self.input_data_:
                            input_idx = 0
                            for dl_entry_name in self.input_data_:
                                if dl_entry_name == dl_tensor_name:
                                    break
                                input_idx += 1
                            out_c.write (f'#ifdef _DUMP_INPUTS_ \n')
                            out_c.write (f'  TVMDumpBuffer (\"{dl_tensor_name}\", inputs[{input_idx}], DumpFile_p); \n')
                            out_c.write (f'#endif \n')
                        out_c.write (f'\n')
                        
                        curr_idx += 1

                    for idx in range(num_outputs):
                        dl_tensor_name = get_output_tensor_name (node_name, idx);
                        #
                        # If this output is not an activation => find the output
                        #
                        if dl_tensor_name not in self.activations_:

                            assert (dl_tensor_name in self.output_data_)

                            output_idx = 0
                            for dl_exit_name in self.output_data_:
                                if dl_exit_name == dl_tensor_name:
                                    break
                                output_idx += 1
                            out_c.write (f'  {dl_args_name}[{curr_idx}].v_handle = &outputs[{output_idx}]->dltensor; \n')
                        else:
                            out_c.write (f'  {dl_args_name}[{curr_idx}].v_handle = &{dl_tensor_name}.dltensor; \n')
                        out_c.write (f'  {dl_arg_types_name}[{curr_idx}] = kTVMNDArrayHandle; \n')
                        out_c.write (f'\n')
                    
                        curr_idx += 1

                    #
                    # call this function
                    #
                    out_c.write (f'#if (_VERBOSE_ > 0) \n')
                    out_c.write (f'  printf (\"  {func_name}  ... \\r\\n\"); \n')
                    out_c.write (f'#endif \n')

                    out_c.write (f'  if ({func_name} ({dl_args_name}, {dl_arg_types_name}, {num_args})) {{ \n')
                    out_c.write (f'    TVMAPISetLastError(\"Invalid handle\");\n')
                    out_c.write (f'    return AI_STATUS_ERROR; \n')

                    out_c.write (f'  }} \n')
                    out_c.write (f'#if (_VERBOSE_ > 0) \n')
                    out_c.write (f'  printf (\"  {func_name}  Done.\\r\\n\"); \n')
                    out_c.write (f'#endif \n')
                    out_c.write (f'\n')

                    out_c.write (f'#ifdef _DUMP_{node_name_upper}_ \n')
                    for idx in range(num_outputs):
                        dl_tensor_name = get_output_tensor_name (node_name, idx);
                        if dl_tensor_name in self.activations_:
                            out_c.write (f'  TVMDumpBuffer (\"{dl_tensor_name}\", &{dl_tensor_name}, DumpFile_p); \n')
                        else:
                            assert (dl_tensor_name in self.output_data_)
                            output_idx = 0
                            for dl_exit_name in self.output_data_:
                                if dl_exit_name == dl_tensor_name:
                                    break
                                output_idx += 1
                            out_c.write (f'  TVMDumpBuffer (\"{dl_tensor_name}\", outputs[{output_idx}], DumpFile_p); \n')
                    out_c.write (f'#endif \n')

            nid += 1

        out_c.write (f'\n')

        out_c.write (f'#if defined(_DUMP_INPUTS_) ')
        for node in self.nodes_:
            node_name = node['name']
            if node['op'] != 'null':
                out_c.write (f'|| defined(_DUMP_{node_name_upper}_) ')
        out_c.write (f'\n')
        out_c.write (f'  fclose(DumpFile_p); \n')
        out_c.write (f'#endif \n')
        out_c.write (f'\n')
        
        out_c.write (f'  return AI_STATUS_OK; \n')
        out_c.write (f'}} \n')
        out_c.write (f'\n')
    
    # ==========================================================
    #   __emit_create_destroy
    # ==========================================================
    def __emit_create_destroy (self, name, out_h, out_c):

        out_h.write (f'AI_API_ENTRY \n')
        out_h.write (f'ai_status ai_{name}_create ( \n')
        out_h.write (f'  const ai_ptr weights, \n')
        out_h.write (f'  const ai_ptr activations \n')
        out_h.write (f'); \n')
        out_h.write (f'\n')
    
        out_h.write (f'AI_API_ENTRY \n')
        out_h.write (f'ai_status ai_{name}_destroy (); \n')
        out_h.write (f'\n')
    
        out_c.write (f'// {DBar} \n')
        out_c.write (f'//   ai_{name}_create \n')
        out_c.write (f'// {DBar} \n')
        out_c.write (f'AI_API_ENTRY \n')
        out_c.write (f'ai_status ai_{name}_create( \n')
        out_c.write (f'  const ai_ptr weights, \n')
        out_c.write (f'  const ai_ptr activations \n')
        out_c.write (f') \n')
        out_c.write (f'{{ \n')
        out_c.write (f'  ai_status status = AI_STATUS_OK;\n')
        out_c.write (f'  status = {name}_configure_weights (weights); \n')
        out_c.write (f'  if (status != AI_STATUS_OK) {{\n')
        out_c.write (f'    return status;\n')
        out_c.write (f'  }}\n')
        out_c.write (f'  status = {name}_configure_activations (activations); \n')
        out_c.write (f'  if (status != AI_STATUS_OK) {{\n')
        out_c.write (f'    return status;\n')
        out_c.write (f'  }}\n')
        out_c.write (f'  return AI_STATUS_OK; \n')
        out_c.write (f'}} \n')
        out_c.write (f'\n')

        out_c.write (f'// {DBar} \n')
        out_c.write (f'//   ai_{name}_destroy \n')
        out_c.write (f'// {DBar} \n')
        out_c.write (f'AI_API_ENTRY \n')
        out_c.write (f'ai_status ai_{name}_destroy () \n')
        out_c.write (f'{{ \n')
        out_c.write (f'  return AI_STATUS_OK; \n')
        out_c.write (f'}} \n')

    # ==================================================================
    #   emit_code
    # ==================================================================
    def emit_code (self, dest_dir, model_name):
        """ Emit the C code implementing the model. """

        if not os.path.exists (dest_dir):
            os.makedirs (dest_dir)

        prefix = dest_dir+'/'+model_name

        #
        # Write the C code:
        #
        src = "#include <stdio.h>\n" \
              "#include <math.h>\n" \
              "#include \"stm32lib.h\"\n\n"
        for m in self.lib_.imported_modules:
            src = src + m.get_source(fmt="c")

        with open (prefix+'_lib.c', "w") as f:
            f.write(src)

        #
        # Save params as bynary data
        #
        saved_params = relay.save_param_dict(self.params_)
        with open (prefix+'_params.dat', 'wb') as f:
            f.write(saved_params)
        
        #
        # Write the .json
        #
        with open (prefix+'_graph.json', 'w') as f:
            f.write(self.graph_)

        #
        # emit X_data[c,h]
        #
        data_h = open('{}_data.h'.format(prefix), 'w')
        data_c = open('{}_data.c'.format(prefix), 'w')
        out_h = open('{}.h'.format(prefix), 'w')
        out_c = open('{}.c'.format(prefix), 'w')

        #
        # emit X[c,h]
        #

        self.__emit_params_data (model_name, data_h, data_c)
    
        self.__emit_open (model_name, out_h, out_c)
        self.__emit_params_buffers (model_name, out_h, out_c)
        self.__emit_activation_buffers (model_name, out_h, out_c)
        self.__emit_network (model_name, out_h, out_c)

        self.__emit_init (model_name, out_h, out_c)
        self.__emit_create_destroy (model_name, out_h, out_c)
        self.__emit_run (model_name, out_h, out_c)

        self.__emit_close (model_name, out_h, out_c)
        
        #
        # Close files
        #
        out_c.close()
        out_h.close()
        data_c.close()
        data_h.close()

        #self.__emit_linker_script (dest_dir)

    # ==========================================================
    #   __emit_gcc_linker_script
    # ==========================================================
    def __emit_linker_script (self, dest_dir, target=None):

        min_stack_size = 0x8000
        #
        # 1MB 0x4000
        #
        min_heap_size = 0x20000
        
        #print ("== emit_linker_script: weights size = {}".format(self.weights_size_))
        #print ("== emit_linker_script: activations size = {}".format(self.activations_size_))
        
        #
        # Emit the linker script
        # TODO: one for each possible target compiler supported, eg. GCC, ARMCC, Keil
        #
        with open (dest_dir+'/STM32H747_gcc.ld', 'w') as f:

            t = Template(STM32_H747_GCC_LINKER_SCRIPT)

            #
            # TODO: 
            #   stack placement perhaps AXIRAM ?
            #   heap placement ?
            #   activations/weights placement should depend on
            #     what compiler had to say.
            #
            s = t.substitute(stack_section = 'DTCMRAM',
                             heap_section = 'SDRAM',
                             activations_section = 'SDRAM',
                             weights_section = 'QSPIFLASH',
                             stack_size = hex(min_stack_size),
                             heap_size = hex(min_heap_size)
                             )

            f.write (s)

STM32_H747_GCC_LINKER_SCRIPT = """
/* Entry Point */
ENTRY(Reset_Handler)

/* Highest address of the user mode stack */
/* _estack = 0x24080000; */  /* end of AXIRAM */
/*_estack = 0x20020000; */   /* end of DTCMRAM */
/* Generate a link error if heap and stack don't fit into RAM */
_Min_Stack_Size = $stack_size; /* required amount of stack */
_Min_Heap_Size =  $heap_size; /* required amount of heap  */

/* Specify the memory areas */
MEMORY
{
ITCMRAM (xrw)   : ORIGIN = 0x00000000, LENGTH = 64K
FLASH (rx)      : ORIGIN = 0x08000000, LENGTH = 2048K
DTCMRAM (xrw)   : ORIGIN = 0x20000000, LENGTH = 128K
AXIRAM (xrw)    : ORIGIN = 0x24000000, LENGTH = 512K
SRAM123 (xrw)   : ORIGIN = 0x30000000, LENGTH = 288K
SRAM4   (xrw)   : ORIGIN = 0x38000000, LENGTH = 64K
BKPSRAM (xrw)   : ORIGIN = 0x38800000, LENGTH = 64K
QSPIFLASH (rx)  : ORIGIN = 0x90000000, LENGTH = 128M
SDRAM (xrw)     : ORIGIN = 0xD0000000, LENGTH = 32M
}

/* Define output sections */
SECTIONS
{
  /* The startup code goes first into FLASH */
  .isr_vector :
  {
    . = ALIGN(4);
    KEEP(*(.isr_vector)) /* Startup code */
    . = ALIGN(4);
  } >FLASH
  
  
  /* The program code and other data goes into FLASH */
  .text :
  {
    . = ALIGN(4);
    *(.text)           /* .text sections (code) */
    *(.text*)          /* .text* sections (code) */
    *(.glue_7)         /* glue arm to thumb code */
    *(.glue_7t)        /* glue thumb to arm code */
    *(.eh_frame)

    KEEP (*(.init))
    KEEP (*(.fini))

    . = ALIGN(4);
    _etext = .;        /* define a global symbols at end of code */
  } >FLASH

  .params_external_flash :
  {
    *(.nn_weights*)
  } >$weights_section   /* QSPIFLASH FLASH*/

  /* Constant data goes into FLASH */
  .rodata :
  {
    . = ALIGN(4);
    *(.rodata)         /* .rodata sections (constants, strings, etc.) */
    *(.rodata*)        /* .rodata* sections (constants, strings, etc.) */
    . = ALIGN(4);
  } >FLASH

  .ARM.extab   : { *(.ARM.extab* .gnu.linkonce.armextab.*) } >FLASH
  .ARM : {
    __exidx_start = .;
    *(.ARM.exidx*)
    __exidx_end = .;
  } >FLASH

  .preinit_array     :
  {
    PROVIDE_HIDDEN (__preinit_array_start = .);
    KEEP (*(.preinit_array*))
    PROVIDE_HIDDEN (__preinit_array_end = .);
  } >FLASH
  .init_array :
  {
    PROVIDE_HIDDEN (__init_array_start = .);
    KEEP (*(SORT(.init_array.*)))
    KEEP (*(.init_array*))
    PROVIDE_HIDDEN (__init_array_end = .);
  } >FLASH
  .fini_array :
  {
    PROVIDE_HIDDEN (__fini_array_start = .);
    KEEP (*(SORT(.fini_array.*)))
    KEEP (*(.fini_array*))
    PROVIDE_HIDDEN (__fini_array_end = .);
  } >FLASH

  .nn_models :
  {
    . = ALIGN(4);
    __models_section_start__ = .;
    KEEP (*(.nn_models*))
    __models_section_end__ = .;
  } >DTCMRAM  /*AXIRAM -- for some reason cannot initialize AXIRAM*/

  /* User_heap_stack section, used to check that there is enough RAM left */
  ._user_stack :
  {
    . = ALIGN(8);
    _sstack = .;
    . = . + _Min_Stack_Size;
    _estack = .;
    . = ALIGN(8);
  } >$stack_section

  /* Check if MY_MEMORY usage exceeds MY_MEMORY size */
/*
  ASSERT( LENGTH(MY_MEMORY) >= (__mysection_end__ - __mysection_start__), "MY_MEMORY memory overflowed !")
*/

  /* used by the startup to initialize data */
  _sidata = LOADADDR(.data);

  /* Initialized data sections goes into RAM, load LMA copy after code */
  .data : 
  {
    . = ALIGN(4);
    _sdata = .;        /* create a global symbol at data start */
    *(.data)           /* .data sections */
    *(.data*)          /* .data* sections */

    . = ALIGN(4);
    _edata = .;        /* define a global symbol at data end */
  } >DTCMRAM AT> FLASH

  
  /* Uninitialized data section */
  . = ALIGN(4);
  .bss :
  {
    /* This is used by the startup in order to initialize the .bss secion */
    _sbss = .;         /* define a global symbol at bss start */
    __bss_start__ = _sbss;
    *(.bss)
    *(.bss*)
    *(COMMON)

    . = ALIGN(4);
    _ebss = .;         /* define a global symbol at bss end */
    __bss_end__ = _ebss;
  } >DTCMRAM

  /* Warning: looks like there is no heap overlow check in sbrk */
  ._user_heap :
  {
    . = ALIGN(8);
    PROVIDE ( end = . );
    PROVIDE ( _end = . );
    . = . + _Min_Heap_Size;
    . = ALIGN(8);
  } >$heap_section

  .external_sdram (NOLOAD) :
  {
    *(.nn_data_act*)
  }  >$activations_section

  /* Remove information from the standard libraries */
  /DISCARD/ :
  {
    libc.a ( * )
    libm.a ( * )
    libgcc.a ( * )
  }

  .ARM.attributes 0 : { *(.ARM.attributes) }
}

"""

