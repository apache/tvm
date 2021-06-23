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
# pylint: disable=useless-super-delegation

"""Code emission for the STM32 targets."""

import json
import os
import shutil

import tvm

from tvm.contrib.stm32.BaseEmitter import BaseEmitter
from tvm.contrib.stm32.BaseEmitter import write_operators_lib
from tvm.contrib.stm32.BaseEmitter import DBAR


class CodeEmitter(BaseEmitter):
    """Code emitter class/utility."""

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

        super().__init__(model_name, include_activations, include_inputs, include_outputs)

    def emit_network(self, out_c):
        """Emits prototypes for the network operator functions."""

        name = self.model_name_

        out_c.write(f"// \n")
        out_c.write(f"// Network: {name}\n")
        out_c.write(f"// \n")

        for node in self.nodes_:
            if node["op"] == "null":
                continue
            assert node["op"] == "tvm_op", f"###Error: Only TVM ops are supported."
            self.emit_operator_proto(node, out_c)
        out_c.write(f"\n")

    def emit_run(self, out_h, out_c):
        """Emits the run function code."""

        name = self.model_name_

        out_h.write(f"AI_API_ENTRY \n")
        out_h.write(f"ai_status ai_{name}_run ( \n")
        out_h.write(f"  ai_tensor *inputs[], \n")
        out_h.write(f"  ai_tensor *outputs[] \n")
        out_h.write(f"); \n")
        out_h.write(f"\n")

        out_c.write(f"// {DBAR} \n")
        out_c.write(f"//   ai_{name}_run \n")
        out_c.write(f"// {DBAR} \n")
        out_c.write(f"AI_API_ENTRY \n")
        out_c.write(f"ai_status ai_{name}_run ( \n")
        out_c.write(f"  ai_tensor *inputs[], \n")
        out_c.write(f"  ai_tensor *outputs[] \n")
        out_c.write(f") \n")
        out_c.write(f"{{ \n")

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
        for node in self.nodes_:

            if node["op"] == "null":
                continue
            assert node["op"] == "tvm_op", f"###Error: Only TVM ops are supported."
            self.emit_operator_call(node, out_c)

        out_c.write(f"\n")

        out_c.write(f"#if defined(_DUMP_INPUTS_) ")
        for node in self.nodes_:
            node_name = node["name"]
            if node["op"] != "null":
                out_c.write(f"|| defined(_DUMP_{node_name_upper}_) ")
        out_c.write(f"\n")
        out_c.write(f"  fclose(DumpFile_p); \n")
        out_c.write(f"#endif \n")
        out_c.write(f"\n")

        out_c.write(f"  return AI_STATUS_OK; \n")
        out_c.write(f"}} \n")
        out_c.write(f"\n")

    def emit_create_destroy(self, out_h, out_c):
        """Emits the create/destroy functions."""
        name = self.model_name_
        out_h.write(f"AI_API_ENTRY \n")
        out_h.write(f"ai_status ai_{name}_create ( \n")
        out_h.write(f"  const ai_ptr weights, \n")
        out_h.write(f"  const ai_ptr activations \n")
        out_h.write(f"); \n")
        out_h.write(f"\n")

        out_h.write(f"AI_API_ENTRY \n")
        out_h.write(f"ai_status ai_{name}_destroy (); \n")
        out_h.write(f"\n")

        out_c.write(f"// {DBAR} \n")
        out_c.write(f"//   ai_{name}_create \n")
        out_c.write(f"// {DBAR} \n")
        out_c.write(f"AI_API_ENTRY \n")
        out_c.write(f"ai_status ai_{name}_create( \n")
        out_c.write(f"  const ai_ptr weights, \n")
        out_c.write(f"  const ai_ptr activations \n")
        out_c.write(f") \n")
        out_c.write(f"{{ \n")
        out_c.write(f"  ai_status status = AI_STATUS_OK;\n")
        out_c.write(f"  status = {name}_configure_weights (weights); \n")
        out_c.write(f"  if (status != AI_STATUS_OK) {{\n")
        out_c.write(f"    return status;\n")
        out_c.write(f"  }}\n")
        out_c.write(f"  status = {name}_configure_activations (activations); \n")
        out_c.write(f"  if (status != AI_STATUS_OK) {{\n")
        out_c.write(f"    return status;\n")
        out_c.write(f"  }}\n")
        out_c.write(f"  return AI_STATUS_OK; \n")
        out_c.write(f"}} \n")
        out_c.write(f"\n")

        out_c.write(f"// {DBAR} \n")
        out_c.write(f"//   ai_{name}_destroy \n")
        out_c.write(f"// {DBAR} \n")
        out_c.write(f"AI_API_ENTRY \n")
        out_c.write(f"ai_status ai_{name}_destroy () \n")
        out_c.write(f"{{ \n")
        out_c.write(f"  return AI_STATUS_OK; \n")
        out_c.write(f"}} \n")

    def emit_code(self, dest_dir, quantization_map=None):
        """Emits the C code implementing the model."""

        model_name = self.model_name_

        #
        # Build the directory structure
        #
        if os.path.exists(dest_dir):
            print(f'Removing existing "{dest_dir}" directory')
            try:
                shutil.rmtree(dest_dir)
            except OSError as err:
                raise ValueError(f"emit_code.Error: {dest_dir} : {err.strerror}")

        # Make a new one
        os.mkdir(dest_dir)

        #
        # Build BaseEmitter internal data structures
        #
        self.compute_data_placement()

        #
        # Write the C code: we can parse the string
        #
        write_operators_lib(model_name, self.lib_, dest_dir)

        #
        # Save params as bynary data
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
        model_h_name = os.path.join(dest_dir, model_name + ".h")
        model_c_name = os.path.join(dest_dir, model_name + ".c")
        out_h = open(model_h_name, "w")
        out_c = open(model_c_name, "w")

        #
        # emit X[c,h]
        #
        self.emit_params_data(dest_dir)
        self.emit_open(out_h, out_c)
        self.emit_params_buffers(quantization_map, out_c)
        self.emit_activation_buffers(quantization_map, out_c)
        self.emit_network(out_c)
        self.emit_init(out_c)
        self.emit_create_destroy(out_h, out_c)
        self.emit_run(out_h, out_c)
        self.emit_close(out_h, out_c)

        #
        # Close files
        #
        out_c.close()
        out_h.close()
