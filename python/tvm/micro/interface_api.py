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

"""Defines functions for generating a C interface header"""

import os

from tvm.relay.backend.utils import mangle_module_name


def _emit_brief(header_file, model_name, description):
    header_file.write("/*!\n")
    header_file.write(f" * \\brief TVM {model_name} model {description} \n")
    header_file.write(" */\n")


def generate_c_interface_header(model_name, inputs, outputs, output_path):
    """Generates a C interface header for a given models inputs and outputs

    Parameters
    ----------
    model_name : str
        Name of the model to be used in defining structs and naming the header
    inputs : list[str]
        List of model input names to be placed in generated structs
    outputs : list[str]
        List of model output names to be placed in generated structs
    output_path : str
        Path to the output folder to generate the header into
    """

    mangled_name = mangle_module_name(model_name)
    metadata_header = os.path.join(output_path, f"{mangled_name}.h")
    with open(metadata_header, "w") as header_file:
        _emit_brief(header_file, model_name, "input tensors")
        header_file.write(f"struct {mangled_name}_inputs {{\n")
        for input_name in inputs:
            header_file.write(f"  void* {input_name};\n")
        header_file.write("};\n\n")

        _emit_brief(header_file, model_name, "output tensors")
        header_file.write(f"struct {mangled_name}_outputs {{\n")
        for output_name in outputs:
            header_file.write(f"  void* {output_name};\n")
        header_file.write("};\n\n")

        _emit_brief(header_file, model_name, "memory blocks")
        header_file.write(f"struct {mangled_name}_memory {{\n")
        header_file.write("};\n\n")

        _emit_brief(header_file, model_name, "device configurations")
        header_file.write(f"struct {mangled_name}_devices {{\n")
        header_file.write("};\n\n")

        header_file.write("/*!\n")
        header_file.write(f" * \\brief TVM {model_name} model run function \n")
        header_file.write(" * \\param inputs Input tensors for the model \n")
        header_file.write(" * \\param outputs Output tensors for the model \n")
        header_file.write(" * \\param memory Memory blocks for the model to use \n")
        header_file.write(" * \\param devices Devices for the model to use \n")
        header_file.write(" */\n")
        header_file.write(f"int {mangled_name}_run(\n")
        header_file.write(f"  struct {mangled_name}_inputs* inputs,\n")
        header_file.write(f"  struct {mangled_name}_outputs* outputs,\n")
        header_file.write(f"  struct {mangled_name}_memory* memory,\n")
        header_file.write(f"  struct {mangled_name}_devices* devices\n")
        header_file.write(");\n")
