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

import os
import argparse
import pathlib
import sys
from tflite.Model import Model
import tflite


def create_header_file(name, tensor_name, tensor_shape, tensor_type_size, output_path):
    """
    This function generates a header file containing the data placesholder.
    """

    file_path = pathlib.Path(f"{output_path}/" + name).resolve()
    # Create header file as a C array
    raw_path = file_path.with_suffix(".h").resolve()

    with open(raw_path, "w") as header_file:
        header_file.write(
            "\n"
            + f"const size_t {tensor_name}_len = {tensor_shape.prod()};\n"
            + f'__attribute__((section(".data.tvm"), aligned(32))) int8_t {tensor_name}[{tensor_name}_len * {tensor_type_size}];\n\n'
        )


def tflite_type_size(type):
    if type in [tflite.TensorType.INT8, tflite.TensorType.UINT8]:
        return 1
    elif type in [tflite.TensorType.FLOAT16, tflite.TensorType.INT16]:
        return 2
    elif type in [tflite.TensorType.INT32, tflite.TensorType.FLOAT32]:
        return 4
    assert False, "Invalid type!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-B", "--build-dir", default=".")
    opts = parser.parse_args()

    model_path = opts.file
    assert model_path.endswith(".tflite"), "Not a TFLite model!"
    with open(model_path, "rb") as fi:
        model = Model.GetRootAsModel(bytearray(fi.read()), 0)
    subgraph = model.Subgraphs(0)
    input_tensor = subgraph.Tensors(subgraph.Inputs(0))
    output_tensor = subgraph.Tensors(subgraph.Outputs(0))

    # Create input header file
    include_dir = os.path.join(opts.build_dir, "include")
    os.makedirs(include_dir, exist_ok=True)
    create_header_file(
        "inputs",
        "input",
        input_tensor.ShapeAsNumpy(),
        tflite_type_size(input_tensor.Type()),
        include_dir,
    )
    # Create output header file
    create_header_file(
        "outputs",
        "output",
        output_tensor.ShapeAsNumpy(),
        tflite_type_size(output_tensor.Type()),
        include_dir,
    )
