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
import pathlib
import re
import sys
from PIL import Image
import numpy as np

from tvm.micro import copy_crt_config_header


def create_header_file(name, tensor_name, tensor_data, output_path):
    """
    This function generates a header file containing the data from the numpy array provided.
    """
    file_path = pathlib.Path(f"{output_path}/" + name).resolve()
    # Create header file with npy_data as a C array
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write(
            "\n"
            + f"const size_t {tensor_name}_len = {tensor_data.size};\n"
            + f'__attribute__((section(".data.tvm"), aligned(16))) int8_t {tensor_name}[] = "'
        )

        data_hexstr = tensor_data.tobytes().hex()
        for i in range(0, len(data_hexstr), 2):
            header_file.write(f"\\x{data_hexstr[i:i+2]}")
        header_file.write('";\n\n')


def create_headers(image_name):
    """
    This function generates C header files for the input and output arrays required to run inferences
    """
    img_path = os.path.join("./", f"{image_name}")

    # Resize image to 224x224
    resized_image = Image.open(img_path).resize((224, 224))
    img_data = np.asarray(resized_image).astype("float32")

    # # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    img_data = np.expand_dims(img_data, axis=0)

    # Create input header file
    input_data = img_data - 128
    input_data = input_data.astype(np.int8)
    create_header_file("inputs", "input", input_data, "./include")
    # Create output header file
    output_data = np.zeros([2], np.int8)
    create_header_file(
        "outputs",
        "output",
        output_data,
        "./include",
    )


if __name__ == "__main__":
    create_headers(sys.argv[1])

    # Generate crt_config.h
    crt_config_output_path = pathlib.Path(__file__).parent.resolve() / "build" / "crt_config"
    if not crt_config_output_path.exists():
        crt_config_output_path.mkdir()
    copy_crt_config_header("crt", crt_config_output_path)
