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


def create_header_file(name, section, npy_data, output_path):
    """
    This function generates a header file containing the data from the numpy array provided.
    """
    file_path = pathlib.Path(f"{output_path}/" + name).resolve()

    # Create header file with npy_data as a C array
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write("#include <tvmgen_default.h>\n")
        for tensor_name in npy_data.keys():
            sanitized_tensor_name = re.sub(r"\W+", "_", tensor_name)
            header_file.write(
                f"const size_t {sanitized_tensor_name}_len = {npy_data[tensor_name].size};\n"
            )

            # Convert numpy data type to C data type
            if npy_data[tensor_name].dtype == np.uint8:
                c_type = "uint8_t"
            elif npy_data[tensor_name].dtype == np.int8:
                c_type = "int8_t"
            else:
                raise RuntimeError(f"Data type {str(npy_data[tensor_name].dtype)} not supported")

            header_file.write(
                f'{c_type} {sanitized_tensor_name}[] __attribute__((section("{section}"), aligned(16))) = "'
            )

            data_hexstr = npy_data[tensor_name].tobytes().hex()
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

    # Convert input to NCHW
    img_data = np.transpose(img_data, (2, 0, 1))

    # Create input header file
    input_data = {"input": img_data.astype(np.uint8)}
    create_header_file("inputs", "ethosu_scratch", input_data, "./include")

    # Create output header file
    output_data = {"output": np.zeros([1001], np.uint8)}
    create_header_file(
        "outputs",
        "output_data_sec",
        output_data,
        "./include",
    )


if __name__ == "__main__":
    create_headers(sys.argv[1])
