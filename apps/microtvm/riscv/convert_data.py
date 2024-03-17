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
import sys
import numpy as np
import argparse


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
            + f'__attribute__((section(".data.tvm"), aligned(32))) int8_t {tensor_name}[] = "'
        )

        data_hexstr = tensor_data.tobytes().hex()
        for i in range(0, len(data_hexstr), 2):
            header_file.write(f"\\x{data_hexstr[i:i+2]}")
        header_file.write('";\n\n')


def create_headers(name, type, output_len):
    """
    This function generates C header files for the input and output arrays required to run inferences
    """
    path = os.path.join("./", f"{name}")
    _, ext = os.path.splitext(path)

    if ext in [".npy"]:
        data = np.load(path)
    elif ext in [".txt"]:
        data = np.loadtxt(path)
    else:
        from PIL import Image

        image = Image.open(path)
        if opts.width and opts.height:
            # Resize image
            image = image.resize((opts.width, opts.height))
        img_data = np.asarray(image).astype("float32")
        # Add the batch dimension, as we are expecting 4-dimensional input: NHWC.
        img_data = np.expand_dims(img_data, axis=0)
        data = img_data - 128

    # Create input header file
    type = np.dtype(type)
    input_data = data.astype(type)
    create_header_file("inputs", "input", input_data, "./include")
    # Create output header file
    output_data = np.zeros([int(output_len)], type)
    create_header_file(
        "outputs",
        "output",
        output_data,
        "./include",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-W", "--width", type=int, default=0)
    parser.add_argument("-H", "--height", type=int, default=0)
    parser.add_argument("-t", "--type", required=True)
    parser.add_argument("-o", "--output-len", type=int, required=True)
    opts = parser.parse_args()

    create_headers(opts.file, opts.type, opts.output_len)
