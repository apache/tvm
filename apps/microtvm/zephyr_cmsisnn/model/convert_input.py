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


def create_file(name, prefix, tensor_name, tensor_data, output_path):
    """
    This function generates a header file containing the data from the numpy array provided.
    """
    file_path = pathlib.Path(f"{output_path}/" + name).resolve()
    # Create header file with npy_data as a C array
    raw_path = file_path.with_suffix(".c").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write(
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            f"const size_t {tensor_name}_len = {tensor_data.size};\n"
            f"{prefix} float {tensor_name}_storage[] = "
        )
        header_file.write("{")
        for i in np.ndindex(tensor_data.shape):
            header_file.write(f"{tensor_data[i]}, ")
        header_file.write("};\n\n")


def create_files(input_file, output_dir):
    """
    This function generates C files for the input and output arrays required to run inferences
    """
    # Create out folder
    os.makedirs(output_dir, exist_ok=True)

    # Create input header file
    input_data = np.loadtxt(input_file)
    create_file("inputs", "const", "input", input_data, output_dir)

    # Create output header file
    output_data = np.zeros([12], np.float32)
    create_file(
        "outputs",
        "",
        "output",
        output_data,
        output_dir,
    )


if __name__ == "__main__":
    create_files(sys.argv[1], sys.argv[2])
