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
import shutil
import pathlib

from tvm.micro import copy_crt_config_header


def create_labels_header(labels_file, output_path):
    """
    This function generates a header file containing the ImageNet labels as an array of strings
    """
    labels_path = pathlib.Path(labels_file).resolve()
    file_path = pathlib.Path(f"{output_path}/labels.c").resolve()

    with open(labels_path) as f:
        labels = f.readlines()

    with open(file_path, "w") as header_file:
        header_file.write(f"char* labels[] = {{")

        for _, label in enumerate(labels):
            header_file.write(f'"{label.rstrip()}",')

        header_file.write("};\n")


def prepare_crt_config():
    crt_config_output_path = (
        pathlib.Path(__file__).parent.resolve().parent() / "build" / "crt_config"
    )
    if not crt_config_output_path.exists():
        crt_config_output_path.mkdir()
    copy_crt_config_header("zephyr", crt_config_output_path)


if __name__ == "__main__":
    create_labels_header(sys.argv[1], sys.argv[2])
    prepare_crt_config()
