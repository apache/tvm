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

"""
    UMA Command Line Interface (CLI)

    Tool to create code skeletons for an easy integration of
    new AI hardware accelerators/libraries into TVM using UMA
"""

import argparse
import os
import shutil
import sys
import pathlib
from inflection import camelize, underscore


def _parse_args():
    parser = argparse.ArgumentParser(description="UMA Interface command line interface")
    parser.add_argument(
        "--add_hardware",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tutorial",
        type=str,
    )
    args = parser.parse_args()
    return args


def replace_template_name(
    files: list, template_name: str, add_hw_name: str, template_source: str = "_template"
) -> None:
    """
    Replace names in template skeleton code by new name
    """
    for f in files:
        with open(f) as read_file:
            data = read_file.read()
        for case in [underscore, camelize]:
            data = data.replace(case(template_name), case(add_hw_name))
        data = data.replace(template_source, underscore(add_hw_name))
        with open(f, "w") as write_file:
            write_file.write(data)


def main():
    """
    UMA Command Line Interface (CLI)
    """
    args = _parse_args()
    add_hw_name = args.add_hardware
    uma_template_path = pathlib.Path(os.getcwd(), "_template").absolute()

    add_hw_path = os.path.join(uma_template_path.parent, add_hw_name)
    if os.path.exists(add_hw_path):
        print(
            f"Hardware with name {add_hw_name} already exists in UMA file structure: {add_hw_path}"
        )
        sys.exit(-1)
    else:
        os.mkdir(add_hw_path)

    uma_files = ["backend.py", "codegen.py", "passes.py", "patterns.py", "run.py", "strategies.py"]
    if args.tutorial == "vanilla":
        uma_files.append("conv2dnchw.cc")

    source_files = [os.path.join(uma_template_path, f) for f in uma_files]
    destination_files = [os.path.join(add_hw_path, f) for f in uma_files]

    for src, dst in zip(source_files, destination_files):
        shutil.copyfile(src, dst)

    template_name = "my_ai_hw"
    replace_template_name(destination_files, template_name, add_hw_name)

    print(f"Success: added {add_hw_name} to {add_hw_path}")


if __name__ == "__main__":
    main()
