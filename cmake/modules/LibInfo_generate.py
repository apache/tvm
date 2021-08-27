#!/usr/bin/env python3

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

import argparse
from pathlib import Path
import sys

template = r"""
/* WARNING: THIS IS AN AUTO-GENERATED FILE.
 *
 * This file contains cmake build information, to be exposed to TVM.
 * Any changes made to this file will be overwritten.  The list of
 * variables exposed is defined in cmake, which are passed to a code
 * generator in python.  The locations of these files are below.
 *
 * Generator script: {python_filename}
 * CMake file: {cmake_filename}
 *
 */

#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

namespace tvm {{

/*!
 * \brief Get a dictionary containing compile-time info, including cmake flags and git commit hash
 * \return The compile-time info
 */

TVM_DLL Map<String, String> GetLibInfo() {{
  Map<String, String> result = {{
    {entries}
  }};
  return result;
}}

TVM_REGISTER_GLOBAL("support.GetLibInfo").set_body_typed(GetLibInfo);

}}  // namespace tvm
"""

entry_template = """{{ "{key}", "{value}" }}"""


def parse_args():
    parser = argparse.ArgumentParser()

    def key_value(s):
        key, val = s.split("=", maxsplit=1)
        return (key, val)

    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        default=None,
        help="The output file to generate.  If unspecified, will print to stdout",
    )
    parser.add_argument(
        "libinfo",
        nargs=argparse.REMAINDER,
        type=key_value,
        help="Key-value pairs that should be exposed to python.  (e.g. USE_CUDA=ON)",
    )

    return parser.parse_args()


def generate_text(libinfo):
    python_filename = Path(__file__).absolute()
    cmake_filename = python_filename.with_name("LibInfo.cmake")

    entries = []
    for key, value in libinfo:
        key = key.replace('"', '\\"')
        value = value.replace('"', '\\"')
        entries.append(entry_template.format(key=key, value=value))

    return template.format(
        python_filename=str(python_filename),
        cmake_filename=str(cmake_filename),
        entries=",\n".join(entries),
    )


def main():
    args = parse_args()
    text = generate_text(args.libinfo)

    if args.output_file:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
