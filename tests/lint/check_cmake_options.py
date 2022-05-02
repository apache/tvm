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
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent

LIBINFO_CC = REPO_ROOT / "src" / "support" / "libinfo.cc"
LIBINFO_CMAKE = REPO_ROOT / "cmake" / "modules" / "LibInfo.cmake"
CMAKELISTS = REPO_ROOT / "CMakeLists.txt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check that CMake options are mirrored to libinfo.cc"
    )

    with open(CMAKELISTS) as f:
        cmake = f.readlines()

    with open(LIBINFO_CC) as f:
        libinfo = f.read()

    with open(LIBINFO_CMAKE) as f:
        libinfo_cmake = f.read()

    # Read tvm_options from CMakeLists.txt
    options = []
    for line in cmake:
        m = re.search(r"tvm_option\((.*?) ", line)
        if m is not None:
            options.append(m.groups()[0])

    # Check that each option is present in libinfo.cc
    missing_lines = []
    for option in options:
        expected_line = f'      {{"{option}", TVM_INFO_{option}}},'
        if expected_line not in libinfo:
            missing_lines.append(expected_line)

    error = False
    if len(missing_lines) > 0:
        missing_lines = "\n".join(missing_lines)
        print(
            f"Missing these lines from {LIBINFO_CC.relative_to(REPO_ROOT)}, please update it\n{missing_lines}"
        )
        error = True

    # Check that each option has a compile defintion in LibInfo.cmake
    missing_cmake_lines = []
    for option in options:
        expected_line = f'    TVM_INFO_{option}="${{{option}}}"'
        if expected_line not in libinfo_cmake:
            missing_cmake_lines.append(expected_line)

    if len(missing_cmake_lines) > 0:
        missing_cmake_lines = "\n".join(missing_cmake_lines)
        print(
            f"Missing these lines from {LIBINFO_CMAKE.relative_to(REPO_ROOT)}, please update it\n{missing_cmake_lines}"
        )
        error = True

    if error:
        exit(1)
