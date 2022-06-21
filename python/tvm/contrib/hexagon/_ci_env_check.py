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

"""Hexagon environment checks for CI usage

These may be required by either tvm.testing or
tvm.contrib.hexagon.pytest_plugin, and are separated here to avoid a
circular dependency.
"""

import os

import tvm

ANDROID_SERIAL_NUMBER = "ANDROID_SERIAL_NUMBER"
HEXAGON_TOOLCHAIN = "HEXAGON_TOOLCHAIN"


def _compile_time_check():
    """Return True if compile-time support for Hexagon is present, otherwise
    error string.

    Designed for use as a the ``compile_time_check`` argument to
    `tvm.testing.Feature`.
    """
    if (
        tvm.testing.utils._cmake_flag_enabled("USE_LLVM")
        and tvm.target.codegen.llvm_version_major() < 7
    ):
        return "Hexagon requires LLVM 7 or later"

    if "HEXAGON_TOOLCHAIN" not in os.environ:
        return f"Missing environment variable {HEXAGON_TOOLCHAIN}."

    return True


def _run_time_check():
    """Return True if run-time support for Hexagon is present, otherwise
    error string.

    Designed for use as a the ``run_time_check`` argument to
    `tvm.testing.Feature`.
    """
    if ANDROID_SERIAL_NUMBER not in os.environ:
        return f"Missing environment variable {ANDROID_SERIAL_NUMBER}."

    return True
