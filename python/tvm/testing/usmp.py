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
""" This file contains USMP tests harnesses."""

import tvm


def is_tvm_backendallocworkspace_calls(mod: tvm.runtime.module) -> bool:
    """TVMBackendAllocWorkspace call check.

    This checker checks whether any c-source produced has TVMBackendAllocWorkspace calls.
    If USMP is invoked, none of them should have TVMBAW calls
    """
    dso_modules = mod._collect_dso_modules()
    for dso_mod in dso_modules:
        if dso_mod.type_key not in ["c", "llvm"]:
            assert (
                False
            ), 'Current AoT codegen flow should only produce type "c" or "llvm" runtime modules'

        source = dso_mod.get_source()
        if source.count("TVMBackendAllocWorkspace") != 0:
            return True

    return False
