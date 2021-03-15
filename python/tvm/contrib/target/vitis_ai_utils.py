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
"""Vitis AI target utilities"""

import importlib

import tvm


def vitis_ai_available():
    """Return whether Vitis AI tools are available"""
    pyxir_spec = importlib.util.find_spec("pyxir")
    if not tvm.get_global_func("tvm.vitis_ai_runtime.from_xgraph", True) or pyxir_spec is None:
        return False
    return True


def init_for_vitis_ai():
    """Initialization function for the Vitis AI codegen"""
    # We need to import the Vitis AI target module to make sure the codegen
    #   is registered
    importlib.import_module("tvm.contrib.target.vitis_ai")
