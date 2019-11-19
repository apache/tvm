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
"""Example extension package of TVM."""
from __future__ import absolute_import
import os
import ctypes
# Import TVM first to get library symbols
import tvm

def load_lib():
    """Load library, the functions will be registered into TVM"""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    # load in as global so the global extern symbol is visible to other dll.
    lib = ctypes.CDLL(
        os.path.join(curr_path, "../../lib/libtvm_ext_json_rt.so"), ctypes.RTLD_GLOBAL)
    return lib

_LIB = load_lib()

create_json_rt = tvm.get_global_func("ext_json_rt.create_json_rt")

