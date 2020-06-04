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
# pylint: disable=unused-import
""" ... """

import tvm._ffi
from tvm.runtime import Object

from . import _ffi_api

@tvm._ffi.register_object("ansor.HardwareParams")
class HardwareParams(Object):
    """
    Parameters
    ----------
    num_cores : Int
    vector_unit_bytes : Int
    cache_line_bytes : Int
    max_unroll_vec : Int
    max_innermost_split_factor : Int
    """

    def __init__(self, num_cores, vector_unit_bytes, cache_line_bytes,
                 max_unroll_vec, max_innermost_split_factor):
        self.__init_handle_by_constructor__(_ffi_api.HardwareParams, num_cores,
                vector_unit_bytes, cache_line_bytes, max_unroll_vec,
                max_innermost_split_factor)


@tvm._ffi.register_object("ansor.SearchTask")
class SearchTask(Object):
    """
    Parameters
    ----------
    dag : ComputeDAG
    workload_key : Str
    target : tvm.target
    target_host : tvm.target
    hardware_params : HardwareParams
    """

    def __init__(self, dag, workload_key, target, target_host=None,
                 hardware_params=None):
        self.__init_handle_by_constructor__(_ffi_api.SearchTask, dag,
                workload_key, target, target_host, hardware_params)
