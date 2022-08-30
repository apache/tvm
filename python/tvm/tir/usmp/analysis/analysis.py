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
"""USMP Analysis Python API for passes"""
# pylint: disable=invalid-name
from . import _ffi_api
from ...function import PrimFunc
from ....ir.module import IRModule


def extract_buffer_info(main_func: PrimFunc, mod: IRModule):
    """Convert Parallel For Loop to Serial.

    Parameters
    ----------
    main_func: tvm.tir.PrimFunc
        The main function containing calls to operator PrimFuncs.
    mod : tvm.ir.IRModule
        The full IRModule containing all PrimFuncs

    Returns
    -------
    Map<tir::Stmt, BufferInfo>
        extracted buffer info objects
    """
    return _ffi_api.extract_buffer_info(main_func, mod)
