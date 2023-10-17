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
# pylint: disable=invalid-name, unused-wildcard-import, wildcard-import, pointless-exception-statement
"""Abstract Class to generate kernel."""
from abc import ABC, abstractmethod

from tvm import relax
from tvm.topi.utils import get_const_tuple
import tvm._ffi
from tvm.runtime import Object
from . import _ffi_api as ffi

from . import library


dtype_map = {
    # "int8": library.DataType.s8,
    # "uint8": library.DataType.u8,
    # "int32": library.DataType.s32,
    "float32": library.DataType.f32,
    "float16": library.DataType.f16,
}


def extract_relax_function_signature(f):
    signature = {}

    for i, arg in enumerate(f.params):
        sinfo = arg.struct_info
        if isinstance(sinfo, relax.TensorStructInfo):
            signature["arg%d_shape" % i] = get_const_tuple(sinfo.shape)
            signature["arg%d_dtype" % i] = sinfo.dtype
        elif isinstance(sinfo, relax.ShapeStructInfo):
            signature["arg%d_shape" % i] = get_const_tuple(sinfo.values)
        else:
            raise NotImplementedError()

    ret_sinfo = f.ret_struct_info
    if ret_sinfo.shape is not None:
        signature["ret_shape"] = get_const_tuple(ret_sinfo.shape)
    else:
        signature["ret_shape"] = None
    signature["ret_dtype"] = ret_sinfo.dtype

    return signature


def extract_arg_idx(pattern_name, f):
    extract_func = tvm.get_global_func("relax.contrib.extract_arg_idx")
    arg_indices = extract_func(pattern_name, f)
    return {k: int(v) for k, v in arg_indices.items()}


class CodegenResult(Object):
    """The holder for the generated code and required headers."""

    def __init__(self, code, headers):
        self.__init_handle_by_constructor__(ffi.CodegenResult, code, headers)
