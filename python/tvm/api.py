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
"""Functions defined in TVM."""
# pylint: disable=invalid-name,unused-import,redefined-builtin
import tvm._ffi
import tvm.ir
import tvm.tir

from tvm.runtime import convert, const, DataType
from tvm.ir import container as _container, Range
from tvm.tir import decl_buffer, layout, bijective_layout
from tvm.tir import min_value, max_value, indexdiv, indexmod, all, any
from tvm.te import placeholder, compute, scan, extern, var, size_var, thread_axis, reduce_axis


from ._ffi.base import string_types, TVMError
from ._ffi.registry import register_func, get_global_func, extract_ext_funcs

from . import make as _make

int8 = "int8"
int32 = "int32"
float32 = "float32"
handle = "handle"
