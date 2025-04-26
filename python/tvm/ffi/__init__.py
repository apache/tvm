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
"""TVM FFI binding module.

This module binds the TVM FFI C API to python.
This is a standalone module that can be
"""

from .dtype import dtype
from .registry import register_object, get_global_func
from .core import Object, ObjectGeneric, Function
from .convert import convert
from .string import String, Bytes
from .error import register_error

__all__ = [
    "dtype",
    "Object",
    "register_object",
    "get_global_func",
    "Object",
    "ObjectGeneric",
    "Function",
    "convert",
    "String",
    "Bytes",
    "register_error",
]
