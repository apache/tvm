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
"""Doc types for TVMScript Unified Printer"""

import tvm._ffi
from tvm.runtime import Object
from tvm.tir import FloatImm, IntImm

from . import _ffi_api


class Doc(Object):
    """Base class of all Docs"""


class ExprDoc(Object):
    """Base class of all expression Docs"""


_literal_constructors = [
    (str, _ffi_api.LiteralDocStr),
    (float, _ffi_api.LiteralDocFloat),
    (int, _ffi_api.LiteralDocInt),
    (IntImm, _ffi_api.LiteralDocIntImm),
    (FloatImm, _ffi_api.LiteralDocFloatImm)
]


@tvm._ffi.register_object("script.printer.LiteralDoc")
class LiteralDoc(ExprDoc):
    """Doc that represents literal value"""

    def __init__(self, value):
        if value is None:
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocNone)
            return

        for (cls, constructor) in _literal_constructors:
            if isinstance(value, cls):
                break
        else:
            raise TypeError(f"Unsupported type {type(value)} for LiteralDoc")

        self.__init_handle_by_constructor__(constructor, value)

