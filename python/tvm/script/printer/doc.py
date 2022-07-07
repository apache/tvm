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

from . import _ffi_api


class Doc(Object):
    """Base class of all Docs"""


class ExprDoc(Object):
    """Base class of all expression Docs"""


@tvm._ffi.register_object("script.printer.LiteralDoc")
class LiteralDoc(ExprDoc):
    """Doc that represents literal value"""

    def __init__(self, value):
        if value is None:
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocNone)  # type: ignore
        elif isinstance(value, str):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocStr, value)  # type: ignore
        elif isinstance(value, float):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocFloat, value)  # type: ignore
        elif isinstance(value, bool):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocBoolean, value)  # type: ignore
        elif isinstance(value, int):
            self.__init_handle_by_constructor__(_ffi_api.LiteralDocInt, value)  # type: ignore
        else:
            raise TypeError(f"Unsupported type {type(value)} for LiteralDoc")
