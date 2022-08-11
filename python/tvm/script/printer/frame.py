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
"""
Frame is the core data structure for semantic information when printing
IR graph into TVMScript code.
"""

from typing import Callable, Sequence

from tvm._ffi import register_object
from tvm.runtime import Object
from tvm.script.printer.doc import StmtDoc

from . import _ffi_api


class Frame(Object):
    """
    Frame is the core data structure for semantic information
    when printing IR graph into TVMScript code.

    Frame base class manages a list of callbacks to be executed
    when frame goes out of scope.
    """

    def add_exit_callback(self, callback: Callable[[], None]) -> None:
        """
        Adds a callback function to be executed when frame goes out of scope.

        Parameters
        ----------
        callback : Callable[[], None]
            The callback function.
        """
        _ffi_api.FrameAddExitCallback(self, callback)  # type: ignore # pylint: disable=no-member

    def __enter__(self):
        _ffi_api.FrameEnterWithScope(self)  # type: ignore # pylint: disable=no-member
        return self

    def __exit__(self, *exception_info):
        _ffi_api.FrameExitWithScope(self)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.MetadataFrame")
class MetadataFrame(Frame):
    """
    MetadataFrame contains information like contant parameter array.
    """

    metadata: Sequence[Object]

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.MetadataFrame)  # type: ignore # pylint: disable=no-member


@register_object("script.printer.VarDefFrame")
class VarDefFrame(Frame):
    """
    VarDefFrame contains information about the free variables that needs to
    be defined at the beginning of the printed snippet.
    """

    stmts: Sequence[StmtDoc]

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.VarDefFrame)  # type: ignore # pylint: disable=no-member
