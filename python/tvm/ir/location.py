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
"""Source-location metadata and the shared unknown location."""

from tvm_ffi import get_global_func, register_object

from tvm.base import _RUNTIME_ONLY
from tvm.runtime import Object

from . import _ffi_api


@register_object("ir.SourceMap")
class SourceMap(Object):
    def add(self, name, content):
        return get_global_func("SourceMapAdd")(self, name, content)


@register_object("ir.SourceName")
class SourceName(Object):
    """An identifier for a source location.

    Parameters
    ----------
    name : str
        The name of the source.
    """

    def __init__(self, name):
        self.__init_handle_by_constructor__(_ffi_api.SourceName, name)  # type: ignore # pylint: disable=no-member


@register_object("ir.Location")
class Location(Object):
    """Non-null source-location metadata attached to IR nodes.

    IR constructors default to the shared :data:`UNKNOWN_LOC`.
    """

    def __init__(self):
        raise TypeError("Location is a base class; use SourceLoc, UnknownLoc, or CallSiteLoc")


@register_object("ir.UnknownLoc")
class UnknownLoc(Location):
    """The canonical immutable location used when source information is unavailable."""

    def __init__(self):
        self.__init_handle_by_constructor__(_ffi_api.UnknownLoc)


@register_object("ir.SourceLoc")
class SourceLoc(Location):
    """A source range with unchanged frontend coordinate units and endpoints.

    Parameters
    ----------
    source_name : SourceName
        The name of the source.
    start_line : int
        The starting line number.
    start_column : int
        The starting column offset.
    end_line : int
        The ending line number.
    end_column : int
        The ending column offset.
    """

    def __init__(self, source_name, start_line, start_column, end_line, end_column):
        self.__init_handle_by_constructor__(
            _ffi_api.SourceLoc, source_name, start_line, start_column, end_line, end_column
        )


@register_object("ir.CallSiteLoc")
class CallSiteLoc(Location):
    """A callee's location together with the location of its caller."""

    def __init__(self, callee: Location, caller: Location):
        self.__init_handle_by_constructor__(_ffi_api.CallSiteLoc, callee, caller)


# Runtime-only imports expose API definitions without compiler IR constructors.
# Compiler mode always provides the shared, non-null native UnknownLoc wrapper.
UNKNOWN_LOC = None if _RUNTIME_ONLY else UnknownLoc()
