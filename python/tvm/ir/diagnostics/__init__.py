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
# pylint: disable=invalid-name
"""
The diagnostic interface to TVM, uses for reporting and rendering
diagnostic information about the compiler. This module exposes
three key abstractions a Diagnostic, the DiagnosticContext,
and the DiagnosticRenderer.
"""
import enum
import tvm._ffi
from . import _ffi_api
from ... import get_global_func, register_func, Object


def get_default_renderer():
    return _ffi_api.DefaultRenderer


def set_default_renderer(render_func):
    def _render_factory():
        return DiagnosticRenderer(render_func)

    register_func("diagnostics.DefaultRenderer", _render_factory, override=True)


class DiagnosticLevel(enum.IntEnum):
    BUG = 10
    ERROR = 20
    WARNING = 30
    NOTE = 40
    HELP = 50


@tvm._ffi.register_object("Diagnostic")
class Diagnostic(Object):
    """A single diagnostic object from TVM."""

    def __init__(self, level, span, message):
        self.__init_handle_by_constructor__(_ffi_api.Diagnostic, level, span, message)


# Register the diagnostic renderer.
@tvm._ffi.register_object("DiagnosticRenderer")
class DiagnosticRenderer(Object):
    def __init__(self, render_func):
        self.__init_handle_by_constructor__(_ffi_api.DiagnosticRenderer, render_func)

    def render(self, ctx):
        return _ffi_api.DiagnosticRendererRender(self, ctx)


# Register the diagnostic context.
@tvm._ffi.register_object("DiagnosticContext")
class DiagnosticContext(Object):
    """
    A diagnostic context which records active errors
    and contains a renderer.
    """

    def __init__(self, module, renderer):
        self.__init_handle_by_constructor__(_ffi_api.DiagnosticContext, module, renderer)

    def emit(self, diagnostic):
        _ffi_api.Emit(self, diagnostic)

    def render(self):
        _ffi_api.DiagnosticContextRender(self)
