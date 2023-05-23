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
import tvm
from tvm import IRModule, get_global_func, register_func, relay
from tvm.error import DiagnosticError
from tvm.ir.diagnostics import get_renderer, override_renderer
from tvm.relay import SpanCheck
from tvm.relay.transform import AnnotateSpans
from tvm.runtime import Object

DEFAULT_RENDERER = get_renderer()

__TESTING__ = None


def testing_renderer(diag_ctx):
    global __TESTING__
    if __TESTING__ and __TESTING__.mirror:
        DEFAULT_RENDERER.render(diag_ctx)

    if __TESTING__:
        __TESTING__._render(diag_ctx)


class DiagnosticTesting:
    def __init__(self, mirror=False):
        self.mirror = mirror
        self.messages = []

    def __enter__(self):
        global __TESTING__
        __TESTING__ = self
        override_renderer(testing_renderer)
        return self

    def __exit__(self, type, value, traceback):
        global __TESTING__
        __TESTING__ = None
        override_renderer(None)
        if type is DiagnosticError and self.matches:
            return True

    def assert_message(self, in_message):
        self.messages.append(in_message)

    def _render(self, diag_ctx):
        self.matches = False
        for diagnostic in diag_ctx.diagnostics:
            message = diagnostic.message
            for partial_msg in self.messages:
                if partial_msg in message:
                    self.matches = True
