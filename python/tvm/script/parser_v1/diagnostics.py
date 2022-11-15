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
"""Bridge from synr's (the library used for parsing the python AST)
   DiagnosticContext to TVM's diagnostics
"""
from synr import DiagnosticContext, ast

import tvm
from tvm.ir.diagnostics import DiagnosticContext as TVMCtx
from tvm.ir.diagnostics import get_renderer, DiagnosticLevel, Diagnostic


class TVMDiagnosticCtx(DiagnosticContext):
    """TVM diagnostics for synr"""

    diag_ctx: TVMCtx

    def __init__(self) -> None:
        self.diag_ctx = TVMCtx(tvm.IRModule(), get_renderer())
        self.source_name = None

    def to_tvm_span(self, src_name, ast_span: ast.Span) -> tvm.ir.Span:
        return tvm.ir.Span(
            src_name,
            ast_span.start_line,
            ast_span.end_line,
            ast_span.start_column,
            ast_span.end_column,
        )

    def add_source(self, name: str, source: str) -> None:
        src_name = self.diag_ctx.module.source_map.add(name, source)
        self.source_name = src_name

    def emit(self, _level, message, span):
        span = self.to_tvm_span(self.source_name, span)
        self.diag_ctx.emit(Diagnostic(DiagnosticLevel.ERROR, span, message))
        self.diag_ctx.render()  # Raise exception on the first error we hit. TODO remove

    def render(self):
        self.diag_ctx.render()
