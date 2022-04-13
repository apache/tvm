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
# pylint: disable=invalid-name, import-outside-toplevel, unused-variable
"""Common utility functions in TVM tir"""
import inspect
import re
import tvm
from tvm.ir.diagnostics import override_renderer


CHECK_ERROR_RE = re.compile(r"^.*# check_error: (.+)$")


def check_error(func, rel_lineno):
    """check if TIR script throws error"""
    # Override the default renderer to accumulate errors
    errors = []

    def render(e):
        for d in e.diagnostics:
            errors.append(d)

    override_renderer(render)
    # The diagnostic context throws an exception when it gets an error
    try:
        source_code = inspect.getsource(func)
        source_code = "@T.prim_func\n" + source_code
        from tvm.script import from_source

        # to avoid cyclic import
        from_source(source_code)
    except tvm.error.DiagnosticError as e:
        pass
    assert len(errors) == 1, errors
    for d in errors:
        assert (
            d.span.line - 1 == rel_lineno
        ), f"Expected error to be on line {rel_lineno}, but it was on {d.span.line - 1}"

    error_line = source_code.split("\n")[rel_lineno]
    m = CHECK_ERROR_RE.match(error_line)
    if m:
        expected_error_text = m.group(1)
        errors = [e.message for e in errors]
        assert (
            expected_error_text in errors
        ), f'check_error expects "{expected_error_text} in str(errors): {errors}'
