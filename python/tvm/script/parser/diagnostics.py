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
"""Source-located construction errors without replacing their original traceback."""

import ast
import linecache
import traceback

from tvm.error import DiagnosticError


def diagnostic_error(error, compiler):
    """Return a DiagnosticError highlighting the original source operation.

    error is the caught Python/builder exception. compiler supplies the source
    filename and original AST for fallback coordinates. Builder errors carry
    plain location tuples; otherwise Python traceback/SyntaxError coordinates
    identify the innermost original operation. Python 3.11+ supplies precise
    traceback columns, while earlier versions fall back to the source line.
    No frame or IR object is inspected; the caller attaches error as __cause__.
    """
    filename = compiler.filename
    location = getattr(error, "__tvm_script_location__", None)
    if location is not None:
        filename, start, end, column, end_column = location
        column, end_column = column - 1, end_column - 1
    elif isinstance(error, SyntaxError) and error.lineno:
        start = error.lineno
        end = error.end_lineno or start
        column = max((error.offset or 1) - 1, 0)
        end_column = max((error.end_offset or column + 2) - 1, column + 1)
    else:
        frames = [
            frame
            for frame in traceback.extract_tb(error.__traceback__)
            if frame.filename == filename
        ]
        if frames:
            frame = frames[-1]
            start, end = frame.lineno, getattr(frame, "end_lineno", None) or frame.lineno
            column = getattr(frame, "colno", None) or 0
            end_column = getattr(frame, "end_colno", None)
            if getattr(frame, "end_lineno", None) is None:
                # Python before 3.11 supplies only a line. Recover the original
                # statement range from source AST, without an IR span context or
                # executing/reparsing a generated program. Prefer the narrowest
                # statement beginning on that line (a nested body over its def).
                candidates = [
                    node
                    for node in ast.walk(compiler.tree)
                    if isinstance(node, ast.stmt) and node.lineno == start
                ]
                if candidates:
                    node = min(candidates, key=lambda item: item.end_lineno - item.lineno)
                    end, column, end_column = node.end_lineno, node.col_offset, node.end_col_offset
        else:
            node = compiler.tree.body[-1]
            start, end = node.lineno, node.lineno
            column, end_column = node.col_offset, None
    lines = [f"{filename}:{start}: {type(error).__name__}: {error}"]
    for number in range(start, end + 1):
        source = linecache.getline(filename, number).rstrip("\n")
        first = column if number == start else len(source) - len(source.lstrip())
        last = end_column if number == end and end_column is not None else len(source)
        lines.extend(
            (
                f" {number} | {source}",
                " " * (len(str(number)) + 4 + first) + "^" * max(last - first, 1),
            )
        )
    return DiagnosticError("\n".join(lines))


def _format_source_snippet(
    source_lines: list,
    lineno: int,
    col_offset: int,
    end_lineno: int,
    end_col_offset: int,
) -> str:
    """Format a source code snippet with a column/span marker.

    Renders every source line spanned by the diagnostic (``lineno`` through
    ``end_lineno``, inclusive) with a per-line gutter, followed by a caret
    underline covering the offending span. For a single-line span
    (``end_lineno == lineno``) only the columns ``col_offset`` (inclusive)
    through ``end_col_offset`` (exclusive) are underlined; for a multi-line
    span the underline covers the start column to end-of-line on the first
    line, the full text of interior lines, and the start of the final line up
    to ``end_col_offset``.

    Parameters
    ----------
    source_lines : list of str
        Lines of the source code.

    lineno : int
        1-based starting line number in the source.

    col_offset : int
        1-based starting column (inclusive) on ``lineno``.

    end_lineno : int
        1-based ending line number in the source (>= ``lineno``).

    end_col_offset : int
        1-based ending column (exclusive) on ``end_lineno``.

    Returns
    -------
    snippet : str
        Formatted source snippet with caret-marker line(s).
    """
    if end_lineno < lineno:
        end_lineno = lineno

    # Determine the gutter width so that all line numbers line up.
    header_width = len(f" {end_lineno} ")
    no_line_header = " " * header_width

    parts = [f"{no_line_header}|  "]
    for cur_lineno in range(lineno, end_lineno + 1):
        idx = cur_lineno - 1
        if not 0 <= idx < len(source_lines):
            continue
        line_text = source_lines[idx].rstrip("\n")
        line_header = f" {cur_lineno} ".rjust(header_width)

        # Compute the underline span [start_col, stop_col) for this line.
        start_col = col_offset if cur_lineno == lineno else 1
        stop_col = end_col_offset if cur_lineno == end_lineno else len(line_text) + 1

        marker = ""
        for i in range(1, len(line_text) + 1):
            if start_col <= i < stop_col:
                marker += "^"
            else:
                marker += " "
        parts.append(f"{line_header}|  {line_text}")
        parts.append(f"{no_line_header}|  {marker}")

    if len(parts) == 1:
        return ""
    return "\n".join(parts)
