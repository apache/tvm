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
"""Source text and canonical one-based IR coordinates."""
import ast
import inspect

from tvm.ir import SourceName, Span

class Source:
    """Source code class for TVMScript.

    It is constructed by source code str or doc AST tree.

    Parameters
    ----------
    source_name : str
        The filename of the file where the source code locates.

    start_line : int
        The first line number of the source code.

    start_column : int
        The first column number of the first line of the source code.

    source : str
        The source code str of source code.

    full_source : str
        The complete source code of the file where the source code locates.
    """

    source_name: str
    start_line: int
    start_column: int
    source: str
    full_source: str

    def __init__(self, program: str | ast.AST):
        if isinstance(program, str):
            self.source_name = "<str>"
            self.start_line = 1
            self.start_column = 0
            self.source = program
            self.full_source = program
            return

        self.source_name = inspect.getsourcefile(program)  # type: ignore
        lines, self.start_line = inspect.getsourcelines(program)  # type: ignore
        if lines:
            self.start_column = len(lines[0]) - len(lines[0].lstrip())
        else:
            self.start_column = 0
        if self.start_column and lines:
            self.source = "\n".join([l[self.start_column :].rstrip() for l in lines])
        else:
            self.source = "".join(lines)
        try:
            # It will cause a problem when running in Jupyter Notebook.
            # `mod` will be <module '__main__'>, which is a built-in module
            # and `getsource` will throw a TypeError
            mod = inspect.getmodule(program)
            if mod:
                self.full_source = inspect.getsource(mod)
            else:
                self.full_source = self.source
        except TypeError:
            # It's a work around for Jupyter problem.
            # Since `findsource` is an internal API of inspect, we just use it
            # as a fallback method.
            src, _ = inspect.findsource(program)  # type: ignore
            self.full_source = "".join(src)

    def as_ast(self) -> ast.AST:
        """Parse the source code into AST.

        Returns
        -------
        res : ast.AST
            The AST of source code.
        """
        return ast.parse(self.source)

    def location(self, node: ast.AST) -> tuple[int, int, int, int]:
        """Return the absolute 1-based source range of an AST node."""
        lineno = getattr(node, "lineno", 1) or 1
        col_offset = getattr(node, "col_offset", self.start_column)
        col_offset = self.start_column if col_offset is None else col_offset
        end_lineno = getattr(node, "end_lineno", lineno) or lineno
        end_col_offset = getattr(node, "end_col_offset", col_offset)
        end_col_offset = col_offset if end_col_offset is None else end_col_offset
        lineno += self.start_line - 1
        end_lineno += self.start_line - 1
        col_offset += self.start_column + 1
        end_col_offset += self.start_column + 1
        return lineno, col_offset, end_lineno, end_col_offset

    def to_span(self, node: ast.AST) -> Span:
        """Convert an AST node to the canonical IR source span."""
        lineno, col_offset, end_lineno, end_col_offset = self.location(node)
        return Span(
            SourceName(self.source_name or "<unknown>"),
            lineno,
            end_lineno,
            col_offset,
            end_col_offset,
        )

